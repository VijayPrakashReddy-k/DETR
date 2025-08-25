# ================== Streamlit-friendly DETR + Detectron2 (CPU) ==================
# - Default: DETR ResNet50 (light)
# - Optional: Detectron2 Faster R-CNN R50-FPN (CPU)
# - Thread caps + headless matplotlib to avoid OOM/crashes
# ================================================================================

import os
# Hard caps to prevent runaway threading / RAM
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import io
import gc
import base64
import itertools
import numpy as np
import seaborn as sns
from PIL import Image
import streamlit as st

import matplotlib
matplotlib.use("Agg")  # headless backend for Streamlit
import matplotlib.pyplot as plt

import torch
import torchvision.transforms as T

from detr import DETRdemo  # your local module (Streamlit_Deployment/detr.py)

# Torch runtime knobs
torch.set_grad_enabled(False)
torch.set_num_threads(1)

# --- UI tweaks ---
st.set_option("deprecation.showPyplotGlobalUse", False)
palette = itertools.cycle(sns.color_palette())

# DETR's 91-class list (with 'N/A' holes)
CLASSES_DETR_91 = [
    'N/A', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
    'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack',
    'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass',
    'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
    'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A',
    'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]

# Correct 80 COCO class names for Detectron2 predictions (contiguous 0..79)
COCO_80 = [
    'person','bicycle','car','motorcycle','airplane','bus','train','truck','boat',
    'traffic light','fire hydrant','stop sign','parking meter','bench','bird','cat',
    'dog','horse','sheep','cow','elephant','bear','zebra','giraffe','backpack','umbrella',
    'handbag','tie','suitcase','frisbee','skis','snowboard','sports ball','kite','baseball bat',
    'baseball glove','skateboard','surfboard','tennis racket','bottle','wine glass','cup','fork',
    'knife','spoon','bowl','banana','apple','sandwich','orange','broccoli','carrot','hot dog',
    'pizza','donut','cake','chair','couch','potted plant','bed','dining table','toilet','tv',
    'laptop','mouse','remote','keyboard','cell phone','microwave','oven','toaster','sink',
    'refrigerator','book','clock','vase','scissors','teddy bear','hair drier','toothbrush'
]

# ---- background image helper (cached) ----
@st.cache_data(show_spinner=False)
def get_base64_of_file(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

def set_page_bg(png_path: str) -> None:
    try:
        bin_str = get_base64_of_file(png_path)
        st.markdown(
            f"""
            <style>
            .stApp {{
                background-image: url("data:image/png;base64,{bin_str}");
                background-size: cover;
            }}
            </style>
            """,
            unsafe_allow_html=True,
        )
    except Exception:
        pass  # background is optional

set_page_bg("Streamlit_Deployment/background.jpeg")

# ---- transforms ----
# Keep short/long side ~800 to limit memory usage
transform = T.Compose([
    T.Resize(800),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# ---- box helpers ----
def box_cxcywh_to_xyxy(x: torch.Tensor) -> torch.Tensor:
    x_c, y_c, w, h = x.unbind(1)
    return torch.stack([x_c - 0.5 * w, y_c - 0.5 * h, x_c + 0.5 * w, y_c + 0.5 * h], dim=1)

def rescale_bboxes(out_bbox: torch.Tensor, size) -> torch.Tensor:
    img_w, img_h = size
    scale = torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return box_cxcywh_to_xyxy(out_bbox) * scale

# ---- drawing helpers ----
# def draw_boxes_matplotlib(pil_img: Image.Image, boxes, labels, scores, color_cycle=COLORS, title=None):
#     fig, ax = plt.subplots(figsize=(12, 8))
#     ax.imshow(pil_img)
#     for (xmin, ymin, xmax, ymax), name, sc, c in zip(boxes, labels, scores, color_cycle * 100):
#         ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
#                                    fill=False, color=c, linewidth=2.5))
#         ax.text(xmin, ymin, f"{name}: {sc:.2f}", fontsize=13,
#                 bbox=dict(facecolor="yellow", alpha=0.5))
#     ax.axis("off")
#     if title:
#         ax.set_title(title)
#     fig.tight_layout(pad=0.0)
#     return fig

def draw_boxes_matplotlib(pil_img, boxes, labels, scores, color_cycle=None, title=None):
    # local default palette if none is provided
    if color_cycle is None:
        color_cycle = [
            (0.000, 0.447, 0.741),
            (0.850, 0.325, 0.098),
            (0.929, 0.694, 0.125),
            (0.494, 0.184, 0.556),
            (0.466, 0.674, 0.188),
            (0.301, 0.745, 0.933),
        ]

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(pil_img)

    # repeat colors to cover all boxes
    colors = color_cycle * 100
    for (xmin, ymin, xmax, ymax), name, sc, c in zip(boxes, labels, scores, colors):
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                   fill=False, color=c, linewidth=2.5))
        ax.text(xmin, ymin, f"{name}: {sc:.2f}", fontsize=13,
                bbox=dict(facecolor="yellow", alpha=0.5))

    ax.axis("off")
    if title:
        ax.set_title(title)
    fig.tight_layout(pad=0.0)
    return fig

# ---- DETR pipeline ----
@st.cache_resource(show_spinner=False)
def load_detr_demo():
    m = DETRdemo(num_classes=91)
    state_dict = torch.hub.load_state_dict_from_url(
        url="https://dl.fbaipublicfiles.com/detr/detr_demo-da2a99e9.pth",
        map_location="cpu", check_hash=True
    )
    m.load_state_dict(state_dict)
    m.eval()
    return m

def infer_detr(pil_img: Image.Image, model, prob_thresh: float):
    img_rgb = pil_img.convert("RGB")
    tensor = transform(img_rgb).unsqueeze(0)
    # Bound absolute size to avoid huge memory spikes
    assert tensor.shape[-2] <= 1600 and tensor.shape[-1] <= 1600, \
        "Image too large (limit 1600x1600). Please upload a smaller image."

    with torch.inference_mode():
        outputs = model(tensor)

    probas = outputs["pred_logits"].softmax(-1)[0, :, :-1]
    keep = probas.max(-1).values > prob_thresh
    boxes = rescale_bboxes(outputs["pred_boxes"][0, keep], img_rgb.size).tolist()
    scores = probas[keep].max(dim=1).values.cpu().numpy().tolist()
    class_ids = probas[keep].argmax(dim=1).cpu().numpy().tolist()
    labels = [CLASSES_DETR_91[i] for i in class_ids]

    del outputs
    gc.collect()
    return img_rgb, boxes, labels, scores

# ---- Detectron2 pipeline (lazy imports) ----
@st.cache_resource(show_spinner=True)
def load_d2_predictor(low_thresh: float = 0.30):
    # Import Detectron2 pieces only when needed
    from detectron2.config import get_cfg
    from detectron2 import model_zoo
    from detectron2.engine import DefaultPredictor

    cfg = get_cfg()
    cfg.MODEL.DEVICE = "cpu"
    # Lighter model than Mask R-CNN + easier to draw ourselves
    # You can switch to mask_rcnn_R_50_FPN_1x.yaml if you want masks (heavier)
    model_yml = "COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml"
    cfg.merge_from_file(model_zoo.get_config_file(model_yml))
    cfg.DATALOADER.NUM_WORKERS = 0
    cfg.INPUT.FORMAT = "BGR"
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_yml)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = low_thresh  # we will post-filter anyway
    predictor = DefaultPredictor(cfg)
    return predictor

def _pil_to_bgr_np(pil_img: Image.Image, max_side: int = 800):
    arr = np.array(pil_img.convert("RGB"))  # HWC, RGB
    h, w = arr.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    if scale < 1.0:
        new_w, new_h = int(w * scale), int(h * scale)
        # Use PIL resize for quality; then convert to numpy/BGR
        arr = np.array(Image.fromarray(arr).resize((new_w, new_h), Image.BILINEAR))
    bgr = arr[:, :, ::-1]  # RGB -> BGR
    return bgr

def infer_d2(pil_img: Image.Image, predictor, prob_thresh: float):
    img_bgr = _pil_to_bgr_np(pil_img, max_side=800)
    with torch.inference_mode():
        outputs = predictor(img_bgr)

    inst = outputs["instances"].to("cpu")
    if len(inst) == 0:
        return pil_img, [], [], []

    boxes_np = inst.pred_boxes.tensor.numpy()
    cls_np = inst.pred_classes.numpy()
    scores_np = inst.scores.numpy()

    # post-filter by UI threshold
    keep = scores_np >= prob_thresh
    boxes = boxes_np[keep].tolist()
    cls_ids = cls_np[keep].tolist()
    scores = scores_np[keep].tolist()
    labels = [COCO_80[i] if 0 <= i < len(COCO_80) else f"id_{i}" for i in cls_ids]

    # clean up
    del outputs
    gc.collect()
    return pil_img, boxes, labels, scores

# ---- sidebar ----
def sidebar_controls():
    with st.sidebar:
        st.subheader("Settings")
        backend = st.selectbox(
            "Backend",
            ["DETR (ResNet50, light)", "Detectron2 Faster R-CNN R50-FPN (CPU)"],
            index=0,
        )
        thresh = st.slider("Confidence threshold", 0.0, 1.0, 0.80, 0.05)
        st.caption("Lower threshold → more boxes (and more false positives).")
    return backend, thresh

# ---- main app ----
def main():
    st.title("Object Detection (DETR / Detectron2)")
    st.caption("Optimized for CPU-only Streamlit deployment. Panoptic is disabled to keep memory low.")

    backend, prob_thresh = sidebar_controls()
    uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if not uploaded:
        st.info("Upload a JPG/PNG to run detection.")
        return

    try:
        image = Image.open(uploaded).convert("RGB")
    except Exception:
        st.error("Could not read the image. Please try another file.")
        return

    st.image(image, caption="Uploaded image", use_column_width=True)

    try:
        if backend.startswith("DETR"):
            model = load_detr_demo()
            img_rgb, boxes, labels, scores = infer_detr(image, model, prob_thresh)
            title = "DETR (ResNet50)"
        else:
            try:
                predictor = load_d2_predictor(low_thresh=0.30)
            except Exception as e:
                st.error("Detectron2 failed to load. Verify the vendored wheel path and opencv-python-headless.")
                st.exception(e)
                return
            img_rgb, boxes, labels, scores = infer_d2(image, predictor, prob_thresh)
            title = "Detectron2 Faster R-CNN R50-FPN (CPU)"
    except AssertionError as e:
        st.error(str(e))
        return
    except RuntimeError as e:
        st.error(f"Inference failed: {e}")
        return

    if len(boxes) == 0:
        st.warning("No detections above the selected threshold.")
        return

    fig = draw_boxes_matplotlib(img_rgb, boxes, labels, scores, title=title)
    st.pyplot(fig)

    with st.expander("Show raw detections"):
        import pandas as pd
        df = pd.DataFrame({
            "label": labels,
            "confidence": [float(x) for x in scores],
            "xmin": [b[0] for b in boxes],
            "ymin": [b[1] for b in boxes],
            "xmax": [b[2] for b in boxes],
            "ymax": [b[3] for b in boxes],
        }).sort_values("confidence", ascending=False).reset_index(drop=True)
        st.dataframe(df, use_container_width=True)

if __name__ == "__main__":
    main()

# import io
# import math
# import torch
# import base64
# import itertools
# import numpy as np
# import seaborn as sns
# from PIL import Image
# import streamlit as st
# from detr import DETRdemo
# from copy import deepcopy
# import matplotlib.pyplot as plt
# import torchvision.transforms as T
# torch.set_grad_enabled(False);
# from detectron2.config import get_cfg
# from detectron2.data import MetadataCatalog
# from detectron2.utils.visualizer import Visualizer

# palette = itertools.cycle(sns.color_palette())
# st.set_option('deprecation.showPyplotGlobalUse', False)

# # Define constants for COCO classes and colors
# CLASSES = [
#     'N/A', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
#     'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A',
#     'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
#     'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack',
#     'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
#     'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
#     'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass',
#     'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
#     'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
#     'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A',
#     'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
#     'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A',
#     'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
#     'toothbrush'
# ]

# COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
#           [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]

# # Define Streamlit configuration options
# st.set_option('deprecation.showPyplotGlobalUse', False)

# @st.cache(allow_output_mutation=True)
# def get_base64_of_bin_file(bin_file):
#     with open(bin_file, 'rb') as f:
#         data = f.read()
#     return base64.b64encode(data).decode()

# def set_png_as_page_bg(png_file):
#     bin_str = get_base64_of_bin_file(png_file)
#     page_bg_img = '''
#     <style>
#     .stApp {
#     background-image: url("data:image/png;base64,%s");
#     background-size: cover;
#     }
#     </style>
#     ''' % bin_str

#     st.markdown(page_bg_img, unsafe_allow_html=True)
#     return

# set_png_as_page_bg('Streamlit_Deployment/background.jpeg')

# # Define image preprocessing transform
# transform = T.Compose([
#     T.Resize(800),
#     T.ToTensor(),
#     T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
# ])

# # for output bounding box post-processing
# def box_cxcywh_to_xyxy(x):
#     x_c, y_c, w, h = x.unbind(1)
#     b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
#          (x_c + 0.5 * w), (y_c + 0.5 * h)]
#     return torch.stack(b, dim=1)

# def rescale_bboxes(out_bbox, size):
#     img_w, img_h = size
#     b = box_cxcywh_to_xyxy(out_bbox)
#     b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
#     return b

# # Define a function to perform object detection
# def perform_object_detection(image, detr, transform, probability_threshold=0.70):
#     try:
#         # mean-std normalize the input image (batch-size: 1)
#         img = transform(image).unsqueeze(0)

#         # Check if the image size exceeds the supported limit
#         assert img.shape[-2] <= 1600 and img.shape[-1] <= 1600, 'Image size exceeds the supported limit (1600x1600 pixels)'

#         # Propagate through the model
#         outputs = detr(img)

#         # keep only predictions with the specified confidence threshold
#         probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
#         keep = probas.max(-1).values > probability_threshold

#         # convert boxes from [0; 1] to image scales
#         bboxes_scaled = rescale_bboxes(outputs['pred_boxes'][0, keep], image.size)

#         # Calculate and return the scores
#         scores = probas[keep]

#         return scores, bboxes_scaled, None  # No error, so set error_message to None

#     except AssertionError as e:
#         return None, None, str(e)
    
# # Define your plot_results function
# def plot_results(pil_img, prob, boxes):
#     fig, ax = plt.subplots(figsize=(16, 10))
#     ax.imshow(pil_img)
#     for p, (xmin, ymin, xmax, ymax), c in zip(prob, boxes.tolist(), COLORS * 100):
#         ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
#                                    fill=False, color=c, linewidth=3))
#         cl = p.argmax()
#         text = f'{CLASSES[cl]}: {p[cl]:0.2f}'
#         ax.text(xmin, ymin, text, fontsize=15,
#                 bbox=dict(facecolor='yellow', alpha=0.5))
#     ax.axis('off')

# # Define a function to load the model and perform inference
# @st.cache_data()
# def load_model():
#     detr = DETRdemo(num_classes=91)
#     state_dict = torch.hub.load_state_dict_from_url(
#         url='https://dl.fbaipublicfiles.com/detr/detr_demo-da2a99e9.pth',
#         map_location='cpu', check_hash=True)
#     detr.load_state_dict(state_dict)
#     detr.eval()
#     return detr

# @st.cache_data()
# def panoptic_load_model():
#     # Load the Panoptic DETR model with a postprocessor
#     model, postprocessor = torch.hub.load(
#         'facebookresearch/detr',
#         'detr_resnet101_panoptic',
#         pretrained=True,
#         return_postprocessor=True,
#         num_classes=250
#     )

#     # Set the model to evaluation mode
#     model.eval()
#     return model, postprocessor

# def side_show():
#     """Shows the sidebar components for the template and returns user inputs as dict."""
#     with st.sidebar:
#         st.write("#### Predictions with the specified confidence threshold")
#         threshold = st.slider("Threshold", value=0.80, min_value=0.00, max_value=1.00, step=0.05)
#     return threshold

# def main():
#     st.title("DETR Object Detection")
#     uploaded_image = st.file_uploader("Upload an image for Object Detection", type=["jpg", "png", "jpeg"])

#      # Set the desired probability threshold
#     probability_threshold = side_show()
    
#     if uploaded_image is not None:
#         image = Image.open(uploaded_image)

#         # Perform object detection or any other processing on the 'image' variable
#         st.image(image, caption='Uploaded Image', use_column_width=True)

#         detr = load_model()

#         # Perform object detection or handle the error
#         scores, bboxes, error_message = perform_object_detection(image, detr, transform, probability_threshold)

#         if scores is not None and bboxes is not None:
#             # Create a Streamlit figure and display the results
#             st.pyplot(plot_results(image, scores, bboxes))

#             ## Comment from here - start
#             # Create a button to load the Panoptic DETR model
#             if st.button("Panoptic Segmentation"):

#                 st.write("Loading Panoptic DETR model...")

#                 model, postprocessor = panoptic_load_model()

#                 # mean-std normalize the input image (batch-size: 1)
#                 img = transform(image).unsqueeze(0)

#                 # Perform inference
#                 outputs = model(img)
#                 scores = outputs["pred_logits"].softmax(-1)[..., :-1].max(-1)[0]
#                 keep = scores > 0.85

#                 # Define the number of columns for displaying masks
#                 ncols = 5

#                 # Calculate the number of rows required based on the number of masks
#                 num_masks = keep.sum().item()
#                 num_rows = math.ceil(num_masks / ncols)

#                 # Define a title for the plot
#                 st.subheader("High Confidence Masks")

#                 # Iterate over the masks and display them
#                 for i, mask in enumerate(outputs["pred_masks"][keep]):
#                     plt.subplot(num_rows, ncols, i + 1)
#                     plt.imshow(mask.cpu().numpy(), cmap="cividis")
#                     plt.axis('off')

#                 # Show the plot using Streamlit
#                 st.pyplot()

#                 # Post-process the results
#                 result = postprocessor(outputs, torch.as_tensor(img.shape[-2:]).unsqueeze(0))[0]

#                 # Decode the segmentation and color each mask individually
#                 panoptic_seg = Image.open(io.BytesIO(result['png_string']))
#                 panoptic_seg = np.array(panoptic_seg, dtype=np.uint8).copy()

#                 def rgb2id(color):
#                     if isinstance(color, np.ndarray) and len(color.shape) == 3:
#                         if color.dtype == np.uint8:
#                             color = color.astype(np.int32)
#                         return color[:, :, 0] + 256 * color[:, :, 1] + 256 * 256 * color[:, :, 2]
#                     return int(color[0] + 256 * color[1] + 256 * 256 * color[2])

#                 # We retrieve the ids corresponding to each mask
#                 panoptic_seg_id = rgb2id(panoptic_seg)

#                 # Finally, we color each mask individually
#                 panoptic_seg[:, :, :] = 0
#                 for id in range(panoptic_seg_id.max() + 1):
#                     panoptic_seg[panoptic_seg_id == id] = np.asarray(next(palette)) * 255

#                 st.subheader("Panoptic Segmentation")

#                 # Display the segmented image using Streamlit
#                 st.image(panoptic_seg, caption='Panoptic Segmentation Result', use_column_width=True)

#                 # We extract the segments info and the panoptic result from DETR's prediction
#                 segments_info = deepcopy(result["segments_info"])       

#                 # Panoptic predictions are stored in a special format png
#                 panoptic_seg = Image.open(io.BytesIO(result['png_string']))
#                 final_w, final_h = panoptic_seg.size
#                 # We convert the png into an segment id map
#                 panoptic_seg = np.array(panoptic_seg, dtype=np.uint8)
#                 panoptic_seg = torch.from_numpy(rgb2id(panoptic_seg))

#                 # Detectron2 uses a different numbering of coco classes, here we convert the class ids accordingly
#                 meta = MetadataCatalog.get("coco_2017_val_panoptic_separated")
#                 for i in range(len(segments_info)):
#                     c = segments_info[i]["category_id"]
#                     segments_info[i]["category_id"] = meta.thing_dataset_id_to_contiguous_id[c] if segments_info[i]["isthing"] else meta.stuff_dataset_id_to_contiguous_id[c]

#                 # Finally we visualize the prediction
#                 v = Visualizer(np.array(image.copy().resize((final_w, final_h)))[:, :, ::-1], meta, scale=1.0)
#                 v._default_font_size = 20
#                 v = v.draw_panoptic_seg_predictions(panoptic_seg, segments_info, area_threshold=0)

#                 st.subheader("Panoptic Segmentation with Labels using Detectron2")

#                 # Display the image using Streamlit
#                 st.image(v.get_image(), caption='Panoptic Detection with Class Labels', use_column_width=True)
#                 ## Till Here - Stop
#         else:
#             # Display the error message to the user
#             st.error("Error: " + error_message)

# if __name__ == "__main__":
#     main()



