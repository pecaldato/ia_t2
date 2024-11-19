import streamlit as st
from src.side_bar import SideBar
from src.camera_masker import CameraMasker
from src.image_masker import ImageMasker
from src.video_masker import VideoMasker
from src.model import Model

YOLO_MODEL = "yolo11n-seg.pt" 
OPTIONS_FILE = "resources/options.json"

model = Model(YOLO_MODEL)
side_bar = SideBar(OPTIONS_FILE)
img_masker = ImageMasker(model)
video_masker = VideoMasker(model)
camera_masker = CameraMasker(model)

# Set page configuration
st.set_page_config(page_title="Masker", layout="wide")

# App title
st.title("Masker tool")

st.info("This tool was created to help you segment images using different mask types.")
st.write("With this tool, you can blur objects, background, or create black and white masks.")


side_bar.generateSideBar()

# Create tabs
tabs = st.tabs(["Image", "Video", "Camera"])

# Content for the "Image" tab
with tabs[0]:
    img_masker.generateImageMasker(side_bar.selected_class, side_bar.selected_mask, side_bar.selected_mask_region)

# Content for the "Video" tab
with tabs[1]:
    video_masker.generateImageMasker(side_bar.selected_class, side_bar.selected_mask, side_bar.selected_mask_region)

# Content for the "Contact" tab
with tabs[2]:
    camera_masker.generateImageMasker(side_bar.selected_class, side_bar.selected_mask, side_bar.selected_mask_region)
