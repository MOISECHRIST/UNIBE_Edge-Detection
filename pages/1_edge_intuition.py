##============================================
## Author : MEKA Moïse Christian Junior
## Email : moise.meka@students.unibe.ch
##============================================

from modules import utils
import streamlit as st
from streamlit_image_coordinates import streamlit_image_coordinates
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
from skimage.measure import profile_line

st.set_page_config(
    page_title="Edge-Detection"
)

st.sidebar.header("First Intuition")

sample_images = {
    "cameraman": "sample_data/cameraman.png",
    "cat1": "sample_data/cat3.png",
    "checkerboard" : "sample_data/checkerboard.png",
    "circles": "sample_data/circles.jpg",
    "cat2": "sample_data/hugo.jpg",
    "owl": "sample_data/owl.jpg",
    "shapes" : "sample_data/shapes.jpg",
    "Upload your image": None
}

image_name = st.sidebar.selectbox(label="Choose your sample image", options=sample_images.keys())

raw_image = None
if image_name == "Upload your image":
    uploaded_file = st.sidebar.file_uploader("Upload an image", type=utils.LIST_EXTENSIONS, 
                                             accept_multiple_files=False)
    if uploaded_file:
        raw_image = utils.read_image(uploaded_file)
elif sample_images[image_name]:
    raw_image = utils.read_image(sample_images[image_name])


if raw_image is not None:
    processed_image = utils.rgb2grayscale(raw_image)
    processed_image = utils.normalize_image(processed_image)
    
    display_image = (processed_image * 255).astype(np.uint8)
    pil_image = Image.fromarray(display_image)

    if "clicked_points" not in st.session_state:
        st.session_state.clicked_points = []

    col1, col2 = st.columns([3,3])
    
    with col1:
        st.write("Click two points on the image to draw a profile line.")
        value = streamlit_image_coordinates(pil_image, key="img_coords", height =400, width=400)

    with col2:
        if value is not None:
            point = (value['x'], value['y'])
            
            if point not in st.session_state.clicked_points:
                st.session_state.clicked_points.append(point)
            
            if len(st.session_state.clicked_points) > 2:
                st.session_state.clicked_points = [point]

        st.write(f"Points selected: {len(st.session_state.clicked_points)}/2")

        if len(st.session_state.clicked_points) == 2:
            p1 = st.session_state.clicked_points[0]
            p2 = st.session_state.clicked_points[1]
            
            start = (p1[1], p1[0]) 
            end = (p2[1], p2[0])
            
            intensity_profile = profile_line(processed_image, start, end, mode='constant')
            
            derivative = np.diff(intensity_profile)

            fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(6, 6))
            
            ax[0].plot(intensity_profile, 'b-', lw=1.5)
            ax[0].fill_between(range(len(intensity_profile)), intensity_profile, alpha=0.15)
            ax[0].set_title(f"Intensity Profile f(x) between {p1} and {p2}")
            ax[0].set_ylabel("f(x)")
            ax[0].grid(True, linestyle='--', alpha=0.6)
            
            ax[1].plot(derivative, 'r-', lw=1.5)
            ax[1].set_title(f"Derivative df/dx between {p1} and {p2}")
            ax[1].set_ylabel("df(x)/dx")
            ax[1].set_xlabel("Distance along line (pixels)")
            ax[1].grid(True, linestyle='--', alpha=0.6)
            
            plt.tight_layout()
            st.pyplot(fig=fig)

            if st.button("Clear Points"):
                st.session_state.clicked_points = []
                st.rerun()
else:
    st.info("Please select a sample image or upload your own to begin.")

st.divider()

col1, col2 = st.columns([3, 2])
with col1:
    st.page_link("app.py", label="**Quick Introduction**")

with col2:
    st.page_link("app.py", label="**Edge detection using dradient**")