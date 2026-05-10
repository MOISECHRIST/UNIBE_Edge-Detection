##============================================
## Author : MEKA Moïse Christian Junior
## Email : moise.meka@students.unibe.ch
##============================================

"""
Deep dive into the Canny Edge Detection algorithm.

This page breaks down the four main steps of the Canny edge detector:
1. Gaussian Smoothing
2. Gradient Computation (Magnitude and Direction)
3. Non-Maximum Suppression (NMS)
4. Hysteresis Thresholding
"""

import streamlit as st
from modules import utils, processing, img_viz
import numpy as np
from scipy.ndimage import gaussian_filter

st.set_page_config(
    page_title="Edge-Detection"
)

image_name = st.sidebar.selectbox(label="Choose your sample image", 
                                  options=utils.sample_images.keys(),
                                  on_change=utils.wipe_everything)
prop_noise = st.sidebar.select_slider(label="Add noises on image", options=np.arange(0, 1.01, 0.01))
sigma = st.sidebar.select_slider(label="The value of sigma ($\sigma$)", options=np.arange(0.01,5.01,0.01))
filter_name = st.sidebar.selectbox(
        label="Choose your filter", 
        options=["Sobel", "Prewitt", "Robert"]
    )
high_thresold = st.sidebar.select_slider(label="Define the thresold", options=np.arange(0.01,5.01,0.01))

st.markdown("## Canny Edge Detector")

st.divider()

st.markdown("""
            The Canny detector (1986) is the gold standard. It chains four steps:
            """)

st.image("images/canny_steps.png")

raw_image = None
if image_name == "Upload your image":
    uploaded_file = st.sidebar.file_uploader("Upload an image", type=utils.LIST_EXTENSIONS, 
                                             accept_multiple_files=False)
    if uploaded_file:
        raw_image = utils.read_image(uploaded_file)
elif utils.sample_images[image_name]:
    raw_image = utils.read_image(utils.sample_images[image_name])


if raw_image is not None:
    processed_image = utils.rgb2grayscale(raw_image)
    processed_image = utils.normalize_image(processed_image)
    if prop_noise>0:
        processed_image = utils.add_noise(image=processed_image, prop=prop_noise)

    with st.expander("1- Apply Gaussian smoothing :"):
        
        blured_image = gaussian_filter(processed_image, sigma=sigma)
        st.pyplot(img_viz.plot_step_img_process(processed_image, blured_image,
                                    titles=["Original Image", "Blured Image"],
                                    cmaps=["grey","grey"]))

    with st.expander("2- Apply Gradient (magnitude and direction) :"):
        _, _,magnitude, direction = processing.gradient_of_gaussian(processed_image, 
                                                                    sigma=sigma, 
                                                                    filtername=filter_name.lower())
        
        st.pyplot(img_viz.plot_gradients_img(processed_image, magnitude, direction, 
                                            titles=["Original Image", 'Gradient magnitude |∇I|', 'Direction $\Theta$'],
                                            cmaps=["grey", "hot", "hsv"], figsize=(25,20)))

    with st.expander("3- Apply Non-Maximum Suppression"):
        st.markdown("""
        - **Problem:** the gradient magnitude map has thick ridges around edges. We want edges that are exactly one pixel wide.
    - **Idea:** at each pixel, look at the two neighbours along the gradient direction. If this pixel is not
    the local maximum, suppress it (set to 0).
        """)

        NMS_result = processing.non_maximum_suppression(magnitude=magnitude, angle=direction)
        st.pyplot(img_viz.plot_step_img_process(processed_image, NMS_result,
                                    titles=["Original Image", "Non-Maximum Suppression"],
                                    cmaps=["grey","hot"]))

    with st.expander("4- Apply Hysteresis Thresholding"):
        st.markdown("""
    After NMS we still have many pixels. A single threshold would either miss weak edges or keep
    too much noise.
    Solution: use two thresholds klo < khi :
    - **:green[Pixels ≥ khi]:** strong edges (always kept).
    - **:red[Pixels ∈ \[klo , khi )]:** weak edges ( kept only if 8-connected to a strong edge pixel).
    - **Pixels < klo:** definitely noise (suppressed).

    Rule of thumb (Canny’s own suggestion): khi /klo ≈ 2
    """)
        thresgolding_result = processing.show_hyteresis_thresholding(NMS_result, high_threshold=high_thresold)
        st.pyplot(img_viz.plot_step_img_process(processed_image, thresgolding_result,
                                    titles=["Original Image", f"Hysteresis Thresholding (ht={high_thresold:.3f})"],
                                    cmaps=["grey",None]))
    with st.expander("Canny final result"):
        canny_result = processing.apply_canny(processed_image, sigma=sigma, high_threshold=high_thresold)
        st.pyplot(img_viz.plot_step_img_process(processed_image, canny_result,
                                    titles=["Original Image", f"Canny (ht={high_thresold:.3f} $\sigma$={sigma:.3f})"],
                                    cmaps=["grey",'grey']))
else:
    st.info("Please select a sample image or upload your own to begin.")
    
col1, col2 = st.columns([3, 2])
with col1:
    st.page_link("pages/2_apply_gradient.py", label="**Edge detection using gradient**")

with col2:
    st.page_link("pages/4_apply_laplacian.py", label="**Edge detection using Laplacian based method**")