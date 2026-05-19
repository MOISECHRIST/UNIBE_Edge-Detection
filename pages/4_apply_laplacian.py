##============================================
## Author : MEKA Moïse Christian Junior
## Email : moise.meka@students.unibe.ch
##============================================

"""
Implementation and visualization of Laplacian-based edge detection.

This page explores second-order derivative filters, including the simple 
Laplacian and the Laplacian of Gaussian (LoG), comparing them with the 
Canny edge detector.
"""

import streamlit as st
from modules import utils, processing, img_viz
import numpy as np
from scipy.ndimage import convolve

st.set_page_config(
    page_title="Edge-Detection"
)

image_name = st.sidebar.selectbox(label="Choose your sample image", 
                                  options=utils.sample_images.keys(),
                                  on_change=utils.wipe_everything)
prop_noise = st.sidebar.select_slider(label="Add noises on image", options=np.arange(0, 1.01, 0.01))
sigma = st.sidebar.select_slider(label="The value of sigma ($\sigma$)", options=np.arange(0.01,5.01,0.01), value=0.5)
high_thresold = st.sidebar.select_slider(label="Define the thresold", options=np.arange(0.01,5.01,0.01), value=0.5)

st.markdown("## Edge detection using Laplacian based method")

st.divider()

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
    tab1, tab2, tab3 = st.tabs(["Apply Laplacian","Apply Laplacian-of-Gaussian (LoG)", "Canny vs Laplacian vs LoG"])
    
    with tab1:
        st.markdown("""
        **Idea behind laplacian :** instead of peaks in the first derivative, look for zero crossings in the second derivative.
        """)
        c1, c2 = st.columns(2)
        with c1:
            st.text("The Laplacian sums the second derivatives in x and y:")
            st.latex(r"\nabla^2 I = \frac{\partial^2 I}{\partial x^2} + \frac{\partial^2 I}{\partial y^2}")
        with c2:
            st.text("To obtain this second derivative of an image, we use the kernel below:")
            st.image("images/laplacian_filter.png", width=150)
        
        laplacian_output = processing.apply_laplacian(processed_image)
        if prop_noise>0:
            st.info("Applied alone it is very noisy, so we combine it with Gaussian smoothing. That is the goal of Laplacian of Gaussian")
        st.pyplot(img_viz.plot_step_img_process(processed_image, laplacian_output,
                                    titles=["Original Image", "Laplacian"],
                                    cmaps=["grey","hsv"]))
        

    with tab2:
        st.markdown("""
        Using the kernel below, we can apply the Laplacian of Gaussian to an image. This allows us to detect edges while reducing noise.
        """)
        st.latex(r"LoG(x, y) = -\frac{1}{\pi\sigma^4}\left(1 - \frac{x^2 + y^2}{2\sigma^2}\right)e^{-\frac{x^2+y^2}{2\sigma^2}}")
        st.info("""
- Small $\sigma$ (e.g., 0.5): Preserves fine details and sharp edges but leaves more noise in the image.
- Large $\sigma$ (e.g., 2.0+): Removes significant noise and detects only "strong," large-scale boundaries 
    (like the outline of a head vs. the individual strands of hair).""")
        LoG_kernel = processing.log_kernel(sigma=sigma)

        LoG_output = convolve(processed_image, LoG_kernel)
        st.pyplot(img_viz.plot_step_img_process(processed_image, LoG_output,
                                    titles=["Original Image", f"LoG $\sigma$={sigma:.3f}"],
                                    cmaps=["grey","hsv"]))

    with tab3:
        canny_result = processing.apply_canny(processed_image, sigma=sigma, high_threshold=high_thresold)
        
        st.pyplot(img_viz.plot_gradients_img(canny_result, laplacian_output, LoG_output,
                                             titles=[f"Canny (ht={high_thresold:.3f} $\sigma$={sigma:.3f})",
                                                     "Laplacian",f"LoG $\sigma$={sigma:.3f}"],
                                                     cmaps=["grey","hsv","hsv"]))
    
else:
    st.info("Please select a sample image or upload your own to begin.")




col1, _ = st.columns([3, 2])
with col1:
    st.page_link("pages/3_apply_canny.py", label="**Canny Edge Detector**")