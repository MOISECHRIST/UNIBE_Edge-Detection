import streamlit as st
from modules import utils, processing, img_viz
import numpy as np
from scipy.ndimage import convolve

st.set_page_config(
    page_title="Edge-Detection"
)
noise_solution = None
image_name = st.sidebar.selectbox(label="Choose your sample image", options=utils.sample_images.keys())

filters_list = ["Sobel", "Prewitt", "Robert"]
filer_name = st.sidebar.selectbox(label="Choose your filter", options=filters_list)

prop_noise = st.sidebar.select_slider(label="Add noises on image", options=np.arange(0,1.01,0.01))

st.markdown("## Edge Detection using Gradient")

st.divider()

with st.expander("Click to see how to use gradients on an image"):
    st.markdown("""
        As we saw earlier, it's possible to use the derivative to detect the variation in pixel intensity along a line.
        Based on this principle, we can consider applying it to an image to identify its edges.
        Since images are 2D objects, we need to calculate this derivative along the X and Y axes (gradient).""")
                
    st.latex(r"\nabla I = \begin{pmatrix} \partial I/\partial x \\ \partial I/\partial y \end{pmatrix}")
    st.markdown("""
        Next, we need to identify:
        - **The magnitude :** how strong is the edge? """)
    st.latex(r"|\nabla I| = \sqrt{I_x^2 + I_y^2}")
    st.markdown("- **The direction :** which way does the edge face?")
    st.latex(r"\theta = \arctan(I_y / I_x)")
    st.markdown("""To do this, we will use convolution with specific kernels :""")

    st.image("images/kernel_img.png")

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
        st.warning("As you can see, this method is very sensitive to noise.")
        processed_image = utils.add_noise(image=processed_image, prop=prop_noise)
        noise_solution = st.sidebar.radio(label="Solve the noise problem using", 
                         options=["nothing","Gaussian smoothing + Gradient", "Derivative of Gaussian (DoG)"])

    if noise_solution is None or noise_solution=='nothing' or noise_solution == "Gaussian smoothing + Gradient":
        if noise_solution == "Gaussian smoothing + Gradient":
            
            sigma = st.sidebar.select_slider(label="The value of sigma", options=np.arange(0.01,5.01,0.01))
            Gx, Gy,magnitude, direction = processing.gradient_of_gaussian(processed_image, sigma=sigma, filtername=filer_name.lower())
        
        else:
            Gx = processing.compute_gradient(processed_image, 
                                                        on="X",
                                                        filtername=filer_name.lower())
            
            Gy = processing.compute_gradient(processed_image, 
                                                        on="Y",
                                                        filtername=filer_name.lower())
            
            magnitude, direction = processing.magnitude_direction(Gradient_x=Gx, Gradient_y=Gy, out="both")
        
        
    else:
        sigma = st.sidebar.select_slider(label="The value of sigma", options=np.arange(0.01,5.01,0.01))
        kernel_Gx = processing.dog_kernel(sigma=sigma, on='X')
        kernel_Gy = processing.dog_kernel(sigma=sigma, on='Y')
        Gx = convolve(processed_image, kernel_Gx)
        Gy = convolve(processed_image, kernel_Gy)
        magnitude, direction = processing.magnitude_direction(Gradient_x=Gx, Gradient_y=Gy, out="both")
    
    st.markdown("### Gradient of image")
    st.pyplot(img_viz.plot_gradients_img(processed_image, Gx, Gy, figsize=(15,10)))
    

    st.markdown("### Magnitude and Direction")
    st.pyplot(img_viz.plot_gradients_img(processed_image, magnitude, direction, 
                                        titles=["Original Image", 'Gradient magnitude |∇I|', 'Direction $\Theta$'],
                                        cmaps=["grey", "hot", "hsv"], figsize=(15,10)))

else:
    st.info("Please select a sample image or upload your own to begin.")


st.divider()

col1, col2 = st.columns([3, 2])
with col1:
    st.page_link("pages/1_edge_intuition.py", label="**Start Exploring: First Intuition**")

with col2:
    st.page_link("pages/2_apply_gradient.py", label="**Edge detection using gradient**")