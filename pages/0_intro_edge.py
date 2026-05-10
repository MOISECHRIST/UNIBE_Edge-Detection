##============================================
## Author : MEKA Moïse Christian Junior
## Email : moise.meka@students.unibe.ch
##============================================

"""
Main entry point for the Edge-Detection learning application.

This Streamlit application provides an interactive platform for learning about
edge detection in image analysis. It covers the basic concepts of what edges are,
why they matter, and provides tools for exploring different edge detection techniques.
"""

import streamlit as st

st.set_page_config(
    page_title="Edge-Detection"
)


st.sidebar.markdown(
"""
[What is an edge?](#what-is-an-edge)

[Why do they matter?](#why-do-they-matter)
"""
)

st.write("## EDGE DETECTION Learning App")

st.divider()

st.markdown(
"""
## A Quick Introduction 

Before we delve into the mechanical processes of detecting edges within an image, we must first establish a 
foundational understanding by answering two critical questions: What exactly is an edge, and why is it so vital to computer vision?

### What is an edge?

At its core, an edge is a localized spatial discontinuity within an image. It typically signifies a definitive 
boundary characterized by a sharp, rapid variation in pixel intensity (brightness), color, or texture. Physically, 
these variations reflect real-world transitions in object geometry, material properties, or illumination gradients (such as a cast shadow).
- **Visually:** It is the discernible contour that separates an object from its background or delineates two distinct surfaces. For instance, 
the sharp contrast between a dark shadow traversing a brightly illuminated face, or the distinct membrane boundary of a pathogen under a microscope.
- **Mathematically:** Because digital images are fundamentally continuous signals sampled into discrete matrices (or tensors), an edge 
occurs where the rate of change in pixel values is exceptionally high. If you were to plot the pixel intensities, an edge manifests as a steep 
gradient or "step" function, where values abruptly spike from a low threshold (dark) to a high threshold (light), or vice versa.
to a high value (light), or vice versa.

"""
)
st.image("images/example.png")

st.markdown(
"""
### Why do they matter?

Edges form the absolute bedrock of how computational systems parse, "see," and interpret visual data. 
Their critical importance is anchored in several key principles:

- **Morphological Object Recognition:** Before a computer vision algorithm can classify a vehicle, authenticate a human face, or identify a specific cellular phenotype in a complex microbiome sample, it must first isolate fundamental geometries. Detecting edges is the indispensable initial step in defining an object’s spatial boundaries and structural morphology.

- **Massive Dimensionality Reduction:** High-resolution digital imaging generates matrices containing millions of pixels, often heavily laden with redundant information (such as a uniform blue sky, a blank wall, or homogeneous extracellular fluid). By extracting solely the edges, the system performs a vital data reduction technique. It filters out non-informative noise and distills the image down to its essential structural topology, drastically conserving memory bandwidth and processing power.

- **Precise Image Segmentation:** Edges enable algorithms to partition a complex image into its discrete, semantic components. This is the underlying mechanism that allows consumer devices to apply depth-of-field (bokeh) effects by isolating the foreground subject, or what allows automated diagnostic tools to separate distinct tissue regions or regions of interest (ROIs) from noisy backgrounds.

- **The Foundational Primitives of AI:** In deep learning architectures, particularly Convolutional Neural Networks (CNNs), the earliest processing layers inherently act as mathematical edge detectors. By applying filters over the input tensors, the network seeks out simple, low-level gradients. Deeper layers subsequently synthesize these rudimentary edges to construct progressively sophisticated spatial hierarchies—moving from basic lines to corners, then to complex motifs, and ultimately to comprehensive object representations.
"""
)

st.divider()

col1, col2 = st.columns([3, 2])
with col2:
    st.page_link("pages/1_edge_intuition.py", label="**Start Exploring: First Intuition**")