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


st.sidebar.header("Quick Introduction")
st.sidebar.markdown(
"""
[What is an edge?](#what-is-an-edge)

[Why do they matter?](#why-do-they-matter)
"""
)

st.write("# EDGE DETECTION Learning App")

st.markdown(
"""
## A Quick Introduction 

Before we start exploring how to detect edges in an image, 
we will first answer the following questions:

### What is an edge?

An edge is a boundary or contour within a picture where there is a sharp, 
rapid change in pixel intensity (brightness), color, or texture.
- **Visually:** It is the visible outline that separates an object from its background 
or distinguishes two distinct surfaces (like a dark shadow falling across a brightly lit face).
- **Mathematically:** In the raw digital data of an image, an edge occurs 
where the pixel brightness values suddenly jump from a low value (dark) 
to a high value (light), or vice versa.

"""
)
st.image("images/example.png")

st.markdown(
"""
### Why do they matter?

Edges are the absolute foundation of how computers "see" and understand visual data. They matter for several crucial reasons:

- **Object Recognition:** Before AI or computer vision can recognize a car, a face, or a specific cell on a microscope slide, 
it must first locate shapes. Finding edges is the first step in defining what an object actually is.

- **Massive Data Reduction:** High-resolution images contain millions of pixels. A solid blue sky or a blank white wall often provides 
redundant data. By extracting only the edges, a computer filters out less useful information and focuses strictly on the structural layout, 
saving memory and processing power.

- **Image Segmentation:** Edges allow algorithms to break an image down into its individual components. This is how your phone's camera knows 
exactly where to apply a blur effect to the background while keeping your face in sharp focus.

- **Building Blocks for AI:** Deep learning models, like Convolutional Neural Networks (CNNs), look for simple edges in their very first 
layers of processing. They then combine these basic edges to understand more complex patterns, like corners, circles, and eventually whole objects.
"""
)

st.divider()

col1, col2 = st.columns([3, 2])
with col2:
    st.page_link("pages/1_edge_intuition.py", label="**Start Exploring: First Intuition**")