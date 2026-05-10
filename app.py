##============================================
## Author : MEKA Moïse Christian Junior
## Email : moise.meka@students.unibe.ch
##============================================
"""
Main application script for the Edge-Detection learning platform.

This script initializes the Streamlit navigation and sets up the page structure 
for the interactive edge detection tutorial.
"""

import streamlit as st


home_page = st.Page("pages/0_intro_edge.py", title="Introduction")
intuition_page = st.Page("pages/1_edge_intuition.py", title="Edge intuition")
gradient_page = st.Page("pages/2_apply_gradient.py", title="Edge detection using gradient")
canny_page = st.Page("pages/3_apply_canny.py", title="Canny Edge Detector")
laplacian_page = st.Page("pages/4_apply_laplacian.py", title="Edge detection using Laplacian")


pg = st.navigation([home_page, intuition_page, gradient_page, canny_page, laplacian_page])


pg.run()