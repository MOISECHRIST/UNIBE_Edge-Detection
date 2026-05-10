import streamlit as st


home_page = st.Page("pages/0_intro_edge.py", title="Introduction")
intuition_page = st.Page("pages/1_edge_intuition.py", title="Edge intuition")
gradient_page = st.Page("pages/2_apply_gradient.py", title="Edge detection using gradient")


pg = st.navigation([home_page, intuition_page, gradient_page])


pg.run()