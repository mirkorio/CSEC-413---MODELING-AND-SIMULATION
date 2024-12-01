import streamlit as st

# Page Configuration
st.set_page_config(
    page_title="Modeling and Simulation with Python",
    #page_icon="📘",
    layout="centered"
)

# Title
st.header("Modeling and Simulation with Python")
st.markdown("---")

# Introduction Section
st.subheader("Introduction")
st.write(
    """
    Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. 
    Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.
    """
)

# Project Overview Section
st.subheader("Project Overview")
st.write(
    """
    Lorem ipsum dolor sit amet, consectetur adipiscing elit. Quisque non risus in orci tristique dapibus. 
    Proin nec lacus et urna scelerisque vulputate. Donec in venenatis risus. Nulla facilisi.
    """
)

# Footer or Additional Information
st.markdown("---")
st.caption("By Team ORION")
with st.expander.caption("By Team ORION"):
            st.subheader("Interpretation of Histograms")
            st.write("""
                - **Text Similarity Histogram**: This histogram shows the distribution of text similarity scores across the dataset. A concentration of bars at higher percentages indicates a greater number of code pairs with similar text.
                - **Structural Similarity Histogram**: The structural similarity histogram visualizes how similar the structures of the code pairs are. Peaks in the lower ranges suggest more varied structural designs, while higher values indicate structural consistency.
                - **Weighted Similarity Histogram**: The weighted similarity metric combines both text and structural similarities. A skew toward higher percentages might suggest that most code pairs are both textually and structurally similar. A balanced distribution across all ranges would indicate varied similarities across the dataset.
            """)

