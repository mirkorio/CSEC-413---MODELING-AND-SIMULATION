import streamlit as st



# Page Configuration
st.set_page_config(
    page_title="Modeling and Simulation with Python",
    page_icon="📘",
    layout="wide"
)

# Add logo to sidebar
gif_url = "https://i.giphy.com/media/v1.Y2lkPTc5MGI3NjExazM2cnB2NGYyaHNnc2ZyaThtNXFnYXg5NHJ4dHpodHFpeXk0Y2ZweCZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9cw/Eq1sxBSFpIBdUyBAsP/giphy.gif"
st.sidebar.image(gif_url)

# Title
st.title("Modeling and Simulation with Python")
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
#st.caption("By Team ORION")
with st.expander("By Team ORION"):
            st.write("""
                - Venn Delos P. Santos
                - John Christian M. Gava
                - Marc Christian D. Tumaneng
            """)

