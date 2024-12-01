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

with st.expander("Project Overview"):
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


