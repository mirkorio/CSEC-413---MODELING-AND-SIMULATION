import streamlit as st
import pandas as pd

# Set the page configuration
st.set_page_config(
    page_title="ML Model Generator",
    page_icon="🤖",
    layout="wide"
)

# Title of the app
st.title("ML Model Generator")

# Section for uploading a CSV file
st.header("Upload Your Dataset")
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

# If a file is uploaded
if uploaded_file is not None:
    try:
        # Load the data
        data = pd.read_csv(uploaded_file)
        
        # Display all data in larger view
        st.subheader("Data Preview")
        st.write("Below is the full dataset. Scroll to view all rows and columns.")
        st.dataframe(uploaded_file, use_container_width=True)

    except Exception as e:
        st.error(f"An error occurred while reading the file: {e}")

# Footer or placeholder for next steps
st.write("---")
st.info("More functionalities for ML model generation will be added.")
