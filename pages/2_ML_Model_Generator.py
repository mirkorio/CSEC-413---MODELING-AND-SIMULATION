import streamlit as st
import pandas as pd
import numpy as np

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
        
        # Display dataset info
        st.subheader("Dataset Information")
        info_dict = {
            column: f"{data[column].dtype}({data[column].count()})"
            for column in data.columns
        }
        info_dict["Total"] = f"np.int64({data.shape[0]})"
        st.json(info_dict)

        # Display all data
        st.subheader("Dataset Preview")
        st.dataframe(data)

    except Exception as e:
        st.error(f"An error occurred while reading the file: {e}")

# Footer or placeholder for next steps
st.write("---")
st.info("More functionalities for ML model generation will be added.")
