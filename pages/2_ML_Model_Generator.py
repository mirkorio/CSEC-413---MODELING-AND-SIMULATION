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
        
        # Number of rows and columns
        st.write(f"**Shape of Dataset:** {data.shape[0]} rows, {data.shape[1]} columns")
        
        # Column names, data types, and non-null counts
        column_info = {
            column: {
                "Data Type": str(data[column].dtype),
                "Non-Null Count": data[column].count()
            }
            for column in data.columns
        }
        st.write("**Column Details:**")
        st.json(column_info)
        
        # Memory usage
        memory_usage = data.memory_usage(deep=True).sum() / 1024 ** 2  # Convert bytes to MB
        st.write(f"**Memory Usage:** {memory_usage:.2f} MB")
        
        # Statistical summary
        st.write("**Statistical Summary (Numerical Columns):**")
        st.dataframe(data.describe())
        
        # Display all data
        st.subheader("Dataset Preview")
        st.dataframe(data)

    except Exception as e:
        st.error(f"An error occurred while reading the file: {e}")

# Footer or placeholder for next steps
st.write("---")
st.info("More functionalities for ML model generation will be added.")
