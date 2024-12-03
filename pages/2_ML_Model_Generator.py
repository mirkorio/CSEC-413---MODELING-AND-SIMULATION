import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

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
        
        # Set up columns for side-by-side display
        col1, col2 = st.columns(2)

        # Dataset Information in an expander
        with col1:
            with st.expander("Dataset Information"):
                st.write(f"**Number of Rows:** {data.shape[0]}")
                st.write(f"**Number of Columns:** {data.shape[1]}")
                # Features
                st.write(f"**Features:** {list(data.columns)}")
                # Classes
                for col in data.columns:
                    unique_values = data[col].unique()
                    unique_count = data[col].nunique()
                    if unique_count <= 10:  # Show classes only for columns with <= 10 unique values
                        st.write(f"**{col}:** {list(unique_values)} ({unique_count} classes)")

        # Dataset Preview in an expander
        with col2:
            with st.expander("Dataset Preview"):
                st.dataframe(data)

        # Train/Test Split Configuration
        st.header("Train/Test Split Configuration")
        split_ratio = st.slider("Select train/test split ratio", min_value=0.1, max_value=0.9, value=0.8, step=0.1)
        
        # Split the dataset
        train_data, test_data = train_test_split(data, test_size=1-split_ratio, random_state=42)
        
        # Display split info
        st.subheader("Dataset Split Information")
        total_samples = data.shape[0]
        train_samples = train_data.shape[0]
        test_samples = test_data.shape[0]
        
        st.write(f"**Total Samples:** {total_samples}")
        st.write(f"**Training Samples:** {train_samples}")
        st.write(f"**Testing Samples:** {test_samples}")
    
    except Exception as e:
        st.error("An error occurred while reading the file. Ensure it's a valid CSV file.")
        st.write(f"Details: {e}")

# Footer or placeholder for next steps
st.write("---")
st.info("""Work in progress.""")
