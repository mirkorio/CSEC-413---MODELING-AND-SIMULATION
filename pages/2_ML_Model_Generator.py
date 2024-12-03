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

        # Side-by-side Train/Test Split Configuration and Info
        st.header("Dataset Split Configuration and Information")
        config_col, info_col = st.columns(2)

        with config_col:
            st.subheader("Train/Test Split Configuration")
            split_ratio = st.slider(
                "Split Ratio (Train %)",
                min_value=10,
                max_value=50,
                value=0.2,
                step=0.05,
                label_visibility="collapsed"  # Hides the main slider label
            )
            st.caption("Select the percentage of data used for training.")

        with info_col:
            # Split the dataset
            train_data, test_data = train_test_split(data, test_size=1 - split_ratio, random_state=42)
            
            # Display split info
            st.subheader("Dataset Split Information")
            total_samples = data.shape[0]
            train_samples = train_data.shape[0]
            test_samples = test_data.shape[0]
            
            st.metric(label="Total Samples", value=total_samples)
            st.metric(label="Training Samples", value=train_samples)
            st.metric(label="Testing Samples", value=test_samples)

    except Exception as e:
        st.error("An error occurred while reading the file. Ensure it's a valid CSV file.")
        st.write(f"Details: {e}")

# Footer or placeholder for next steps
st.write("---")
st.info("""Work in progress.""")
