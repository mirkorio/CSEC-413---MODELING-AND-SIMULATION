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
        
        # Set up columns for side-by-side display
        col1, col2 = st.columns(2)

        # Dataset Information in an expander
        with col1:
            with st.expander("Dataset Information"):
                st.write(f"**Number of Rows:** {data.shape[0]}")
                st.write(f"**Number of Columns:** {data.shape[1]}")
                for col in data.columns:
                    st.write(f"**{col}**: {data[col].dtype} ({data[col].count()} non-null values)")
                
                # Include classes for each column if applicable
                st.write("### Classes (Unique Values)")
                for col in data.columns:
                    unique_values = data[col].nunique()
                    if unique_values <= 10:  # Show classes only for columns with <= 10 unique values
                        st.write(f"**{col}**: {data[col].unique()} ({unique_values} classes)")

        # Dataset Preview in an expander
        with col2:
            with st.expander("Dataset Preview"):
                st.dataframe(data)

    except Exception as e:
        st.error("An error occurred while reading the file. Ensure it's a valid CSV file.")
        st.write(f"Details: {e}")

# Footer or placeholder for next steps
st.write("---")
st.info("""
🚀 Coming soon:
- Feature Selection and Engineering
- Model Training and Evaluation
- Automated Hyperparameter Tuning
""")
