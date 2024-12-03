import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
import plotly.express as px

# Set the page configuration
st.set_page_config(
    page_title="ML Model Generator",
    page_icon="🤖",
    layout="wide"
)

# Title of the app
st.title("ML Model Generator")
st.write("---")

# Section for uploading a CSV file
st.header("Upload Your Dataset")
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

# If a file is uploaded
if uploaded_file is not None:
    try:
        # Load the data
        data = pd.read_csv(uploaded_file)
        
        # Display dataset information and preview
        col1, col2 = st.columns(2)

        with col1:
            with st.expander("Dataset Information"):
                st.write(f"**Number of Rows:** {data.shape[0]}")
                st.write(f"**Number of Columns:** {data.shape[1]}")
                st.write(f"**Features:** {list(data.columns)}")

        with col2:
            with st.expander("Dataset Preview"):
                st.dataframe(data)

        # Dataset Split Configuration and Info
        st.header("Dataset Split Configuration and Information")
        config_col, info_col = st.columns(2)

        with config_col:
            st.subheader("Test Size Configuration")
            test_percentage = st.slider(
                "Test Data Percentage (%)",
                min_value=10,
                max_value=50,
                value=20,
                step=5,
                format="%d%%"
            )
            st.caption("Select the percentage of data used for testing.")

        with info_col:
            test_ratio = test_percentage / 100
            train_data, test_data = train_test_split(data, test_size=test_ratio, random_state=42)
            total_samples = data.shape[0]
            train_samples = train_data.shape[0]
            test_samples = test_data.shape[0]

            st.subheader("Dataset Split Information")
            st.metric(label="Total Samples", value=total_samples)
            st.metric(label="Training Samples", value=f"{train_samples} ({round(train_samples / total_samples * 100)}%)")
            st.metric(label="Testing Samples", value=f"{test_samples} ({round(test_samples / total_samples * 100)}%)")

        # Feature Selection and Visualization
        st.header("Feature Visualization")
        st.subheader("Feature Selection")
        target_column = st.selectbox("Select the target/class column (for coloring):", data.columns)

        # Visualization Type
        viz_type = st.radio("Select the type of visualization:", ["2D", "3D"])

        if viz_type == "2D":
            x_axis = st.selectbox("Select feature for X-axis:", data.columns, index=0)
            y_axis = st.selectbox("Select feature for Y-axis:", data.columns, index=1)

            fig_2d = px.scatter(
                data,
                x=x_axis,
                y=y_axis,
                color=target_column,
                title="2D Feature Visualization",
                labels={x_axis: "X-axis", y_axis: "Y-axis"},
                template="plotly_dark" if st.session_state.get("theme", "light") == "dark" else "plotly"
            )
            st.plotly_chart(fig_2d, use_container_width=True)

        elif viz_type == "3D":
            x_axis = st.selectbox("Select feature for X-axis (3D):", data.columns, index=0)
            y_axis = st.selectbox("Select feature for Y-axis (3D):", data.columns, index=1)
            z_axis = st.selectbox("Select feature for Z-axis (3D):", data.columns, index=2)

            fig_3d = px.scatter_3d(
                data,
                x=x_axis,
                y=y_axis,
                z=z_axis,
                color=target_column,
                title="3D Feature Visualization",
                labels={x_axis: "X-axis", y_axis: "Y-axis", z_axis: "Z-axis"},
                template="plotly_dark" if st.session_state.get("theme", "light") == "dark" else "plotly"
            )
            fig_3d.update_traces(marker=dict(size=5))
            st.plotly_chart(fig_3d, use_container_width=True)

    except Exception as e:
        st.error("An error occurred while processing the file. Ensure it's a valid CSV file.")
        st.write(f"Details: {e}")

# Footer or placeholder for next steps
st.write("---")
st.info("""Work in progress.""")
