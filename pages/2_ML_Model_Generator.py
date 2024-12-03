import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.exceptions import NotFittedError

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

        # Set up columns for side-by-side display
        col1, col2 = st.columns(2)

        # Dataset Information in an expander
        with col1:
            with st.expander("Dataset Information"):
                st.write(f"**Number of Rows:** {data.shape[0]}")
                st.write(f"**Number of Columns:** {data.shape[1]}")
                st.write(f"**Features:** {list(data.columns)}")
                st.write(data.dtypes)

        # Dataset Preview in an expander
        with col2:
            with st.expander("Dataset Preview"):
                st.dataframe(data)

        # Side-by-side Train/Test Split Configuration and Info
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
            # Convert test percentage to ratio
            test_ratio = test_percentage / 100

            # Split the dataset
            train_data, test_data = train_test_split(data, test_size=test_ratio, random_state=42)

            # Display split info
            st.subheader("Dataset Split Information")
            st.metric(label="Total Samples", value=data.shape[0])
            st.metric(label="Training Samples", value=train_data.shape[0])
            st.metric(label="Testing Samples", value=test_data.shape[0])

        # Model Training Section
        st.header("Train a Machine Learning Model")
        with st.expander("Model Configuration"):
            # Select target variable
            target_variable = st.selectbox("Select Target Variable", options=data.columns)

            # Train the model
            if st.button("Train Model"):
                try:
                    # Check if target variable exists
                    if target_variable not in data.columns:
                        raise ValueError("Target variable is not valid.")

                    # Separate features and target
                    X_train = train_data.drop(columns=[target_variable])
                    y_train = train_data[target_variable]
                    X_test = test_data.drop(columns=[target_variable])
                    y_test = test_data[target_variable]

                    # Train a Random Forest Classifier
                    model = RandomForestClassifier(random_state=42)
                    model.fit(X_train, y_train)

                    # Predict on the test set
                    y_pred = model.predict(X_test)

                    # Display classification report
                    st.subheader("Model Evaluation")
                    report = classification_report(y_test, y_pred, output_dict=True)
                    st.dataframe(pd.DataFrame(report).transpose())
                    st.success("Model trained successfully!")

                except ValueError as ve:
                    st.error(f"Error: {ve}")
                except NotFittedError:
                    st.error("Model could not be trained. Ensure the target variable and features are correctly selected.")
                except Exception as e:
                    st.error(f"An unexpected error occurred: {e}")

    except Exception as e:
        st.error("An error occurred while reading the file. Ensure it's a valid CSV file.")
        st.write(f"Details: {e}")

# Footer or placeholder for next steps
st.write("---")
st.info("""Work in progress.""")
