import streamlit as st
import numpy as np
import pandas as pd

def generate_synthetic_data(features, class_settings, n_samples_per_class):
    """Generates synthetic data with specified settings."""
    data = []
    labels = []

    for class_name, settings in class_settings.items():
        means = settings["mean"]
        std_devs = settings["std_dev"]

        class_data = np.random.normal(
            loc=means,
            scale=std_devs,
            size=(n_samples_per_class, len(features))
        )
        data.append(class_data)
        labels.extend([class_name] * n_samples_per_class)

    data = np.vstack(data)
    labels = np.array(labels)

    df = pd.DataFrame(data, columns=features)
    df["Class"] = labels
    return df

# Streamlit app
st.set_page_config(page_title="Synthetic Data Generator", page_icon="📊")

st.title("Synthetic Data Generator")
st.write("Easily generate synthetic datasets for testing and experimentation.")

# Input for feature names
st.header("Step 1: Define Features")
features = st.text_input("Enter feature names separated by commas (e.g., feature1, feature2, feature3):")
if features:
    feature_list = [f.strip() for f in features.split(",")]
else:
    feature_list = []

# Input for class settings
st.header("Step 2: Define Classes and Settings")
class_settings = {}
class_names = st.text_input("Enter class names separated by commas (e.g., ClassA, ClassB):")
if class_names:
    class_list = [c.strip() for c in class_names.split(",")]

    for class_name in class_list:
        with st.expander(f"Settings for {class_name}", expanded=False):
            means = []
            std_devs = []
            for feature in feature_list:
                mean = st.number_input(
                    f"Mean for {feature} ({class_name}):",
                    value=np.random.uniform(0, 10),
                    key=f"{class_name}_{feature}_mean",
                )
                std_dev = st.number_input(
                    f"Std. Dev for {feature} ({class_name}):",
                    value=np.random.uniform(1, 5),
                    key=f"{class_name}_{feature}_std_dev",
                )
                means.append(mean)
                std_devs.append(std_dev)
            class_settings[class_name] = {"mean": means, "std_dev": std_devs}

# Number of samples per class
st.header("Step 3: Generate Data")
n_samples = st.number_input("Number of samples per class:", min_value=1, value=100, step=1)

# Generate data button
if st.button("Generate Data"):
    if not feature_list or not class_list or not class_settings:
        st.error("Please define features, classes, and their settings.")
    else:
        synthetic_data = generate_synthetic_data(feature_list, class_settings, n_samples)
        st.success("Synthetic data generated successfully!")

        # Display data preview
        st.subheader("Data Preview")
        st.dataframe(synthetic_data.head())

        # Download data as CSV
        csv_data = synthetic_data.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Data as CSV",
            data=csv_data,
            file_name="synthetic_data.csv",
            mime="text/csv",
        )
