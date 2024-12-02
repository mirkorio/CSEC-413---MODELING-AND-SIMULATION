import streamlit as st
import numpy as np
import pandas as pd


def generate_synthetic_data(features, class_settings, total_samples):
    """Generates synthetic data with specified settings."""
    class_names = list(class_settings.keys())
    proportions = np.random.dirichlet(np.ones(len(class_names)), size=1).flatten()
    samples_per_class = (proportions * total_samples).astype(int)

    data = []
    labels = []

    for class_name, samples, settings in zip(class_names, samples_per_class, class_settings.values()):
        means = settings["mean"]
        std_devs = settings["std_dev"]

        class_data = np.random.normal(
            loc=means,
            scale=std_devs,
            size=(samples, len(features))
        )
        data.append(class_data)
        labels.extend([class_name] * samples)

    data = np.vstack(data)
    labels = np.array(labels)

    df = pd.DataFrame(data, columns=features)
    df["Class"] = labels
    return df, samples_per_class


# Streamlit app
st.set_page_config(page_title="Synthetic Data Generator", page_icon="📊")

st.title("Synthetic Data Generator")
st.markdown("---")


# Input for feature names
st.subheader("Step 1: Define Features")
features = st.text_input("Enter feature names separated by commas (e.g., feature1, feature2, feature3):")
if features:
    feature_list = [f.strip() for f in features.split(",")]
else:
    feature_list = []

# Input for class settings
st.subheader("Step 2: Define Classes and Settings")
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

# Total number of samples
st.subheader("Step 3: Generate Data")
total_samples = st.number_input("Total number of samples for the dataset:", min_value=1, value=1000, step=1)

# Generate data button
if st.button("Generate Data"):
    if not feature_list or not class_list or not class_settings:
        st.error("Please define features, classes, and their settings.")
    else:
        synthetic_data, samples_per_class = generate_synthetic_data(feature_list, class_settings, total_samples)
        st.success("Synthetic data generated successfully!")

        # Display samples per class
        st.subheader("Samples Per Class")
        class_counts = {class_name: count for class_name, count in zip(class_list, samples_per_class)}
        st.write(class_counts)

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
