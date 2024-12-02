import streamlit as st
import numpy as np
import pandas as pd


# Function to generate synthetic data
def generate_synthetic_data(features, class_settings, total_samples):
    """Generates synthetic data with specified settings."""
    class_names = list(class_settings.keys())
    
    # Randomly distribute samples per class, ensuring they sum up to total_samples
    proportions = np.random.dirichlet(np.ones(len(class_names)), size=1).flatten()
    samples_per_class = (proportions * total_samples).astype(int)
    samples_per_class[-1] += total_samples - np.sum(samples_per_class)  # Adjust the last class to match the total

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


# Initialize session state
if "features" not in st.session_state:
    st.session_state["features"] = []
if "class_settings" not in st.session_state:
    st.session_state["class_settings"] = {}
if "synthetic_data" not in st.session_state:
    st.session_state["synthetic_data"] = None
if "samples_per_class" not in st.session_state:
    st.session_state["samples_per_class"] = None


# Streamlit app
st.set_page_config(page_title="Synthetic Data Generator", page_icon="📊")

st.header("Synthetic Data Generator")
st.markdown("---")


# Input for feature names
st.subheader("Step 1: Define Features")
features = st.text_input("Enter feature names separated by commas (e.g., feature1, feature2, feature3):")
if features:
    feature_list = [f.strip() for f in features.split(",")]
    st.session_state["features"] = feature_list
else:
    feature_list = st.session_state["features"]

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
                cols = st.columns(2)  # Two columns for side-by-side inputs
                with cols[0]:
                    mean = st.number_input(
                        f"Mean for {feature} ({class_name}):",
                        value=np.random.uniform(0, 10),
                        key=f"{class_name}_{feature}_mean",
                    )
                with cols[1]:
                    std_dev = st.number_input(
                        f"Std. Dev for {feature} ({class_name}):",
                        value=np.random.uniform(1, 5),
                        key=f"{class_name}_{feature}_std_dev",
                    )
                means.append(mean)
                std_devs.append(std_dev)
            class_settings[class_name] = {"mean": means, "std_dev": std_devs}
    st.session_state["class_settings"] = class_settings
else:
    class_list = list(st.session_state["class_settings"].keys())
    class_settings = st.session_state["class_settings"]

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

        # Save to session state
        st.session_state["synthetic_data"] = synthetic_data
        st.session_state["samples_per_class"] = samples_per_class

# Display data if it exists
if st.session_state["synthetic_data"] is not None:
    synthetic_data = st.session_state["synthetic_data"]
    samples_per_class = st.session_state["samples_per_class"]

    # Display samples per class
    st.subheader("Samples Per Class")
    class_counts = {class_name: count for class_name, count in zip(class_list, samples_per_class)}
    class_counts["Total"] = sum(samples_per_class)
    st.write(class_counts)

    # Display all data in larger view
    st.subheader("Data Preview")
    st.write("Below is the full dataset. Scroll to view all rows and columns.")
    st.dataframe(synthetic_data, use_container_width=True)

    # Download data as CSV
    csv_data = synthetic_data.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Data as CSV",
        data=csv_data,
        file_name="synthetic_data.csv",
        mime="text/csv",
    )
