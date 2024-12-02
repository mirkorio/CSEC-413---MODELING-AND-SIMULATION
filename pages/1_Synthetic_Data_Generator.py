import streamlit as st
import numpy as np
import pandas as pd


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

# Streamlit app
st.set_page_config(page_title="Synthetic Data Generator", page_icon="📊")

st.title("Synthetic Data Generator")
st.write("Easily generate synthetic datasets for testing and experimentation.")

# Input for feature names
st.header("Step 1: Define Features")
feature_input = st.text_input(
    "Enter feature names separated by commas (e.g., feature1, feature2, feature3):",
    value=", ".join(st.session_state.features),
)
st.session_state.features = [f.strip() for f in feature_input.split(",")]

# Input for class settings
st.header("Step 2: Define Classes and Settings")
class_names = st.text_input(
    "Enter class names separated by commas (e.g., classA, classB, classC):",
    value=", ".join(st.session_state.class_settings.keys()),
)
class_list = [c.strip() for c in class_names.split(",")]

# Update or create settings for classes
for class_name in class_list:
    if class_name not in st.session_state.class_settings:
        st.session_state.class_settings[class_name] = {
            "mean": [np.random.uniform(0, 10) for _ in st.session_state.features],
            "std_dev": [np.random.uniform(1, 5) for _ in st.session_state.features],
        }

for class_name, settings in st.session_state.class_settings.items():
    if class_name in class_list:
        with st.expander(f"Settings for {class_name}"):
            cols = st.columns(2)
            mean_list = []
            std_dev_list = []
            for feature, mean, std_dev in zip(st.session_state.features, settings["mean"], settings["std_dev"]):
                mean_val = cols[0].number_input(
                    f"Mean for {feature}",
                    value=mean,
                    key=f"{class_name}_{feature}_mean",
                )
                std_dev_val = cols[1].number_input(
                    f"Std. Dev for {feature}",
                    value=std_dev,
                    key=f"{class_name}_{feature}_std_dev",
                )
                mean_list.append(mean_val)
                std_dev_list.append(std_dev_val)
            st.session_state.class_settings[class_name]["mean"] = mean_list
            st.session_state.class_settings[class_name]["std_dev"] = std_dev_list

# Total number of samples
st.header("Step 3: Generate Data")
total_samples = st.number_input("Total number of samples for the dataset:", min_value=1, value=1000, step=1)

# Generate data button
if st.button("Generate Data"):
    if not st.session_state.features or not class_list:
        st.error("Please define features and classes.")
    else:
        synthetic_data, samples_per_class = generate_synthetic_data(
            st.session_state.features, st.session_state.class_settings, total_samples
        )
        st.success("Synthetic data generated successfully!")

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
