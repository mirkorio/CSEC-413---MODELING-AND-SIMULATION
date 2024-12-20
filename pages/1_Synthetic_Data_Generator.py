import streamlit as st
import numpy as np
import pandas as pd


# Page Configuration
st.set_page_config(
    page_title="Synthetic Data Generator", 
    page_icon="📊",
    layout="wide"
)

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


st.title("Synthetic Data Generator")
st.markdown("---")

# Initialize session state if not already set
if 'class_settings' not in st.session_state:
    st.session_state.class_settings = {}

if 'feature_list' not in st.session_state:
    st.session_state.feature_list = []

if 'generated_data' not in st.session_state:
    st.session_state.generated_data = None

if 'samples_per_class' not in st.session_state:
    st.session_state.samples_per_class = None

# Input for feature names with default values
st.subheader("Define Features")
default_features = "average_viewership, player_count, prize_pool, game_duration, downloads"
features = st.text_input("Enter feature names separated by commas (e.g., average_viewership, player_count):", 
                         value=default_features if not st.session_state.feature_list else ", ".join(st.session_state.feature_list))

if features:
    st.session_state.feature_list = [f.strip() for f in features.split(",")]

# Input for class names with default values
st.subheader("Define Classes")
default_classes = "League_of_Legends, Dota_2, Counter_Strike, Valorant, Mobile_Legends"
class_names = st.text_input("Enter class names separated by commas (e.g., League_of_Legends, Dota_2):", 
                            value=default_classes if not st.session_state.class_settings else ", ".join(st.session_state.class_settings.keys()))

if class_names:
    class_list = [c.strip() for c in class_names.split(",")]

    # Initialize class settings in session state if not already set
    for class_name in class_list:
        if class_name not in st.session_state.class_settings:
            st.session_state.class_settings[class_name] = {"mean": [], "std_dev": []}
    
    for class_name in class_list:
        with st.expander(f"Settings for {class_name}", expanded=False):
            means = st.session_state.class_settings[class_name]["mean"]
            std_devs = st.session_state.class_settings[class_name]["std_dev"]

            for i, feature in enumerate(st.session_state.feature_list):
                cols = st.columns(2)  # Two columns for side-by-side inputs
                with cols[0]:
                    mean = st.number_input(
                        f"Mean for {feature} ({class_name}):",
                        value=means[i] if i < len(means) else np.random.uniform(0, 10),
                        key=f"{class_name}_{feature}_mean",
                    )
                with cols[1]:
                    std_dev = st.number_input(
                        f"Std. Dev for {feature} ({class_name}):",
                        value=std_devs[i] if i < len(std_devs) else np.random.uniform(1, 5),
                        key=f"{class_name}_{feature}_std_dev",
                    )
    
                # Update means and std_devs in session state
                if i >= len(means):
                    means.append(mean)
                    std_devs.append(std_dev)
                else:
                    means[i] = mean
                    std_devs[i] = std_dev
            st.session_state.class_settings[class_name] = {"mean": means, "std_dev": std_devs}

# Total number of samples
st.subheader("Generate Data")
total_samples = st.number_input("Total number of samples for the dataset:", min_value=1, value=1000, step=1,
                                key="total_samples", 
                                )

# Generate data button
if st.button("Generate Data"):
    if not st.session_state.feature_list or not class_list or not st.session_state.class_settings:
        st.error("Please define features, classes, and their settings.")
    else:
        synthetic_data, samples_per_class = generate_synthetic_data(st.session_state.feature_list, st.session_state.class_settings, total_samples)
        st.session_state.generated_data = synthetic_data
        st.session_state.samples_per_class = samples_per_class

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
