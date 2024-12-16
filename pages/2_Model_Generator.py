import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import plotly.express as px
import time
import pickle
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns


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

         # Feature Visualization Section
        with st.expander("Feature Visualization", expanded=True):
            st.subheader("Feature Visualization")
            target_column = st.selectbox("Select the target/class column (for coloring):", data.columns)

            # Visualization Type
            viz_type = st.radio("Select the type of visualization:", ["2D", "3D"])

            if viz_type == "2D":
                col_x, col_y = st.columns(2)

                with col_x:
                    x_axis = st.selectbox("X-axis:", data.columns, index=0)

                with col_y:
                    y_axis = st.selectbox("Y-axis:", data.columns, index=1)

                fig_2d = px.scatter(
                    data,
                    x=x_axis,
                    y=y_axis,
                    color=target_column,
                    title="2D Feature Visualization",
                    labels={x_axis: x_axis, y_axis: y_axis},
                    template="plotly_dark" if st.session_state.get("theme", "light") == "dark" else "plotly"
                )
                st.plotly_chart(fig_2d, use_container_width=True)

            elif viz_type == "3D":
                col_x, col_y, col_z = st.columns(3)

                with col_x:
                    x_axis = st.selectbox("X-axis (3D):", data.columns, index=0)

                with col_y:
                    y_axis = st.selectbox("Y-axis (3D):", data.columns, index=1)

                with col_z:
                    z_axis = st.selectbox("Z-axis (3D):", data.columns, index=2)

                fig_3d = px.scatter_3d(
                    data,
                    x=x_axis,
                    y=y_axis,
                    z=z_axis,
                    color=target_column,
                    title="3D Feature Visualization",
                    labels={x_axis: x_axis, y_axis: y_axis, z_axis: z_axis},
                    template="plotly_dark" if st.session_state.get("theme", "light") == "dark" else "plotly"
                )
                fig_3d.update_traces(marker=dict(size=5))
                st.plotly_chart(fig_3d, use_container_width=True)
        

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

        # Check for Class column
        if "Class" not in data.columns:
            st.error("Dataset must contain a 'Class' column as the target variable.")
            st.stop()
        else:
            target_col = "Class"
            features = data.drop(columns=[target_col])
            target = data[target_col]

        random_state = st.number_input("Set Random State (for reproducibility)", 0, 100, 42)
        train_x, test_x, train_y, test_y = train_test_split(
            features, 
            target, 
            test_size=test_percentage/100,  # Convert percentage to decimal
            random_state=random_state
        )
        
        # Models
        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "K-Nearest Neighbors": KNeighborsClassifier(),
            "Support Vector Classifier": SVC(probability=True),
            "Decision Tree": DecisionTreeClassifier(),
            "Random Forest": RandomForestClassifier()
        }

        # Train models
        if "model_results" not in st.session_state:
            results = {}
            fitted_pipelines = {}
            best_model_name = None
            best_model = None
            best_scaler = None
            best_metrics = None
            best_f1 = -1
            
            # Training and evaluation
            for model_name, model in models.items():
                try:
                    start_time = time.time()
                    pipeline = Pipeline([
                        ('scaler', StandardScaler()),
                        ('model', model)
                    ])
                    pipeline.fit(train_x, train_y)
                    fitted_pipelines[model_name] = pipeline
                    predictions = pipeline.predict(test_x)
                    
                    # Calculate metrics
                    acc = accuracy_score(test_y, predictions)
                    prec = precision_score(test_y, predictions, average="weighted")
                    rec = recall_score(test_y, predictions, average="weighted")
                    f1 = f1_score(test_y, predictions, average="weighted")
                    elapsed_time = time.time() - start_time
                    
                    # Save metrics
                    results[model_name] = {
                        "Accuracy": acc,
                        "Precision": prec,
                        "Recall": rec,
                        "F1-Score": f1,
                        "Training Time": elapsed_time,
                        "Status": "Success"
                    }
                    
                    # Track best model
                    if f1 > best_f1:
                        best_f1 = f1
                        best_model_name = model_name
                        best_model = pipeline
                        best_scaler = pipeline.named_steps["scaler"]
                        best_metrics = {
                            "Accuracy": acc,
                            "Precision": prec,
                            "Recall": rec,
                            "F1-Score": f1
                        }
                except Exception as e:
                    results[model_name] = {
                        "Accuracy": None,
                        "Precision": None,
                        "Recall": None,
                        "F1-Score": None,
                        "Training Time": None,
                        "Status": f"Failed: {e}"
                    }
            
            # Store results in session state
            st.session_state.model_results = results
            st.session_state.fitted_pipelines = fitted_pipelines
            st.session_state.best_model_name = best_model_name
            st.session_state.best_model = best_model
            st.session_state.best_scaler = best_scaler
            st.session_state.best_metrics = best_metrics

        # Use results from session state
        results = st.session_state.model_results
        fitted_pipelines = st.session_state.fitted_pipelines
        best_model_name = st.session_state.best_model_name
        best_model = st.session_state.best_model
        best_scaler = st.session_state.best_scaler
        best_metrics = st.session_state.best_metrics

        # Dataset Download Section
        st.header("Download Dataset")
        col1, col2 = st.columns(2)

        with col1:
            # Download original dataset
            st.download_button(
                label="Download Original Dataset (CSV)",
                data=data.to_csv(index=False),
                file_name="original_dataset.csv",
                mime="text/csv"
            )

        with col2:
            # Download scaled dataset
            scaler = StandardScaler()
            scaled_data = pd.DataFrame(
                scaler.fit_transform(data.select_dtypes(include=['float64', 'int64'])),
                columns=data.select_dtypes(include=['float64', 'int64']).columns
            )
            st.download_button(
                label="Download Scaled Dataset (CSV)",
                data=scaled_data.to_csv(index=False),
                file_name="scaled_dataset.csv",
                mime="text/csv"
            )

        # Display dataset statistics in an expander
        with st.expander("Dataset Statistics", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Original Dataset Statistics:")
                st.dataframe(data.describe())
            
            with col2:
                st.subheader("Scaled Dataset Statistics:")
                st.dataframe(scaled_data.describe())

        # Best model performance
        if best_model_name:
            st.subheader(f"Best Model: {best_model_name}")
            
            # Create columns for metrics and download buttons
            col1, col2 = st.columns([0.3, 0.3])
            
            with col1:
                st.json(best_metrics)
            
            with col2:
                st.download_button(
                    label="Download Best Model",
                    data=pickle.dumps(best_model),
                    file_name="best_model.pkl"
                )
                st.download_button(
                    label="Download Scaler", 
                    data=pickle.dumps(best_scaler),
                    file_name="scaler.pkl"
                )
            
            st.subheader("Classification Report")
            predictions = best_model.predict(test_x)
            report = classification_report(test_y, predictions, output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df)
        
        # Model comparison table
        st.subheader("Model Comparison")
        comparison_df = pd.DataFrame(results).T
        st.dataframe(comparison_df)
        
        # Model performance graph
        st.subheader("Model Performance Comparison")
        
        # Add model selection widget
        available_models = comparison_df.index.tolist()
        selected_models = st.multiselect(
            "Select models to compare:",
            available_models,
            default=available_models,
            key="model_selector"
        )
        
        if selected_models:
            # Prepare the data for plotting with selected models only
            plot_df = comparison_df.loc[selected_models].drop(columns=["Training Time", "Status"])
            
            # Reshape the dataframe for plotly
            plot_df_melted = plot_df.reset_index().melt(
                id_vars=['index'],
                var_name='Metric',
                value_name='Score'
            )
            
            # Create plotly bar chart
            fig = px.bar(
                plot_df_melted,
                x='index',
                y='Score',
                color='Metric',
                barmode='group',
                title="Model Performance Metrics Comparison",
                labels={'index': 'Model', 'Score': 'Value'},
                template="plotly_dark" if st.session_state.get("theme", "light") == "dark" else "plotly"
            )
            
            # Customize the layout
            fig.update_layout(
                xaxis_title="Models",
                yaxis_title="Score",
                xaxis_tickangle=-45,
                showlegend=True,
                legend_title="Metric",
                height=600
            )
            
            # Display the plot
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Please select at least one model to display the comparison chart.")
        
        # Create a table for saved models with download buttons
        st.subheader("Saved Models")
        
        # Calculate number of columns needed for a square-ish layout
        n_models = len(fitted_pipelines)
        n_cols = min(3, n_models)  # Max 3 columns to maintain readability
        n_rows = (n_models + n_cols - 1) // n_cols
        
        # Create the grid layout
        for row in range(n_rows):
            cols = st.columns(n_cols)
            for col in range(n_cols):
                idx = row * n_cols + col
                if idx < n_models:
                    model_name = list(fitted_pipelines.keys())[idx]
                    with cols[col]:
                        # Create a container with border styling
                        with st.container():
                            st.markdown(
                                f"""
                                <div style="
                                    padding: 10px;
                                    border: 1px solid #ccc;
                                    border-radius: 5px;
                                    margin: 5px 0;
                                    text-align: center;
                                ">
                                    <h4>{model_name}</h4>
                                    <p>Accuracy: {results[model_name]['Accuracy']:.4f}</p>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                            
                            # Add centered download buttons
                            st.markdown("<div style='text-align: center; display: flex; justify-content: center; gap: 5px;'>", unsafe_allow_html=True)
                            col1, col2 = st.columns(2)
                            with col1:
                                st.download_button(
                                    label="📥 Download Model",
                                    data=pickle.dumps(fitted_pipelines[model_name]),
                                    file_name=f"{model_name}_model.pkl",
                                    key=f"model_{idx}"
                                )
                            with col2:
                                st.download_button(
                                    label="📥 Download Scaler",
                                    data=pickle.dumps(fitted_pipelines[model_name].named_steps['scaler']),
                                    file_name=f"{model_name}_scaler.pkl",
                                    key=f"scaler_{idx}"
                                )
                            st.markdown("</div>", unsafe_allow_html=True)

        # Learning curves and confusion matrices
        st.subheader("Learning Curves and Confusion Matrices")
        for model_name, metrics in results.items():
            if metrics["Status"] == "Success":
                fig, ax = plt.subplots(1, 2, figsize=(15, 5))
                
                # Learning curve using the fitted pipeline
                pipeline = fitted_pipelines[model_name]
                scores = cross_val_score(pipeline, train_x, train_y, cv=5)
                ax[0].plot(scores)
                ax[0].set_title(f"Learning Curve: {model_name}")
                ax[0].set_xlabel("Fold")
                ax[0].set_ylabel("Accuracy")
                
                # Confusion matrix using the fitted pipeline
                predictions = pipeline.predict(test_x)
                cm = confusion_matrix(test_y, predictions)
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax[1])
                ax[1].set_title(f"Confusion Matrix: {model_name}")
                ax[1].set_xlabel("Predicted")
                ax[1].set_ylabel("True")
                
                st.pyplot(fig)

    except Exception as e:
        st.error("An error occurred while processing the file. Ensure it's a valid CSV file.")
        st.write(f"Details: {e}")


