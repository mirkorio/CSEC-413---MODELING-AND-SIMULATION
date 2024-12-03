import streamlit as st
import pandas as pd
import numpy as np
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

# Set Streamlit page configuration
st.set_page_config(
    page_title="Model Trainer and Evaluator",
    layout="wide",
    page_icon="🤖"
)

st.title("Model Trainer and Evaluator")
st.write("---")

# Upload dataset
uploaded_file = st.file_uploader("Upload your dataset (CSV)", type=["csv"])

if uploaded_file is not None:
    # Load dataset
    data = pd.read_csv(uploaded_file)
    st.header("Original Dataset")
    st.write(data.head())
    st.write(f"Dataset Shape: {data.shape}")
    
    # Dataset statistics
    st.subheader("Dataset Statistics")
    st.write(data.describe())
    
    # Feature selection
    target_col = st.selectbox("Select Target Column", data.columns)
    features = data.drop(columns=[target_col])
    target = data[target_col]
    
    # Train-test split
    test_size = st.slider("Select Test Size (%)", 10, 50, 20) / 100
    random_state = st.number_input("Set Random State (for reproducibility)", 0, 100, 42)
    train_x, test_x, train_y, test_y = train_test_split(
        features, target, test_size=test_size, random_state=random_state
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
    if st.button("Train Models"):
        results = {}
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
        
        # Best model performance
        if best_model_name:
            st.subheader(f"Best Model: {best_model_name}")
            st.json(best_metrics)
            
            # Download the best model and scaler
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
        
        # Model comparison table
        st.subheader("Model Comparison")
        comparison_df = pd.DataFrame(results).T
        st.dataframe(comparison_df)
        
        # Model performance graph
        st.subheader("Model Performance Comparison")
        comparison_df.drop(columns=["Training Time", "Status"]).plot(kind="bar", figsize=(10, 5))
        plt.xticks(rotation=45)
        plt.title("Model Metrics")
        st.pyplot(plt)
        
        # Learning curves and confusion matrices
        st.subheader("Learning Curves and Confusion Matrices")
        for model_name, metrics in results.items():
            if metrics["Status"] == "Success":
                fig, ax = plt.subplots(1, 2, figsize=(15, 5))
                
                # Learning curve
                scores = cross_val_score(models[model_name], train_x, train_y, cv=5)
                ax[0].plot(scores)
                ax[0].set_title(f"Learning Curve: {model_name}")
                ax[0].set_xlabel("Fold")
                ax[0].set_ylabel("Accuracy")
                
                # Confusion matrix
                cm = confusion_matrix(test_y, models[model_name].predict(test_x))
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax[1])
                ax[1].set_title(f"Confusion Matrix: {model_name}")
                ax[1].set_xlabel("Predicted")
                ax[1].set_ylabel("True")
                
                st.pyplot(fig)
