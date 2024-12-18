import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler

# Set the page configuration
st.set_page_config(
    page_title="ML Model Implementation",
    page_icon="💻",
    layout="wide"
)

# App Title 
st.title("ML Model Implementation")
st.write("---")

# After the initial imports and page config, add these lines
if 'model' not in st.session_state:
    st.session_state.model = None
if 'scaler' not in st.session_state:
    st.session_state.scaler = None

# After initializing session state variables, add these lines
model_file = None
scaler_file = None

# Create two columns for file uploaders (model and scaler pkl files)
col1, col2 = st.columns(2)

# Model uploader in first column
with col1:
    if st.session_state.model is None:
        model_file = st.file_uploader("Upload Model (.pkl)", type=['pkl'])
    else:
        st.success("✅ Model already loaded!")
        if st.button("Clear loaded model"):
            st.session_state.model = None
            st.experimental_rerun()
    
# Scaler uploader in second column
with col2:
    if st.session_state.scaler is None:
        scaler_file = st.file_uploader("Upload Scaler (.pkl)", type=['pkl'])
    else:
        st.success("✅ Scaler already loaded!")
        if st.button("Clear loaded scaler"):
            st.session_state.scaler = None
            st.experimental_rerun()

def validate_model(model):
    """Validate if the uploaded file is a valid ML model"""
    return hasattr(model, 'predict_proba') and hasattr(model, 'predict')

def validate_scaler(scaler):
    """Validate if the uploaded file is a valid scaler"""
    return isinstance(scaler, StandardScaler) or (hasattr(scaler, 'transform') and hasattr(scaler, 'fit_transform'))

# Only proceed if both files are uploaded
if (model_file is not None or st.session_state.model is not None) or (scaler_file is not None or st.session_state.scaler is not None):
    try:
        # Load model if not already in session state
        if st.session_state.model is None:
            if model_file is None:
                st.warning("⚠️ Please upload a model file.")
                st.stop()
            try:
                st.session_state.model = pickle.load(model_file)
            except Exception as e:
                st.error("❌ Failed to load model file. Please check if it's a valid pickle file.")
                st.stop()

        # Load scaler if not already in session state
        if st.session_state.scaler is None:
            if scaler_file is None:
                st.warning("⚠️ Please upload a scaler file.")
                st.stop()
            try:
                st.session_state.scaler = pickle.load(scaler_file)
            except Exception as e:
                st.error("❌ Failed to load scaler file. Please check if it's a valid pickle file.")
                st.stop()

        # Use session state variables instead of local variables
        if not validate_model(st.session_state.model):
            st.error("❌ Invalid model file! The uploaded file doesn't seem to be a valid machine learning model.")
            st.session_state.model = None
            st.stop()
        
        if not validate_scaler(st.session_state.scaler):
            st.error("❌ Invalid scaler file! The uploaded file doesn't seem to be a valid scaler.")
            st.session_state.scaler = None
            st.stop()

        # Show success message
        st.success("✅ Model and Scaler files successfully loaded!")
        
        st.write("---")
        
        # Get feature names from the model
        try:
            feature_names = st.session_state.model.feature_names_in_
        except:
            try:
                feature_names = [f"feature_{i}" for i in range(st.session_state.model.n_features_in_)]
            except:
                st.error("❌ Could not determine the number of features required by the model.")
                st.stop()
        
        # Initialize session state for feature values if not exists with dummy values
        if 'feature_values' not in st.session_state:
            st.session_state.feature_values = {feature: np.random.uniform(0, 100) for feature in feature_names}
        
        st.subheader("Enter Feature Values")
        
        # Add random generation button at the top
        if st.button("🎲 Generate Random Values For Prediction"):
            for feature in feature_names:
                st.session_state.feature_values[feature] = np.random.uniform(0, 100)
        
        # Create input fields in a grid layout
        feature_values = {}
        num_cols = 3  # Number of columns in the grid
        cols = st.columns(num_cols)
        for idx, feature in enumerate(feature_names):
            with cols[idx % num_cols]:
                feature_values[feature] = st.number_input(
                    feature,
                    value=st.session_state.feature_values[feature],
                    format="%.2f"
                )
                st.session_state.feature_values[feature] = feature_values[feature]
        
        try:
            # Prepare input data and make prediction
            X = np.array([list(feature_values.values())])
            X_scaled = st.session_state.scaler.transform(X)
            prediction = st.session_state.model.predict_proba(X_scaled)[0]
        except Exception as e:
            st.error("❌ Error making prediction. The model or scaler might be incompatible.")
            st.info("💡 Hint: Check if the model and scaler were trained together and are compatible.")
            st.stop()
        
        st.write("---")
        st.subheader("Prediction Results")
        
        # Create visualization columns
        viz_col1, viz_col2 = st.columns([1, 1])
        
        with viz_col1:
            # Bar chart for class probabilities
            classes = st.session_state.model.classes_
            prob_df = pd.DataFrame({
                'Class': classes,
                'Probability': prediction
            })
            
            st.write("Prediction Probabilities by Class")
            fig = go.Figure(data=[
                go.Bar(
                    x=prob_df['Class'],
                    y=prob_df['Probability'],
                    marker_color='rgb(30, 144, 255)',
                    text=prob_df['Probability'].apply(lambda x: f'{x:.1%}'),
                    textposition='auto',
                )
            ])
            fig.update_layout(
                yaxis_title="Probability",
                yaxis_tickformat='.0%',
                plot_bgcolor='rgba(0,0,0,0)',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with viz_col2:
            # Gauge chart for highest probability
            max_prob = prediction.max()
            max_class = classes[prediction.argmax()]
            
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=max_prob * 100,
                title={'text': f'Confidence for Class {max_class}'},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "rgb(30, 144, 255)"},
                    'steps': [
                        {'range': [0, 33], 'color': "lightgray"},
                        {'range': [33, 66], 'color': "gray"},
                        {'range': [66, 100], 'color': "darkgray"}
                    ],
                }
            ))
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        # Detailed probabilities table
        st.write("Detailed Class Probabilities:")
        prob_df['Probability'] = prob_df['Probability'].apply(lambda x: f'{x:.2%}')
        st.dataframe(prob_df, use_container_width=True)
        
    except Exception as e:
        st.error(f"❌ An unexpected error occurred: {str(e)}")
        st.info("💡 Please try uploading the files again or check your internet connection if the issue persists.")