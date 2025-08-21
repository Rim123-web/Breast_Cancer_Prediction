# app.py
import streamlit as st
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

st.set_page_config(page_title="Breast Cancer Predictor", layout="centered")

st.title("Breast Cancer Predictor (Random Forest)")

st.write("""
This app predicts whether a tumor is **Benign** or **Malignant**.
You only need to input **3 key features**, and the model will estimate the rest automatically.
""")

# --- Load trained models ---
# Ensure these pickle files exist: rf_model.pkl, and helper_models.pkl
# If you haven't saved them yet, save like:
# pickle.dump(rf_model, open("rf_model.pkl", "wb"))
# pickle.dump(helper_models, open("helper_models.pkl", "wb"))
rf_model = pickle.load(open("rf_model.pkl", "rb"))
helper_models = pickle.load(open("helper_models.pkl", "rb"))

# Key features the user inputs
key_features = ['concave points_worst', 'perimeter_worst', 'radius_worst']

# --- User inputs via sliders ---
st.header("User Inputs")
user_input = {}
for feature in key_features:
    # You can adjust min/max values based on your dataset
    user_input[feature] = st.slider(
        feature, 
        min_value=float(0.0), 
        max_value=float(200.0), 
        value=float(50.0),
        step=0.01
    )

# --- Predict button ---
if st.button("Predict"):
    # Convert user input to DataFrame
    df_input = pd.DataFrame([user_input])
    
    # Predict remaining features
    for feature, model in helper_models.items():
        if feature not in key_features:
            df_input[feature] = model.predict(df_input[key_features])
    
    # Ensure all features are in same order as RF model expects
    df_input = df_input[rf_model.feature_names_in_]
    
    # Make prediction
    prediction = rf_model.predict(df_input)[0]
    prediction_proba = rf_model.predict_proba(df_input)[0][prediction]
    
    # Display result
    if prediction == 0:
        st.success(f"Prediction: Benign ✅ (Confidence: {prediction_proba*100:.2f}%)")
    else:
        st.error(f"Prediction: Malignant ⚠️ (Confidence: {prediction_proba*100:.2f}%)")
