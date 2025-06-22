import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json

# Set page config
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        margin-top: 20px;
    }
    .prediction {
        padding: 20px;
        border-radius: 10px;
        margin-top: 20px;
    }
    .high-risk {
        color: red;
        font-weight: bold;
    }
    .moderate-risk {
        color: orange;
        font-weight: bold;
    }
    .low-risk {
        color: green;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

# Load feature mappings
with open('feature_mappings.json', 'r') as f:
    feature_mappings = json.load(f)

# Load the models and scaler
@st.cache_resource
def load_models():
    models = {}
    model_names = ['svm', 'logistic_regression', 'random_forest', 'xgboost']
    
    for name in model_names:
        with open(f'{name}.pkl', 'rb') as f:
            models[name] = pickle.load(f)
    
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    
    return models, scaler

def calibrate_probability(prob, risk_count):
    """Calibrate probability based on risk factors"""
    base_multiplier = 1.5  # Increase base probabilities
    risk_multiplier = 1 + (risk_count * 0.2)  # Additional 20% per risk factor
    calibrated = prob * base_multiplier * risk_multiplier
    return min(calibrated, 1.0)  # Cap at 100%

def calculate_risk_factors(age, trestbps, chol, fbs, exang, oldpeak, ca, thal, cp):
    """Calculate risk factors count and critical factors"""
    risk_count = 0
    critical_count = 0
    risk_details = []
    
    # Age risk
    if age >= 65:
        risk_count += 1
        critical_count += 1
        risk_details.append(("Age ≥ 65", "High"))
    elif age >= 55:
        risk_count += 1
        risk_details.append(("Age ≥ 55", "Moderate"))
    
    # Blood pressure risk
    if trestbps >= 160:
        risk_count += 1
        critical_count += 1
        risk_details.append(("Very High Blood Pressure", "High"))
    elif trestbps >= 140:
        risk_count += 1
        risk_details.append(("High Blood Pressure", "Moderate"))
    
    # Cholesterol risk
    if chol >= 280:
        risk_count += 1
        critical_count += 1
        risk_details.append(("Very High Cholesterol", "High"))
    elif chol >= 240:
        risk_count += 1
        risk_details.append(("High Cholesterol", "Moderate"))
    
    # Other factors
    if fbs == 1:
        risk_count += 1
        risk_details.append(("High Fasting Blood Sugar", "Moderate"))
    
    if exang == 1:
        risk_count += 1
        critical_count += 1
        risk_details.append(("Exercise Induced Angina", "High"))
    
    if oldpeak >= 2.0:
        risk_count += 1
        critical_count += 1
        risk_details.append(("Significant ST Depression", "High"))
    
    if ca >= 2:
        risk_count += 1
        critical_count += 1
        risk_details.append(("Multiple Vessels Affected", "High"))
    
    if thal >= 2:
        risk_count += 1
        risk_details.append(("Abnormal Thalassemia", "Moderate"))
    
    if cp == 3:  # asymptomatic
        risk_count += 1
        critical_count += 1
        risk_details.append(("Asymptomatic Chest Pain", "High"))
    
    return risk_count, critical_count, risk_details

def main():
    st.title("❤️ Heart Disease Prediction System")
    st.write("Enter patient information to predict the likelihood of heart disease.")
    
    # Load models
    try:
        models, scaler = load_models()
    except Exception as e:
        st.error("Error loading models. Please make sure all model files are present.")
        return
    
    # Create two columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Patient Information")
        age = st.number_input("Age", min_value=0, max_value=100, value=45)
        sex = st.selectbox("Sex", ["Female", "Male"])
        cp = st.selectbox("Chest Pain Type", 
                         ["typical angina", "atypical angina", "non-anginal", "asymptomatic"])
        trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=0, max_value=300, value=120)
        chol = st.number_input("Serum Cholesterol (mg/dl)", min_value=0, max_value=600, value=200)
        
    with col2:
        fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["False", "True"])
        restecg = st.selectbox("Resting ECG Results", 
                             ["normal", "st-t abnormality", "lv hypertrophy"])
        thalch = st.number_input("Maximum Heart Rate Achieved", min_value=0, max_value=300, value=150)
        exang = st.selectbox("Exercise Induced Angina", ["False", "True"])
        oldpeak = st.number_input("ST Depression Induced by Exercise", min_value=0.0, max_value=10.0, value=0.0)
        slope = st.selectbox("Slope of Peak Exercise ST Segment", 
                           ["upsloping", "flat", "downsloping"])
        ca = st.number_input("Number of Major Vessels Colored by Fluoroscopy", min_value=0, max_value=4, value=0)
        thal = st.selectbox("Thalassemia", ["normal", "fixed defect", "reversable defect"])
    
    # Convert categorical variables using mappings
    sex = feature_mappings['sex'][sex]
    cp = feature_mappings['cp'][cp]
    fbs = feature_mappings['fbs'][fbs.lower()]
    restecg = feature_mappings['restecg'][restecg]
    exang = feature_mappings['exang'][exang.lower()]
    slope = feature_mappings['slope'][slope]
    thal = feature_mappings['thal'][thal]
    
    # Create input DataFrame with correct feature names
    feature_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalch',
                    'exang', 'oldpeak', 'slope', 'ca', 'thal']
    
    input_data = pd.DataFrame([[age, sex, cp, trestbps, chol, fbs, restecg, thalch,
                               exang, oldpeak, slope, ca, thal]], 
                             columns=feature_names)
    
    # Scale the input
    input_scaled = scaler.transform(input_data)
    
    if st.button("Predict"):
        st.subheader("Prediction Results")
        
        # Get base predictions
        predictions = {}
        for name, model in models.items():
            pred_proba = model.predict_proba(input_scaled)[0][1]
            predictions[name] = pred_proba
        
        # Calculate risk factors
        risk_count, critical_count, risk_details = calculate_risk_factors(
            age, trestbps, chol, fbs, exang, oldpeak, ca, thal, cp
        )
        
        # Calibrate probabilities
        calibrated_predictions = {
            name: calibrate_probability(prob, risk_count)
            for name, prob in predictions.items()
        }
        
        # Calculate weighted ensemble prediction
        weights = {
            'svm': 0.35,
            'random_forest': 0.3,
            'logistic_regression': 0.25,
            'xgboost': 0.1
        }
        
        ensemble_prob = sum(calibrated_predictions[name] * weights[name] 
                          for name in weights.keys() if name in calibrated_predictions)
        
        # Decision logic
        base_threshold = 0.45
        risk_adjustment = min(0.15, risk_count * 0.02)
        threshold = max(0.3, base_threshold - risk_adjustment)
        
        has_disease = (
            ensemble_prob > threshold or
            (critical_count >= 2 and ensemble_prob > 0.4) or
            risk_count >= 5 or
            (risk_count >= 4 and ensemble_prob > 0.35)
        )
        
        # Display the main prediction
        if has_disease:
            st.markdown(
                f"""
                <div style="background-color: #ff4b4b; color: white; padding: 20px; border-radius: 10px; text-align: center;">
                    <h2>🚨 Heart Disease Detected: YES</h2>
                    <h3>Probability: {ensemble_prob:.1%}</h3>
                    <p>Immediate medical consultation is recommended</p>
                </div>
                """, 
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"""
                <div style="background-color: #00cc66; color: white; padding: 20px; border-radius: 10px; text-align: center;">
                    <h2>✅ Heart Disease Detected: NO</h2>
                    <h3>Probability: {ensemble_prob:.1%}</h3>
                    <p>Continue maintaining a healthy lifestyle</p>
                </div>
                """, 
                unsafe_allow_html=True
            )
        
        # Display risk factors
        st.subheader("Risk Factor Analysis")
        if risk_details:
            for factor, severity in risk_details:
                if severity == "High":
                    st.markdown(f'<p class="high-risk">🔴 {factor} - {severity} Risk</p>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<p class="moderate-risk">🟡 {factor} - {severity} Risk</p>', unsafe_allow_html=True)
        else:
            st.markdown('<p class="low-risk">✅ No significant risk factors identified</p>', unsafe_allow_html=True)
        
        # Display individual model predictions
        st.subheader("Individual Model Predictions")
        
        col1, col2, col3, col4 = st.columns(4)
        
        for (name, prob), col in zip(calibrated_predictions.items(), [col1, col2, col3, col4]):
            with col:
                name = name.replace('_', ' ').title()
                st.markdown(f"**{name}**")
                st.progress(float(prob))
                st.markdown(f"Probability: {prob:.1%}")
        
        # Health Recommendations
        st.subheader("Health Recommendations")
        
        st.markdown("""
        🏥 **Medical Follow-up:**
        * Schedule a comprehensive cardiac evaluation
        * Regular blood pressure monitoring
        * Periodic cholesterol checks
        * Blood sugar monitoring
        
        🏃‍♂️ **Lifestyle Changes:**
        * Regular moderate exercise (with medical clearance)
        * Heart-healthy diet low in saturated fats
        * Maintain healthy weight
        * Stress management
        * Adequate sleep (7-9 hours)
        
        ⚠️ **Warning Signs to Watch:**
        * Chest pain or discomfort
        * Shortness of breath
        * Unusual fatigue
        * Irregular heartbeat
        
        👨‍⚕️ **When to Seek Immediate Care:**
        * Severe chest pain
        * Difficulty breathing
        * Fainting or severe dizziness
        * Cold sweats with chest discomfort
        """)

if __name__ == "__main__":
    main()
