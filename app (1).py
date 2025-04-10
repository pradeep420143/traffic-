import streamlit as st
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf

# Set page config
st.set_page_config(page_title="🚦 Traffic Signal Predictor", layout="centered")

# Title
st.title("🚦 AI-based Traffic Signal Predictor")

# Load models and encoders
@st.cache_resource
def load_models():
    model_signal = tf.keras.models.load_model('traffic_signal_classification.h5')
    model_timer = tf.keras.models.load_model('traffic_signal_timer.h5')
    preprocessor = joblib.load('preprocessor.pkl')
    label_encoder = joblib.load('label_encoder.pkl')
    return model_signal, model_timer, preprocessor, label_encoder

model_signal, model_timer, preprocessor, label_encoder = load_models()

# Input form
with st.form("traffic_form"):
    st.subheader("📝 Enter Traffic Information:")

    vehicle_count = st.number_input("Vehicle Count", min_value=0, value=10)
    pedestrian_count = st.number_input("Pedestrian Count", min_value=0, value=5)
    emergency_vehicle = st.selectbox("Emergency Vehicle Presence", [0, 1], format_func=lambda x: "Yes" if x else "No")
    weather_condition = st.selectbox("Weather Condition", ['Clear', 'Rain', 'Fog', 'Snow'])
    time_of_day = st.selectbox("Time of Day", ['Morning', 'Afternoon', 'Evening', 'Night'])
    traffic_flow_rate = st.slider("Traffic Flow Rate", 0, 100, value=50)

    submit = st.form_submit_button("Predict")

# Prediction
if submit:
    # Prepare input
    input_dict = {
        'Vehicle_Count': [vehicle_count],
        'Pedestrian_Count': [pedestrian_count],
        'Emergency_Vehicle_Presence': [emergency_vehicle],
        'Weather_Condition': [weather_condition],
        'Time_of_Day': [time_of_day],
        'Traffic_Flow_Rate': [traffic_flow_rate]
    }

    input_df = pd.DataFrame(input_dict)
    input_transformed = preprocessor.transform(input_df)

    # Predict signal
    signal_prediction = model_signal.predict(input_transformed)
    signal_label = label_encoder.inverse_transform([np.argmax(signal_prediction)])[0]

    # Predict timer
    timer_prediction = model_timer.predict(input_transformed)
    timer_value = np.round(timer_prediction[0][0], 2)

    # Results
    st.success("✅ Prediction Complete!")
    st.markdown(f"**🚦 Predicted Signal State:** `{signal_label}`")
    st.markdown(f"**⏱ Timer Duration:** `{timer_value} seconds`")
