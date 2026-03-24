import streamlit as st
import pandas as pd
import joblib
from sklearn.ensemble import GradientBoostingRegressor

st.title("⚡ Smart Microgrid Energy System")

# Load training data
X_train = joblib.load("X_train.pkl")
y_train = joblib.load("y_train.pkl")

# Rebuild model (SAFE)
model = GradientBoostingRegressor()
model.fit(X_train, y_train)

st.subheader("🔮 Predict Energy Consumption")

# User inputs
solar = st.number_input("Solar Generation", value=0.0)
wind = st.number_input("Wind Generation", value=50.0)
hour = st.slider("Hour", 0, 23, 12)
day = st.slider("Day", 1, 31, 1)
month = st.slider("Month", 1, 12, 1)
weekday = st.slider("Weekday (0=Mon)", 0, 6, 0)
lag_1 = st.number_input("Last Hour Consumption", value=20.0)
lag_24 = st.number_input("Yesterday Same Hour", value=20.0)
lag_48 = st.number_input("2 Days Ago", value=20.0)
lag_72 = st.number_input("3 Days Ago", value=20.0)

# Input dataframe
input_data = pd.DataFrame([[
    solar, wind, hour, day, month, weekday,
    lag_1, lag_24, lag_48, lag_72
]], columns=[
    'Solar', 'Wind', 'hour', 'day', 'month', 'weekday',
    'lag_1', 'lag_24', 'lag_48', 'lag_72'
])

# Prediction
if st.button("Predict"):
    prediction = model.predict(input_data)[0]
    st.success(f"Predicted Consumption: {prediction:.2f}")
