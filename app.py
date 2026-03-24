import streamlit as st
import pandas as pd
import joblib
from sklearn.ensemble import GradientBoostingRegressor
import matplotlib.pyplot as plt

st.set_page_config(page_title="Smart Microgrid", layout="wide")

st.title("⚡ Smart Microgrid Energy System")
st.markdown("### AI-powered Energy Prediction & Optimization")

# Load training data
X_train = joblib.load("X_train.pkl")
y_train = joblib.load("y_train.pkl")

model = GradientBoostingRegressor()
model.fit(X_train, y_train)

# Layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("🔮 Input Parameters")

    solar = st.number_input("Solar Generation", value=0.0)
    wind = st.number_input("Wind Generation", value=50.0)
    hour = st.slider("Hour", 0, 23, 12)
    day = st.slider("Day", 1, 31, 1)
    month = st.slider("Month", 1, 12, 1)
    weekday = st.slider("Weekday", 0, 6, 0)

    lag_1 = st.number_input("Last Hour Consumption", value=20.0)
    lag_24 = st.number_input("Yesterday Same Hour", value=20.0)
    lag_48 = st.number_input("2 Days Ago", value=20.0)
    lag_72 = st.number_input("3 Days Ago", value=20.0)

with col2:
    st.subheader("⚡ System Output")

if st.button("🚀 Run Smart System"):

    input_data = pd.DataFrame([[
        solar, wind, hour, day, month, weekday,
        lag_1, lag_24, lag_48, lag_72
    ]], columns=[
        'Solar', 'Wind', 'hour', 'day', 'month', 'weekday',
        'lag_1', 'lag_24', 'lag_48', 'lag_72'
    ])

    prediction = model.predict(input_data)[0]

    total_generation = solar + wind
    energy_diff = total_generation - prediction

    # Battery system
    battery_capacity = 100
    battery_level = 50

    if energy_diff > 0:
        battery_level = min(battery_capacity, battery_level + energy_diff)
        grid_usage = 0
        status = "🔋 Charging Battery"
    else:
        needed = abs(energy_diff)
        if battery_level >= needed:
            battery_level -= needed
            grid_usage = 0
            status = "🔋 Using Battery"
        else:
            grid_usage = needed - battery_level
            battery_level = 0
            status = "⚡ Using Grid"

    # METRICS DISPLAY
    colA, colB, colC = st.columns(3)

    colA.metric("Predicted Consumption", f"{prediction:.2f}")
    colB.metric("Total Generation", f"{total_generation:.2f}")
    colC.metric("Energy Difference", f"{energy_diff:.2f}")

    st.success(status)

    colD, colE = st.columns(2)
    colD.metric("Battery Level", f"{battery_level:.2f}")
    colE.metric("Grid Usage", f"{grid_usage:.2f}")

    # GRAPH
    fig, ax = plt.subplots()
    ax.bar(['Generation', 'Consumption'], [total_generation, prediction])
    st.pyplot(fig)
