import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import plotly.express as px

# Simulated functions for loading model and predictions (Replace with your actual model)
def load_model():
    # This function loads the trained forecasting model (e.g., a machine learning model)
    pass

def predict_wind_power(input_data):
    # This function predicts wind power generation based on input data
    # For now, let's simulate some predictions
    return np.random.normal(loc=100, scale=20, size=24)  # Simulated forecast for 24 hours

def load_actual_data():
    # Simulate actual wind power data for the past 24 hours (replace with your actual data)
    return np.random.normal(loc=100, scale=15, size=24)

# Streamlit UI components
st.title("Wind Power Generation Forecasting")

# Sidebar inputs
st.sidebar.header("Input Parameters")
wind_speed = st.sidebar.slider("Wind Speed (m/s)", min_value=0, max_value=25, value=10)
temperature = st.sidebar.slider("Temperature (°C)", min_value=-20, max_value=40, value=15)
humidity = st.sidebar.slider("Humidity (%)", min_value=0, max_value=100, value=60)

# Prediction and Data Display
st.header("Forecasted vs Actual Power Generation")

# Generate predictions and actual data
predictions = predict_wind_power({'wind_speed': wind_speed, 'temperature': temperature, 'humidity': humidity})
actual_data = load_actual_data()

# Display forecast vs actual plot
time = np.arange(1, 25)  # Simulated 24 hours

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(time, predictions, label="Forecasted", color="blue", linestyle='--')
ax.plot(time, actual_data, label="Actual", color="green")
ax.set_title("Wind Power Generation: Forecast vs Actual")
ax.set_xlabel("Time (Hours)")
ax.set_ylabel("Power Generation (MW)")
ax.legend()
st.pyplot(fig)

# Metrics display
mae = mean_absolute_error(actual_data, predictions)
rmse = np.sqrt(mean_squared_error(actual_data, predictions))
r2 = r2_score(actual_data, predictions)

st.subheader("Model Performance")
st.write(f"Mean Absolute Error (MAE): {mae:.2f} MW")
st.write(f"Root Mean Squared Error (RMSE): {rmse:.2f} MW")
st.write(f"R-squared: {r2:.2f}")

# Interactive Map for wind conditions (can integrate real weather data API or geo data)
st.subheader("Wind Conditions Overview")
fig_map = px.scatter_geo(lat=[40.7128], lon=[-74.0060], text=["New York"], title="Wind Speed Location Map")
st.plotly_chart(fig_map)

# Forecasting Confidence Intervals
st.subheader("Confidence Interval for Predictions")
lower_bound = predictions - 20  # Placeholder: lower bound of forecast confidence
upper_bound = predictions + 20  # Placeholder: upper bound of forecast confidence
st.line_chart(pd.DataFrame({
    "Forecasted Power": predictions,
    "Lower Bound": lower_bound,
    "Upper Bound": upper_bound
}))

# Scenario Simulation: Adjust inputs and see forecast changes
st.subheader("Scenario Simulation")
simulated_wind_speed = st.slider("Simulate Wind Speed (m/s)", min_value=0, max_value=25, value=10)
simulated_forecast = predict_wind_power({'wind_speed': simulated_wind_speed, 'temperature': temperature, 'humidity': humidity})
st.line_chart(simulated_forecast)

# Feature Importance (For ML models)
st.subheader("Feature Importance Visualization")
# Example bar chart for feature importance (replace with your actual model's feature importance)
feature_importance = {'Wind Speed': 0.4, 'Temperature': 0.3, 'Humidity': 0.3}
fig_feature_importance = px.bar(x=list(feature_importance.keys()), y=list(feature_importance.values()), title="Feature Importance")
st.plotly_chart(fig_feature_importance)

# Run the app with:
# streamlit run wind_power_forecasting_app.py
