# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# Hardcoded dataset
data = pd.DataFrame({
    'Temperature': [24.5, 25.0, 23.0, 24.5, 26.0, 27.5, 23.5, 25.5, 26.5, 22.0],
    'Humidity': [40, 42, 38, 41, 45, 48, 37, 43, 46, 35],
    'Occupancy': [1, 0, 1, 1, 0, 0, 1, 1, 0, 0],
    'EnergyConsumption': [120, 110, 130, 125, 105, 100, 135, 121, 108, 95]
})

# Split data into features (X) and target variable (y)
X = data.drop('EnergyConsumption', axis=1)
y = data['EnergyConsumption']

# Create a Random Forest Regressor
rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the regressor on the dataset
rf_regressor.fit(X, y)

# Simulate real-time control with PID
def pid_control(target_temperature, current_temperature):
    # PID control parameters
    Kp = 0.1  # Proportional gain
    Ki = 0.01  # Integral gain
    Kd = 0.01  # Derivative gain

    # Initialize PID variables
    integral = 0
    previous_error = 0

    for _ in range(100):  # Simulate control for a limited number of time steps
        # Calculate the error
        error = target_temperature - current_temperature

        # Calculate PID terms
        proportional = Kp * error
        integral += Ki * error
        derivative = Kd * (error - previous_error)

        # Calculate the control signal
        control_signal = proportional + integral + derivative

        # Apply the control signal to the HVAC system (simulated control)
        current_temperature += control_signal

        # Update previous error for the next iteration
        previous_error = error

        # Use the trained model to predict energy consumption
        predicted_energy = rf_regressor.predict(pd.DataFrame({'Temperature': [current_temperature], 'Humidity': [42], 'Occupancy': [1]}))
        print(f"Current Temperature: {current_temperature:.2f}°C, Predicted Energy Consumption: {predicted_energy[0]:.2f} kWh")

# Example usage of PID control with prediction from Random Forest
target_temp = 24.0  # Set the desired temperature
initial_temp = 25.0  # Initial room temperature
pid_control(target_temp, initial_temp)
