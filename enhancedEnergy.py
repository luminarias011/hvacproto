# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# Hardcoded dataset (as shown earlier)
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

# Simulate real-time control with PID and energy optimization
def pid_control(target_energy, current_temperature):
    # PID control parameters
    Kp = 0.1  # Proportional gain
    Ki = 0.01  # Integral gain
    Kd = 0.01  # Derivative gain

    # Initialize PID variables
    integral = 0
    previous_error = 0

    for _ in range(100):  # Simulate control for a limited number of time steps
        # Measure current temperature (replace with actual sensor reading)
        current_temperature = get_current_temperature()

        # Calculate the error
        error = target_energy - predict_energy(current_temperature)

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

        # Print information for visualization (replace with actual control action)
        print(f"Current Temperature: {current_temperature:.2f}°C, Predicted Energy Consumption: {predict_energy(current_temperature):.2f} kWh")

# Function to predict energy consumption using the trained Random Forest model
def predict_energy(current_temperature):
    return rf_regressor.predict(pd.DataFrame({'Temperature': [current_temperature], 'Humidity': [42], 'Occupancy': [1] }))[0]

# Simulated function for getting current temperature (replace with actual sensor reading)
def get_current_temperature():
    return np.random.uniform(23.0, 28.0)

# Example usage of PID control with prediction from Random Forest
target_energy = 110  # Set the desired energy consumption
initial_temp = 25.0  # Initial room temperature
pid_control(target_energy, initial_temp)
