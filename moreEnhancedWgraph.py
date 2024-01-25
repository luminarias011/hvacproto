import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

# Hardcoded dataset
data = pd.DataFrame({
    'Temperature': [24.5, 25.0, 23.0, 24.5, 26.0, 27.5, 23.5, 25.5, 26.5, 22.0],
    'Humidity': [40, 42, 38, 41, 45, 48, 37, 43, 46, 35],
    'Occupancy': [1, 0, 1, 1, 0, 0, 1, 1, 0, 0],
    'EnergyConsumption': [120, 110, 130, 125, 105, 100, 135, 121, 108, 95]
})

X = data.drop('EnergyConsumption', axis=1)
y = data['EnergyConsumption']

# Random Forest Regressor
rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)

# Train sa ML - regressor on the dataset
rf_regressor.fit(X, y)

# Simulate real-time control with PID
def pid_control(target_temperature, current_temperature):
    
    Kp = 0.1  # Proportional gain
    Ki = 0.01  # Integral gain
    Kd = 0.01  # Derivative gain
    
    integral = 0
    previous_error = 0

    # Initialize lists to store data for plotting
    desired_temperatures = []
    current_temperatures = []
    predicted_energy_consumption = []

    for _ in range(100):  # Simulate control for a limited number of time steps
        # errors
        error = target_temperature - current_temperature

        # Calculation ng PID with error
        proportional = Kp * error
        integral += Ki * error
        derivative = Kd * (error - previous_error)

        # Calculation sa PID control signal
        control_signal = proportional + integral + derivative

        # Apply the control signal to the HVAC system (simulated control)
        current_temperature += control_signal

        # Use the trained model to predict energy consumption
        predicted_energy = rf_regressor.predict(pd.DataFrame({'Temperature': [current_temperature], 'Humidity': [42], 'Occupancy': [1]}))

        print(f"Desired Temperature: {target_temperature:.2f}°C, Current Temperature: {current_temperature:.2f}°C, Predicted Energy Consumption: {predicted_energy[0]:.2f} kWh")

        #  for plotting
        desired_temperatures.append(target_temperature)
        current_temperatures.append(current_temperature)
        predicted_energy_consumption.append(predicted_energy[0])

        # Update previous error for the next iteration
        previous_error = error

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()
    
    ax1.plot(desired_temperatures, label='Desired Temperature', linestyle='--', marker='o', color='tab:blue')
    ax1.plot(current_temperatures, label='Current Temperature', marker='x', color='tab:orange')
    ax2.plot(predicted_energy_consumption, label='Energy Consumption', linestyle='-', marker='s', color='tab:green')

    ax1.set_xlabel('Time Steps')
    ax1.set_ylabel('Temperature (°C)', color='tab:blue')
    ax2.set_ylabel('Energy Consumption (kWh)', color='tab:green')

    plt.title('Optimized Energy Consumption with Hybrid Algorithm')
    fig.legend(loc="upper left")
    plt.show()
    
    # Plot the data
    # plt.figure(figsize=(10, 6))
    # plt.plot(desired_temperatures, label='Desired Temperature', linestyle='--', marker='o')
    # plt.plot(current_temperatures, label='Current Temperature', marker='x')
    # plt.xlabel('Time Steps')
    # plt.ylabel('Temperature (°C)')
    # plt.title('Temperature Control with PID')
    # plt.legend()
    # plt.show()

# Example usage of PID control with prediction from Random Forest
target_temp = 18.0  # Set the desired temperature
initial_temp = 25.0  # Initial room temperature
pid_control(target_temp, initial_temp)
