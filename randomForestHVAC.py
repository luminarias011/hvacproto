import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

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

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Random Forest Regressor
rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the regressor on the training data
rf_regressor.fit(X_train, y_train)

# Make predictions on the test data
y_pred = rf_regressor.predict(X_test)

# Calculate the Mean Squared Error
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error: {:.2f}".format(mse))

# Use the trained model for HVAC energy optimization
# You can use the trained model to predict energy consumption based on current sensor readings
current_sensors = pd.DataFrame({'Temperature': [25.5], 'Humidity': [40.0], 'Occupancy': [1]})
predicted_energy = rf_regressor.predict(current_sensors)
print("Predicted Energy Consumption: {:.2f}".format(predicted_energy[0]))

# Create a line graph for initial temperature, desired temperature, and optimized energy consumption
initial_temperature = data['Temperature']
desired_temperature = [18.5] * len(initial_temperature)
optimized_energy_consumption = rf_regressor.predict(data[['Temperature', 'Humidity', 'Occupancy']])

plt.figure(figsize=(10, 5))
plt.plot(initial_temperature, label='Initial Temperature')
plt.plot(desired_temperature, label='Desired Temperature')
plt.plot(optimized_energy_consumption, label='Optimized Energy Consumption')
plt.xlabel('Steps')
plt.ylabel('Value')
plt.legend()
plt.title('Random Forest HVAC Energy Optimization')
plt.show()
