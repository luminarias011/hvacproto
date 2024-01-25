import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Create a synthetic dataset with the given parameters
num_samples = 1000

mean_outdoor_temperature = 7
heating_temperature_setting = 25
heating_energy_efficiency_ratio = 2.5
indoor_lighting_power = 5
teaching_equipment_power = 4.5
occupant_density = 0.3

# Generate synthetic data
data = pd.DataFrame({
    'temperature': np.random.normal(mean_outdoor_temperature, 5, num_samples),
    'heating_setting': heating_temperature_setting,
    'energy_efficiency_ratio': heating_energy_efficiency_ratio,
    'lighting_power': indoor_lighting_power,
    'equipment_power': teaching_equipment_power,
    'occupant_density': occupant_density,
    'heating_load': np.random.normal(50, 10, num_samples),  # Adjust this based on your actual data distribution
    'cooling_load': np.random.normal(30, 5, num_samples)  # Adjust this based on your actual data distribution
})

# Choose the target variable: 'heating_load' or 'cooling_load'
target_variable = 'heating_load'  # or 'cooling_load'
target = data[target_variable]

# Features
features = data[['temperature', 'heating_setting', 'energy_efficiency_ratio', 'lighting_power', 'equipment_power', 'occupant_density']]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Create a Random Forest Regressor model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model
rf_model.fit(X_train, y_train)

# Make predictions on the test set
predictions = rf_model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, predictions)
print(f'Mean Squared Error: {mse}')

# Visualize the predictions against the actual values
plt.scatter(y_test, predictions)
plt.xlabel(f'Actual {target_variable.capitalize()}')
plt.ylabel(f'Predicted {target_variable.capitalize()}')
plt.title(f'Actual vs Predicted {target_variable.capitalize()}')
plt.show()

# Use the trained model for energy consumption optimization
enhanced_energy_consumption = rf_model.predict(features)

# Add the enhanced energy consumption to the original dataset
data['enhanced_energy_consumption'] = enhanced_energy_consumption

# Display the enhanced energy consumption results
print(data[['temperature', 'heating_setting', 'energy_efficiency_ratio', 'lighting_power', 'equipment_power', 'occupant_density', target_variable, 'enhanced_energy_consumption']])
