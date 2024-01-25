import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


class HVACSimulatorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("HVAC Simulator")

        self.dataset_path = tk.StringVar()
        self.target_variable = tk.StringVar(value='Heating Load')

        self.mean_outdoor_temp_var = tk.StringVar(value="7")
        self.heating_temp_setting_var = tk.StringVar(value="25")
        self.heating_energy_efficiency_ratio_var = tk.StringVar(value="2.5")
        self.indoor_lighting_power_var = tk.StringVar(value="5")
        self.teaching_equipment_power_var = tk.StringVar(value="4.5")
        self.occupant_density_var = tk.StringVar(value="0.3")
        self.setpoint_var = tk.StringVar(value="22")

        self.create_widgets()

    def create_widgets(self):
        # User Input for Mean Outdoor Temperature
        outdoor_temp_label = ttk.Label(
            self.root, text="Mean Outdoor Temperature (°C):")
        outdoor_temp_label.grid(row=0, column=0, sticky="w", padx=10, pady=5)

        outdoor_temp_entry = ttk.Entry(
            self.root, textvariable=self.mean_outdoor_temp_var)
        outdoor_temp_entry.grid(row=0, column=1, padx=5, pady=5)

        # User Input for Heating Temperature Setting
        heating_temp_label = ttk.Label(
            self.root, text="Heating Temperature Setting (°C):")
        heating_temp_label.grid(row=1, column=0, sticky="w", padx=10, pady=5)

        heating_temp_entry = ttk.Entry(
            self.root, textvariable=self.heating_temp_setting_var)
        heating_temp_entry.grid(row=1, column=1, padx=5, pady=5)

        # User Input for Heating Energy Efficiency Ratio
        efficiency_ratio_label = ttk.Label(
            self.root, text="Heating Energy Efficiency Ratio:")
        efficiency_ratio_label.grid(
            row=2, column=0, sticky="w", padx=10, pady=5)

        efficiency_ratio_entry = ttk.Entry(
            self.root, textvariable=self.heating_energy_efficiency_ratio_var)
        efficiency_ratio_entry.grid(row=2, column=1, padx=5, pady=5)

        # User Input for Indoor Lighting Power
        lighting_power_label = ttk.Label(
            self.root, text="Indoor Lighting Power (W/m2):")
        lighting_power_label.grid(row=3, column=0, sticky="w", padx=10, pady=5)

        lighting_power_entry = ttk.Entry(
            self.root, textvariable=self.indoor_lighting_power_var)
        lighting_power_entry.grid(row=3, column=1, padx=5, pady=5)

        # User Input for Teaching Equipment Power
        equipment_power_label = ttk.Label(
            self.root, text="Teaching Equipment Power (W/m2):")
        equipment_power_label.grid(
            row=4, column=0, sticky="w", padx=10, pady=5)

        equipment_power_entry = ttk.Entry(
            self.root, textvariable=self.teaching_equipment_power_var)
        equipment_power_entry.grid(row=4, column=1, padx=5, pady=5)

        # User Input for Occupant Density
        occupant_density_label = ttk.Label(
            self.root, text="Occupant Density (People/m2):")
        occupant_density_label.grid(
            row=5, column=0, sticky="w", padx=10, pady=5)

        occupant_density_entry = ttk.Entry(
            self.root, textvariable=self.occupant_density_var)
        occupant_density_entry.grid(row=5, column=1, padx=5, pady=5)

        # User Input for Setpoint Temperature
        setpoint_label = ttk.Label(
            self.root, text="Setpoint Temperature (°C):")
        setpoint_label.grid(row=6, column=0, sticky="w", padx=10, pady=5)

        setpoint_entry = ttk.Entry(
            self.root, textvariable=self.setpoint_var)
        setpoint_entry.grid(row=6, column=1, padx=5, pady=5)

        # File Selection
        file_label = ttk.Label(self.root, text="Select Dataset:")
        file_label.grid(row=7, column=0, sticky="w", padx=10, pady=5)

        file_entry = ttk.Entry(
            self.root, textvariable=self.dataset_path, width=40, state="readonly")
        file_entry.grid(row=7, column=1, padx=5, pady=5)

        file_button = ttk.Button(
            self.root, text="Browse", command=self.browse_file)
        file_button.grid(row=7, column=2, padx=5, pady=5)

        # Target Variable Selection
        target_label = ttk.Label(self.root, text="Select Target Loads:")
        target_label.grid(row=8, column=0, sticky="w", padx=10, pady=5)

        target_combobox = ttk.Combobox(self.root, textvariable=self.target_variable, values=[
                                       'Heating Load', 'Cooling Load'])
        target_combobox.grid(row=8, column=1, padx=5, pady=5)

        # Run Simulation Button
        simulate_button = ttk.Button(
            self.root, text="Run Simulation", command=self.run_simulation)
        simulate_button.grid(row=9, column=0, columnspan=3, pady=10)

    def browse_file(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("CSV files", "*.csv")])
        self.dataset_path.set(file_path)

    def run_simulation(self):
        mean_outdoor_temperature = float(self.mean_outdoor_temp_var.get())

        heating_temperature_setting = float(
            self.heating_temp_setting_var.get())

        heating_energy_efficiency_ratio = float(
            self.heating_energy_efficiency_ratio_var.get())

        indoor_lighting_power = float(self.indoor_lighting_power_var.get())

        teaching_equipment_power = float(
            self.teaching_equipment_power_var.get())

        occupant_density = float(self.occupant_density_var.get())

        setpoint = float(self.setpoint_var.get())

        dataset_path = self.dataset_path.get()

        if not dataset_path:
            tk.messagebox.showerror("Error", "Please select a dataset.")
            return

        # Load dataset
        data = pd.read_csv(dataset_path)

        # Basic applied settings
        # mean_outdoor_temperature = 7
        # heating_temperature_setting = 25
        # heating_energy_efficiency_ratio = 2.5
        # indoor_lighting_power = 5
        # teaching_equipment_power = 4.5
        # occupant_density = 0.3

        # Feature engineering
        # Assuming outdoor temperature is constant
        data['temperature'] = mean_outdoor_temperature
        data['heating_setting'] = heating_temperature_setting
        data['energy_efficiency_ratio'] = heating_energy_efficiency_ratio
        data['lighting_power'] = indoor_lighting_power
        data['equipment_power'] = teaching_equipment_power
        data['occupant_density'] = occupant_density

        # Choose the target variable: 'heating_load' or 'cooling_load'
        target_variable = self.target_variable.get()

        # Split the data into features and target variable
        features = data[['temperature', 'heating_setting', 'energy_efficiency_ratio',
                        'lighting_power', 'equipment_power', 'occupant_density']]
        target = data[target_variable]

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            features, target, test_size=0.2, random_state=42)

        # Create a Random Forest Regressor model
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

        # Train the model
        rf_model.fit(X_train, y_train)

        # Initialize simulation variables
        # setpoint = 22  # Desired temperature
        current_temperature = mean_outdoor_temperature  # Initial temperature
        integral = 0
        prev_error = 0

        # Lists to store simulation results
        simulated_temperatures = [current_temperature]
        enhanced_energy_consumption = [0]

        for i in range(len(X_test)):
            # Simulate PID control
            error = setpoint - current_temperature
            integral += error
            derivative = error - prev_error

            # Calculate control signal using PID
            pid_control = 0.1 * error + 0.01 * integral + 0.01 * derivative

            # Adjust HVAC settings based on control signal
            heating_setting = max(
                0, min(30, heating_temperature_setting + pid_control))

            # Use Random Forest model to predict energy consumption
            features_current = np.array([[mean_outdoor_temperature, heating_setting, heating_energy_efficiency_ratio,
                                          indoor_lighting_power, teaching_equipment_power, occupant_density]])
            predicted_energy_consumption = rf_model.predict(features_current)

            # Update simulated temperature
            current_temperature += (heating_energy_efficiency_ratio *
                                    pid_control) / 10  # Simplified temperature update

            # Record simulation results
            simulated_temperatures.append(current_temperature)
            enhanced_energy_consumption.append(
                predicted_energy_consumption[0])

            # Update previous error for PID control
            prev_error = error

        # Plot results
        plt.figure(figsize=(12, 6))

        # Plot Temperature
        plt.subplot(2, 1, 1)
        plt.plot(simulated_temperatures,
                 label='Simulated Temperature', color='blue')
        plt.axhline(y=setpoint, linestyle='--', color='red', label='Setpoint')
        plt.title('Simulated Temperature with Hybrid PID + Random Forest Control')
        plt.xlabel('Time')
        plt.ylabel('Temperature (°C)')
        plt.legend()

        # Plot Enhanced Energy Consumption
        plt.subplot(2, 1, 2)
        plt.plot(enhanced_energy_consumption,
                 label='Enhanced Energy Consumption', color='orange')
        plt.title(
            'Enhanced Energy Consumption with Hybrid PID + Random Forest Control')
        plt.xlabel('Time')
        plt.ylabel('Enhanced Energy Consumption')
        plt.legend()

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    root = tk.Tk()
    app = HVACSimulatorApp(root)
    root.mainloop()
