import matplotlib.pyplot as plt

# Constants for the PID controller
Kp = 0.5  # Proportional gain
Ki = 0.1  # Integral gain
Kd = 0.2  # Derivative gain

# Setpoints
desired_temperature = 25.0  # Desired temperature in Celsius

# Initial values
initial_temperature = 30.0  # Initial temperature in Celsius
current_temperature = initial_temperature
energy_consumption = 0.0

# Lists to store data for the graph
time_steps = []
temperatures = []
energy_consumptions = []

# PID controller parameters
previous_error = 0
integral = 0

# Simulation parameters
time = 0
end_time = 60  # Total simulation time in minutes

# Simulation loop
while time < end_time:
    error = desired_temperature - current_temperature

    # Calculate control signal
    control_signal = Kp * error + Ki * integral + Kd * (error - previous_error)

    # Update integral
    integral += error

    # Simulate HVAC system and energy consumption
    energy_consumption += abs(control_signal)
    current_temperature += control_signal

    # Store data for the graph
    time_steps.append(time)
    temperatures.append(current_temperature)
    energy_consumptions.append(energy_consumption)

    # Update time
    time += 1

    # Update previous error
    previous_error = error

# Create the line graph
plt.figure(figsize=(10, 6))
plt.plot(time_steps, [initial_temperature] * len(time_steps), label="Initial Temperature", linestyle='--')
plt.plot(time_steps, [desired_temperature] * len(time_steps), label="Desired Temperature", linestyle='--')
plt.plot(time_steps, energy_consumptions, label="Optimized Energy Consumption")
plt.xlabel("Time (minutes)")
plt.ylabel("Temperature (°C) / Energy Consumption")
plt.legend()
plt.title("HVAC PID Control")
plt.grid(True)

plt.show()
