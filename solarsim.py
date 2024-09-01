import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.optimize import curve_fit
import matplotlib.animation as animation
import pandas as pd

G=6.67*10**-11

class CelestialBody:
    def __init__(self, mass, pos, vel):
        self.mass = mass
        self.pos = np.array(pos)
        self.vel = np.array(vel)

    def update(self, force, dt, method):
        if method == 'euler':
            acc = force / self.mass
            self.vel += acc * dt
            self.pos += self.vel * dt
        elif method == 'rk4':
            # Here you would need to implement the RK4 method for updating velocity and position
            # This implementation assumes that force is a function of time and position
            k1_v = dt * force / self.mass
            k1_p = dt * self.vel
            k2_v = dt * force / (self.mass + 0.5 * k1_v)
            k2_p = dt * (self.vel + 0.5 * k1_v)
            k3_v = dt * force / (self.mass + 0.5 * k2_v)
            k3_p = dt * (self.vel + 0.5 * k2_v)
            k4_v = dt * force / (self.mass + k3_v)
            k4_p = dt * (self.vel + k3_v)
            
            self.vel += (k1_v + 2*k2_v + 2*k3_v + k4_v) / 6
            self.pos += (k1_p + 2*k2_p + 2*k3_p + k4_p) / 6


class SolarSystem:
    def __init__(self, bodies):
        self.bodies = bodies

    def calculate_forces(self):
        forces = []
        for body1 in self.bodies:
            force = 0
            for body2 in self.bodies:
                if body1 != body2:
                    r = body2.pos - body1.pos
                    force += G * body1.mass * body2.mass * r / np.linalg.norm(r)**3
            forces.append(force)
        return forces

    def update(self, method, dt):
        forces = self.calculate_forces()
        for body, force in zip(self.bodies, forces):
            body.update(force, dt, method)


def run_simulation(solar_system, method, dt, steps):
    # Store the trajectories of each body
    trajectories = {body: [] for body in solar_system.bodies}
    
    for step in range(steps):
        solar_system.update(method, dt)
        for body in solar_system.bodies:
            trajectories[body].append(body.pos.copy())
    
    return trajectories

def plot_histogram(trajectories):
    # Flatten the trajectories and combine all positions into a single list
    positions = [pos for trajectory in trajectories.values() for pos in trajectory]
    
    # Separate the positions into x, y, and z coordinates
    xs = [pos[0] for pos in positions]
    ys = [pos[1] for pos in positions]
    zs = [pos[2] for pos in positions]
    
    # Plot histograms for each coordinate
    plt.figure(figsize=(18, 6))
    
    plt.subplot(131)
    plt.hist(xs, bins=50)
    plt.title('X Positions')
    
    plt.subplot(132)
    plt.hist(ys, bins=50)
    plt.title('Y Positions')
    
    plt.subplot(133)
    plt.hist(zs, bins=50)
    plt.title('Z Positions')
    
    plt.show()


def gradient_descent(cost_function, gradient_function, initial_positions, learning_rate=0.01, num_iterations=1000):
    positions = initial_positions
    cost_history = []

    for i in range(num_iterations):
        # Calculate cost and gradient
        cost = cost_function(positions)
        gradient = gradient_function(positions)
        
        # Update positions
        positions = positions - learning_rate * gradient

        # Save cost
        cost_history.append(cost)

    return positions, cost_history


def newton_method(initial_positions, cost_function, gradient_function, hessian_function, num_iterations):
    positions = initial_positions
    cost_history = []

    for i in range(num_iterations):
        # Calculate cost, gradient and Hessian
        cost = cost_function(positions)
        gradient = gradient_function(positions)
        hessian = hessian_function(positions)
        
        # Add regularization term to the hessian
        hessian += 1e-6 * np.eye(len(hessian))

        # Update positions
        positions = positions - np.linalg.inv(hessian) @ gradient

        # Save cost
        cost_history.append(cost)

    return positions, cost_history


def central_limit_demo(trajectories, num_samples=1000):
    # Flatten the trajectories and combine all positions into a single list
    positions = [pos for trajectory in trajectories.values() for pos in trajectory]

    # Extract the position values from CelestialBody objects
    positions = np.array(positions)

    # Number of positions per sample
    sample_size = positions.shape[0]

    # Number of samples to generate
    num_samples = min(num_samples, sample_size)

    # Generate sample means
    sample_means = []
    for _ in range(num_samples):
        sample = np.random.choice(positions.flatten(), size=sample_size, replace=True)
        sample_mean = np.mean(sample)
        sample_means.append(sample_mean)

    # Plot a histogram of the sample means
    plt.hist(sample_means, bins=50)
    plt.title('Central Limit Theorem Demonstration')
    plt.xlabel('Sample Mean')
    plt.ylabel('Frequency')
    plt.show()


def cost_function(actual_state, desired_state):
    return np.linalg.norm(np.array(actual_state) - np.array(desired_state))

def fit_function(x, a, b):
    return a * x + b  # Example of a linear function


# Read the data from the CSV file
data = pd.read_csv('planets_initial_conditions.csv')

# Create the celestial bodies from the data
bodies = []
for _, row in data.iterrows():
    mass = row['Mass']
    pos = [row['Position_X'], row['Position_Y'], row['Position_Z']]
    vel = [row['Velocity_X'], row['Velocity_Y'], row['Velocity_Z']]
    body = CelestialBody(mass, pos, vel)
    bodies.append(body)

# Create an instance of SolarSystem with the celestial bodies
solar_system = SolarSystem(bodies)

# Define the simulation parameters
method = 'euler'
dt = 0.01
steps = 1000

# Run the simulation and store the trajectories
trajectories = run_simulation(solar_system, method, dt, steps)

# Flatten the trajectories and combine all x-coordinates into a single list
x_coords = [pos[0] for trajectory in trajectories.values() for pos in trajectory]

# Generate some "time" data
times = np.arange(len(x_coords))

# Use curve_fit to find the best-fit parameters
params, params_covariance = curve_fit(fit_function, times, x_coords)

print('Best-fit parameters:', params)




