import pandas as pd

# Define the initial conditions for the planets
initial_data = {
    'Planet': ['Mercury', 'Venus', 'Earth', 'Mars', 'Jupiter', 'Saturn', 'Uranus', 'Neptune'],
    'Position_X': [0.387, 0.723, 1.0, 1.524, 5.203, 9.539, 19.18, 30.07],
    'Position_Y': [0, 0, 0, 0, 0, 0, 0, 0],
    'Position_Z': [0, 0, 0, 0, 0, 0, 0, 0],
    'Velocity_X': [0, 0, 0, 0, 0, 0, 0, 0],
    'Velocity_Y': [0.2056, 0.1745, 0.0167, 0.0933, 0.0473, 0.0348, 0.0119, 0.0061],
    'Velocity_Z': [0, 0, 0, 0, 0, 0, 0, 0],
    'Mass': [0.330, 4.87, 5.97, 0.642, 1898, 568, 86.8, 102]
}

# Create a DataFrame from the initial conditions
df = pd.DataFrame(initial_data)

# Save the DataFrame to a .csv file
df.to_csv('planets_initial_conditions.csv', index=False)
