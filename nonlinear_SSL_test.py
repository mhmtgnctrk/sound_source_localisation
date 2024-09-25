from sympy.solvers import solve
from sympy import var, Eq
import numpy as np
import time

b1=time.time()

### 4 vectors that defines the x-y-z coordinates of a 2-D square that lies on X-Z plane an its center is at (0,0,0)
# Define the side length of the square
side_length = 20

# Define the coordinates of the four vertices
v1 = np.array([-side_length/2, 0, -side_length/2])
v2 = np.array([side_length/2, 0, -side_length/2])
v3 = np.array([side_length/2, 0, side_length/2])
v4 = np.array([-side_length/2, 0, side_length/2])

#print("v1:", v1)
#print("v2:", v2)
#print("v3:", v3)
#print("v4:", v4)

### create sound location 
# Choose random x and z values between -100 and 100
x = np.random.randint(-100, 101)
z = np.random.randint(-100, 101)

#Choose random y value between 500 and 300
y = np.random.randint(300, 500)

# Create the sound location array
sound_location = np.array([x, y, z])

print("Sound location:", sound_location)

### distance between microphones and sound location in an array
distances = []
for mic in [v1, v2, v3, v4]:
  distance = np.linalg.norm(sound_location - mic)
  distances.append(distance)

print("Distances:", distances)

### C is sound speed in m/s C = 331.3*(1+T/273.15)^(1/2)
T = 20  # Temperature in Celsius
C_mps = 331.3 * (1 + T / 273.15) ** 0.5  # Sound speed in m/s
C_mms = C_mps * 1000  # Convert sound speed to mm/s

print("Sound speed (C) in mm/s:", C_mms)

### the travel time of sound from source to each microphone
travel_times = []
for distance in distances:
  travel_time = distance / C_mms
  travel_times.append(travel_time)

time_differences = []
for travel_time in travel_times:
  time_difference = min(travel_times) - travel_time
  time_differences.append(time_difference)
print("Travel times:", travel_times)
print("Travle time differences:", time_differences)

### the reference micrphone decision
# Find the minimum travel time and its index
min_travel_time = min(travel_times)
reference_mic_index = travel_times.index(min_travel_time)

# Print the reference microphone index
print("Reference microphone index:", reference_mic_index)

# Get the corresponding microphone coordinates
reference_mic = [v1, v2, v3, v4][reference_mic_index]
print("Reference microphone coordinates:", reference_mic)

### An array that have range differences of the sound travel distances from the reference microphone
range_differences = []
for i in range(len(distances)):
  range_difference = distances[i] - distances[reference_mic_index]
  range_differences.append(range_difference)

print("Range differences:", range_differences)

### Building 3 equations using sympy. 
x_s, y_s, z_s = var('x, y, z')

# Reference microphone coordinates
ref_x, ref_y, ref_z = reference_mic

# Build the equations
equations = []
for i in range(len(distances)):
  if i != reference_mic_index:
    mic_coords = [v1, v2, v3, v4][i]
    mic_x, mic_y, mic_z = mic_coords
    equation = Eq(
        ((x_s - ref_x)**2 + (y_s - ref_y)**2 + (z_s - ref_z)**2)**0.5 -
        ((x_s - mic_x)**2 + (y_s - mic_y)**2 + (z_s - mic_z)**2)**0.5+
        range_differences[i],0
    )
    equations.append(equation)
b2=time.time()
print("Ön hesap süresi: ", b2-b1," sn")
a1=time.time()
# Solve the equations
solution = solve(equations, (x_s, y_s, z_s))
a2=time.time()
print("Solution:", solution, "\n", "Time (s):", a2-a1)