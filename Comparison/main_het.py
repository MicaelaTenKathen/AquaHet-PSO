import pandas as pd

from Environment.plot import Plots
import time
from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning

simplefilter("ignore", category=ConvergenceWarning)

from PSO.psohet_function import PSOEnvironment
import numpy as np

# Configuration

"""
resolution: map resolution
xs: size on the x-axis of the map
ys: size on the y-axis of the map
GEN: maximum number of code iterations
"""

resolution = 1
xs = 100
ys = 150
navigation_map = np.genfromtxt('../Image/ypacarai_map_bigger.csv')

# Map

"""
grid_or: map grid of the surface without security limits
grid_min: minimum limit of the map
grid_max: maximum limit of the map
grid_max_x: maximum limit on the x-axis of the map
grid_max_y: maximum limit on the y-axis of the map
"""

# Benchmark function

"""
n: number of the ground truth
bench_function: benchmark function values
X_test: coordinates of the points that are analyzed by the benchmark function
secure: map grid of the surface with security limits 
df_bounds: data of the limits of the surface where the drone can travel
"""

# Variables initialization


#action = np.array([3.6845, 1.5614, 0, 3.1262]) # Syracuse (Exploitation)
#action = np.array([2.0187, 0, 3.2697, 0]) #Rome (Exploration)
action = np.array([2, 2, 0, 0])

initial_position = np.array([[8, 56],
                             [37, 16],
                             [78, 81],
                             [74, 124],
                             [20, 40],
                             [32, 92],
                             [64, 60],
                             [52, 10],
                             [91, 113],
                             [49, 51],
                             [8, 56],
                             [37, 16],
                             [78, 81],
                             [74, 124],
                             [20, 40],
                             [32, 92],
                             [64, 60],
                             [52, 10],
                             [91, 113],
                             [49, 51],
                             ])
sensors = np.array([['s1'],
                    ['s2'],
                    ['s3'],
                    ['s2', 's1'],
                    ['s3', 's4'],
                    ['s1', 's3'],
                    ['s2', 's4'],
                    ['s1', 's5'],
                    ['s3', 's5'],
                    ['s4', 's5'],
                    ['s1', 's5', 's3'],
                    ['s1', 's2', 's3'],
                    ['s2', 's4', 's5'],
                    ['s1', 's4', 's3'],
                    ['s2', 's4', 's5', 's3'],
                    ['s2', 's4', 's5', 's1'],
                    ['s2', 's4', 's3', 's1'],
                    ['s1', 's2', 's3', 's5'],
                    ['s2', 's4', 's5', 's1', 's3'],
                    ['s4'],
                    ['s5']])
start_time = time.time()
weights = {'1': {'Explore': np.array([2.7460, 3.3385, 0.0732, 3.0006]),
                 'Exploit': np.array([2.7460, 3.3385, 0.0732, 3.0006])},
           '2': {'Explore': np.array([0.2517, 3.4080, 1.4596, 0.0201]),
                 'Exploit': np.array([2.7460, 3.3385, 0.0732, 3.0006])},
           '3': {'Explore': np.array([1.2131, 3.1476, 1.5157, 0]),
                 'Exploit': np.array([1.3588, 1.4528, 0, 4])},
           '4': {'Explore': np.array([2.0187, 0, 3.2697, 0]),
                 'Exploit': np.array([3.6845, 1.5614, 0, 3.1262])}}

# PSO initialization
vehicles = -2

stage = 'exploration'
weights_b = True
# stage = 'no_exploitation'
method = 0
pso = PSOEnvironment(resolution, ys, method, method_pso='coupled', initial_seed=1000000,
                     initial_position=initial_position, sensor_vehicle=sensors, weights=weights, weights_b=weights_b,
                     vehicles=vehicles, exploration_distance=100, exploitation_distance=200, action=False,
                     reward_function='inc_mse', type_error='all_map', stage=stage, final_model='federated')

# Gaussian process initialization


# First iteration of PSO
import matplotlib.pyplot as plt

error_vec = []
last_error = []


for i in range(1):
    print(i)
    time_init = time.time()
    done = False
    pso.reset()
    R_vec = []
    n = 0
    mean_error = []

    # Main part of the code
    while not done:
        done = pso.step(action)

pso.data_out()
print("Total time:", time.time() - start_time)