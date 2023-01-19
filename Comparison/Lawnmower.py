import pandas as pd

from Environment.plot import Plots
import time
from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning

simplefilter("ignore", category=ConvergenceWarning)

from Comparison.lawnmower_simulatorhet import LawnmowerEnvironment

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

initial_position = np.array([[8, 56],
                             [37, 16],
                             [78, 81],
                             [74, 124],
                             [20, 40],
                             [32, 92],
                             [64, 60],
                             [52, 10],
                             [91, 113],
                             [49, 51]])
start_time = time.time()

# PSO initialization
vehicles = 4
lwm = LawnmowerEnvironment(ys, resolution, vehicles=vehicles, initial_seed=1000000, initial_position=initial_position,
                     exploration_distance=200, type_error='all_map')

# Gaussian process initialization


# First iteration of PSO
import matplotlib.pyplot as plt

error_vec = []
last_error = []

for i in range(30):
    print(i)
    time_init = time.time()
    done = False
    lwm.reset()
    R_vec = []
    n = 0
    mean_error = []

    # Main part of the code
    while not done:
        done = lwm.simulator()

    # print('Bench:', bench_max)
    # print('Std:', std_total)
    # print('Conf:', conf_total)

lwm.data_out()
