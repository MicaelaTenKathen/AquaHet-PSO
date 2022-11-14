import pandas as pd

from Environment.plot import Plots
import time
from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning

simplefilter("ignore", category=ConvergenceWarning)

from Comparison.lawnmower_simulator import LawnmoverEnvironment

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
lwm = LawnmoverEnvironment(ys, resolution, vehicles=vehicles, initial_seed=1000009, initial_position=initial_position,
                     exploration_distance=200, type_error='all_map')

# Gaussian process initialization


# First iteration of PSO
import matplotlib.pyplot as plt

error_vec = []
last_error = []

for i in range(1):
    time_init = time.time()
    done = False
    lwm.reset()
    R_vec = []
    n = 0
    mean_error = []

    # Main part of the code
    while not done:
        done = lwm.simulator()

        error_data = np.array(lwm.error_value())
        error_actual = error_data[n:]
        mean_error.append(np.mean(error_actual))
        n = error_data.shape[0] - 1
    X_test, secure, bench_function, grid_min, sigma, mu, MSE_data, it, part_ant, y_data, grid = lwm.data_out()
    plot = Plots(xs, ys, X_test, secure, bench_function, grid_min, grid, 'no_exploitation')
    centers_bench, dict_limits_bench, dict_coord = lwm.return_bench()
    #plot.movement_exploration(mu, sigma, part_ant)
    #plot.benchmark()
    #distances = pso.distances_data()
    plot.plot_classic(mu, sigma, part_ant)

    print('GT:', i)
    print('MSE:', error_data[-1])
    print('Time:', time.time() - time_init)
    # print('Bench:', bench_max)
    # print('Std:', std_total)
    # print('Conf:', conf_total)

#pso.save_excel()
