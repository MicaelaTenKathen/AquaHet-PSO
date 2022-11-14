from Environment.plot import Plots
import time
from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning

simplefilter("ignore", category=ConvergenceWarning)

from PSO.pso_function import PSOEnvironment
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


action = np.array([3.1286, 2.568, 0.79, 0])
initial_position = np.array([[0, 0],
                             [8, 56],
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

seed = 1000001
seed_epsilon = 1000000
vehicles = 4
#stage = 'exploration'
stage = 'no_exploitation'
method = 0
pso = PSOEnvironment(resolution, ys, method, initial_seed=1000009, initial_position=initial_position, vehicles=vehicles,
                     exploration_distance=100, exploitation_distance=200, reward_function='inc_mse',
                     type_error='all_map', stage=stage, final_model='federated')
# Gaussian process initialization


# First iteration of PSO
import matplotlib.pyplot as plt

mse_vec = []
epsilon = 0
delta_epsilon = 0

for i in range(1):

    done = False
    state = pso.reset()
    R_vec = []
    delta_epsilon = 0.13
    epsilon_array = []

    # Main part of the code

    while not done:
        seed_epsilon += 1
        distances_array = pso.distances_data()
        distances = np.max(distances_array)
        if distances <= 65:
            epsilon = 0.95
        elif distances >= 135:
            epsilon = 0.05
        else:
            epsilon = epsilon_ant - delta_epsilon
        val = np.random.RandomState(seed_epsilon).rand()
        #print(val)
        if epsilon >= val:
            action = np.array([2.0187, 0, 3.2697, 0])
            #action = np.array([1, 4, 4, 1])
        else:
            action = np.array([3.6845, 1.5614, 0, 3.1262])
            #action = np.array([2, 1, 1, 4])
        epsilon_array.append(epsilon)
        epsilon_ant = epsilon

        state, reward, done, dic = pso.step(action)

        R_vec.append(-reward)

    #print('Time', time.time() - start_time)

    #plt.plot(epsilon_array)
    MSE_data = np.array(pso.error_value())
    #plt.grid()
    #plt.show()
    print('GT:', i)
    print('MSE:', MSE_data[-1])

X_test, secure, bench_function, grid_min, sigma, mu, MSE_data, it, part_ant, y_data, grid, bench_max, dict_mu, \
dict_sigma, centers, part_ant_exploit, dict_centers, assig_center, part_ant_explore, final_mu, final_sigma, dict_limits = pso.data_out()
plot = Plots(xs, ys, X_test, secure, bench_function, grid_min, grid, stage)
centers_bench, dict_limits_bench, dict_coord = pso.return_bench()
# plot.gaussian(mu, sigma, part_ant)
# plot.movement_exploration(mu, sigma, part_ant_explore)
# plot.movement_exploration(final_mu, final_sigma, part_ant)
plot.benchmark()
plot.detection_areas(mu, sigma)
# plot.mu_exploitation(dict_mu, dict_sigma, centers)
# distances = pso.distances_data()
# plot.movement_exploitation(vehicles, dict_mu, dict_sigma, centers, dict_centers, part_ant_exploit, assig_center)
plot.plot_classic(mu, sigma, part_ant)
# plot.zoom_action_zone(centers_bench, dict_limits_bench, mu, sigma, final_mu, final_sigma)
# print(centers_bench, dict_limits_bench, dict_coord)
#pso.save_excel()

