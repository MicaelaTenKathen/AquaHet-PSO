import copy
import math
import os
import sys

import gym
import networkx as nx
import openpyxl
from deap import base
from deap import creator
from deap import tools
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.metrics import mean_squared_error

# from Benchmark.benchmark_functions import Benchmark_function
from Benchmark.bench_functions import *
from Data.limits import Limits
from Data.utils import Utils
from Environment.bounds import Bounds
from Environment.contamination_areas import DetectContaminationAreas
from Environment.plot import Plots

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

"""[https://deap.readthedocs.io/en/master/examples/pso_basic.html]"""


def createPart():
    """
    Creation of the objects "FitnessMax" and "Particle"
    """
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Particle", np.ndarray, fitness=creator.FitnessMax, speed=None, smin=None, smax=None,
                   best=None, node=None)
    creator.create("BestGP", np.ndarray, fitness=creator.FitnessMax)


class PSOEnvironment(gym.Env):

    def __init__(self, resolution, ys, method, initial_seed, initial_position, sensor_vehicle, vehicles=4,
                 exploration_distance=100,
                 exploitation_distance=200, reward_function='mse', behavioral_method=0, type_error='all_map',
                 stage='exploration', final_model='samples'):
        self.list_S_sf = []
        self.p_vehicles = ['p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7', 'p8']
        self.P = nx.MultiGraph()
        self.sub_fleets = None
        self.sensor_vehicle = sensor_vehicle
        self.type_error = type_error
        self.final_model = final_model
        self.initial_stage = stage
        self.exploration_distance_initial = exploration_distance
        self.exploitation_distance_initial = exploitation_distance
        self.exploration_distance = exploration_distance
        self.exploitation_distance = exploitation_distance
        self.stage = stage
        self.dist_pre = 0
        self.f = None
        self.k = None
        self.initial = True
        self.dict_ = {}
        self.max_bench = list()
        self.file = 'MultiPSO/2Vehicles/Explore150Exploit300'
        self.dict_impo_ = {}
        self.dict_bench = {}
        self.dict_coord_ = {}
        self.dict_sample_x = {}
        self.dict_error_peak = {}
        self.dict_error_peak_explore = {}
        self.dict_sample_y = {}
        self.dict_fitness = {}
        self.dict_mu = {}
        self.dict_error_explore = {}
        self.dict_index = {}
        self.dict_sigma = {}
        self.dict_max_sigma = {}
        self.dict_max_mu = {}
        self.dict_global_best = {}
        self.dict_error = {}
        self.dict_centers = {}
        self.dict_error_comparison = {}
        self.dict_limits = {}
        self.coord_centers = []
        self.len_scale = 0
        self.max_centers_bench = []
        self.centers = 0
        self.vehicles = vehicles
        self.assig_centers = np.zeros((self.vehicles, 1))
        self.population = vehicles
        self.resolution = resolution
        self.smin = 0
        self.smax = 3
        self.size = 2
        self.wmin = 0.4 / (15000 / ys)
        self.wmax = 0.9 / (15000 / ys)
        self.xs = int(10000 / (15000 / ys))
        self.ys = ys
        self.max_peaks_bench = list()
        self.max_peaks_mu = list()
        ker = RBF(length_scale=10, length_scale_bounds=(1e-1, 10 ^ 5))
        self.gpr = GaussianProcessRegressor(kernel=ker, alpha=1e-6)  # optimizer=None)
        self.x_h = []
        self.y_h = []
        self.x_p = []
        self.y_p = []
        self.fitness = []
        self.y_data = []
        self.mu_max = 0
        self.x_bench = None
        self.y_bench = None
        self.n_plot = float(1)
        self.s_n = np.full(self.vehicles, True)
        self.s_ant = np.zeros(self.vehicles)
        self.samples = None
        self.dist_ant = None
        self.sigma_best = []
        self.mu_best = []
        self.action_zone = list()
        self.action_zone_index = list()
        self.n_data = 0
        self.num = 0
        self.save = 0
        self.fail_exp = False
        self.num_of_peaks = 0
        self.save_dist = [25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300, 325, 350, 375, 400, 425, 450, 475,
                          500, 525, 550, 575, 600, 625, 650, 675, 700]
        self.seed = initial_seed
        self.initial_seed = initial_seed
        self.mu = []
        self.sigma = []
        self.final_mu = []
        self.final_sigma = []
        self.g = 0
        self.g_exploit = 0
        self.fail = False
        self.post_array = np.ones((1, self.vehicles))
        self.distances = np.zeros(self.vehicles)
        self.distances_exploit = np.zeros(self.vehicles)
        self.lam = 0.375
        self.part_ant = np.zeros((1, self.vehicles * 2))
        self.part_ant_exploit = copy.copy(self.part_ant)
        self.part_ant_explore = copy.copy(self.part_ant)
        self.last_sample, self.k, self.f, self.samples, self.ok = 0, 0, 0, 0, False
        self.error_data = []
        self.error_data1 = []
        self.it = []
        self.coordinate_bench_max = []
        self.bench_max = 0
        self.method = method
        self.error = []
        self.index_a = list()
        self.peaks = list()
        self.ERROR_data = []
        self.error_comparison = []
        self.error_comparison1 = []
        self.error_comparison2 = []
        self.error_comparison3 = []
        self.error_comparison4 = []
        self.error_comparison5 = []
        self.error_comparison6 = []
        self.error_comparison7 = []
        self.error_comparison8 = []
        self.print_warning = True
        self.error_comparison9 = []
        self.bench_array = []
        self.error_distance = []
        self.duplicate = False
        self.array_part = np.zeros((1, self.vehicles * 2))
        self.reward_function = reward_function
        self.behavioral_method = behavioral_method
        self.initial_position = initial_position
        if self.method == 0:
            self.state = np.zeros(self.vehicles * 8, )
        else:
            self.state = np.zeros((6, self.xs, self.ys))

        self.grid_or = Map(self.xs, ys).black_white()

        self.grid_min, self.grid_max, self.grid_max_x, self.grid_max_y = 0, self.ys, self.xs, self.ys

        self.p = 1
        self.max_peaks = None

        self.df_bounds, self.X_test, self.bench_limits = Bounds(self.resolution, self.xs, self.ys,
                                                                load_file=False).map_bound()
        self.secure, self.df_bounds = Bounds(self.resolution, self.xs, self.ys).interest_area()

        self.X_test_y = self.X_test[1]
        self.bench_function = None

        self.plot = Plots(self.xs, self.ys, self.X_test, self.secure, self.bench_function, self.grid_min, self.grid_or,
                          self.stage)

        self.util = Utils(self.vehicles)

        # self.mpb = movingpeaks.MovingPeaks(dim=2, npeaks=4, lambda_=0.6,
        #                                    min_height=0,
        #                                    pfunc=self.function1,
        #                                    max_height=1,
        #                                    uniform_height=1,
        #                                    move_severity=1,
        #                                    height_severity=0.005,
        #                                    min_width=0.0001,
        #                                    max_width=0.01,
        #                                    uniform_width=0.005,
        #                                    width_severity=0.0001,
        #                                    min_coord=0.0,
        #                                    max_coord=max(self.grid_or.shape),
        #                                    X_test=self.X_test, seed=self.seed)

        createPart()

    def function1(self, individual, position, height, width):
        return height / (1 + width * sum(((individual[j] - position[j]) ** 2) for j in range(len(individual))))

    def generatePart(self):

        """
        Generates a random position and a random speed for the particles (drones).
        """
        list_part = list()
        part = creator.Particle([self.initial_position[self.p, i] for i in range(self.size)])
        list_part.append(part)
        part.speed = np.array([random.uniform(self.smin, self.smax) for _ in range(self.size)])
        part.smin = self.smin
        part.smax = self.smax
        self.P.add_node(self.p_vehicles[self.p], S_p=dict.fromkeys(self.sensor_vehicle[self.p], []),
                                    U_p=list_part)
        # part.node = add_node retorna un void, aqui solo deberia guardarse el nombre del nodo
        # part.node =

        self.p += 1

        return part

    def set_sensor(self):
        i = 0
        for node_p in self.P.nodes(data=True):
            i += 1
            j = 0
            for node_q in self.P.nodes(data=True):
                j += 1
                if i <= j:
                    print(node_q[1]["S_p"])
                    intersection = node_p[1]["S_p"].keys() & node_q[1]["S_p"].keys()
                    if node_q != node_p and len(intersection) > 0:
                        if not self.P.has_edge(node_p[0], node_q[0]):
                            self.P.add_edge(node_p[0], node_q[0], S_pq=intersection)
        self.sub_fleets = list(nx.connected_components(self.P))
        for i, sub_fleet in enumerate(self.sub_fleets):
            S_sf = set()
            for particle in sub_fleet:
                S_sf = S_sf | self.P.nodes[particle]['S_p'].keys()
            print(f'Subfleet {i} contains {S_sf} y se usa en eqs. 13c y 13d')
            self.list_S_sf.append(S_sf)

            for particle in sub_fleet:
                print(f'Particle {particle} contains {self.P.nodes[particle]["S_p"]} y se usa en eqs. 13a y 13b')

    def gp_update(self):
        # S_n = {"sensor1": {"mu": [], "sigma": []}}
        measuring_positions = nx.get_node_attributes(self.P, "Q_p")
        for i, sub_fleet in enumerate(self.sub_fleets):
            for s, sensor in enumerate(self.list_S_sf[i]):
                measures_for_sensor = []
                coordinates_for_sensor = []
                for p, particle in enumerate(sub_fleet):
                    s_p = self.P.nodes(data=True)[particle]['S_p']
                    if sensor in s_p:
                        measures_for_sensor.extend(s_p[sensor])
                        coordinates_for_sensor.append(measuring_positions[particle])
                print(s, sensor)
                print(coordinates_for_sensor)
                print(measures_for_sensor)
                # GP.fit(coordinates_for_sensor, measures_for_sensor)
                # S_n[sensor][mu], S_n[sensor][sigma] = GP.predict(X_Mapa, return_std=true)

        # for i, sub_fleet in enumerate(self.sub_fleets):
        #     sensor_sf = self.list_S_sf[i]
        #     for s, sensor in enumerate(sensor_sf):
        #         measures = []
        #         coordinates = []
        #         for p, particle in enumerate(sub_fleet):
        #             part_sensor = self.P.nodes[particle]['S_p']

    # def take_measures(self):
    ## take water quality measures
    ## aqui deberiamos actualizar Q_p

    def updateParticle_n(self, p, c1, c2, c3, c4, part):

        """
        Calculates the speed and the position of the particles (drones).
        """
        list_part = nx.get_node_attributes(self.P, 'U_p')[p]
        if self.behavioral_method == 0:
            u1 = np.array([random.uniform(0, c1) for _ in range(len(part))])
            u2 = np.array([random.uniform(0, c2) for _ in range(len(part))])
            u3 = np.array([random.uniform(0, c3) for _ in range(len(part))])
            u4 = np.array([random.uniform(0, c4) for _ in range(len(part))])
        else:
            u1 = c1
            u2 = c2
            u3 = c3
            u4 = c4

        v_u1 = u1 * (part.best - part)
        v_u2 = u2 * (self.best - part)
        v_u3 = u3 * (self.sigma_best - part)
        v_u4 = u4 * (self.mu_best - part)
        w = 1
        part.speed = v_u1 + v_u2 + v_u3 + v_u4 + part.speed * w
        for i, speed in enumerate(part.speed):
            if abs(speed) < part.smin:
                part.speed[i] = math.copysign(part.smin, speed)
            elif abs(speed) > part.smax:
                part.speed[i] = math.copysign(part.smax, speed)
        part[:] = part + part.speed
        self.P[p]['U_p'] = list_part

        return part

    def tool(self):

        """
        The operators are registered in the toolbox with their parameters.
        """
        self.toolbox = base.Toolbox()
        self.toolbox.register("particle", self.generatePart)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.particle)
        self.toolbox.register("update", self.updateParticle_n)
        # self.toolbox.register("evaluate", shekel)

        return self.toolbox

    def swarm(self):

        """
        Creates a population.
        """
        toolbox = self.tool()
        self.pop = toolbox.population(n=self.population)
        self.best = self.pop[0]

        return self.best, self.pop

    def statistic(self):

        """
        Visualizes the stats of the code.
        """

        self.stats = tools.Statistics(lambda ind: ind.fitness.values)
        self.stats.register("avg", np.mean)
        self.stats.register("std", np.std)
        self.stats.register("min", np.min)
        self.stats.register("max", np.max)

        self.logbook = tools.Logbook()
        self.logbook.header = ["gen", "evals"] + self.stats.fields

        return self.stats, self.logbook

    def reset(self):

        """
        Initialization of the pso.
        """
        self.reset_variables()
        # self.bench_function, self.bench_array, self.num_of_peaks, self.index_a = Benchmark_function(self.grid_or,
        #                                                                                             self.resolution,
        #                                                                                             self.xs, self.ys,
        #                                                                                             self.X_test,
        #                                                                                             self.seed, self.vehicles).create_new_map()
        self.bench_function, self.bench_array, self.num_of_peaks, self.index_a = Benchmark_function(self.grid_or, 1,
                                                                                                    self.xs, self.ys,
                                                                                                    None, self.seed, 0,
                                                                                                    base_benchmark="ackley",
                                                                                                    randomize_shekel=True).create_new_map()

        self.max_contamination()
        self.generatePart()
        self.tool()
        random.seed(self.seed)
        self.swarm()
        self.statistic()
        # self.peaks_bench()
        self.detect_areas = DetectContaminationAreas(self.X_test, self.bench_array, vehicles=self.vehicles,
                                                     area=self.xs)
        self.centers_bench, self.dict_index_bench, self.dict_bench, self.dict_coord_bench, self.center_peaks_bench, \
        self.max_bench_list, self.dict_limits_bench, self.action_zone_bench, self.dict_impo_bench, \
        self.index_center_bench = self.detect_areas.benchmark_areas()
        self.max_peaks = self.detect_areas.real_peaks()
        self.state = self.first_values()
        # self.mpb = movingpeaks.MovingPeaks(dim=2, npeaks=4, lambda_=0.6,
        #                                    min_height=0,
        #                                    pfunc=self.function1,
        #                                    max_height=1,
        #                                    uniform_height=1,
        #                                    move_severity=1,
        #                                    height_severity=0.005,
        #                                    min_width=0.0001,
        #                                    max_width=0.01,
        #                                    uniform_width=0.005,
        #                                    width_severity=0.0001,
        #                                    min_coord=0.0,
        #                                    max_coord=max(self.grid_or.shape),
        #                                    X_test=self.X_test, seed=self.seed)
        return self.state

    def reset_variables(self):
        self.f = None
        self.list_S_sf = []
        self.k = None
        self.repeat = False
        self.x_h = []
        self.fail = False
        self.fail_exp = False
        self.y_h = []
        self.print_warning = True
        self.stage = self.initial_stage
        self.exploration_distance = self.exploration_distance_initial
        self.exploitation_distance = self.exploitation_distance_initial
        self.coord_centers = []
        self.max_centers_bench = []
        self.x_p = []
        self.len_scale = 0
        self.y_p = []
        self.action_zone = list()
        self.max_bench = list()
        self.action_zone_index = list()
        self.fitness = []
        self.final_mu = []
        self.final_sigma = []
        self.y_data = []
        self.dict_ = {}
        self.dict_error_peak_explore = {}
        self.dict_impo_ = {}
        self.dict_limits = {}
        self.dict_error_peak = {}
        self.dict_error_explore = {}
        self.assig_centers = np.zeros((self.vehicles, 1))
        self.dict_coord_ = {}
        self.dict_sample_x = {}
        self.dict_error = {}
        self.dict_sample_y = {}
        self.dict_index = {}
        self.dict_fitness = {}
        self.dict_mu = {}
        self.dict_bench = {}
        self.dict_sigma = {}
        self.dict_centers = {}
        self.dict_max_sigma = {}
        self.dict_max_mu = {}
        self.dict_global_best = {}
        self.centers = 0
        self.x_bench = None
        self.y_bench = None
        self.n_plot = float(1)
        self.s_n = np.full(self.vehicles, True)
        self.s_ant = np.zeros(self.vehicles)
        self.samples = None
        self.dist_ant = None
        self.sigma_best = []
        self.mu_best = []
        self.coordinate_bench_max = []
        self.n_data = 0
        self.mu = []
        self.max_peaks_bench = list()
        self.max_peaks_mu = list()
        self.p = 0
        self.sigma = []
        self.post_array = np.ones(self.vehicles)
        self.distances = np.zeros(self.vehicles)
        self.part_ant = np.zeros((1, self.vehicles * 2))
        self.distances_exploit = copy.copy(self.distances)
        self.part_ant_explore = copy.copy(self.part_ant)
        self.part_ant_exploit = copy.copy(self.part_ant)
        self.last_sample, self.k, self.f, self.samples, self.ok = 0, 0, 0, 0, False
        self.error_data = []
        self.save = 0
        self.error_comparison = []
        self.error_distance = []
        self.error = None

        self.it = []
        self.error = []
        self.duplicate = False
        self.array_part = np.zeros((1, self.vehicles * 2))
        self.seed += 1

        if self.method == 0:
            self.state = np.zeros(self.vehicles * 8, )
        else:
            self.state = np.zeros((6, self.xs, self.ys))

        self.num += 1
        self.g = 0
        self.g_exploit = 0

    def max_contamination(self):
        self.bench_max, self.coordinate_bench_max = self.obtain_max(self.bench_array)

    def pso_fitness(self, part):

        """
        Obtains the local best (part.best) of each particle (drone) and the global best (best) of the swarm (fleet).
        """

        part.fitness.values = self.new_fitness(part)

        if self.ok:
            self.check_duplicate(part)
        else:
            if part.best.fitness < part.fitness:
                part.best = creator.Particle(part)
                part.best.fitness.values = part.fitness.values
            if self.stage != "exploitation":
                if self.best.fitness < part.fitness:
                    self.best = creator.Particle(part)
                    self.best.fitness.values = part.fitness.values

        return self.ok, part

    def new_fitness(self, part):
        part, self.s_n = Limits(self.secure, self.xs, self.ys).new_limit(self.g, part, self.s_n, self.n_data,
                                                                         self.s_ant, self.part_ant)
        self.x_bench = int(part[0])
        self.y_bench = int(part[1])

        new_fitness_value = [self.bench_function[self.x_bench][self.y_bench]]
        # fit = self.toolbox.evaluate(part)
        return new_fitness_value

    def gp_regression(self):

        """
        Fits the gaussian process.
        """

        x_a = np.array(self.x_h).reshape(-1, 1)
        y_a = np.array(self.y_h).reshape(-1, 1)
        x_train = np.concatenate([x_a, y_a], axis=1).reshape(-1, 2)
        y_train = np.array(self.fitness).reshape(-1, 1)

        self.gpr.fit(x_train, y_train)
        self.gpr.get_params()

        self.mu, self.sigma = self.gpr.predict(self.X_test, return_std=True)
        post_ls = round(np.min(np.exp(self.gpr.kernel_.theta[0])), 1)
        r = self.n_data
        self.post_array[r] = post_ls
        if self.post_array[-1] == 0.1:
            self.len_scale += 1
            self.fail = True
            print("Warning: reached the minimum value of length scale (Exploration)")
        else:
            self.len_scale = 0

        return self.post_array

    def sort_index(self, array, rev=True):
        index = range(len(array))
        s = sorted(index, reverse=rev, key=lambda i: array[i])
        return s

    def peaks_bench(self):
        for i in range(len(self.index_a)):
            self.max_peaks_bench.append(self.bench_array[round(self.index_a[i])])

    def peaks_mu(self):
        self.max_peaks_mu = list()
        for i in range(len(self.index_a)):
            self.max_peaks_mu.append(self.mu[round(self.index_a[i])])

    def sigma_max(self):

        """
        Returns the coordinates of the maximum uncertainty (sigma_best) and the maximum contamination (mu_best).
        """

        sigma_max, self.sigma_best = self.obtain_max(self.sigma)
        mu_max, self.mu_best = self.obtain_max(self.mu)

        return self.sigma_best, self.mu_best

    def calculate_reward(self):
        if self.reward_function == 'mse':
            reward = -self.error_data[-1]
        elif self.reward_function == 'inc_mse':
            reward = self.error_data[-2] - self.error_data[-1]
        return reward

    def check_duplicate(self, part):
        self.duplicate = False
        for i in range(len(self.x_h)):
            if self.x_h[i] == self.x_bench and self.y_h[i] == self.y_bench:
                self.duplicate = True
                self.fitness[i] = part.fitness.values
                break
            else:
                self.duplicate = False
        if self.duplicate:
            pass
        else:
            self.x_h.append(int(part[0]))
            self.y_h.append(int(part[1]))
            self.fitness.append(part.fitness.values)

    def obtain_max_explotaition(self, array_function, action_zone):
        max_value = np.max(array_function)
        index_1 = np.where(array_function == max_value)
        index_x1 = index_1[0]

        coord = self.dict_coord_["action_zone%s" % action_zone]
        index_x2 = index_x1[0]
        index_x = int(coord[index_x2][0])
        index_y = int(coord[index_x2][1])

        index_xy = [index_x, index_y]
        coordinate_max = np.array(index_xy)

        return max_value, coordinate_max

    def take_sample(self, part, action_zone):

        """
        Obtains the local best (part.best) of each particle (drone) and the global best (best) of the swarm (fleet).
        """

        part.fitness.values = self.new_fitness(part)
        x_bench = part[0]
        y_bench = part[1]
        duplicate = False
        x_l = copy.copy(self.dict_sample_x["action_zone%s" % action_zone])
        y_l = copy.copy(self.dict_sample_y["action_zone%s" % action_zone])
        fitness = copy.copy(self.dict_fitness["action_zone%s" % action_zone])
        index_action_zone = copy.copy(self.dict_index["action_zone%s" % action_zone])
        for i in range(len(x_l)):
            if x_l[i] == x_bench and y_l[i] == y_bench:
                duplicate = True
                fitness[i] = part.fitness.values
                break
            else:
                duplicate = False
        if duplicate:
            pass
        else:
            x_l.append(int(part[0]))
            y_l.append(int(part[1]))
            fitness.append(part.fitness.values)

        x_a = np.array(x_l).reshape(-1, 1)
        y_a = np.array(y_l).reshape(-1, 1)
        x_train = np.concatenate([x_a, y_a], axis=1).reshape(-1, 2)
        y_train = np.array(fitness).reshape(-1, 1)

        self.gpr.fit(x_train, y_train)
        self.gpr.get_params()

        mu, sigma = self.gpr.predict(self.X_test, return_std=True)
        post_ls = round(np.min(np.exp(self.gpr.kernel_.theta[0])), 1)
        r = self.n_data
        self.post_array[r] = post_ls
        if post_ls == 0.1:
            self.len_scale += 1
        else:
            self.len_scale = 0

        mu_available = list()
        sigma_available = list()

        for i in range(len(index_action_zone)):
            mu_available.append(mu[index_action_zone[i]])
            sigma_available.append(sigma[index_action_zone[i]])

        sigma_max, sigma_best = self.obtain_max_explotaition(sigma_available, action_zone)
        mu_max, mu_best = self.obtain_max_explotaition(mu_available, action_zone)

        self.dict_sample_x["action_zone%s" % action_zone] = copy.copy(x_l)
        self.dict_sample_y["action_zone%s" % action_zone] = copy.copy(y_l)
        self.dict_fitness["action_zone%s" % action_zone] = copy.copy(fitness)
        self.dict_mu["action_zone%s" % action_zone] = copy.copy(mu)
        self.dict_sigma["action_zone%s" % action_zone] = copy.copy(sigma)
        self.dict_max_sigma["action_zone%s" % action_zone] = copy.copy(sigma_best)
        self.dict_max_mu["action_zone%s" % action_zone] = copy.copy(mu_best)

    def first_values(self):

        """
        The output "out" of the method "initcode" is the positions of the particles (drones) after the first update of the
        gaussian process (initial state).
        method = 0 -> out = scalar vector
        out = [px_1, py_1, px_2, py_2, px_3, py_3, px_4, py_4, lbx_1, lby_1, lbx_2, lby_2, lbx_3, lby_3, lbx_4, lby_4, gbx, gby,
               sbx, sgy, mbx, mby]
               where:
               px: x coordinate of the drone position
               py: y coordinate of the drone position
               lbx: x coordinate of the local best
               lby: y coordinate of the local best
               gbx: x coordinate of the global best
               gby: y coordinate of the global best
               sbx: x coordinate of the sigma best (maximum uncertainty)
               sby: y coordinate of the sigma best (maximum uncertainty)
               mbx: x coordinate of the mean best (maximum contamination)
               mby: y coordinate of the mean best (maximum contamination)
        method = 1 -> out = images
        :param c1: weight that determinate the importance of the local best component
        :param c2: weight that determinate the importance of the global best component
        :param c3: weight that determinate the importance of the maximum uncertainty component
        :param c4: weight that determinate the importance of the maximum contamination component
        :param lam: ratio of one of the different length scales [Equation 7
        (https://doi.org/10.3390/electronics10131605)]
        :param post_array: refers to the posterior length scale of the surrogate model [Equation 7
        (https://doi.org/10.3390/electronics10131605)]
        """

        for part in self.pop:

            part.fitness.values = self.new_fitness(part)

            if self.n_plot > self.vehicles:
                self.n_plot = float(1)

            part.best = creator.Particle(part)
            part.best.fitness.values = part.fitness.values

            if self.best.fitness < part.fitness:
                self.best = creator.Particle(part)
                self.best.fitness.values = part.fitness.values

            self.part_ant, self.distances = self.util.distance_part(self.g, self.n_data, part, self.part_ant,
                                                                    self.distances, self.array_part, dfirst=True)

            self.check_duplicate(part)

            self.post_array = self.gp_regression()

            self.samples += 1

            self.n_data += 1
            if self.n_data > self.vehicles - 1:
                self.n_data = 0

        self.error = self.calculate_error()
        self.error_data.append(self.error)
        self.it.append(self.g)

        self.sigma_best, self.mu_best = self.sigma_max()

        self.return_state()

        self.k = self.vehicles
        self.ok = False

        return self.state

    def save_data(self):
        if self.save < (self.exploitation_distance_initial / 25):
            mult = self.save_dist[self.save]
            mult_min = mult - 5
            mult_max = mult + 5
            if mult_min <= np.max(self.distances) < mult_max:
                if self.seed == self.initial_seed + 1:
                    if self.initial:
                        for i in range(int(self.exploitation_distance_initial / 25) + 1):
                            self.dict_error_comparison["Distance%s" % i] = list()
                        self.initial = False
                error_list = copy.copy(self.dict_error_comparison["Distance%s" % self.save])
                self.ERROR_data = self.calculate_error()
                error_list.append(self.ERROR_data)
                self.dict_error_comparison["Distance%s" % self.save] = copy.copy(error_list)
                self.save += 1

    def obtain_max(self, array_function):
        max_value = np.max(array_function)
        index_1 = np.where(array_function == max_value)
        index_x1 = index_1[0]

        index_x2 = index_x1[0]
        index_x = int(self.X_test[index_x2][0])
        index_y = int(self.X_test[index_x2][1])

        index_xy = [index_x, index_y]
        coordinate_max = np.array(index_xy)

        return max_value, coordinate_max

    def calculate_error(self, dfirts=False):
        if self.type_error == 'all_map':
            if self.stage == 'exploration' or self.stage == 'no_exploitation':
                self.error = mean_squared_error(y_true=self.bench_array, y_pred=self.mu)
            else:
                if self.final_model == 'centralized':
                    self.final_mu = copy.copy(self.mu)
                    self.final_sigma = copy.copy(self.sigma)
                elif self.final_model == 'federated':
                    self.replace_action_zones()
                self.error = mean_squared_error(y_true=self.bench_array, y_pred=self.final_mu)
        elif self.type_error == 'peaks':
            if dfirts and self.stage != 'no_exploitation':
                for i in range(len(self.index_center_bench)):
                    max_az = self.final_mu[self.index_center_bench[i]]
                    # for i in range(len(self.center_peaks_bench)):
                    #   coord = self.center_peaks_bench[i]
                    #  for j in range(len(self.X_test)):
                    #     coord_xtest = self.X_test[j]
                    #    if coord[0] == coord_xtest[0] and coord[1] == coord_xtest[1]:
                    #       max_az = self.mu[j]
                    #      break
                    self.dict_error_peak_explore["action_zone%s" % i] = abs(self.max_bench_list[i] - max_az)
            elif self.stage == 'no_exploitation':
                for i in range(len(self.index_center_bench)):
                    max_az = self.mu[self.index_center_bench[i]]
                    # for i in range(len(self.center_peaks_bench)):
                    #   coord = self.center_peaks_bench[i]
                    #  for j in range(len(self.X_test)):
                    #     coord_xtest = self.X_test[j]
                    #    if coord[0] == coord_xtest[0] and coord[1] == coord_xtest[1]:
                    #       max_az = self.mu[j]
                    #      break
                    self.dict_error_peak["action_zone%s" % i] = abs(self.max_bench_list[i] - max_az)
            else:
                if self.final_model == 'centralized':
                    self.final_mu = copy.copy(self.mu)
                    self.final_sigma = copy.copy(self.sigma)
                elif self.final_model == 'federated':
                    self.replace_action_zones()
                for i in range(len(self.index_center_bench)):
                    max_az = self.final_mu[self.index_center_bench[i]]
                    # for i in range(len(self.center_peaks_bench)):
                    #   coord = self.center_peaks_bench[i]
                    #  for j in range(len(self.X_test)):
                    #     coord_xtest = self.X_test[j]
                    #    if coord[0] == coord_xtest[0] and coord[1] == coord_xtest[1]:
                    #       max_az = self.final_mu[j]
                    #      break
                    self.dict_error_peak["action_zone%s" % i] = abs(self.max_bench_list[i] - max_az)
        elif self.type_error == 'contamination_1':
            index_mu_max = [i for i in range(len(self.X_test)) if (self.X_test[i] == self.coordinate_bench_max).all()]
            index_mu_max = index_mu_max[0]
            mu_max = self.mu[index_mu_max]
            mu_max = mu_max[0]
            self.error = self.bench_max - mu_max
        elif self.type_error == 'contamination':
            self.peaks_mu()
            self.error = mean_squared_error(y_true=self.max_peaks_bench, y_pred=self.max_peaks_mu)
        elif self.type_error == 'action_zone':
            if dfirts and self.stage != 'no_exploitation':
                estimated_all = list()
                for i in range(len(self.center_peaks_bench)):
                    bench_action = copy.copy(self.dict_bench["action_zone%s" % i])
                    estimated_action = list()
                    index_action = copy.copy(self.dict_index_bench["action_zone%s" % i])
                    for j in range(len(index_action)):
                        estimated_action.append(self.mu[index_action[j]])
                        estimated_all.append(self.mu[index_action[j]])
                    error_action = mean_squared_error(y_true=bench_action, y_pred=estimated_action)
                    self.dict_error_explore["action_zone%s" % i] = copy.copy(error_action)
                self.error = mean_squared_error(y_true=self.action_zone_bench, y_pred=estimated_all)
            elif self.stage == 'no_exploitation':
                estimated_all = list()
                for i in range(len(self.center_peaks_bench)):
                    bench_action = copy.copy(self.dict_bench["action_zone%s" % i])
                    estimated_action = list()
                    index_action = copy.copy(self.dict_index_bench["action_zone%s" % i])
                    for j in range(len(index_action)):
                        value = self.mu[index_action[j]]
                        estimated_action.append(value[0])
                        estimated_all.append(self.mu[index_action[j]])
                    error_action = mean_squared_error(y_true=bench_action, y_pred=estimated_action)
                    self.dict_error["action_zone%s" % i] = copy.copy(error_action)
                self.error = mean_squared_error(y_true=self.action_zone_bench, y_pred=estimated_all)
            else:
                if self.final_model == 'centralized':
                    self.final_mu = copy.copy(self.mu)
                    self.final_sigma = copy.copy(self.sigma)
                elif self.final_model == 'federated':
                    self.replace_action_zones()
                estimated_all = list()
                for i in range(len(self.center_peaks_bench)):
                    bench_action = copy.copy(self.dict_bench["action_zone%s" % i])
                    estimated_action = list()
                    index_action = copy.copy(self.dict_index_bench["action_zone%s" % i])
                    for j in range(len(index_action)):
                        estimated_action.append(self.final_mu[index_action[j]])
                        estimated_all.append(self.final_mu[index_action[j]])
                    error_action = mean_squared_error(y_true=bench_action, y_pred=estimated_action)
                    self.dict_error["action_zone%s" % i] = copy.copy(error_action)
                self.error = mean_squared_error(y_true=self.action_zone_bench, y_pred=estimated_all)
        return self.error

    def allocate_vehicles(self):
        center_zone = copy.copy(self.coord_centers)
        population = copy.copy(self.pop)
        num = 0
        asvs = np.arange(0, self.vehicles, 1)
        elements_repeat_all = math.trunc(self.vehicles / self.centers)
        elements_repeat = self.vehicles - elements_repeat_all * self.centers
        repeat_all = np.full((self.centers, 1), elements_repeat_all)
        repeat = np.zeros((self.centers, 1))
        while num < elements_repeat:
            repeat[num] = 1
            num += 1
        for i in range(len(repeat_all)):
            assig_vehicles = list()
            z = repeat_all[i] + repeat[i]
            o = 0
            while o < z:
                asv = 0
                for part in population:
                    if asv == 0:
                        low = math.sqrt((center_zone[i, 0] - part[0]) ** 2 + (center_zone[i, 1] - part[1]) ** 2)
                        index = 0
                    else:
                        dista = math.sqrt((center_zone[i, 0] - part[0]) ** 2 + (center_zone[i, 1] - part[1]) ** 2)
                        if dista < low:
                            low = copy.copy(dista)
                            index = copy.copy(asv)
                    asv += 1
                assig_vehicles.append(asvs[index])
                self.assig_centers[asvs[index]] = i
                del population[index]
                asvs = np.delete(asvs, index)
                o += 1
            self.dict_centers["action_zone%s" % i] = assig_vehicles
        zones = 0
        while zones < self.centers:
            self.dict_max_mu["action_zone%s" % zones] = center_zone[zones]
            self.dict_max_sigma["action_zone%s" % zones] = self.sigma_best
            self.dict_sample_x["action_zone%s" % zones] = self.x_h
            self.dict_sample_y["action_zone%s" % zones] = self.y_h
            self.dict_fitness["action_zone%s" % zones] = self.fitness
            zones += 1

    def step_stage_exploration(self, action):

        """
        The output "out" of the method "step" is the positions of the particles (drones) after traveling 1000 m
        (scaled).

        method = 0 -> out = scalar vector
        out = [px_1, py_1, px_2, py_2, px_3, py_3, px_4, py_4, lbx_1, lby_1, lbx_2, lby_2, lbx_3, lby_3, lbx_4, lby_4,
               gbx, gby, sbx, sgy, mbx, mby]
               where:
               px: x coordinate of the drone position
               py: y coordinate of the drone position
               lbx: x coordinate of the local best
               lby: y coordinate of the local best
               gbx: x coordinate of the global best
               gby: y coordinate of the global best
               sbx: x coordinate of the sigma best (maximum uncertainty)
               sby: y coordinate of the sigma best (maximum uncertainty)
               mbx: x coordinate of the mean best (maximum contamination)
               mby: y coordinate of the mean best (maximum contamination)

        method = 1 -> out = images

        :param c1: weight that determinate the importance of the local best component
        :param c2: weight that determinate the importance of the global best component
        :param c3: weight that determinate the importance of the maximum uncertainty component
        :param c4: weight that determinate the importance of the maximum contamination component
        :param lam: ratio of one of the different length scales [Equation 7
        (https://doi.org/10.3390/electronics10131605)]
        :param post_array: refers to the posterior length scale of the surrogate model [Equation 7
        (https://doi.org/10.3390/electronics10131605)]
        """
        dis_steps = 0
        dist_ant = np.mean(self.distances)
        self.dist_pre = np.max(self.distances)
        self.n_data = 0
        self.f += 1

        while dis_steps < 10:

            previous_dist = np.max(self.distances)

            for part in self.pop:
                self.toolbox.update(action[0], action[1], action[2], action[3], part)

            for part in self.pop:
                self.ok, part = self.pso_fitness(part)
                self.part_ant, self.distances = self.util.distance_part(self.g, self.n_data, part, self.part_ant,
                                                                        self.distances, self.array_part, dfirst=False)

                self.n_data += 1
                if self.n_data > self.vehicles - 1:
                    self.n_data = 0

            if (np.mean(self.distances) - self.last_sample) >= (np.min(self.post_array) * self.lam):
                self.k += 1
                self.ok = True
                self.last_sample = np.mean(self.distances)

                for part in self.pop:
                    self.ok, part = self.pso_fitness(part)

                    self.check_duplicate(part)

                    self.post_array = self.gp_regression()

                    self.samples += 1

                    self.n_data += 1
                    if self.n_data > self.vehicles - 1:
                        self.n_data = 0

                self.it.append(self.g)
                self.error = self.calculate_error()
                self.error_data.append(self.error)

                # self.save_data()

                self.sigma_best, self.mu_best = self.sigma_max()

                self.ok = False
            if self.len_scale >= 3 and self.print_warning:
                print("Warning: reached the minimum value of length scale (Exploration)")
                self.print_warning = False
            dis_steps = np.mean(self.distances) - dist_ant
            if np.max(self.distances) == previous_dist:
                break

            self.g += 1
        self.return_state()

        # reward = self.calculate_reward()
        reward = 0

        self.logbook.record(gen=self.g, evals=len(self.pop), **self.stats.compile(self.pop))

        if (self.distances >= self.exploration_distance).any() or np.max(self.distances) == self.dist_pre:
            if self.stage == 'no_exploitation':
                done = True
            else:
                done = False
                # if max(self.mu) < 0.33:
                # done = True
                #   self.exploration_distance = self.exploration_distance + self.exploration_distance_initial
                #  self.exploitation_distance = self.exploitation_distance_initial - self.exploration_distance_initial
                # else:
                # self.plot.movement_exploration(self.mu, self.sigma, self.part_ant_explore)
                self.dict_, self.dict_coord_, self.dict_impo_, self.centers, self.coord_centers, self.dict_index, \
                self.action_zone, self.dict_limits = self.detect_areas.areas_levels(self.mu)
                self.part_ant_explore = copy.copy(self.part_ant)
                self.allocate_vehicles()
                self.obtain_global()
                for part in self.pop:
                    self.part_ant_exploit, self.distances_exploit = self.util.distance_part(self.g_exploit, self.n_data,
                                                                                            part,
                                                                                            self.part_ant_exploit,
                                                                                            self.distances_exploit,
                                                                                            self.array_part,
                                                                                            dfirst=True)
                    self.n_data += 1
                    if self.n_data > self.vehicles - 1:
                        self.n_data = 0
        else:
            done = False
        return self.state, reward, done, {}

    def obtain_global(self):
        for i in range(len(self.dict_centers)):
            list_vehicle = self.dict_centers["action_zone%s" % i]
            for j in range(len(list_vehicle)):
                part = self.pop[list_vehicle[j]]
                if j == 0:
                    best = part.best
                    best.fitness.values = part.fitness.values
                else:
                    if best.fitness < part.fitness:
                        best = part.best
                        best.fitness.values = part.fitness.values
            self.dict_global_best["action_zone%s" % i] = best

    def step_stage_exploitation_federated(self, action):

        """
        The output "out" of the method "step" is the positions of the particles (drones) after traveling 1000 m
        (scaled).

        method = 0 -> out = scalar vector
        out = [px_1, py_1, px_2, py_2, px_3, py_3, px_4, py_4, lbx_1, lby_1, lbx_2, lby_2, lbx_3, lby_3, lbx_4, lby_4,
               gbx, gby, sbx, sgy, mbx, mby]
               where:
               px: x coordinate of the drone position
               py: y coordinate of the drone position
               lbx: x coordinate of the local best
               lby: y coordinate of the local best
               gbx: x coordinate of the global best
               gby: y coordinate of the global best
               sbx: x coordinate of the sigma best (maximum uncertainty)
               sby: y coordinate of the sigma best (maximum uncertainty)
               mbx: x coordinate of the mean best (maximum contamination)
               mby: y coordinate of the mean best (maximum contamination)

        method = 1 -> out = images

        :param c1: weight that determinate the importance of the local best component
        :param c2: weight that determinate the importance of the global best component
        :param c3: weight that determinate the importance of the maximum uncertainty component
        :param c4: weight that determinate the importance of the maximum contamination component
        :param lam: ratio of one of the different length scales [Equation 7
        (https://doi.org/10.3390/electronics10131605)]
        :param post_array: refers to the posterior length scale of the surrogate model [Equation 7
        (https://doi.org/10.3390/electronics10131605)]
        """
        dis_steps = 0
        dist_ant = np.mean(self.distances)
        self.dist_pre = np.max(self.distances)
        self.n_data = 0
        self.f += 1

        while dis_steps < 10:

            previous_dist = np.max(self.distances)

            asv = 0
            for part in self.pop:
                action_zone = int(self.assig_centers[asv])
                self.mu_best = self.dict_max_mu["action_zone%s" % action_zone]
                self.sigma_best = self.dict_max_sigma["action_zone%s" % action_zone]
                self.best = self.dict_global_best["action_zone%s" % action_zone]
                self.toolbox.update(action[0], action[1], action[2], action[3], part)
                asv += 1

            for part in self.pop:
                self.ok, part = self.pso_fitness(part)

            self.obtain_global()

            for part in self.pop:
                self.part_ant, self.distances = self.util.distance_part(self.g, self.n_data, part, self.part_ant,
                                                                        self.distances, self.array_part, dfirst=False)

                self.part_ant_exploit, self.distances_exploit = self.util.distance_part(self.g_exploit, self.n_data,
                                                                                        part,
                                                                                        self.part_ant_exploit,
                                                                                        self.distances_exploit,
                                                                                        self.array_part,
                                                                                        dfirst=False)

                self.n_data += 1
                if self.n_data > self.vehicles - 1:
                    self.n_data = 0
            if (np.mean(self.distances) - self.last_sample) >= (np.min(self.post_array) * self.lam):
                self.k += 1
                self.last_sample = np.mean(self.distances)

                for i in range(len(self.dict_centers)):
                    list_vehicle = self.dict_centers["action_zone%s" % i]
                    for j in range(len(list_vehicle)):
                        part = self.pop[list_vehicle[j]]
                        self.take_sample(part, i)

                        self.n_data += 1
                        if self.n_data > self.vehicles - 1:
                            self.n_data = 0

                        self.samples += 1

                self.it.append(self.g)
                self.error = self.calculate_error(dfirts=False)
                self.error_data.append(self.error)

                # self.save_data()

            if self.len_scale >= 3 and self.print_warning:
                print("Warning: reached the minimum value of length scale (Exploitation)")
                self.fail_exp = True
                self.print_warning = False

            dis_steps = np.mean(self.distances) - dist_ant
            if np.max(self.distances) == previous_dist:
                break

            self.g += 1
            self.g_exploit += 1
        self.return_state()

        # reward = self.calculate_reward()
        reward = 0

        self.logbook.record(gen=self.g, evals=len(self.pop), **self.stats.compile(self.pop))

        if (self.distances >= self.exploitation_distance).any() or np.max(self.distances) == self.dist_pre:
            # while self.save < (self.exploitation_distance_initial / 25):
            #    error_list = copy.copy(self.dict_error_comparison["Distance%s" % self.save])
            #   self.ERROR_data = self.calculate_error()
            #  error_list.append(self.ERROR_data)
            # self.dict_error_comparison["Distance%s" % self.save] = copy.copy(error_list)
            # self.save += 1
            done = True
        # if np.max(self.distances) == self.dist_pre:
        #   self.repeat = True
        #  self.save_data()
        else:
            done = False
        return self.state, reward, done, {}

    def step_stage_exploitation_centralized(self, action):

        """
        The output "out" of the method "step" is the positions of the particles (drones) after traveling 1000 m
        (scaled).

        method = 0 -> out = scalar vector
        out = [px_1, py_1, px_2, py_2, px_3, py_3, px_4, py_4, lbx_1, lby_1, lbx_2, lby_2, lbx_3, lby_3, lbx_4, lby_4,
               gbx, gby, sbx, sgy, mbx, mby]
               where:
               px: x coordinate of the drone position
               py: y coordinate of the drone position
               lbx: x coordinate of the local best
               lby: y coordinate of the local best
               gbx: x coordinate of the global best
               gby: y coordinate of the global best
               sbx: x coordinate of the sigma best (maximum uncertainty)
               sby: y coordinate of the sigma best (maximum uncertainty)
               mbx: x coordinate of the mean best (maximum contamination)
               mby: y coordinate of the mean best (maximum contamination)

        method = 1 -> out = images

        :param c1: weight that determinate the importance of the local best component
        :param c2: weight that determinate the importance of the global best component
        :param c3: weight that determinate the importance of the maximum uncertainty component
        :param c4: weight that determinate the importance of the maximum contamination component
        :param lam: ratio of one of the different length scales [Equation 7
        (https://doi.org/10.3390/electronics10131605)]
        :param post_array: refers to the posterior length scale of the surrogate model [Equation 7
        (https://doi.org/10.3390/electronics10131605)]
        """
        dis_steps = 0
        dist_ant = np.mean(self.distances)
        self.dist_pre = np.max(self.distances)
        self.n_data = 0
        self.f += 1

        while dis_steps < 10 and not self.fail:

            previous_dist = np.max(self.distances)
            asv = 0
            for part in self.pop:
                action_zone = int(self.assig_centers[asv])
                self.obtain_max_centralized(action_zone)
                self.best = self.dict_global_best["action_zone%s" % action_zone]

                self.toolbox.update(action[0], action[1], action[2], action[3], part)
                asv += 1

            for part in self.pop:
                self.ok, part = self.pso_fitness(part)
                self.part_ant, self.distances = self.util.distance_part(self.g, self.n_data, part, self.part_ant,
                                                                        self.distances, self.array_part, dfirst=False)

                self.n_data += 1
                if self.n_data > self.vehicles - 1:
                    self.n_data = 0

            self.obtain_global()

            if (np.mean(self.distances) - self.last_sample) >= (np.min(self.post_array) * self.lam):
                self.k += 1
                self.ok = True
                self.last_sample = np.mean(self.distances)

                for part in self.pop:
                    self.ok, part = self.pso_fitness(part)
                    self.check_duplicate(part)

                    self.post_array = self.gp_regression()

                    self.samples += 1

                    self.n_data += 1
                    if self.n_data > self.vehicles - 1:
                        self.n_data = 0

                self.error = self.calculate_error()
                self.error_data.append(self.error)
                self.it.append(self.g)

                self.ok = False

            dis_steps = np.mean(self.distances) - dist_ant
            if np.max(self.distances) == previous_dist or self.fail:
                break

            # self.save_data()
            self.g += 1

        self.return_state()

        reward = self.calculate_reward()

        self.logbook.record(gen=self.g, evals=len(self.pop), **self.stats.compile(self.pop))

        if (self.distances >= self.exploitation_distance).any() or np.max(self.distances) == self.dist_pre:
            done = True
            # if np.max(self.distances) == self.dist_pre:
            # self.repeat = True
            # self.save_data()
        else:
            done = False
        return self.state, reward, done, {}

    def obtain_max_centralized(self, action_zone):
        index = copy.copy(self.dict_index["action_zone%s" % action_zone])
        mu_list = list()
        sigma_list = list()
        for i in range(len(index)):
            mu_list.append(self.mu[index[i]])
            sigma_list.append(self.sigma[index[i]])

        mu_max = max(mu_list)
        sigma_max = max(sigma_list)
        index_mu = mu_list.index(mu_max)
        index_sigma = sigma_list.index(sigma_max)
        self.mu_best = self.dict_coord_["action_zone%s" % action_zone][index_mu]
        self.sigma_best = self.dict_coord_["action_zone%s" % action_zone][index_sigma]

    def return_state(self):
        z = 0
        for part in self.pop:
            self.state = self.state
            if self.method == 0:
                self.state[z] = part[0]
                z += 1
                self.state[z] = part[1]
                z += 1
                self.state[z + 6] = part.best[0]
                self.state[z + 7] = part.best[1]
                # if self.n_data == self.vehicles - 1:
                #   self.state[16] = self.best[0]
                ##  self.state[17] = self.best[1]
                # self.state[18] = self.sigma_best[0]
                # self.state[19] = self.sigma_best[1]
                # self.state[20] = self.mu_best[0]
                # self.state[21] = self.mu_best[1]
            else:
                posx = 2 * z
                posy = (2 * z) + 1
                self.state = self.plot.part_position(self.part_ant[:, posx], self.part_ant[:, posy], self.state, z)
                z += 1
                if self.n_data == self.vehicles - 1:
                    self.state = self.plot.state_sigma_mu(self.mu, self.sigma, self.state)
            self.n_data += 1
            if self.n_data > self.vehicles - 1:
                self.n_data = 0

    def step(self, action):
        if self.stage == "exploration":
            action = np.array([2.0187, 0, 3.2697, 0])
            self.state, reward, done, dic = self.step_stage_exploration(action)
            if (self.distances >= self.exploration_distance).any() or np.max(self.distances) == self.dist_pre:
                self.stage = "exploitation"
        elif self.stage == "exploitation":
            action = np.array([3.6845, 1.5614, 0, 3.1262])
            if self.final_model == 'federated':
                self.state, reward, done, dic = self.step_stage_exploitation_federated(action)
            elif self.final_model == 'centralized':
                self.state, reward, done, dic = self.step_stage_exploitation_centralized(action)
        elif self.stage == "no_exploitation":
            action = action
            self.exploration_distance = self.exploitation_distance
            self.state, reward, done, dic = self.step_stage_exploration(action)
            if (self.distances >= self.exploration_distance).any() or np.max(self.distances) == self.dist_pre:
                done = True
        if done and not self.fail:
            if self.stage != 'no_exploitation':
                if self.final_model == 'centralized':
                    self.final_mu = copy.copy(self.mu)
                elif self.final_model == 'federated':
                    self.replace_action_zones()
            # self.plot.action_areas(self.dict_coord_, self.dict_impo_, self.centers)
            # self.plot.action_areas(self.dict_coord_bench, self.dict_impo_bench, self.centers_bench)
            self.type_error = 'action_zone'
            self.calculate_error()
            print("MSE az:", self.dict_error)
            # self.type_error = 'peaks'
            # self.calculate_error()
            # print("Error peak:", self.dict_error_peak)
            # self.type_error = 'all_map'
            # self.calculate_error()
        return self.state, reward, done, {}

    def final_gaussian(self):
        final_sample_x = copy.copy(self.x_h)
        final_sample_y = copy.copy(self.y_h)
        final_fitness = copy.copy(self.fitness)
        for i in range(len(self.dict_mu)):
            x = copy.copy(self.dict_sample_x["action_zone%s" % i])
            y = copy.copy(self.dict_sample_y["action_zone%s" % i])
            fitness = copy.copy(self.dict_fitness["action_zone%s" % i])
            for j in range(len(final_sample_x)):
                for k in range(len(x)):
                    if final_sample_x[j] == x[k] and final_sample_y[j] == y[k]:
                        final_fitness[j] = fitness[k]
                        del x[k]
                        del y[k]
                        del fitness[k]
                        break
            final_sample_x = [*final_sample_x, *x]
            final_sample_y = [*final_sample_y, *y]
            final_fitness = [*final_fitness, *fitness]

        x_a = np.array(final_sample_x).reshape(-1, 1)
        y_a = np.array(final_sample_y).reshape(-1, 1)
        x_train = np.concatenate([x_a, y_a], axis=1).reshape(-1, 2)
        y_train = np.array(final_fitness).reshape(-1, 1)

        self.gpr.fit(x_train, y_train)
        self.gpr.get_params()

        self.final_mu, self.final_sigma = self.gpr.predict(self.X_test, return_std=True)

    def replace_action_zones(self):
        self.final_mu = copy.copy(self.mu)
        self.final_sigma = copy.copy(self.sigma)
        for i in range(len(self.dict_index)):
            action_zone_index = copy.copy(self.dict_index["action_zone%s" % i])
            action_zone_mu = copy.copy(self.dict_mu["action_zone%s" % i])
            action_zone_sigma = copy.copy(self.dict_sigma["action_zone%s" % i])
            for j in range(len(action_zone_index)):
                self.final_mu[action_zone_index[j]] = action_zone_mu[action_zone_index[j]]
                self.final_sigma[action_zone_index[j]] = action_zone_sigma[action_zone_index[j]]

    def data_out(self):

        """
        Return the first and the last position of the particles (drones).
        """

        return self.X_test, self.secure, self.bench_function, self.grid_min, self.sigma, \
               self.mu, self.error_data, self.it, self.part_ant, self.bench_array, self.grid_or, self.bench_max, \
               self.dict_mu, self.dict_sigma, self.centers, self.part_ant_exploit, self.dict_centers, self.assig_centers, \
               self.part_ant_explore, self.final_mu, self.final_sigma, self.dict_limits

    def error_value(self):
        return self.error_data

    def return_bench(self):
        return self.centers_bench, self.dict_limits_bench, self.center_peaks_bench

    def return_seed(self):
        return self.seed

    def distances_data(self):
        return self.distances

    def save_excel(self):
        for i in range(int(self.exploitation_distance_initial / 25)):
            wb = openpyxl.Workbook()
            hoja = wb.active
            hoja.append(self.dict_error_comparison["Distance%s" % i])
            wb.save('../Test/' + self.file + '/ALLCONError_' + str(self.save_dist[i]) + '.xlsx')

        # wb2 = openpyxl.Workbook()
        # hoja2 = wb2.active
        # hoja2.append(self.error_comparison2)
        # wb2.save('../Test/' + self.file + '/ALLCONError_50.xlsx')

        # wb3 = openpyxl.Workbook()
        # hoja3 = wb3.active
        # hoja3.append(self.error_comparison3)
        # wb3.save('../Test/' + self.file + '/ALLCONError_75.xlsx')

        # wb4 = openpyxl.Workbook()
        # hoja4 = wb4.active
        # hoja4.append(self.error_comparison4)
        # wb4.save('../Test/' + self.file + '/ALLCONError_100.xlsx')

        # wb5 = openpyxl.Workbook()
        # hoja5 = wb5.active
        # hoja5.append(self.error_comparison5)
        # wb5.save('../Test/' + self.file + '/ALLCONError_125.xlsx')

        # wb6 = openpyxl.Workbook()
        # hoja6 = wb6.active
        # hoja6.append(self.error_comparison6)
        # wb6.save('../Test/' + self.file + '/ALLCONError_150.xlsx')

        # wb7 = openpyxl.Workbook()
        # hoja7 = wb7.active
        # hoja7.append(self.error_comparison7)
        # wb7.save('../Test/' + self.file + '/ALLCONErrorE_175.xlsx')

        # wb8 = openpyxl.Workbook()
        # hoja8 = wb8.active
        # hoja8.append(self.error_comparison8)
        # wb8.save('../Test/' + self.file + '/ALLCONErrorE_200.xlsx')

        # wb9 = openpyxl.Workbook()
        # hoja9 = wb9.active
        # hoja9.append(self.error_comparison9)
        # wb9.save('../Test/' + self.file + '/ALLCONErrorE_225.xlsx')
