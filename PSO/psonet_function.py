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
    creator.create("Particle", np.ndarray, fitness=creator.FitnessMax, speed=None, smin=None,
                   smax=None, gfitness=creator.FitnessMax, node=None, con=None, un=None)
    creator.create("BestGP", np.ndarray, fitness=creator.FitnessMax)


class PSOEnvironment(gym.Env):

    def __init__(self, resolution, ys, method, method_pso, initial_seed, initial_position, sensor_vehicle, vehicles=4,
                 exploration_distance=100,
                 exploitation_distance=200, reward_function='mse', behavioral_method=0, type_error='all_map',
                 stage='exploration', final_model='samples'):
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
        ker = RBF(length_scale=10, length_scale_bounds=(1e-1, 10 ^ 5))
        self.gpr = GaussianProcessRegressor(kernel=ker, alpha=1e-6)  # optimizer=None)
        self.seed = initial_seed
        self.initial_seed = initial_seed
        self.lam = 0.375
        self.reward_function = reward_function
        self.behavioral_method = behavioral_method
        self.initial_position = initial_position
        self.method = method
        self.method_pso = method_pso

        self.dict_sensors_ = {}
        self.dict_benchs_ = {}
        self.mu_best = []
        self.sigma_best = []
        self.best = []
        self.g = 0
        self.n_data = 0
        self.dist_pre = 0
        self.last_sample = 0
        self.part_ant = np.zeros((1, self.vehicles * 2))
        self.s_ant = np.zeros(self.vehicles)
        self.s_n = np.full(self.vehicles, True)
        self.distances = np.zeros(self.vehicles)
        self.s_sf = list()
        self.array_part = np.zeros((1, self.vehicles * 2))
        self.post_array = None

        if self.method == 0:
            self.state = np.zeros(self.vehicles * 8, )
        else:
            self.state = np.zeros((6, self.xs, self.ys))

        self.grid_or = Map(self.xs, ys).black_white()

        self.grid_min, self.grid_max, self.grid_max_x, self.grid_max_y = 0, self.ys, self.xs, self.ys

        self.p = 0
        self.max_peaks = None

        self.df_bounds, self.X_test, self.bench_limits = Bounds(self.resolution, self.xs, self.ys,
                                                                load_file=False).map_bound()
        self.secure, self.df_bounds = Bounds(self.resolution, self.xs, self.ys).interest_area()

        #self.plot = Plots(self.xs, self.ys, self.X_test, self.secure, self.bench_function, self.grid_min, self.grid_or,
         #                 self.stage)

        self.util = Utils(self.vehicles)

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
        part.node = self.p_vehicles[self.p]
        self.P.add_node(part.node, S_p=dict.fromkeys(self.sensor_vehicle[self.p], []),
                        U_p=list_part, Q_p=list())
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
                    intersection = node_p[1]["S_p"].keys() & node_q[1]["S_p"].keys()
                    if node_q != node_p and len(intersection) > 0:
                        if not self.P.has_edge(node_p[0], node_q[0]):
                            self.P.add_edge(node_p[0], node_q[0], S_pq=intersection)
        self.sub_fleets = list(nx.connected_components(self.P))
        for i, sub_fleet in enumerate(self.sub_fleets):
            S_sf = set()
            for particle in sub_fleet:
                S_sf = S_sf | self.P.nodes[particle]['S_p'].keys()
            for i, sensor in enumerate(S_sf):
                self.dict_sensors_[sensor] = {}
                self.dict_sensors_[sensor]['mu'] = {}
                self.dict_sensors_[sensor]['mu']['data'] = []
                self.dict_sensors_[sensor]['mu']['max'] = []
                self.dict_sensors_[sensor]['sigma'] = {}
                self.dict_sensors_[sensor]['sigma']['data'] = []
                self.dict_sensors_[sensor]['sigma']['max'] = []
                self.dict_sensors_[sensor]['cant'] = 0
                self.dict_benchs_[sensor] = {}
                self.seed += 1
                self.dict_benchs_[sensor]['map_created'], self.dict_benchs_[sensor]['original'] = Benchmark_function_het(self.grid_or, 1, self.xs, self.ys, None, self.seed, 0,
                                                                 base_benchmark="ackley", randomize_shekel=True).create_new_map()
            self.s_sf.append(S_sf)
            print(f'Subfleet {i} contains {S_sf} y se usa en eqs. 13c y 13d')

            for particle in sub_fleet:
                sensors = self.P.nodes[particle]['S_p'].keys()
                for s, sensor in enumerate(sensors):
                    cant = self.dict_sensors_[sensor]['cant']
                    cant += 1
                    self.dict_sensors_[sensor]['cant'] = cant
                print(f'Particle {particle} contains {self.P.nodes[particle]["S_p"]} y se usa en eqs. 13a y 13b')

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
        self.generatePart()
        self.tool()
        random.seed(self.seed)
        self.swarm()
        self.statistic()
        self.set_sensor()
        self.first_values()

    def reset_variables(self):
        self.dict_sensors_ = {}
        self.dict_benchs_ = {}
        self.mu_best = []
        self.sigma_best = []
        self.best = []
        self.g = 0
        self.n_data = 0
        self.dist_pre = 0
        self.last_sample = 0
        self.part_ant = np.zeros((1, self.vehicles * 2))
        self.s_ant = np.zeros(self.vehicles)
        self.s_n = np.full(self.vehicles, True)
        self.distances = np.zeros(self.vehicles)
        self.s_sf = list()
        self.array_part = np.zeros((1, self.vehicles * 2))
        self.post_array = None

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

        v_u1 = u1 * (part.pbest - part)
        v_u2 = u2 * (part.gbest - part)
        v_u3 = u3 * (part.un - part)
        v_u4 = u4 * (part.con - part)
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

    def take_measures(self):
        for i, subfleet in enumerate(self.sub_fleets):
            for p, particle in enumerate(subfleet):
                sensors = self.P.nodes[particle]['S_p']
                sensor_list = list(sensors.keys())
                q_p = self.P.nodes[particle]['Q_p']
                u_p = self.P.nodes[particle]['U_p']
                last = u_p[-1]
                q_p.append(last)
                self.P.nodes[particle]['Q_p'] = q_p
                x = last[0]
                y = last[1]
                for j in range(len(sensor_list)):
                    key_sensor = sensor_list[j]
                    measure = sensors[key_sensor]
                    print(measure)
                    bench = self.dict_benchs_[key_sensor]['map_created']
                    new_measure = bench[x][y]
                    measure.append(new_measure)
                    sensors[key_sensor] = measure
                    print(sensors)
                self.P.nodes[particle]['S_p'] = sensors


    def gp_update(self):
        # S_n = {"sensor1": {"mu": [], "sigma": []}}
        measuring_positions = nx.get_node_attributes(self.P, "Q_p")
        for i, sub_fleet in enumerate(self.sub_fleets):
            sensors = self.s_sf[i]
            for s, sensor in enumerate(sensors):
                measures_for_sensor = []
                coordinates_for_sensor = []
                for p, particle in enumerate(sub_fleet):
                    s_p = self.P.nodes(data=True)[particle]['S_p']
                    sensor_key = s_p.keys()
                    for r, sensorp in enumerate(sensor_key):
                        if sensor in sensorp:
                            measures_for_sensor.extend(s_p[sensorp])
                            coordinates_for_sensor.append(measuring_positions[particle])
                print(s, sensor)
                print(coordinates_for_sensor)
                print(measures_for_sensor)
                self.gpr.fit(coordinates_for_sensor, measures_for_sensor)
                self.dict_sensors_[sensor]['mu']['data'], self.dict_sensors_[sensor]['sigma']['data'] = self.gpr.predict(self.X_test, return_std=True)
                self.post_array = round(np.min(np.exp(self.gpr.kernel_.theta[0])), 1)

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

    def model_max(self):
        for i, sub_fleet in enumerate(self.sub_fleets):
            sensors_key = self.s_sf[i]
            for s, sensor in enumerate(sensors_key):
                mu = self.dict_sensors_[sensor]['mu']['data']
                sigma = self.dict_sensors_[sensor]['sigma']['data']
                max_mu, coord_mu = self.obtain_max(mu)
                self.dict_sensors_[sensor]['mu']['max'] = [max_mu, coord_mu]
                max_sigma, coord_sigma = self.obtain_max(sigma)
                self.dict_sensors_[sensor]['sigma']['max'] = [max_sigma, coord_sigma]

    def w_values(self):
        cant_type = len(self.s_sf)
        for s in range(len(self.s_sf)):
            sensors = self.s_sf[s]
            for i, sensor in enumerate(sensors):
                cant_sensor = self.dict_sensors_[sensor]['cant']

    def method_coupled(self):
        for i, subfleet in enumerate(self.sub_fleets):
            sensors = self.s_sf[i]
            max_con = []
            max_un = []
            for s, sensor in enumerate(sensors):
                max_mu, coord_mu = self.dict_sensors_[sensor]['mu']['max']
                max_sigma, coord_sigma = self.dict_sensors_[sensor]['sigma']['max']
                w_value = self.dict_sensors_[sensor]['w']
                max_con = max_con + w_value * max_mu
                max_un = max_un +w_value * max_sigma

    def method_decoupled(self):
        for i, subfleet in enumerate(self.sub_fleets):
            sensors = self.s_sf[i]
            max_con_value = 0
            max_un_value = 0
            max_con = []
            max_un = []
            for s, sensor in enumerate(sensors):
                max_mu, coord_mu = self.dict_sensors_[sensor]['mu']['max']
                max_sigma, coord_sigma = self.dict_sensors_[sensor]['sigma']['max']
                if s == 0:
                    max_con = coord_mu
                    max_con_value = max_mu
                    max_un = coord_sigma
                    max_un_value = max_sigma
                else:
                    if max_con_value < max_mu:
                        max_con = coord_mu
                    if max_un_value < max_sigma:
                        max_un = coord_sigma
            for p, particle in enumerate(subfleet):
                for part in self.pop:
                    if particle == part.node:
                        part.con = max_con
                        part.un = max_un
                        break

    def local_best(self, part, dfirst):
        part, self.s_n = Limits(self.secure, self.xs, self.ys).new_limit(self.g, part, self.s_n, self.n_data,
                                                                         self.s_ant, self.part_ant)
        x_bench = int(part[0])
        y_bench = int(part[1])

        sensors = self.P.nodes[part.node]['S_p']
        sensor_list = list(sensors.keys())

        for i in range(len(sensor_list)):
            bench = self.dict_benchs_[sensor_list[i]]['map_created']
            pbest = [bench[x_bench][y_bench]]
            if i == 0:
                new_pbest = pbest
            else:
                if new_pbest < pbest:
                    new_pbest = pbest

        part.fitness.values = new_pbest
        if dfirst:
            part.pbest = creator.Particle(part)
            part.pbest.fitness.values = part.fitness.values
        else:
            if part.pbest.fitness < part.fitness:
                part.pbest = creator.Particle(part)
                part.pbest.fitness.values = part.fitness.values

    def global_best(self, part, dfirst=False):
        name = part.node
        p_sf = None
        for i, subfleet in enumerate(self.sub_fleets):
            for p, particle in enumerate(subfleet):
                if name == particle:
                    p_sf = subfleet
                    break
        for p, particle in enumerate(p_sf):
            if p == 0:
                new_gbest = part.pbest.fitness.values
                if dfirst:
                    part.gbest = creator.Particle(part)
                    part.gbest.gfitness.values = part.fitness.values
            else:
                if new_gbest < part.pbest.fitness.values:
                    new_gbest = part.pbest.fitness.values
        part.gfitness.values = new_gbest
        if part.gbest.gfitness < part.gfitness:
            part.gbest = creator.Particle(part)
            part.gbest.gfitness.values = part.gfitness.values

    def first_values(self):
        for part in self.pop:
            self.local_best(part, dfirst=True)
            self.global_best(part, dfirst=True)

            self.part_ant, self.distances = self.util.distance_part(self.g, self.n_data, part, self.part_ant,
                                                                    self.distances, self.array_part, dfirst=True)

            self.n_data += 1
            if self.n_data > self.vehicles - 1:
                self.n_data = 0

        self.take_measures()
        self.gp_update()
        self.model_max()

        if self.method_pso == 'coupled':
            self.method_coupled()
        elif self.method_pso == 'decoupled':
            self.method_decoupled()

    def step(self, action):
        dis_steps = 0
        dist_ant = np.mean(self.distances)
        self.dist_pre = np.max(self.distances)
        self.n_data = 0

        while dis_steps < 10:

            previous_dist = np.max(self.distances)

            for part in self.pop:
                self.toolbox.update(part.node, action[0], action[1], action[2], action[3], part)

            for part in self.pop:
                self.local_best(part, dfirst=False)

            for part in self.pop:
                self.global_best(part)

            for part in self.pop:
                self.part_ant, self.distances = self.util.distance_part(self.g, self.n_data, part, self.part_ant,
                                                                        self.distances, self.array_part, dfirst=False)
                self.n_data += 1
                if self.n_data > self.vehicles - 1:
                    self.n_data = 0

            if (np.mean(self.distances) - self.last_sample) >= (np.min(self.post_array) * self.lam):
                self.last_sample = np.mean(self.distances)
                self.take_measures()
                self.gp_update()
                self.model_max()

                if self.method_pso == 'coupled':
                    self.method_coupled()
                elif self.method_pso == 'decoupled':
                    self.method_decoupled()

            dis_steps = np.mean(self.distances) - dist_ant
            if np.max(self.distances) == previous_dist:
                break
        self.g += 1

        if (self.distances >= self.exploration_distance).any() or np.max(self.distances) == self.dist_pre:
            done = True
        else:
            done = False

        return done
