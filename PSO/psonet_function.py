import copy
import math
import os
import sys
import operator
from operator import add

import gym
import networkx as nx
import pandas as pd
import openpyxl
from deap import base
from deap import creator
from deap import tools
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.metrics import mean_squared_error, r2_score

from Benchmark.benchmark_functions import *
from Data.limits import Limits
from Data.utils import Utils
from Environment.bounds import Bounds
from Environment.contamination_areas import DetectContaminationAreas
from Environment.map import *
from Environment.plot_het import Plots

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

"""[https://deap.readthedocs.io/en/master/examples/pso_basic.html]"""


def createPart():
    """
    Creation of the objects "FitnessMax" and "Particle"
    """
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Particle", np.ndarray, fitness=creator.FitnessMax, speed=None, smin=None,
                   smax=None, node=None)
    creator.create("BestGP", np.ndarray, fitness=creator.FitnessMax)


class PSOEnvironment(gym.Env):

    def __init__(self, resolution, ys, method, method_pso, initial_seed, initial_position, sensor_vehicle, vehicles=4,
                 exploration_distance=100,
                 exploitation_distance=200, reward_function='mse', behavioral_method=0, type_error='all_map',
                 stage='exploration', final_model='samples'):
        self.p_vehicles = ['p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7', 'p8']
        self.s_sensor = ['s1', 's2', 's3', 's4', 's5']
        self.sensor_v = sensor_vehicle
        self.P = nx.MultiGraph()
        self.sub_fleets = None
        self.sensor_vehicle = None
        self.type_error = type_error
        self.final_model = final_model
        self.initial_stage = stage
        self.exploration_distance_initial = exploration_distance
        self.exploitation_distance_initial = exploitation_distance
        self.exploration_distance = exploration_distance
        self.exploitation_distance = exploitation_distance
        self.stage = stage
        self.vehicles = None
        self.population = None
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
        self.mean_error = list()
        self.conf_error = list()
        self.mean_peak_error = list()
        self.conf_peak_error = list()
        self.sensor = list()
        self.w = list()
        self.cant_sensor = list()
        self.seed_bench = initial_seed

        self.dict_sensors_ = {}
        self.dict_benchs_ = {}
        self.mu_best = []
        self.sigma_best = []
        self.best = []
        self.g = 0
        self.n_data = 0
        self.dist_pre = 0
        self.last_sample = 0
        self.part_ant = None
        self.s_ant = None
        self.s_n = None
        self.distances = None
        self.s_sf = list()
        self.array_part = None
        self.post_array = None
        self.bench_function = None
        self.data_particle = ['pbest', 'pbest_fitness', 'gbest', 'un', 'con']
        self.p = 0
        self.max_peaks = None
        self.r2_sensor = []
        self.error_peak_sensor = []

        if self.method == 0:
            self.state = None
        else:
            self.state = np.zeros((6, self.xs, self.ys))

        self.grid_or = Map(self.xs, ys).black_white()

        self.grid_min, self.grid_max, self.grid_max_x, self.grid_max_y = 0, self.ys, self.xs, self.ys

        self.tim = 0
        if self.tim == 0:
            self.df_bounds, self.X_test_no, self.bench_limits_no = Bounds(self.resolution, self.xs, self.ys,
                                                                load_file=False).map_bound()
            self.X_test, self.bench_limits = Bounds(self.resolution, self.xs, self.ys,
                                                       load_file=False).available_xtest()
             # = Bounds(self.resolution, self.xs, self.ys,
             #                                                    load_file=False).available_xtest()
            self.tim = 1
        self.secure = Bounds(self.resolution, self.xs, self.ys).interest_area()

        self.plot = Plots(self.xs, self.ys, self.X_test, self.secure, self.grid_min, self.grid_or,
                          self.stage)

        self.util = None

        createPart()

    def function1(self, individual, position, height, width):
        return height / (1 + width * sum(((individual[j] - position[j]) ** 2) for j in range(len(individual))))

    def generatePart(self):

        """
        Generates a random position and a random speed for the particles (drones).
        """
        list_part = list()
        part = creator.Particle([self.initial_position[self.p, i] for i in range(self.size)])
        list_part.append(np.array(part))
        part.speed = np.array([random.uniform(self.smin, self.smax) for _ in range(self.size)])
        part.smin = self.smin
        part.smax = self.smax
        part.node = self.p_vehicles[self.p]
        self.P.add_node(part.node, S_p=dict.fromkeys(self.sensor_vehicle[self.p], []),
                        U_p=list_part, Q_p=list(), D_p=dict.fromkeys(self.data_particle), index=self.p,
                        pbest=dict.fromkeys(self.sensor_vehicle[self.p]),
                        fitness=dict.fromkeys(self.sensor_vehicle[self.p]),
                        fitness_list=dict.fromkeys(self.sensor_vehicle[self.p], []))
        self.p += 1

        return part

    def fleet_configuration(self):
        random.seed(self.seed)
        # self.vehicles = random.randint(2, 8)
        self.vehicles = 4
        self.population = copy.copy(self.vehicles)
        i = 0
        sensors = []
        # while i < self.vehicles:
        #     list_s = []
        #     sensor = random.randint(1, 5)
        #     index_s = random.sample(range(5), sensor)
        #     for j in range(len(index_s)):
        #         list_s.append(self.s_sensor[index_s[j]])
        #     list_s = sorted(list_s)
        #     sensors.append(list_s)
        #     i += 1
        while i < self.vehicles:
            number = random.randint(0, 20)
            # list_s = sorted(self.sensor_v[number])
            list_s = ['s1']
            sensors.append(list_s)
            i += 1
        self.sensor_vehicle = sensors
        print(self.vehicles, self.sensor_vehicle)

    def create_dictionaries(self, sensor):
        self.dict_sensors_[sensor] = {}
        self.dict_sensors_[sensor]['U_sf'] = []
        self.dict_sensors_[sensor]['fitness'] = []
        self.dict_sensors_[sensor]['vehicles'] = {}
        self.dict_sensors_[sensor]['mu'] = {}
        self.dict_sensors_[sensor]['mu']['data'] = []
        self.dict_sensors_[sensor]['mu']['max'] = []
        self.dict_sensors_[sensor]['mu']['peaks'] = []
        self.dict_sensors_[sensor]['sigma'] = {}
        self.dict_sensors_[sensor]['sigma']['data'] = []
        self.dict_sensors_[sensor]['sigma']['max'] = []
        self.dict_sensors_[sensor]['cant'] = 0
        self.dict_sensors_[sensor]['w'] = 0
        self.dict_sensors_[sensor]['error'] = {}
        self.dict_sensors_[sensor]['error']['data'] = []
        self.dict_sensors_[sensor]['error']['mean'] = []
        self.dict_sensors_[sensor]['error']['conf'] = []
        self.dict_sensors_[sensor]['error']['peak'] = {}
        self.dict_sensors_[sensor]['error']['peak']['data'] = []
        self.dict_sensors_[sensor]['error']['peak']['mean'] = []
        self.dict_sensors_[sensor]['error']['peak']['conf'] = []
        self.dict_benchs_[sensor] = {}
        self.dict_benchs_[sensor]['map_created'], self.dict_benchs_[sensor][
            'original'], self.dict_benchs_[sensor]['num_peaks'], self.dict_benchs_[sensor][
            'index_peaks'] = Benchmark_function(self.grid_or, self.resolution, self.xs, self.ys, self.X_test,
                                                self.seed_bench, self.vehicles).create_new_map()
        self.dict_benchs_[sensor]['peaks'] = []

    def peaks_bench(self):
        for i, subfleet in enumerate(self.sub_fleets):
            sensors = self.s_sf[i]
            for s, sensor in enumerate(sensors):
                peaks = []
                index_bench = copy.copy(self.dict_benchs_[sensor]['index_peaks'])
                bench = copy.copy(self.dict_benchs_[sensor]['original'])
                for j, ind in enumerate(index_bench):
                    peaks.append(bench[round(ind)])
                self.dict_benchs_[sensor]['peaks'] = copy.copy(peaks)

    def set_sensor(self):
        i = 0
        for node_p in self.P.nodes(data=True):
            j = 0
            for node_q in self.P.nodes(data=True):
                j += 1
                if i < j:
                    intersection = sorted(node_p[1]["S_p"].keys() & node_q[1]["S_p"].keys())
                    if node_p != node_q and len(intersection) > 0:
                        if not self.P.has_edge(node_p[0], node_q[0]):
                            self.P.add_edge(node_p[0], node_q[0], S_pq=intersection)

            i += 1
        sub = sorted(nx.connected_components(self.P))
        self.sub_fleets = [sorted(item) for item in sub]
        for i, sub_fleet in enumerate(self.sub_fleets):
            sub_fleet = sorted(sub_fleet)
            S_sf = set()
            for particle in sub_fleet:
                S_sf = S_sf | self.P.nodes[particle]['S_p'].keys()
            S_sf = sorted(S_sf)
            for j, sensor in enumerate(S_sf):
                self.create_dictionaries(sensor)
                list_vehicles = list()
                for particle in sub_fleet:
                    v_sensors = self.P.nodes[particle]['S_p'].keys()
                    for key in v_sensors:
                        if key == sensor:
                            list_vehicles.append(particle)
                self.dict_sensors_[sensor]['vehicles'] = copy.copy(list_vehicles)
                self.seed_bench += 130
            self.s_sf.append(S_sf)
            # print(f'Subfleet {i} contains {S_sf} y se usa en eqs. 13c y 13d')

            for particle in sub_fleet:
                sensors = self.P.nodes[particle]['S_p'].keys()
                for s, sensor in enumerate(sensors):
                    cant = self.dict_sensors_[sensor]['cant']
                    cant += 1
                    self.dict_sensors_[sensor]['cant'] = cant
                # print(f'Particle {particle} contains {self.P.nodes[particle]["S_p"]} y se usa en eqs. 13a y 13b')
        self.w_values()
        # print(self.sub_fleets)
        # print('sf', self.s_sf)

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
        # self.best = self.pop[0]

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
        self.seed += 1
        self.fleet_configuration()
        self.reset_variables()
        random.seed(self.seed)
        self.tool()
        self.swarm()
        self.statistic()
        self.set_sensor()
        self.peaks_bench()
        self.first_values()

    def reset_variables(self):
        self.p_vehicles = ['p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7', 'p8']
        self.P = nx.MultiGraph()
        self.sub_fleets = None
        self.dict_sensors_ = {}
        self.dict_benchs_ = {}
        self.mu_best = []
        self.sigma_best = []
        self.best = []
        self.g = 0
        self.p = 0
        self.max_peaks = None
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
        self.bench_function = None
        self.data_particle = ['pbest', 'pbest_fitness', 'gbest', 'un', 'con']
        self.r2_sensor = []
        self.error_peak_sensor = []
        self.sensor = list()
        self.cant_sensor = list()
        self.w = list()

        self.util = Utils(self.vehicles)

    def updateParticle_n(self, p, c1, c2, c3, c4, part):

        """
        Calculates the speed and the position of the particles (drones).
        """

        d_p = copy.copy(self.P.nodes[part.node]['D_p'])
        pbest = d_p['pbest']
        gbest = d_p['gbest']
        max_con = d_p['con']
        max_un = d_p['un']
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
        v_u1 = u1 * (pbest - part)
        v_u2 = u2 * (gbest - part)
        v_u3 = u3 * (max_un - part)
        v_u4 = u4 * (max_con - part)
        w = 1
        part.speed = v_u1 + v_u2 + v_u3 + v_u4 + part.speed * w
        for i, speed in enumerate(part.speed):
            if abs(speed) < part.smin:
                part.speed[i] = math.copysign(part.smin, speed)
            elif abs(speed) > part.smax:
                part.speed[i] = math.copysign(part.smax, speed)
        part[:] = part + part.speed

        return part

    def take_measures(self):
        for i, subfleet in enumerate(self.sub_fleets):
            for p, particle in enumerate(subfleet):
                sensors = copy.copy(self.P.nodes[particle]['S_p'])
                sensor_list = sensors.keys()
                q_p = copy.copy(self.P.nodes[particle]['Q_p'])
                u_p = copy.copy(self.P.nodes[particle]['U_p'])
                last = u_p[-1]
                x = last[0]
                y = last[1]
                q_p.append(last)
                # print('q_p', particle, q_p)
                self.P.nodes[particle]['Q_p'] = copy.copy(q_p)
                for j, key in enumerate(sensor_list):
                    measure = copy.copy(sensors[key])
                    bench = copy.copy(self.dict_benchs_[key]['map_created'])
                    new_measure = bench[x][y]
                    measure.append(new_measure)
                    sensors[key] = copy.copy(measure)
                self.P.nodes[particle]['S_p'] = copy.copy(sensors)
                # print(sensors)

    def gp_update(self):
        # S_n = {"sensor1": {"mu": [], "sigma": []}}
        for i, sub_fleet in enumerate(self.sub_fleets):
            sensors = self.s_sf[i]
            for s, sensor in enumerate(sensors):
                measures_for_sensor = []
                coordinates_for_sensor = []
                for p, particle in enumerate(sub_fleet):
                    s_p = copy.copy(self.P.nodes[particle]['S_p'])
                    q_p = copy.copy(self.P.nodes[particle]['Q_p'])
                    sensor_key = s_p.keys()
                    for r, sensorp in enumerate(sensor_key):
                        if sensorp == sensor:
                            # print(sensorp, s_p[sensorp])
                            measures_for_sensor.extend(s_p[sensorp])
                            coordinates_for_sensor.extend(q_p)
                coordinates_for_sensor = np.array(coordinates_for_sensor).reshape(-1, 2)
                # print(s, sensor)
                # print(coordinates_for_sensor)
                # print(measures_for_sensor)
                # print('coord', coordinates_for_sensor)
                # print('measure', measures_for_sensor)
                self.gpr.fit(coordinates_for_sensor, measures_for_sensor)
                self.dict_sensors_[sensor]['mu']['data'], self.dict_sensors_[sensor]['sigma'][
                    'data'] = self.gpr.predict(self.X_test, return_std=True)
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
                mu = copy.copy(self.dict_sensors_[sensor]['mu']['data'])
                sigma = copy.copy(self.dict_sensors_[sensor]['sigma']['data'])
                max_mu, coord_mu = self.obtain_max(mu)
                self.dict_sensors_[sensor]['mu']['max'] = [max_mu, coord_mu]
                max_sigma, coord_sigma = self.obtain_max(sigma)
                self.dict_sensors_[sensor]['sigma']['max'] = [max_sigma, coord_sigma]
                # print('max_mu', max_mu, 'max_sigma', max_sigma)

    def w_values(self):
        for s in range(len(self.s_sf)):
            inver = 0
            sensors = self.s_sf[s]
            for i, sensor in enumerate(sensors):
                cant_sensor = self.dict_sensors_[sensor]['cant']
                h = 1 / cant_sensor
                inver = inver + h
            x_value = 1 / inver
            for i, sensor in enumerate(sensors):
                self.dict_sensors_[sensor]['w'] = x_value / self.dict_sensors_[sensor]['cant']

    def method_coupled(self):
        for i, subfleet in enumerate(self.sub_fleets):
            sensors = self.s_sf[i]
            max_con = [0, 0]
            max_un = [0, 0]
            # for s, sensor in enumerate(sensors):
            #     max_mu, coord_mu = self.dict_sensors_[sensor]['mu']['max']
            #     max_sigma, coord_sigma = self.dict_sensors_[sensor]['sigma']['max']
            #     w_value = self.dict_sensors_[sensor]['w']
            #     max_con = max_con + w_value * coord_mu
            #     max_un = max_un + w_value * coord_sigma
            #     # print('max', max_mu, coord_mu)
            summatory_mu = list()
            summatory_sigma = list()
            for s, sensor in enumerate(sensors):
                mu = list(copy.copy(self.dict_sensors_[sensor]['mu']['data']))
                sigma = list(copy.copy(self.dict_sensors_[sensor]['sigma']['data']))
                w_value = self.dict_sensors_[sensor]['w']
                if s == 0:
                    summatory_mu = [data_m * w_value for data_m in mu]
                    summatory_sigma = [data_s * w_value for data_s in sigma]
                else:
                    data_mu = [data_m * w_value for data_m in mu]
                    data_sigma = [data_s * w_value for data_s in sigma]
                    summatory_mu = list(map(add, summatory_mu, data_mu))
                    summatory_sigma = list(map(add, summatory_sigma, data_sigma))
                    max_mu, coord_mu = copy.copy(self.dict_sensors_[sensor]['mu']['max'])
                # print('max', max_mu, coord_mu)
            ind_mu = summatory_mu.index(max(summatory_mu))
            ind_sigma = summatory_sigma.index(max(summatory_sigma))
            for p, particle in enumerate(subfleet):
                self.P.nodes[particle]['D_p']['con'] = self.X_test[ind_mu]
                self.P.nodes[particle]['D_p']['un'] = self.X_test[ind_sigma]

    def method_decoupled(self):
        for i, subfleet in enumerate(self.sub_fleets):
            sensors = self.s_sf[i]
            max_con_value = 0
            max_un_value = 0
            max_con = []
            max_un = []
            for s, sensor in enumerate(sensors):
                max_mu, coord_mu = copy.copy(self.dict_sensors_[sensor]['mu']['max'])
                max_sigma, coord_sigma = copy.copy(self.dict_sensors_[sensor]['sigma']['max'])
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
                self.P.nodes[particle]['D_p']['con'] = max_con
                self.P.nodes[particle]['D_p']['un'] = max_un

    def local_best_coupled(self, part):
        part, self.s_n = Limits(self.secure, self.xs, self.ys, self.vehicles).new_limit(self.g, part, self.s_n, self.n_data,
                                                                         self.s_ant, self.part_ant)
        x_bench = int(part[0])
        y_bench = int(part[1])

        list_part = copy.copy(self.P.nodes[part.node]['U_p'])
        list_part.append(np.array(part))

        sensors = self.P.nodes[part.node]['S_p'].keys()

        for i, key in enumerate(sensors):
            list_f = copy.copy(self.P.nodes[part.node]['fitness_list'][key])
            bench = copy.copy(self.dict_benchs_[key]['map_created'])
            pbest = [bench[x_bench][y_bench]]
            list_f.append(pbest[0])
            self.P.nodes[part.node]['fitness_list'][key] = copy.copy(list_f)

        summatory = list()
        for i, key in enumerate(sensors):
            w = copy.copy(self.dict_sensors_[key]['w'])
            value = copy.copy(self.P.nodes[part.node]['fitness_list'][key])
            if i == 0:
                summatory = [data * w for data in value]
            else:
                list1 = [data * w for data in value]
                summatory = list(map(add, summatory, list1))
        ind = summatory.index(max(summatory))
        self.P.nodes[part.node]['D_p']['pbest'] = list_part[ind]
        self.P.nodes[part.node]['U_p'] = copy.copy(list_part)

    def local_best_decoupled(self, part, dfirst):
        part, self.s_n = Limits(self.secure, self.xs, self.ys, self.vehicles).new_limit(self.g, part, self.s_n,
                                                                                        self.n_data,
                                                                                        self.s_ant, self.part_ant)
        x_bench = int(part[0])
        y_bench = int(part[1])

        list_part = copy.copy(self.P.nodes[part.node]['U_p'])
        list_part.append(np.array(part))

        sensors = self.P.nodes[part.node]['S_p'].keys()

        for i, key in enumerate(sensors):
            bench = copy.copy(self.dict_benchs_[key]['map_created'])
            pbest = [bench[x_bench][y_bench]]
            list_f = copy.copy(self.P.nodes[part.node]['fitness_list'][key])
            list_f.append(pbest[0])
            self.P.nodes[part.node]['fitness_list'][key] = copy.copy(list_f)

            if dfirst:
                self.P.nodes[part.node]['pbest'][key] = part
                self.P.nodes[part.node]['fitness'][key] = pbest
            else:
                pbest_fitness = self.P.nodes[part.node]['fitness'][key]
                if pbest_fitness < pbest:
                    self.P.nodes[part.node]['pbest'][key] = part
                    self.P.nodes[part.node]['fitness'][key] = pbest

        key_max = max(self.P.nodes[part.node]['fitness'].items(), key=operator.itemgetter(1))[0]
        self.P.nodes[part.node]['D_p']['pbest'] = self.P.nodes[part.node]['pbest'][key_max]

        self.P.nodes[part.node]['U_p'] = copy.copy(list_part)

    def global_best_decoupled(self):
        for i, subfleet in enumerate(self.sub_fleets):
            for p, particle in enumerate(subfleet):
                gbest_part = list()
                fitness_part = list()
                p_sensor = self.P.nodes[particle]['S_p'].keys()
                for s, sensor in enumerate(p_sensor):
                    gbest_sensor = list()
                    fitness_sensor = list()
                    vehicles = self.dict_sensors_[sensor]['vehicles']
                    for v, vehicle in enumerate(vehicles):
                        gbest = self.P.nodes[vehicle]['pbest'][sensor]
                        fitness = self.P.nodes[vehicle]['fitness'][sensor]
                        gbest_sensor.append(gbest)
                        fitness_sensor.append(fitness)
                    max_fitness_s = max(fitness_sensor)
                    index_s = fitness_sensor.index(max_fitness_s)
                    gbest_s = gbest_sensor[index_s]
                    gbest_part.append(gbest_s)
                    fitness_part.append(max_fitness_s)
                index = fitness_part.index(max(fitness_part))
                self.P.nodes[particle]['D_p']['gbest'] = gbest_part[index]

    def global_best_coupled(self):
        for i, subfleet in enumerate(self.sub_fleets):
            for p, particle in enumerate(subfleet):
                sensors = self.P.nodes[particle]['S_p'].keys()
                summatory = list()
                for i, key in enumerate(sensors):
                    usf = copy.copy(self.dict_sensors_[key]['U_sf'])
                    w = copy.copy(self.dict_sensors_[key]['w'])
                    value = copy.copy(self.dict_sensors_[key]['fitness'])
                    if i == 0:
                        summatory = [data * w for data in value]
                    else:
                        list1 = [data * w for data in value]
                        summatory = list(map(add, summatory, list1))
                ind = summatory.index(max(summatory))
                self.P.nodes[particle]['D_p']['gbest'] = usf[ind]

            #     if p == 0:
            #         new_gbest = copy.copy(self.P.nodes[particle]['D_p']['pbest_fitness'])
            #         coord_gbest = copy.copy(self.P.nodes[particle]['D_p']['pbest'])
            #     else:
            #         if new_gbest < self.P.nodes[particle]['D_p']['pbest_fitness']:
            #             new_gbest = copy.copy(self.P.nodes[particle]['D_p']['pbest_fitness'])
            #             coord_gbest = copy.copy(self.P.nodes[particle]['D_p']['pbest'])
            # for p, particle in enumerate(subfleet):
            #     self.P.nodes[particle]['D_p']['gbest'] = coord_gbest

    def u_sf(self):
        for i, subfleet in enumerate(self.sub_fleets):
            sensors = self.s_sf[i]
            for s, sensor in enumerate(sensors):
                usf = copy.copy(self.dict_sensors_[sensor]['U_sf'])
                fitness_sf = copy.copy(self.dict_sensors_[sensor]['fitness'])
                for p, particle in enumerate(subfleet):
                    u_p = copy.copy(self.P.nodes[particle]['U_p'])
                    usf.append(u_p[-1])
                    s_p = self.P.nodes[particle]['S_p'].keys()
                    on_board = False
                    for a, key in enumerate(s_p):
                        if key == sensor:
                            on_board = True
                            break
                        else:
                            on_board = False
                    if on_board:
                        fitness = copy.copy(self.P.nodes[particle]['fitness_list'][sensor])
                        fitness_sf.append(fitness[-1])
                    else:
                        fitness_sf.append(0)
                self.dict_sensors_[sensor]['U_sf'] = copy.copy(usf)
                self.dict_sensors_[sensor]['fitness'] = copy.copy(fitness_sf)

    def peaks_mu(self, sensor):
        peaks = []
        index_bench = copy.copy(self.dict_benchs_[sensor]['index_peaks'])
        bench = copy.copy(self.dict_sensors_[sensor]['mu']['data'])
        for j, ind in enumerate(index_bench):
            peaks.append(bench[round(ind)])
        self.dict_sensors_[sensor]['mu']['peaks'] = copy.copy(peaks)

    def calculate_error(self):
        if self.type_error == 'all_map':
            r2_simulation = []
            for i, subfleet in enumerate(self.sub_fleets):
                sensors = self.s_sf[i]
                for s, sensor in enumerate(sensors):
                    bench = copy.copy(self.dict_benchs_[sensor]['original'])
                    mu = copy.copy(self.dict_sensors_[sensor]['mu']['data'])
                    # r2 = r2_score(y_true=bench, y_pred=mu)
                    r2 = mean_squared_error(y_true=bench, y_pred=mu)
                    cant_sensor = self.dict_sensors_[sensor]['cant']
                    w = self.dict_sensors_[sensor]['w']
                    r2_simulation.append(r2)
                    self.r2_sensor.append(r2)
                    self.sensor.append(sensor)
                    self.cant_sensor.append(cant_sensor)
                    self.w.append(w)
            r2_simulation = np.array(r2_simulation)
            r2_mean = np.mean(r2_simulation)
            r2_std = np.std(r2_simulation)
            self.mean_error.append(r2_mean)
            self.conf_error.append(r2_std * 1.96)
        elif self.type_error == 'peaks':
            error_simulation = []
            conf_simulation = []
            for i, subfleet in enumerate(self.sub_fleets):
                sensors = self.s_sf[i]
                for s, sensor in enumerate(sensors):
                    error_peaks = []
                    self.peaks_mu(sensor)
                    bench = copy.copy(self.dict_benchs_[sensor]['peaks'])
                    mu = copy.copy(self.dict_sensors_[sensor]['mu']['peaks'])
                    for j, be in enumerate(bench):
                        error_peak = be - mu[j]
                        error_peaks.append(error_peak)
                    error_peaks = np.array(error_peaks)
                    error_mean_s = np.mean(error_peaks)
                    self.error_peak_sensor.append(error_mean_s)
                    error_std_s = np.std(error_peaks)
                    error_simulation.append(error_mean_s)
                    conf_simulation.append(error_std_s * 1.96)
            error_simulation = np.array(error_simulation)
            error_mean = np.mean(error_simulation)
            error_conf = np.std(error_simulation)
            self.mean_peak_error.append(error_mean)
            self.conf_peak_error.append(error_conf * 1.96)

    def first_values(self):
        for part in self.pop:
            if self.method_pso == 'coupled':
                self.local_best_coupled(part)
            elif self.method_pso == 'decoupled':
                self.local_best_decoupled(part, dfirst=True)

        self.u_sf()

        if self.method_pso == 'coupled':
            self.global_best_coupled()
        elif self.method_pso == 'decoupled':
            self.global_best_decoupled()

        for part in self.pop:
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

        if np.mean(self.distances) <= self.exploration_distance:
            action = np.array([2.0187, 0, 3.2697, 0])
        else:
            action = np.array([3.6845, 1.5614, 0, 3.1262])

        while dis_steps < 10:

            previous_dist = np.max(self.distances)

            for part in self.pop:
                self.toolbox.update(part.node, action[0], action[1], action[2], action[3], part)

            for part in self.pop:
                if self.method_pso == 'coupled':
                    self.local_best_coupled(part)
                elif self.method_pso == 'decoupled':
                    self.local_best_decoupled(part, dfirst=False)

                self.n_data += 1
                if self.n_data > self.vehicles - 1:
                    self.n_data = 0

            self.u_sf()

            if self.method_pso == 'coupled':
                self.global_best_coupled()
            elif self.method_pso == 'decoupled':
                self.global_best_decoupled()

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

        if (np.max(self.distances) >= self.exploitation_distance) or np.max(self.distances) == self.dist_pre:
            done = True
            self.type_error = 'all_map'
            self.calculate_error()
            self.type_error = 'peaks'
            self.calculate_error()
            df1 = {'Sensor': self.sensor, 'R2_sensor': self.r2_sensor, 'Error_peak_sensor': self.error_peak_sensor, 'Number': self.cant_sensor, 'w': self.w}
            df1 = pd.DataFrame(data=df1)
            # df2 = {'Sensors': self.sensor_vehicle}
            # df2 = pd.DataFrame(data=df2)
            # new = pd.concat([df1, df2], axis=1)
            df1.to_excel('Sensors_data_' + str(self.seed) + '.xlsx')

            # for i, subfleet in enumerate(self.sub_fleets):
            #     sensors = self.s_sf[i]
            #     for s, sensor in enumerate(sensors):
            #         bench = copy.copy(self.dict_benchs_[sensor]['map_created'])
            #         self.plot.benchmark(bench, sensor)
            #         mu = copy.copy(self.dict_sensors_[sensor]['mu']['data'])
            #         sigma = copy.copy(self.dict_sensors_[sensor]['sigma']['data'])
            #         vehicles = copy.copy(self.dict_sensors_[sensor]['vehicles'])
            #         trajectory = list()
            #         first = True
            #         list_ind = list()
            #         for veh in vehicles:
            #             list_ind.append(self.P.nodes[veh]['index'])
            #             if first:
            #                 trajectory = np.array(self.P.nodes[veh]['U_p'])
            #                 first = False
            #             else:
            #                 new = np.array(self.P.nodes[veh]['U_p'])
            #                 trajectory = np.concatenate((trajectory, new), axis=1)
            #         self.plot.plot_classic(mu, sigma, trajectory, sensor, list_ind)
        else:
            done = False

        return done

    def data_out(self):
        data1 = {'R2': self.mean_error, 'Conf_R2': self.conf_error, 'Mean_Error': self.mean_peak_error, 'Conf_Error': self.conf_peak_error}
        df = pd.DataFrame(data=data1)
        df.to_excel('Main_results.xlsx')
        fig1, ax1 = plt.subplots()
        ax1.set_title('R2 All Map')
        ax1.boxplot(self.mean_error, notch=True)
        fig2, ax2 = plt.subplots()
        ax2.set_title('Error peaks')
        ax2.boxplot(self.mean_peak_error, notch=True)
        plt.show()

        print('R2:', np.mean(np.array(self.mean_error)), '+-', np.std(np.array(self.mean_error)) * 1.96)
        print('Error:', np.mean(np.array(self.mean_peak_error)), '+-', np.std(np.array(self.mean_peak_error)) * 1.96)
