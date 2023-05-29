import numpy as np
import random
import math
import gym
import copy
import warnings
from random import shuffle

import openpyxl
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.metrics import mean_squared_error, r2_score
from deap import base, creator, tools, algorithms
from scipy.spatial.distance import euclidean as eu_d
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt

from Data.limits import Limits
from Environment.map import Map
from Benchmark.benchmark_functions import Benchmark_function
from Environment.bounds import Bounds
from Data.utils import Utils, obtain_prefabricated_vehicles
from Environment.contamination_areas import DetectContaminationAreas
from Environment.plot_het import Plots


class LawnmowerEnvironment:

    def __init__(self, ys, resolution, vehicles, initial_seed, initial_position, exploration_distance, type_error):
        self.resolution = resolution
        self.exploration_distance = exploration_distance
        self.file = 'Test/Lawnmower'
        self.type_error = type_error
        self.initial_type_error = type_error
        self.xs = int(10000 / (15000 / ys))
        self.ys = ys
        self.n_data = 0
        self.grid_or = Map(self.xs, ys).black_white()

        self.grid_min, self.grid_max, self.grid_max_x, self.grid_max_y = 0, self.ys, self.xs, self.ys
        self.vehicles = None
        self.n_vehicles = vehicles
        self.seed = initial_seed
        self.initial_seed = initial_seed
        self.initial_position = initial_position
        self.vel = [10, 10]
        self.dict_direction = {}
        self.dict_turn = {}
        self.asv = 1
        self.check = True
        self.turn = False
        self.g = 0
        self.post_array = None
        self.distances = None
        self.part_ant = None
        self.array_part = None
        self.x_h = []
        self.y_h = []
        self.water_samples = []
        self.dict_n_pos = {}
        self.dict_c_pos = {}
        self.dict_error_comparison = {}
        self.new_initial_position = []
        self.x_bench = None
        self.y_bench = None
        self.mu = []
        self.sigma = []
        self.samples = 0
        self.dict_error = {}
        self.dict_error_peak = {}
        self.dict_error_az = {}
        self.type_error = self.initial_type_error
        self.error = []
        self.ERROR_data = []
        self.error_data = []
        self.dict_check_turn = {}
        self.it = []
        self.f = 0
        self.lam = 0.375
        self.last_sample = 0
        self.initial = True
        self.save = 0
        self.save_dist = [25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300, 325, 350, 375, 400, 425, 450, 475,
                          500, 525, 550, 575, 600, 625, 650, 675, 700]

        self.p_vehicles = ['p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7', 'p8']
        self.s_sensor = ['s1', 's2', 's3', 's4', 's5']
        self.subfleet_number = 1
        self.P = nx.MultiGraph()
        self.simulation = 0
        self.sub_fleets = None
        self.sensor_vehicle = None
        self.type_error = type_error
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
        self.positions = initial_position
        self.initial_position = initial_position
        self.mean_error = list()
        self.conf_error = list()
        self.mean_mse_error = list()
        self.conf_mse_error = list()
        self.mean_peak_error = list()
        self.conf_peak_error = list()
        self.sensor = list()
        self.w = list()
        self.cant_sensor = list()
        self.mse_sensor = list()
        self.sensor_mse = list()
        self.w_mse = list()
        self.cant_sensor_mse = list()
        self.seed_bench = initial_seed
        self.array_error = list()
        self.error_subfleet_1 = list()
        self.error_subfleet_2 = list()
        self.error_subfleet_3 = list()
        self.array_r2 = list()
        self.r2_subfleet_1 = list()
        self.r2_subfleet_2 = list()
        self.r2_subfleet_3 = list()
        self.mean_az_mse = []
        self.conf_az_mse = []
        self.first_1 = True
        self.first_2 = True
        self.first_3 = True

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

        self.limits = None

        self.plot = Plots(self.xs, self.ys, self.X_test, self.secure, self.grid_min, self.grid_or,
                          'no_exploitation')

        self.util = None

    def fleet_configuration(self):
        self.P = nx.MultiGraph()
        random.seed(self.seed)
        # self.vehicles = random.randint(4, 8)
        # print(self.n_vehicles, self.subfleet_number)
        self.p_vehicles, self.sensor_vehicle = obtain_prefabricated_vehicles(self.n_vehicles,
                                                                                   self.subfleet_number)
        self.vehicles = len(self.p_vehicles)
        self.population = copy.copy(self.vehicles)
        # print(self.p_vehicles)
        # print(self.sensor_vehicle)
        for i, particle in enumerate(self.p_vehicles):
            self.P.add_node(particle, S_p=dict.fromkeys(self.sensor_vehicle[i], []),
                            Q_p=list(), U_p=list(), index=self.p)
        # sensors = []
        # while i < self.vehicles:
        #     list_s = []
        #     sensor = random.randint(1, 5)
        #     index_s = random.sample(range(5), sensor)
        #     for j in range(len(index_s)):
        #         list_s.append(self.s_sensor[index_s[j]])
        #     list_s = sorted(list_s)
        #     sensors.append(list_s)
        #     i += 1
        # while i < self.vehicles:
        #     number = random.randint(0, 20)
        #     # list_s = sorted(self.sensor_v[number])
        #     list_s = ['s1']
        #     sensors.append(list_s)
        #     i += 1
        # self.sensor_vehicle = sensors
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
        self.detect = DetectContaminationAreas(self.X_test)

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
        # print(self.sub_fleets)
        # print('sf', self.s_sf)

    def reset(self):
        # self.seed += 1
        # self.bench_function, self.bench_array, self.num_of_peaks, self.index_a = Benchmark_function(self.grid_or,
        #                                                                                             self.resolution,
        #                                                                                             self.xs, self.ys,
        #                                                                                             self.X_test,
        #                                                                                             self.seed,
        #                                                                                             self.vehicles).create_new_map()
        # random.seed(self.seed)
        # self.detect_areas = DetectContaminationAreas(self.X_test, self.bench_array, vehicles=self.vehicles,
        #                                              area=self.xs)
        # self.centers_bench, self.dict_index_bench, self.dict_bench, self.dict_coord_bench, self.center_peaks_bench, \
        # self.max_bench_list, self.dict_limits_bench, self.action_zone_bench, self.dict_impo_bench, \
        # self.index_center_bench = self.detect_areas.benmchark_areas()
        # self.reset_variables()
        # self.first_values()

        self.seed += 1
        self.simulation += 1
        # if self.simulation <= 10:
        self.subfleet_number = 1
        if self.first_1:
            self.array_error = list()
            self.array_r2 = list()
            self.first_1 = False
        # elif 10 < self.simulation <= 20:
        #     self.subfleet_number = 2
        #     if self.first_2:
        #         self.error_subfleet_1 = copy.copy(self.array_error)
        #         self.array_error = list()
        #         self.r2_subfleet_1 = copy.copy(self.array_r2)
        #         self.array_r2 = list()
        #         self.first_2 = False
        # else:
        #     self.subfleet_number = 3
        #     if self.first_3:
        #         self.error_subfleet_2 = copy.copy(self.array_error)
        #         self.array_error = list()
        #         self.r2_subfleet_2 = copy.copy(self.array_r2)
        #         self.array_r2 = list()
        #         self.first_3 = False
        self.fleet_configuration()
        self.util = Utils(self.vehicles)
        self.reset_variables()
        self.limits = Limits(self.secure, self.xs, self.ys, self.vehicles)
        random.seed(self.seed)
        self.set_sensor()
        self.init_positions_tests()
        self.peaks_bench()
        self.first_values()

    def init_positions(self):

        edge_labels = [(u, v, len(d['S_pq'])) for u, v, d in self.P.edges(data=True)]

        def init_shuffle(icls, i_size):
            ind = list(range(i_size))
            shuffle(ind)
            return icls(ind)

        def evaluate_disp(ind):
            dist = 0
            for vehi, vehj, w in edge_labels:
                id_pos_i = ind[index_.index(vehi)]
                id_pos_j = ind[index_.index(vehj)]
                dist += w * eu_d(self.positions[id_pos_i], self.positions[id_pos_j])
            return dist,

        def similar(ind1, ind2):
            for i in range(len(ind1)):
                if ind1[i] - ind2[i] != 0:
                    return False
            return True

        index_ = list(self.P.nodes)
        IND_SIZE = len(index_)
        POP_SIZE = IND_SIZE * 10  # 10
        CXPB, MUTPB, NGEN = 0.5, 0.5, IND_SIZE * 10
        indmutpb = 0.05
        creator.create('FitnessMax', base.Fitness, weights=(1.0,))
        creator.create('Individual', list, fitness=creator.FitnessMax)

        toolbox = base.Toolbox()
        toolbox.register("individual", init_shuffle, creator.Individual, i_size=len(index_))
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        toolbox.register("evaluate", evaluate_disp)
        toolbox.register("mate", tools.cxOrdered)
        toolbox.register("mutate", tools.mutShuffleIndexes, indpb=indmutpb)
        toolbox.register("select", tools.selTournament, tournsize=3)

        pop = toolbox.population(POP_SIZE)
        hof = tools.ParetoFront(similar)

        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)

        warnings.filterwarnings("ignore")
        pop, logbook = algorithms.eaMuPlusLambda(pop, toolbox, mu=len(pop), lambda_=len(pop), cxpb=CXPB,
                                                 mutpb=MUTPB,
                                                 ngen=NGEN, stats=stats, verbose=True, halloffame=hof)

        dist = 0
        new_pos = list()
        for index, posicion in enumerate(hof[0]):
            print(f'el vehículo {index_[index]}, debe estar en la posición {self.positions[posicion]}')
            new_pos.append(self.positions[posicion])

        self.initial_position = np.array(new_pos)

    def init_positions_tests(self):
        init_pos = copy.copy(self.initial_position)
        new_pos = []
        # print(init_pos)
        for i in range(self.vehicles):
            index = random.randint(0, len(init_pos) - 1)
            new_pos.append(init_pos[index])
            init_pos = np.delete(init_pos, index, axis=0)
        self.new_initial_position = new_pos

    def reset_variables(self):
        self.dict_direction = {}
        self.dict_turn = {}
        self.dict_check_turn = {}
        # self.initial_position = list()
        self.asv = 1
        self.check = True
        self.turn = False
        self.g = 0
        self.post_array = np.ones(self.vehicles)
        self.distances = np.zeros(self.vehicles)
        self.part_ant = np.zeros((1, self.vehicles * 2))
        self.array_part = np.zeros((1, self.vehicles * 2))
        self.x_h = []
        self.y_h = []
        self.water_samples = []
        self.dict_n_pos = {}
        self.dict_c_pos = {}
        self.dict_error_comparison = {}
        self.x_bench = None
        self.y_bench = None
        self.mu = []
        self.sigma = []
        self.samples = 0
        self.dict_error = {}
        self.dict_error_peak = {}
        self.dict_error_az = {}
        self.type_error = self.initial_type_error
        self.error = []
        self.ERROR_data = []
        self.error_data = []
        self.it = []
        self.f = 0
        self.lam = 0.375
        self.last_sample = 0
        self.initial = True
        self.save = 0
        self.save_dist = [25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300, 325, 350, 375, 400, 425, 450, 475,
                          500, 525, 550, 575, 600, 625, 650, 675, 700]
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
        self.mse_sensor = []
        self.error_peak_sensor = []
        self.sensor = list()
        self.cant_sensor = list()
        self.w = list()
        self.sensor_mse = list()
        self.cant_sensor_mse = list()
        self.w_mse = list()
        self.util = Utils(self.vehicles)

    def moving_direction(self):
        for i in range(self.vehicles):
            # if i % 2 == 0:
            # print(type(self.initial_position))
            if (self.bench_limits_no[1] - self.initial_position[i, 0]) <= (
                    self.initial_position[i, 0] - self.bench_limits_no[0]):
                self.dict_direction["vehicle%s" % i] = [-1, 0]
            else:
                self.dict_direction["vehicle%s" % i] = [1, 0]

    def moving_turn(self, i, dfirst):
        if dfirst:
            self.dict_check_turn["vehicle%s" % i] = False
            # if i % 2 == 0:
            x_turn = self.dict_direction["vehicle%s" % i][0]
            if (self.bench_limits_no[3] - self.initial_position[i, 1]) <= (
                    self.initial_position[i, 1] - self.bench_limits_no[2]):
                self.dict_turn["vehicle%s" % i] = [0, -1]
            else:
                self.dict_turn["vehicle%s" % i] = [0, 1]

    def move_vehicle(self, c_pos, vehicle):
        direction = copy.copy(self.dict_direction["vehicle%s" % vehicle])
        n_pos = copy.copy(list(map(lambda x, y, z: x + y * z, c_pos, direction, self.vel)))
        self.check = self.limits.check_lm_limits(n_pos, vehicle)
        if not self.check:
            n_pos = copy.copy(list(map(lambda x, y, z: x * y + z, self.dict_turn["vehicle%s" % vehicle], self.vel,
                             c_pos)))
            self.dict_direction["vehicle%s" % vehicle] = copy.copy(list(
                map(lambda x, y: x * y, self.dict_direction["vehicle%s" % vehicle], [-1, -1])))
            self.moving_turn(vehicle, dfirst=True)
            self.check2 = self.limits.check_lm_limits(n_pos, vehicle)
            if not self.check2:
                n_pos = copy.copy(list(map(lambda w, x, y, z: (w + x) * y + z, self.dict_turn["vehicle%s" % vehicle], self.dict_direction["vehicle%s" % vehicle], self.vel,
                                           c_pos)))
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
        return n_pos

    # def take_sample(self, n_pos):
    #     self.x_bench = n_pos[0]
    #     self.y_bench = n_pos[1]
    #     sample_value = [self.bench_function[self.x_bench][self.y_bench]]
    #     return sample_value

    # def check_duplicate(self, n_pos, sample_value):
    #     self.duplicate = False
    #     for i in range(len(self.x_h)):
    #         if self.x_h[i] == self.x_bench and self.y_h[i] == self.y_bench:
    #             self.duplicate = True
    #             self.water_samples[i] = sample_value
    #             break
    #         else:
    #             self.duplicate = False
    #     if self.duplicate:
    #         pass
    #     else:
    #         self.x_h.append(int(n_pos[0]))
    #         self.y_h.append(int(n_pos[1]))
    #         self.water_samples.append(sample_value)

    # def gp_regression(self):
    #
    #     """
    #     Fits the gaussian process.
    #     """
    #
    #     x_a = np.array(self.x_h).reshape(-1, 1)
    #     y_a = np.array(self.y_h).reshape(-1, 1)
    #     x_train = np.concatenate([x_a, y_a], axis=1).reshape(-1, 2)
    #     y_train = np.array(self.water_samples).reshape(-1, 1)
    #
    #     self.gpr.fit(x_train, y_train)
    #     self.gpr.get_params()
    #
    #     self.mu, self.sigma = self.gpr.predict(self.X_test, return_std=True)
    #     post_ls = np.min(np.exp(self.gpr.kernel_.theta[0]))
    #     r = self.n_data
    #     self.post_array[r] = post_ls
    #
    #     return self.post_array

    def take_measures(self):
        for i, subfleet in enumerate(self.sub_fleets):
            for p, particle in enumerate(subfleet):
                sensors = copy.copy(self.P.nodes[particle]['S_p']).keys()
                q_p = copy.copy(self.P.nodes[particle]['Q_p'])
                u_p = copy.copy(self.P.nodes[particle]['U_p'])
                last = u_p[-1]
                x = last[0]
                y = last[1]
                q_p.append(last)
                # print('q_p', particle, q_p)
                self.P.nodes[particle]['Q_p'] = copy.copy(q_p)
                for j, key in enumerate(sensors):
                    measure = copy.copy(self.P.nodes[particle]['S_p'][key])
                    bench = copy.copy(self.dict_benchs_[key]['map_created'])
                    new_measure = bench[x][y]
                    measure.append(new_measure)
                    self.P.nodes[particle]['S_p'][key] = copy.copy(measure)
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

    #
    # def calculate_error(self):
    #     if self.type_error == 'all_map':
    #         self.error = mean_squared_error(y_true=self.bench_array, y_pred=self.mu)
    #     elif self.type_error == 'peaks':
    #         for i in range(len(self.index_center_bench)):
    #             max_az = self.mu[self.index_center_bench[i]]
    #             self.dict_error_peak["action_zone%s" % i] = abs(self.max_bench_list[i] - max_az)
    #     elif self.type_error == 'action_zone':
    #         estimated_all = list()
    #         for i in range(len(self.center_peaks_bench)):
    #             bench_action = copy.copy(self.dict_bench["action_zone%s" % i])
    #             estimated_action = list()
    #             index_action = copy.copy(self.dict_index_bench["action_zone%s" % i])
    #             for j in range(len(index_action)):
    #                 value = self.mu[index_action[j]]
    #                 estimated_action.append(value[0])
    #                 estimated_all.append(self.mu[index_action[j]])
    #             error_action = mean_squared_error(y_true=bench_action, y_pred=estimated_action)
    #             self.dict_error["action_zone%s" % i] = copy.copy(error_action)
    #         self.error = mean_squared_error(y_true=self.action_zone_bench, y_pred=estimated_all)
    #     return self.error

    # def save_data(self):
    #     if self.save < (self.exploration_distance / 25):
    #         mult = self.save_dist[self.save]
    #         mult_min = mult - 5
    #         mult_max = mult + 5
    #         if mult_min <= np.max(self.distances) < mult_max:
    #             if self.seed == self.initial_seed + 1:
    #                 if self.initial:
    #                     for i in range(len(self.save_dist)):
    #                         self.dict_error_comparison["Distance%s" % i] = list()
    #                     self.initial = False
    #             error_list = copy.copy(self.dict_error_comparison["Distance%s" % self.save])
    #             self.ERROR_data = self.calculate_error()
    #             error_list.append(self.ERROR_data)
    #             self.dict_error_comparison["Distance%s" % self.save] = copy.copy(error_list)
    #             self.save += 1
    #
    # def save_excel(self):
    #     for i in range(int(self.exploration_distance / 25)):
    #         wb = openpyxl.Workbook()
    #         hoja = wb.active
    #         hoja.append(self.dict_error_comparison["Distance%s" % i])
    #         wb.save('../Test/' + self.file + '/ALLCONError_' + str(self.save_dist[i]) + '.xlsx')

    def peaks_mu(self, sensor):
        peaks = []
        index_bench = copy.copy(self.dict_benchs_[sensor]['index_peaks'])
        bench = copy.copy(self.dict_sensors_[sensor]['mu']['data'])
        for j, ind in enumerate(index_bench):
            peaks.append(bench[round(ind)])
        self.dict_sensors_[sensor]['mu']['peaks'] = copy.copy(peaks)

    def calculate_error(self):
        if self.type_error == 'all_map_mse':
            mse_simulation = []
            for i, subfleet in enumerate(self.sub_fleets):
                sensors = self.s_sf[i]
                for s, sensor in enumerate(sensors):
                    bench = copy.copy(self.dict_benchs_[sensor]['original'])
                    mu = copy.copy(self.dict_sensors_[sensor]['mu']['data'])
                    mse = mean_squared_error(y_true=bench, y_pred=mu)
                    cant_sensor = self.dict_sensors_[sensor]['cant']
                    w = self.dict_sensors_[sensor]['w']
                    mse_simulation.append(mse)
                    self.mse_sensor.append(mse)
                    self.sensor_mse.append(sensor)
                    self.cant_sensor_mse.append(cant_sensor)
                    self.w_mse.append(w)
            mse_simulation = np.array(mse_simulation)
            mse_mean = np.mean(mse_simulation)
            mse_std = np.std(mse_simulation)
            self.mean_mse_error.append(mse_mean)
            self.array_error.append(mse_mean)
            self.conf_mse_error.append(mse_std * 1.96)
        elif self.type_error == 'all_map_r2':
            r2_simulation = []
            for i, subfleet in enumerate(self.sub_fleets):
                sensors = self.s_sf[i]
                for s, sensor in enumerate(sensors):
                    bench = copy.copy(self.dict_benchs_[sensor]['original'])
                    mu = copy.copy(self.dict_sensors_[sensor]['mu']['data'])
                    r2 = r2_score(y_true=bench, y_pred=mu)
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
            self.array_r2.append(r2_mean)
            self.conf_error.append(r2_std * 1.96)
        elif self.type_error == 'peaks':
            error_simulation = []
            conf_simulation = []
            for i, subfleet in enumerate(self.sub_fleets):
                sensors = self.s_sf[i]
                for s, sensor in enumerate(sensors):
                    self.dict_benchs_[sensor] = self.detect.benchmark_areas(self.dict_benchs_[sensor], self.vehicles,
                                                                            10)
                    error_peaks = []
                    index_peaks = self.dict_benchs_[sensor]['action_zones']['peaks_index']
                    mu_ = copy.copy(self.dict_sensors_[sensor]['mu']['data'])
                    bench_ = copy.copy(self.dict_benchs_[sensor]['peaks'])
                    for j, be in enumerate(bench_):
                        error_peak = abs(be - mu_[index_peaks[j]])
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
        elif self.type_error == 'zones':
            zone_error = []
            for i, subfleet in enumerate(self.sub_fleets):
                sensors = self.s_sf[i]
                for s, sensor in enumerate(sensors):
                    number = self.dict_benchs_[sensor]['action_zones']['number']
                    estimated_all = list()
                    real = list()
                    mu_ = self.dict_sensors_[sensor]['mu']['data']
                    bench_ = self.dict_benchs_[sensor]['original']
                    for j in range(number):
                        bench_ind = self.dict_benchs_[sensor]['action_zones']["action_zone%s" % j]['index']
                        for k, index in enumerate(bench_ind):
                            real.append(bench_[index])
                            estimated_all.append(mu_[index])
                    mse = mean_squared_error(y_true=real, y_pred=estimated_all)
                    zone_error.append(mse)
            self.mean_az_mse.append(np.mean(zone_error))
            self.conf_az_mse.append(np.std(zone_error) * 1.96)

    def first_values(self):
        self.moving_direction()
        for i in range(len(self.p_vehicles)):
            self.moving_turn(i, dfirst=True)
            self.dict_c_pos["vehicle%s" % i] = list(self.initial_position[i])
            n_pos = list(self.initial_position[i])
            u_p = copy.copy(self.P.nodes[self.p_vehicles[i]]['U_p'])
            u_p.append(n_pos)
            self.P.nodes[self.p_vehicles[i]]['U_p'] = copy.copy(u_p)
            self.dict_n_pos["vehicle%s" % i] = n_pos
            self.part_ant, self.distances = self.util.distance_part(self.g, self.n_data, n_pos, self.part_ant,
                                                                    self.distances, self.array_part, dfirst=True)

            self.samples += 1

            self.n_data += 1
            if self.n_data > self.vehicles - 1:
                self.n_data = 0

        self.take_measures()
        self.gp_update()

        # self.calculate_error()
        # self.error_data.append(self.error)
        # self.it.append(self.g)

        self.k = self.vehicles
        self.ok = False

    def simulator(self):
        dis_steps = 0
        dist_ant = np.mean(self.distances)
        self.dist_pre = np.max(self.distances)
        self.n_data = 0
        self.f += 1
        while dis_steps < 10:

            previous_dist = np.max(self.distances)

            for i in range(self.vehicles):
                n_pos = copy.copy(self.move_vehicle(self.dict_c_pos["vehicle%s" % i], i))
                u_p = copy.copy(self.P.nodes[self.p_vehicles[i]]['U_p'])
                u_p.append(n_pos)
                self.P.nodes[self.p_vehicles[i]]['U_p'] = copy.copy(u_p)
                self.dict_n_pos["vehicle%s" % i] = copy.copy(n_pos)
                self.dict_c_pos["vehicle%s" % i] = copy.copy(n_pos)

            for i in range(self.vehicles):
                n_pos = self.dict_n_pos["vehicle%s" % i]
                self.part_ant, self.distances = self.util.distance_part(self.g, self.n_data, n_pos, self.part_ant,
                                                                        self.distances, self.array_part, dfirst=False)

                self.n_data += 1
                if self.n_data > self.vehicles - 1:
                    self.n_data = 0

            if (np.mean(self.distances) - self.last_sample) >= (np.min(self.post_array) * self.lam):
                self.k += 1
                self.ok = True
                self.last_sample = np.mean(self.distances)

                self.take_measures()
                self.gp_update()
                # self.it.append(self.g)
                # self.error = self.calculate_error()
                # self.error_data.append(self.error)

                # self.save_data()

                self.ok = False

            dis_steps = np.mean(self.distances) - dist_ant
            if np.max(self.distances) == previous_dist:
                break

            self.g += 1

        if (self.distances >= self.exploration_distance).any():
            done = True
            self.type_error = 'all_map_r2'
            self.calculate_error()
            self.type_error = 'all_map_mse'
            self.calculate_error()
            self.type_error = 'peaks'
            self.calculate_error()
            self.type_error = 'zones'
            self.calculate_error()
            df1 = {'Sensor': self.sensor, 'R2_sensor': self.r2_sensor, 'MSE_sensor': self.mse_sensor,
                   'Error_peak_sensor': self.error_peak_sensor,
                   'Number': self.cant_sensor, 'w': self.w}
            df1 = pd.DataFrame(data=df1)
            df2 = {'Sensors': self.sensor_vehicle}
            df2 = pd.DataFrame(data=df2)
            new = pd.concat([df1, df2], axis=1)
            df1.to_excel('../Test/Lawnmower/T2/Sensors_data_' + str(self.seed) + '.xlsx')
            if self.simulation == 30:
                self.error_subfleet_3 = copy.copy(self.array_error)
                self.r2_subfleet_3 = copy.copy(self.array_r2)

            # if self.simulation >= 10:
            # for i, subfleet in enumerate(self.sub_fleets):
            #     sensors = self.s_sf[i]
            #     for s, sensor in enumerate(sensors):
            #         bench = copy.copy(self.dict_benchs_[sensor]['map_created'])
            #         # self.plot.benchmark(bench, sensor)
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

    # def error_value(self):
    #     return self.error_data

    def data_out(self):
        print(len(self.error_subfleet_1), len(self.error_subfleet_2), len(self.error_subfleet_3))
        print('MSE 1 Subfleet:', np.mean(np.array(self.error_subfleet_1)), '+-',
              np.std(np.array(self.error_subfleet_1)) * 1.96)
        print('R2 1 Subfleet:', np.mean(np.array(self.r2_subfleet_1)), '+-',
              np.std(np.array(self.r2_subfleet_1)) * 1.96)
        print('MSE 2 Subfleets:', np.mean(np.array(self.error_subfleet_2)), '+-',
              np.std(np.array(self.error_subfleet_2)) * 1.96)
        print('R2 2 Subfleets:', np.mean(np.array(self.r2_subfleet_2)), '+-',
              np.std(np.array(self.r2_subfleet_2)) * 1.96)
        print('MSE 3 Subfleets:', np.mean(np.array(self.error_subfleet_3)), '+-',
              np.std(np.array(self.error_subfleet_3)) * 1.96)
        print('R2 3 Subfleets:', np.mean(np.array(self.r2_subfleet_3)), '+-',
              np.std(np.array(self.r2_subfleet_3)) * 1.96)
        # data1 = {'R2': self.mean_error, 'Conf_R2': self.conf_error, 'Mean_Error': self.mean_peak_error, 'Conf_Error': self.conf_peak_error}
        # df = pd.DataFrame(data=data1)
        # df.to_excel('../Test/Lawnmower/T2/Main_results.xlsx')
        fig1, ax1 = plt.subplots()
        ax1.set_title('MSE MAP')
        ax1.boxplot(self.mean_mse_error, notch=True)
        fig2, ax2 = plt.subplots()
        ax2.set_title('Error peaks')
        ax2.boxplot(self.mean_peak_error, notch=True)
        fig3, ax3 = plt.subplots()
        ax3.set_title('MSE AZ')
        ax3.boxplot(self.mean_az_mse, notch=True)
        # ax3.set_xticklabels(['1 Subfleet', '2 Subfleets', '3 Subfleets'], rotation=45, fontsize=8)
        fig4, ax4 = plt.subplots()
        ax4.set_title('R2 MAP')
        ax4.boxplot(self.mean_error, notch=True)
        # ax4.set_xticklabels(['1 Subfleet', '2 Subfleets', '3 Subfleets'], rotation=45, fontsize=8)
        # # plt.show()
        df1 = pd.DataFrame(self.mean_peak_error)
        df1.to_excel('../Test/Results/Error/ErrorLawnmower.xlsx')
        df2 = pd.DataFrame(self.mean_az_mse)
        df2.to_excel('../Test/Results/MSEAZ/MSEAZLawnmower.xlsx')
        df3 = pd.DataFrame(self.mean_mse_error)
        df3.to_excel('../Test/Results/MSEM/MSEMLawnmower.xlsx')
        df4 = pd.DataFrame(self.mean_error)
        df4.to_excel('../Test/Results/R2M/R2MLawnmower.xlsx')

        print('R2:', np.mean(np.array(self.mean_error)), '+-', np.std(np.array(self.mean_error)) * 1.96)
        print('MSE:', np.mean(np.array(self.mean_mse_error)), '+-', np.std(np.array(self.mean_mse_error) * 1.96))
        print('Error:', np.mean(np.array(self.mean_peak_error)), '+-', np.std(np.array(self.mean_peak_error)) * 1.96)
        print('AZ:', np.mean(np.array(self.mean_az_mse)), '+-', np.std(np.array(self.mean_az_mse)) * 1.96)

