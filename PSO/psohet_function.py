import copy
import math
import os
import sys
import operator
from operator import add
import warnings
from random import shuffle

import gym
import networkx as nx
import pandas as pd
import openpyxl
from statistics import mean
from deap import base, creator, tools, algorithms
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.metrics import mean_squared_error, r2_score
from scipy.spatial.distance import euclidean as eu_d

from Benchmark.benchmark_functions import *
from Data.limits import Limits
from Data.utils import Utils, obtain_prefabricated_vehicles
from Environment.bounds import Bounds
from Environment.map import *
from Environment.plot_het import Plots
from Environment.contamination_areas import *

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

    def __init__(self, resolution, ys, method, method_pso, initial_seed, initial_position, sensor_vehicle, weights,
                 weights_b, vehicles=4,
                 exploration_distance=100,
                 exploitation_distance=200, action=False, reward_function='mse', behavioral_method=0,
                 type_error='all_map',
                 stage='exploration', final_model='samples'):
        self.p_vehicles = []
        self.s_sensor = ['s1', 's2', 's3', 's4', 's5']
        self.weights = weights
        self.action_explore = np.array([2.0187, 0, 3.2697, 0])
        self.action_exploit = np.array([3.6845, 1.5614, 0, 3.1262])
        self.weights_b = weights_b
        self.sensor_v = sensor_vehicle
        self.n_vehicles = vehicles
        self.mean_un = []
        self.case = 0
        self.entropy = {}
        self.sim_entropy = []
        self.r2_data = np.array([0, 0])
        self.mse_data = np.array([0, 0])
        self.caz_mse_data = np.array([0, 0])
        self.peak_error_data = np.array([1, 0])
        self.r2_ = None
        self.mse_ = None
        self.caz_mse_ = None
        self.peak_error_ = None
        self.max_un = []
        self.new_initial_position = list()
        self.subfleet_number = 1
        self.simulation = 0
        self.cant = 0
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
        ker = RBF(length_scale=10, length_scale_bounds=(1e-1, 10))
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
        self.mean_mse_error = list()
        self.conf_mse_error = list()
        self.mean_peak_error = list()
        self.mean_az_mse = list()
        self.conf_az_mse = list()
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
        self.first_1 = True
        self.first_2 = True
        self.first_3 = True

        self.dict_sensors_ = {}
        self.u_subf = {}
        self.fit_sf = {}
        self.z_ = {}
        self.zones = 0
        self.dict_subfleet_ = {}
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
        self.f_en = True
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
        self.explore = True
        self.summatory = []

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
        self.detect = DetectContaminationAreas(self.X_test)

        self.util = None

        createPart()

    def function1(self, individual, position, height, width):
        return height / (1 + width * sum(((individual[j] - position[j]) ** 2) for j in range(len(individual))))

    def generatePart(self):

        """
        Generates a random position and a random speed for the particles (drones).
        """
        list_part = list()
        # for i in range(self.size):
        # print(self.p)
        # print(self.new_initial_position)
        part = creator.Particle(self.new_initial_position[self.p])
        list_part.append(np.array(part))
        part.speed = np.array([random.uniform(self.smin, self.smax) for _ in range(self.size)])
        part.smin = self.smin
        part.smax = self.smax
        part.node = self.p_vehicles[self.p]
        self.P.add_nodes_from([part.node], Reach=False,
                              U_p=list_part, Up_exploit=[], Q_p=list(), Qp_exploit=list(),
                              D_p=dict.fromkeys(self.data_particle),
                              action_explore=[],
                              action_exploit=[],
                              pbest=dict.fromkeys(self.sensor_vehicle[self.p]),
                              fitness=dict.fromkeys(self.sensor_vehicle[self.p]),
                              fitness_list=dict.fromkeys(self.sensor_vehicle[self.p], []),
                              fitness_exploit=dict.fromkeys(self.sensor_vehicle[self.p], []))
        self.p += 1

        return part

    def fleet_configuration(self):
        random.seed(self.seed)
        # self.vehicles = random.randint(2, 8)
        self.p_vehicles, self.sensor_vehicle = obtain_prefabricated_vehicles(self.n_vehicles,
                                                                                   self.subfleet_number)
        # self.p_vehicles, self.sensor_vehicle = ['v1', 'v2', 'v3'], [['s1', 's2'], ['s1'], ['s2']]
        self.vehicles = len(self.p_vehicles)
        # self.vehicles = 4
        # self.p_vehicles = ['v1', 'v2', 'v3', 'v4']
        # self.sensor_vehicle = [['s1'], ['s1'], ['s1'], ['s1']]
        self.population = copy.copy(self.vehicles)
        self.P = nx.MultiGraph()
        for p_, (part, sen) in enumerate(zip(self.p_vehicles, self.sensor_vehicle)):
            self.P.add_node(part, S_p=dict.fromkeys(sen, []), Sp_exploit=dict.fromkeys(sen, []), index=p_, )
        i = 0
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
        self.dict_sensors_[sensor]['rate'] = []
        self.dict_sensors_[sensor]['entropy'] = []
        self.dict_sensors_[sensor]['cant'] = 0
        self.dict_sensors_[sensor]['w'] = {}
        self.dict_sensors_[sensor]['w_mc'] = {}
        self.dict_sensors_[sensor]['w_init'] = {}
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
        self.dict_benchs_[sensor] = self.detect.benchmark_areas(self.dict_benchs_[sensor], self.vehicles, 10)

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
            self.u_subf['subfleet%s' % i] = []
            self.fit_sf['subfleet%s' % i] = []
            S_sf = set()
            for particle in sub_fleet:
                S_sf = S_sf | self.P.nodes[particle]['S_p'].keys()
            S_sf = sorted(S_sf)
            arr = len(S_sf)
            self.dict_subfleet_[i] = {}
            self.dict_subfleet_[i]['x'] = np.zeros((arr, 2))
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
                    self.cant += 1
                # print(f'Particle {particle} contains {self.P.nodes[particle]["S_p"]} y se usa en eqs. 13a y 13b')
        self.wv_values_u()
        # print(self.sub_fleets)
        # print('sf', self.s_sf)

    def obtain_weights(self):
        if self.weights_b:
            for i, sub_fleet in enumerate(self.sub_fleets):
                for p, particle in enumerate(sub_fleet):
                    s_p = self.P.nodes[particle]['S_p']
                    for s, sensor in enumerate(s_p):
                        if s == 0:
                            cant = self.dict_sensors_[sensor]['cant']
                        else:
                            if cant > self.dict_sensors_[sensor]['cant']:
                                cant = self.dict_sensors_[sensor]['cant']
                    if cant > 4:
                        cant = 4
                    self.P.nodes[particle]['action_explore'] = self.weights[str(cant)]['Explore']
                    self.P.nodes[particle]['action_exploit'] = self.weights[str(cant)]['Exploit']
                    # print(self.P.nodes[particle]['action_exploit'])
        else:
            self.action_explore = np.array([2.0187, 0, 3.2697, 0])
            self.action_exploit = np.array([3.6845, 1.5614, 0, 3.1262])

        # print('explore', self.action_explore)
        # print('exploit', self.action_exploit)

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
                dist += w * eu_d(self.initial_position[id_pos_i], self.initial_position[id_pos_j])
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
        pop, logbook = algorithms.eaMuPlusLambda(pop, toolbox, mu=len(pop), lambda_=len(pop), cxpb=CXPB, mutpb=MUTPB,
                                                 ngen=NGEN, stats=stats, verbose=True, halloffame=hof)

        dist = 0
        new_pos = list()
        for index, posicion in enumerate(hof[0]):
            print(f'el vehículo {index_[index]}, debe estar en la posición {self.initial_position[posicion]}')
            new_pos.append(list(self.initial_position[posicion]))

        self.new_initial_position = new_pos

    def init_positions_tests(self):
        init_pos = copy.copy(self.initial_position)
        new_pos = []
        # print(init_pos)
        for i in range(self.vehicles):
            index = random.randint(0, len(init_pos) - 1)
            new_pos.append(init_pos[index])
            init_pos = np.delete(init_pos, index, axis=0)
        self.new_initial_position = new_pos

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
        self.detect = DetectContaminationAreas(self.X_test)
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
        self.reset_variables()
        random.seed(self.seed)
        self.set_sensor()
        self.init_positions()
        # self.init_positions_tests()
        self.tool()
        self.swarm()
        self.statistic()
        self.obtain_weights()
        # for part in self.pop:
        #     print(part)
        self.peaks_bench()
        self.first_values()

    def reset_variables(self):
        self.sub_fleets = None
        self.r2_data = np.array([[0, 0]])
        self.mse_data = np.array([[0, 0]])
        self.caz_mse_data = np.array([[0, 0]])
        self.peak_error_data = np.array([[1, 0]])
        self.f_en = True
        self.entropy[self.simulation] = {}
        self.entropy[self.simulation]['max'] = []
        self.entropy[self.simulation]['distance'] = []
        self.entropy[self.simulation]['rate'] = []
        self.sim_entropy = []
        self.stage = 'exploration'
        self.explore = True
        self.z_ = {}
        self.zones = 0
        self.new_initial_position = list()
        self.dict_sensors_ = {}
        self.u_subf = {}
        self.fit_sf = {}
        self.dict_subfleet_ = {}
        self.dict_benchs_ = {}
        self.mu_best = []
        self.cant = 0
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

    def updateParticle_n(self, p, c1, c2, c3, c4, part):

        """
        Calculates the speed and the position of the particles (drones).
        """

        d_p = copy.copy(self.P.nodes[part.node]['D_p'])
        pbest = d_p['pbest']
        gbest = d_p['gbest']
        max_con = d_p['con']
        max_un = d_p['un']
        # if self.simulation > 10:
        #     print(part.node, d_p)
        # print(d_p)
        # print(pbest, gbest, max_con, max_un)
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
        self.post_array = []
        entropy = []
        first = True
        self.max_un = []
        rate_max = copy.copy(self.entropy[self.simulation]['max'])
        rate_ = []
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
                self.max_un.append(mean(self.dict_sensors_[sensor]['sigma']['data']))
                # if first:
                #     entropy = map(lambda x: math.log(x)/2 + math.log(2*math.pi*math.e),
                #                                                      self.dict_sensors_[sensor]['sigma']['data'])
                #     first = False
                # else:
                #     sen_entropy = map(lambda x: math.log(x)/2 + math.log(2*math.pi*math.e),
                #                                                      self.dict_sensors_[sensor]['sigma']['data'])
                #     entropy = list(map(add, entropy, sen_entropy))

                self.post_array.append(round(np.min(np.exp(self.gpr.kernel_.theta[0])), 1))
                # self.dict_sensors_[sensor]['w'] = mean(self.dict_sensors_[sensor]['sigma']['data'])
                # rate_sensor = copy.copy(self.dict_sensors_[sensor]['rate'])
                # entropy_ = copy.copy(self.dict_sensors_[sensor]['entropy'])
                # entropy = map(lambda x: math.log(x) / 2 + math.log(2 * math.pi * math.e),
                #               self.dict_sensors_[sensor]['sigma']['data'])
                # entropy_.append(mean(entropy))
                # self.dict_sensors_[sensor]['entropy'] = copy.copy(entropy_)
                # if self.f_en:
                #     rate = entropy_[-1]
                # else:
                #     rate = entropy_[-1] - entropy_[-2]
                # rate_sensor.append(rate)
                # rate_.append(rate)
                # self.dict_sensors_[sensor]['rate'] = copy.copy(rate_sensor)

        # entro_rate = copy.copy(self.entropy[self.simulation]['rate'])
        # entro_dist = copy.copy(self.entropy[self.simulation]['distance'])
        # entro_rate.append(max(rate_))
        # entro_dist.append(mean(list(self.distances)))
        # self.entropy[self.simulation]['rate'] = copy.copy(entro_rate)
        # self.entropy[self.simulation]['distance'] = copy.copy(entro_dist)
        # if self.f_en:
        #     self.f_en = False
        # entro_rate = copy.copy(self.entropy[self.simulation]['rate'])
        # rate = entro_max[-1]
        # entro_rate.append(rate)
        # self.entropy[self.simulation]['rate'] = copy.copy(entro_rate)
        # else:
        #     entro_rate = copy.copy(self.entropy[self.simulation]['rate'])
        #     rate = entro_max[-1] - entro_max[-2]
        #     entro_rate.append(rate)
        #     self.entropy[self.simulation]['rate'] = copy.copy(entro_rate)

    def obtain_max(self, array_function, coord):
        max_value = np.max(array_function)
        index_1 = np.where(array_function == max_value)
        index_x1 = index_1[0]

        index_x2 = index_x1[0]
        index_x = int(coord[index_x2][0])
        index_y = int(coord[index_x2][1])

        index_xy = [index_x, index_y]
        coordinate_max = np.array(index_xy)

        return max_value, coordinate_max

    def model_max(self, mu, sigma, coord):
        max_mu, coord_mu = self.obtain_max(mu, coord)
        max_sigma, coord_sigma = self.obtain_max(sigma, coord)
        # print('max_mu', max_mu, 'max_sigma', max_sigma)
        return [max_mu, coord_mu], [max_sigma, coord_sigma]

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
                self.dict_sensors_[sensor]['w_init'] = x_value / self.dict_sensors_[sensor]['cant']

    def wv_values(self):
        for i, subfleet in enumerate(self.sub_fleets):
            for p, particle in enumerate(subfleet):
                sensors = self.P.nodes[particle]['S_p'].keys()
                inver = 0
                for j, sensor in enumerate(sensors):
                    self.dict_sensors_[sensor]['w'][particle] = 0
                    cant_sensor = self.dict_sensors_[sensor]['cant']
                    h = 1 / cant_sensor
                    inver = inver + h
                x_value = 1 / inver
                for j, sensor in enumerate(sensors):
                    self.dict_sensors_[sensor]['w'][particle] = x_value / self.dict_sensors_[sensor]['cant']
                    self.dict_sensors_[sensor]['w_init'][particle] = x_value / self.dict_sensors_[sensor]['cant']

    def wv_values_u(self):
        for i, subfleet in enumerate(self.sub_fleets):
            for p, particle in enumerate(subfleet):
                sensors = self.P.nodes[particle]['S_p'].keys()
                inver = 0
                only = False
                for j, sensor in enumerate(sensors):
                    # self.dict_sensors_[sensor]['w'][particle] = 0
                    cant_sensor = self.dict_sensors_[sensor]['cant']
                    if cant_sensor == 1:
                        only = True
                        only_sensor = sensor
                        break
                    else:
                        only = False
                    h = 1 / cant_sensor
                    inver = inver + h
                for j, sensor in enumerate(sensors):
                    if only:
                        if sensor == only_sensor:
                            self.dict_sensors_[sensor]['w'][particle] = 1
                            self.dict_sensors_[sensor]['w_init'][particle] = 1
                        else:
                            self.dict_sensors_[sensor]['w'][particle] = 0
                            self.dict_sensors_[sensor]['w_init'][particle] = 0
                    else:
                        x_value = 1 / inver
                        self.dict_sensors_[sensor]['w'][particle] = x_value / self.dict_sensors_[sensor]['cant']
                        self.dict_sensors_[sensor]['w_init'][particle] = x_value / self.dict_sensors_[sensor]['cant']

    def w_values_mc(self):
        for s in range(len(self.s_sf)):
            if len(self.s_sf[s]) > 1:
                sensors = self.s_sf[s]
                x = copy.copy(self.dict_subfleet_[s]['x'])
                for j, sensor in enumerate(sensors):
                    x[j, 0] = self.dict_sensors_[sensor]['w_init']
                    x[j, 1] = np.mean(self.dict_sensors_[sensor]['sigma']['data'])
                self.dict_subfleet_[s]['x'] = copy.copy(x)
                r = copy.copy(x)
                sum_ = np.sum(x, axis=0)
                for j in range(2):
                    for i in range(len(sensors)):
                        r[i, j] = x[i, j] / sum_[j]
                # print(len(sensors))
                k = 1 / math.log(len(sensors))
                d = np.zeros((2, 1))
                for j in range(2):
                    _sum = 0
                    for i in range(len(sensors)):
                        _sum = _sum + r[i, j] * math.log(r[i, j])
                    d[j] = 1 - (-k * _sum)
                w = np.zeros((2, 1))
                for i in range(len(w)):
                    w[i] = d[i] / np.sum(d)
                # print(w)
                for i, sensor in enumerate(sensors):
                    w_sum = 0
                    for j in range(2):
                        w_sum = w_sum + x[i, j] * w[j]
                    self.dict_sensors_[sensor]['w'] = w_sum
                    # print(sensor, w_sum, self.dict_sensors_[sensor]['w_init'])

    def method_coupled(self):
        for i, subfleet in enumerate(self.sub_fleets):
            sensors = self.s_sf[i]
            summatory_mu = list()
            summatory_sigma = list()
            for s, sensor in enumerate(sensors):
                mu = list(copy.copy(self.dict_sensors_[sensor]['mu']['data']))
                sigma = list(copy.copy(self.dict_sensors_[sensor]['sigma']['data']))
                w_value = self.dict_sensors_[sensor]['w'][particle]
                if s == 0:
                    summatory_mu = [data_m * w_value for data_m in mu]
                    summatory_sigma = [data_s * w_value for data_s in sigma]
                else:
                    data_mu = [data_m * w_value for data_m in mu]
                    data_sigma = [data_s * w_value for data_s in sigma]
                    summatory_mu = list(map(add, summatory_mu, data_mu))
                    summatory_sigma = list(map(add, summatory_sigma, data_sigma))
                # print('max', max_mu, coord_mu)
            ind_mu = summatory_mu.index(max(summatory_mu))
            ind_sigma = summatory_sigma.index(max(summatory_sigma))
            for p, particle in enumerate(subfleet):
                self.P.nodes[particle]['D_p']['con'] = self.X_test[ind_mu]
                self.P.nodes[particle]['D_p']['un'] = self.X_test[ind_sigma]

    def method_coupled_sp(self, dfirst=True):
        for i, subfleet in enumerate(self.sub_fleets):
            # sensors = self.s_sf[i]
            for p, particle in enumerate(subfleet):
                sensors = self.P.nodes[particle]['S_p'].keys()
                summatory_mu = list()
                summatory_sigma = list()
                for i, sensor in enumerate(sensors):
                    w_value = copy.copy(self.dict_sensors_[sensor]['w'][particle])
                    # print(sensor, w_value)
                    mu = list(copy.copy(self.dict_sensors_[sensor]['mu']['data']))
                    sigma = list(copy.copy(self.dict_sensors_[sensor]['sigma']['data']))
                    if i == 0:
                        summatory_mu = [data_m * w_value for data_m in mu]
                        summatory_sigma = [data_s * w_value for data_s in sigma]
                    else:
                        data_mu = [data_m * w_value for data_m in mu]
                        data_sigma = [data_s * w_value for data_s in sigma]
                        summatory_mu = list(map(lambda x, y: x + y, summatory_mu, data_mu))
                        summatory_sigma = list(map(lambda x, y: x + y, summatory_sigma, data_sigma))
                        self.summatory = copy.copy(summatory_sigma)
                ind_mu = np.argwhere(summatory_mu == np.amax(summatory_mu))
                # ind_mu = np.amax(summatory_mu)
                ind_mu = ind_mu.flatten().tolist()
                indmu = random.randint(0, len(ind_mu) - 1)
                ind_sigma = np.argwhere(summatory_sigma == np.amax(summatory_sigma))
                ind_sigma = ind_sigma.flatten().tolist()
                indsigma = random.randint(0, len(ind_sigma) - 1)
                # ind_mu = summatory_mu.index(np.max(np.array(summatory_mu)))
                # ind_sigma = summatory_sigma.index(np.max(np.array(summatory_sigma)))
                if dfirst:
                    prev_mu = ind_mu[indmu]
                    index_mu = prev_mu
                    prev_sigma = ind_sigma[indsigma]
                    index_sigma = prev_sigma
                    self.P.nodes[particle]['D_p']['con_index'] = prev_mu
                    self.P.nodes[particle]['D_p']['un_index'] = prev_sigma
                else:
                    prev_mu = self.P.nodes[particle]['D_p']['con_index']
                    prev_sigma = self.P.nodes[particle]['D_p']['un_index']
                    if prev_mu in ind_mu:
                        index_mu = prev_mu
                    else:
                        prev_mu = ind_mu[indmu]
                        index_mu = prev_mu
                        self.P.nodes[particle]['D_p']['con_index'] = prev_mu
                    if prev_sigma in ind_sigma:
                        index_sigma = prev_sigma
                    else:
                        prev_sigma = ind_sigma[indsigma]
                        index_sigma = prev_sigma
                        self.P.nodes[particle]['D_p']['un_index'] = prev_sigma
                self.P.nodes[particle]['D_p']['con'] = self.X_test[index_mu]
                self.P.nodes[particle]['D_p']['un'] = self.X_test[index_sigma]
                # if np.mean(self.distances) > self.exploration_distance - 10 and self.simulation > 10:
                #     self.plot.plot_summatory(particle, summatory_mu, summatory_sigma, np.array(self.P.nodes[particle]['U_p']), self.P.nodes[particle]['index'], mu_=False)
                #     self.plot.plot_summatory(particle, summatory_mu, summatory_sigma, np.array(self.P.nodes[particle]['U_p']), self.P.nodes[particle]['index'], mu_=True)

    def method_decoupled(self):
        for i, subfleet in enumerate(self.sub_fleets):
            sensors = self.s_sf[i]
            max_con_value = 0
            max_un_value = 0
            max_con = []
            max_un = []
            for s, sensor in enumerate(sensors):
                self.dict_sensors_[sensor]['mu']['max'], self.dict_sensors_[sensor]['sigma'][
                    'max'] = self.model_max(self.dict_sensors_[sensor]['mu']['data'],
                                            self.dict_sensors_[sensor]['sigma']['data'], self.X_test)
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

    def method_decoupled_sp(self):
        for i, subfleet in enumerate(self.sub_fleets):
            for p, particle in enumerate(subfleet):
                max_con_value = 0
                max_un_value = 0
                max_con = []
                max_un = []
                p_sensor = self.P.nodes[particle]['S_p'].keys()
                for s, sensor in enumerate(p_sensor):
                    self.dict_sensors_[sensor]['mu']['max'], self.dict_sensors_[sensor]['sigma'][
                        'max'] = self.model_max(self.dict_sensors_[sensor]['mu']['data'],
                                                self.dict_sensors_[sensor]['sigma']['data'], self.X_test)
                    max_mu, coord_mu = list(copy.copy(self.dict_sensors_[sensor]['mu']['max']))
                    max_sigma, coord_sigma = list(copy.copy(self.dict_sensors_[sensor]['sigma']['max']))
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
                self.P.nodes[particle]['D_p']['con'] = max_con
                self.P.nodes[particle]['D_p']['un'] = max_un

    def local_best_coupled(self, part, dfirst=False):
        # index = -1
        # while index == -1:
        part, self.s_n = Limits(self.secure, self.xs, self.ys, self.vehicles).new_limit(self.g, part, self.s_n,
                                                                                        self.n_data,
                                                                                        self.s_ant, self.part_ant)
        x_bench = int(part[0])
        y_bench = int(part[1])

        for c in range(len(self.X_test)):
            coord_ = self.X_test[c]
            if x_bench == coord_[0] and y_bench == coord_[1]:
                index = c
                break

        list_part = copy.copy(self.P.nodes[part.node]['U_p'])
        list_part.append(np.array(part))
        sensors = self.P.nodes[part.node]['S_p'].keys()

        for i, key in enumerate(sensors):
            list_f = copy.copy(self.P.nodes[part.node]['fitness_list'][key])
            bench = list(copy.copy(self.dict_sensors_[key]['mu']['data']))
            list_f.append(bench[index])
            self.P.nodes[part.node]['fitness_list'][key] = copy.copy(list_f)

        summatory = list()
        for i, key in enumerate(sensors):
            w = copy.copy(self.dict_sensors_[key]['w'][part.node])
            value = copy.copy(self.P.nodes[part.node]['fitness_list'][key])
            if i == 0:
                summatory = [data * w for data in value]
            else:
                list1 = [data * w for data in value]
                summatory = list(map(lambda x, y: x + y, summatory, list1))
        ind = summatory.index(np.max(np.array(summatory)))
        self.P.nodes[part.node]['D_p']['pbest'] = copy.copy(list_part[ind])
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

        for c in range(len(self.X_test)):
            coord_ = self.X_test[c]
            if x_bench == coord_[0] and y_bench == coord_[1]:
                index = c
                # print(index)
                break

        for i, key in enumerate(sensors):
            list_f = copy.copy(self.P.nodes[part.node]['fitness_list'][key])
            bench = copy.copy(self.dict_sensors_[key]['mu']['data'])
            list_f.append(bench[index])
            self.P.nodes[part.node]['fitness_list'][key] = copy.copy(list_f)

            if i == 0:
                fitness = max(list_f)
                ind = list_f.index(fitness)
                coord_pbest = list_part[ind]
            else:
                if max(list_f) > fitness:
                    fitness = max(list_f)
                    ind = list_f.index(fitness)
                    coord_pbest = list_part[ind]
        self.P.nodes[part.node]['D_p']['pbest'] = coord_pbest
        self.P.nodes[part.node]['U_p'] = copy.copy(list_part)

    def global_best_decoupled(self):
        for i, subfleet in enumerate(self.sub_fleets):
            for p, particle in enumerate(subfleet):
                sensors = self.P.nodes[particle]['S_p'].keys()
                for s, sensor in enumerate(sensors):
                    usf = copy.copy(self.dict_sensors_[sensor]['U_sf'])
                    value = copy.copy(self.dict_sensors_[sensor]['fitness'])
                    if s == 0:
                        fitness = max(value)
                        ind = value.index(fitness)
                        coord_gbest = usf[ind]
                    else:
                        if max(value) > fitness:
                            fitness = max(value)
                            ind = value.index(fitness)
                            coord_gbest = usf[ind]
                self.P.nodes[particle]['D_p']['gbest'] = coord_gbest

    def global_best_coupled(self):
        for i, subfleet in enumerate(self.sub_fleets):
            for p, particle in enumerate(subfleet):
                sensors = self.P.nodes[particle]['S_p'].keys()
                summatory = list()
                for j, key in enumerate(sensors):
                    usf = copy.copy(self.dict_sensors_[key]['U_sf'])
                    w = copy.copy(self.dict_sensors_[key]['w'][particle])
                    value = copy.copy(self.dict_sensors_[key]['fitness'])
                    if j == 0:
                        summatory = [data * w for data in value]
                    else:
                        list1 = [data * w for data in value]
                        summatory = list(map(lambda x, y: x + y, summatory, list1))
                ind = summatory.index(np.max(np.array(summatory)))
                self.P.nodes[particle]['D_p']['gbest'] = usf[ind]

    def u_sf(self):
        for i, subfleet in enumerate(self.sub_fleets):
            sensors = self.s_sf[i]
            for s, sensor in enumerate(sensors):
                usf = copy.copy(self.dict_sensors_[sensor]['U_sf'])
                fitness_sf = copy.copy(self.dict_sensors_[sensor]['fitness'])
                for p, particle in enumerate(subfleet):
                    s_p = self.P.nodes[particle]['S_p'].keys()
                    u_p = copy.copy(self.P.nodes[particle]['U_p'])
                    usf.append(u_p[-1])
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

    def u_sf_exploit(self):
        for i, subfleet in enumerate(self.sub_fleets):
            for j, zo in enumerate(self.z_['subfleet%s' % i].keys()):
                zone = self.z_['subfleet%s' % i][zo]
                vehicles = zone['vehicles']
                sensors = zone['sensors'].keys()
                for s, sensor in enumerate(sensors):
                    usf = copy.copy(zone['sensors'][sensor]['u_sf'])
                    fitness_sf = copy.copy(zone['sensors'][sensor]['fitness'])
                    for p, particle in enumerate(vehicles):
                        u_p = copy.copy(self.P.nodes[particle]['Up_exploit'])
                        s_p = self.P.nodes[particle]['S_p'].keys()
                        usf.append(u_p[-1])
                        on_board = False
                        for a, key in enumerate(s_p):
                            if key == sensor:
                                on_board = True
                                break
                            else:
                                on_board = False
                        if on_board:
                            fitness = copy.copy(self.P.nodes[particle]['fitness_exploit'][sensor])
                            fitness_sf.append(fitness[-1])
                        else:
                            fitness_sf.append(0)
                    zone['sensors'][sensor]['u_sf'] = copy.copy(usf)
                    zone['sensors'][sensor]['fitness'] = copy.copy(fitness_sf)

    def peaks_mu(self, sensor):
        peaks = []
        index_bench = copy.copy(self.dict_benchs_[sensor]['index_peaks'])
        bench = copy.copy(self.dict_sensors_[sensor]['mu']['data'])
        for j, ind in enumerate(index_bench):
            peaks.append(bench[round(ind)])
        self.dict_sensors_[sensor]['mu']['peaks'] = copy.copy(peaks)

    def configuration_exploit(self, sf, s_sf):
        for i, zo in enumerate(self.z_['subfleet%s' % sf].keys()):
            zone = self.z_['subfleet%s' % sf][zo]
            vehicles = zone['vehicles']
            sensors = zone['sensors'].keys()
            # print(zo, sensors)
            for j, sensor in enumerate(sensors):
                num = 0
                for v, vehicle in enumerate(vehicles):
                    s_p = self.P.nodes[vehicle]['S_p'].keys()
                    self.P.add_node(vehicle, Zone=zo, Subfleet=sf, )
                    up = []
                    up.append(self.P.nodes[vehicle]['U_p'][-1])
                    self.P.nodes[vehicle]['Up_exploit'] = up
                    for s, key in enumerate(s_p):
                        if sensor == key:
                            fitness = []
                            fitness.append(self.P.nodes[vehicle]['fitness_list'][key][-1])
                            self.P.nodes[vehicle]['fitness_exploit'][key] = fitness
                            num += 1
                zone['sensors'][sensor] = {}
                zone['sensors'][sensor]['cant'] = num
                zone['sensors'][sensor]['measure'] = []
                zone['sensors'][sensor]['q_p'] = []
                zone['sensors'][sensor]['u_sf'] = []
                zone['sensors'][sensor]['fitness'] = []
                zone['sensors'][sensor]['mu'] = {}
                zone['sensors'][sensor]['sigma'] = {}
                zone['sensors'][sensor]['mu']['data'] = copy.copy(self.dict_sensors_[sensor]['mu']['data'])
                zone['sensors'][sensor]['sigma']['data'] = copy.copy(self.dict_sensors_[sensor]['sigma']['data'])
                index = zone['index']
                mu = []
                sigma = []
                for b in range(len(index)):
                    mu.append(self.dict_sensors_[sensor]['mu']['data'][index[b]])
                    sigma.append(self.dict_sensors_[sensor]['sigma']['data'][index[b]])
                zone['sensors'][sensor]['mu']['zone'] = copy.copy(mu)
                zone['sensors'][sensor]['sigma']['zone'] = copy.copy(sigma)
            self.wn_exploit(sensors, zone, s_sf, vehicles)
            # print('nuevo', zo, sensors)

    def take_measures_exploit(self):
        for i, subfleet in enumerate(self.sub_fleets):
            for h, zo in enumerate(self.z_['subfleet%s' % i].keys()):
                zone = self.z_['subfleet%s' % i][zo]
                vehicles = zone['vehicles']
                for p, particle in enumerate(vehicles):
                    sensors = copy.copy(self.P.nodes[particle]['S_p']).keys()
                    q_p = copy.copy(self.P.nodes[particle]['Q_p'])
                    u_p = copy.copy(self.P.nodes[particle]['U_p'])
                    last = u_p[-1]
                    x = last[0]
                    y = last[1]
                    q_p.append(last)
                    # print('q_p', particle, q_p)
                    self.P.nodes[particle]['Q_p'] = copy.copy(q_p)
                    qp_exploit = copy.copy(self.P.nodes[particle]['Qp_exploit'])
                    lastp_exploit = u_p[-1]
                    qp_exploit.append(lastp_exploit)
                    # print('q_p', particle, q_p)
                    self.P.nodes[particle]['Qp_exploit'] = copy.copy(qp_exploit)
                    for j, key in enumerate(sensors):
                        measure = copy.copy(self.P.nodes[particle]['S_p'][key])
                        bench = copy.copy(self.dict_benchs_[key]['map_created'])
                        new_measure = bench[x][y]
                        measure.append(new_measure)
                        self.P.nodes[particle]['S_p'][key] = copy.copy(measure)
                        measurep_exploit = copy.copy(self.P.nodes[particle]['Sp_exploit'][key])
                        measurep_exploit.append(new_measure)
                        self.P.nodes[particle]['Sp_exploit'][key] = copy.copy(measurep_exploit)
                # print(sensors)

    def gp_update_exploit(self):
        # S_n = {"sensor1": {"mu": [], "sigma": []}}
        self.post_array = []
        for i, sub_fleet in enumerate(self.sub_fleets):
            for h, zo in enumerate(self.z_['subfleet%s' % i].keys()):
                zone = self.z_['subfleet%s' % i][zo]
                vehicles = zone['vehicles']
                sensors = zone['sensors'].keys()
                for s, sensor in enumerate(sensors):
                    measures_for_sensor = []
                    coordinates_for_sensor = []
                    for p, particle in enumerate(vehicles):
                        s_p = copy.copy(self.P.nodes[particle]['S_p'])
                        s_p_keys = copy.copy(self.P.nodes[particle]['S_p']).keys()
                        q_p = copy.copy(self.P.nodes[particle]['Q_p'])
                        for r, key in enumerate(s_p_keys):
                            if key == sensor:
                                # print(sensorp, s_p[sensorp])
                                measures_for_sensor.extend(s_p[sensor])
                                coordinates_for_sensor.extend(q_p)
                    coordinates_for_sensor = np.array(coordinates_for_sensor).reshape(-1, 2)
                    # print(s, sensor)
                    # print(coordinates_for_sensor)
                    # print(measures_for_sensor)
                    # print('coord', coordinates_for_sensor)
                    # print('measure', measures_for_sensor)
                    if len(coordinates_for_sensor) != 0:
                        self.gpr.fit(coordinates_for_sensor, measures_for_sensor)
                        zone['sensors'][sensor]['mu']['data'], zone['sensors'][sensor]['sigma'][
                            'data'] = self.gpr.predict(self.X_test, return_std=True)
                        index = zone['index']
                        mu_s = []
                        sigma_s = []
                        for b, ind in enumerate(index):
                            mu_s.append(zone['sensors'][sensor]['mu']['data'][ind])
                            sigma_s.append(zone['sensors'][sensor]['sigma']['data'][ind])
                        zone['sensors'][sensor]['mu']['zone'] = copy.copy(mu_s)
                        zone['sensors'][sensor]['sigma']['zone'] = copy.copy(sigma_s)
                        self.post_array.append(round(np.min(np.exp(self.gpr.kernel_.theta[0])), 1))

    def wn_exploit(self, sensors, zone, s_sf, vehicles):
        for i, sensor in enumerate(sensors):
            zone['sensors'][sensor]['w'] = {}
            zone['sensors'][sensor]['w_init'] = {}
        for p, particle in enumerate(vehicles):
            sp = self.P.nodes[particle]['S_p'].keys()
            inver = 0
            for s, sen in enumerate(sp):
                for i, sensor in enumerate(sensors):
                    if sen == sensor:
                        cant_sensor = zone['sensors'][sensor]['cant']
                        if cant_sensor == 0:
                            h = 0
                            only = False
                        elif cant_sensor == 1:
                            only = True
                            only_sensor = sen
                            break
                        else:
                            only = False
                            h = 1 / cant_sensor
                            inver = inver + h
                            x_value = 1 / inver
            for s, sen in enumerate(sp):
                for i, sensor in enumerate(sensors):
                    if sen == sensor:
                        if only:
                            if sen == only_sensor:
                                zone['sensors'][sen]['w'][particle] = 1
                                zone['sensors'][sen]['w_init'][particle] = 1
                            else:
                                zone['sensors'][sen]['w'][particle] = 0
                                zone['sensors'][sen]['w_init'][particle] = 0
                        else:
                            if zone['sensors'][sen]['cant'] == 0:
                                zone['sensors'][sen]['w'][particle] = 0
                                zone['sensors'][sen]['w_init'][particle] = 0
                            else:
                                zone['sensors'][sen]['w'][particle] = x_value / zone['sensors'][sen]['cant']
                                zone['sensors'][sen]['w_init'][particle] = x_value / zone['sensors'][sen]['cant']

    def w_exploit(self, sensors, zone, s_sf):
        inver = 0
        for i, sensor in enumerate(sensors):
            cant_sensor = zone['sensors'][sensor]['cant']
            if cant_sensor == 0:
                h = 0
            else:
                h = 1 / cant_sensor
            inver = inver + h
        x_value = 1 / inver
        for i, sensor in enumerate(sensors):
            if zone['sensors'][sensor]['cant'] == 0:
                zone['sensors'][sensor]['w'] = 0
                zone['sensors'][sensor]['w_init'] = 0
            else:
                zone['sensors'][sensor]['w'] = x_value / zone['sensors'][sensor]['cant']
                zone['sensors'][sensor]['w_init'] = x_value / zone['sensors'][sensor]['cant']

    def local_best_coupled_exploit(self, part):
        index = -1
        while index == -1:
            part, self.s_n = Limits(self.secure, self.xs, self.ys, self.vehicles).new_limit(self.g, part, self.s_n,
                                                                                            self.n_data,
                                                                                            self.s_ant, self.part_ant)
            x_bench = int(part[0])
            y_bench = int(part[1])
            for c in range(len(self.X_test)):
                coord_ = self.X_test[c]
                if x_bench == coord_[0] and y_bench == coord_[1]:
                    index = c
                    # print(index)
                    break

        list_part = copy.copy(self.P.nodes[part.node]['Up_exploit'])
        listpart = copy.copy(self.P.nodes[part.node]['U_p'])
        list_part.append(np.array(part))
        listpart.append(np.array(part))

        s_p = self.P.nodes[part.node]['S_p'].keys()
        subfleet = self.P.nodes[part.node]['Subfleet']
        zone = self.P.nodes[part.node]['Zone']
        s_n = self.z_['subfleet%s' % subfleet][zone]['sensors'].keys()
        coord_zone = self.z_['subfleet%s' % subfleet][zone]['coord']

        for t, coo in enumerate(coord_zone):
            if coo[0] == x_bench and coo[1] == y_bench:
                reach = True
                break
            else:
                reach = False

        self.P.nodes[part.node]['Reach'] = reach

        for i, key in enumerate(s_p):
            for j, sensor in enumerate(s_n):
                if key == sensor:
                    list_f = copy.copy(self.P.nodes[part.node]['fitness_exploit'][key])
                    bench = copy.copy(self.z_['subfleet%s' % subfleet][zone]['sensors'][key]['mu']['data'])
                    list_f.append(bench[index])
                    self.P.nodes[part.node]['fitness_exploit'][key] = copy.copy(list_f)

        summatory = list()
        t = 0
        for i, key in enumerate(s_n):
            for j, sensor in enumerate(s_p):
                # print('sn', s_n, 'sp', s_p)
                if key == sensor:
                    w = copy.copy(self.z_['subfleet%s' % subfleet][zone]['sensors'][key]['w'][part.node])
                    value = copy.copy(self.P.nodes[part.node]['fitness_exploit'][key])
                    if t == 0:
                        summatory = [data * w for data in value]
                        t += 1
                    else:
                        list1 = [data * w for data in value]
                        summatory = list(map(lambda x, y: x + y, summatory, list1))
        ind = summatory.index(np.max(np.array(summatory)))
        # print('fit', len(summatory), 'lis', len(list_part))
        self.P.nodes[part.node]['D_p']['pbest'] = copy.copy(list_part[ind])
        self.P.nodes[part.node]['U_p'] = copy.copy(listpart)
        self.P.nodes[part.node]['Up_exploit'] = copy.copy(list_part)

    def local_best_decoupled_exploit(self, part, dfirst):
        part, self.s_n = Limits(self.secure, self.xs, self.ys, self.vehicles).new_limit(self.g, part, self.s_n,
                                                                                        self.n_data,
                                                                                        self.s_ant, self.part_ant)
        x_bench = int(part[0])
        y_bench = int(part[1])

        list_part = copy.copy(self.P.nodes[part.node]['Up_exploit'])
        listpart = copy.copy(self.P.nodes[part.node]['U_p'])
        list_part.append(np.array(part))
        listpart.append(np.array(part))

        s_p = self.P.nodes[part.node]['S_p'].keys()
        subfleet = self.P.nodes[part.node]['Subfleet']
        zone = self.P.nodes[part.node]['Zone']
        s_n = self.z_['subfleet%s' % subfleet][zone]['sensors'].keys()
        coord_zone = self.z_['subfleet%s' % subfleet][zone]['coord']

        for c in range(len(self.X_test)):
            coord_ = self.X_test[c]
            if x_bench == coord_[0] and y_bench == coord_[1]:
                index = c
                # print(index)
                break

        # for t, coo in enumerate(coord_zone):
        #     if coo[0] == x_bench and coo[1] == y_bench:
        #         reach = True
        #         break
        #     else:
        #         reach = False
        #
        # self.P.nodes[part.node]['Reach'] = reach

        t = 0
        for i, key in enumerate(s_p):
            for j, sensor in enumerate(s_n):
                if key == sensor:
                    list_f = copy.copy(self.P.nodes[part.node]['fitness_exploit'][key])
                    bench = copy.copy(self.z_['subfleet%s' % subfleet][zone]['sensors'][key]['mu']['data'])
                    list_f.append(bench[index])
                    self.P.nodes[part.node]['fitness_exploit'][key] = copy.copy(list_f)
                    if t == 0:
                        fitness = max(list_f)
                        ind = list_f.index(fitness)
                        coord_pbest = list_part[ind]
                        t += 1
                    else:
                        if max(list_f) > fitness:
                            fitness = max(list_f)
                            ind = list_f.index(fitness)
                            coord_pbest = list_part[ind]
        self.P.nodes[part.node]['D_p']['pbest'] = coord_pbest
        self.P.nodes[part.node]['U_p'] = copy.copy(listpart)
        self.P.nodes[part.node]['Up_exploit'] = copy.copy(list_part)

    def global_best_coupled_exploit(self):
        for i, subfleet in enumerate(self.sub_fleets):
            for j, zo in enumerate(self.z_['subfleet%s' % i].keys()):
                zone = self.z_['subfleet%s' % i][zo]
                vehicles = zone['vehicles']
                sensors = zone['sensors'].keys()
                summatory = []
                for p, particle in enumerate(vehicles):
                    sp = self.P.nodes[particle]['S_p'].keys()
                    t = 0
                    for sr, sens in enumerate(sp):
                        for s, sensor in enumerate(sensors):
                            if sens == sensor:
                                usf = copy.copy(zone['sensors'][sensor]['u_sf'])
                                w = copy.copy(zone['sensors'][sensor]['w'][particle]
                                              )
                                value = copy.copy(zone['sensors'][sensor]['fitness'])
                                if t == 0:
                                    summatory = [data * w for data in value]
                                    t += 1
                                else:
                                    list1 = [data * w for data in value]
                                    summatory = list(map(lambda x, y: x + y, summatory, list1))
                    ind = summatory.index(np.max(np.array(summatory)))
                    self.P.nodes[particle]['D_p']['gbest'] = usf[ind]

    def global_best_decoupled_exploit(self):
        for i, subfleet in enumerate(self.sub_fleets):
            for j, zo in enumerate(self.z_['subfleet%s' % i].keys()):
                zone = self.z_['subfleet%s' % i][zo]
                vehicles = zone['vehicles']
                sensors = zone['sensors'].keys()
                for p, particle in enumerate(vehicles):
                    sp = self.P.nodes[particle]['S_p'].keys()
                    t = 0
                    for sr, sens in enumerate(sp):
                        for s, sensor in enumerate(sensors):
                            if sens == sensor:
                                usf = copy.copy(zone['sensors'][sensor]['u_sf'])
                                value = copy.copy(zone['sensors'][sensor]['fitness'])
                                if t == 0:
                                    fitness = max(value)
                                    ind = value.index(fitness)
                                    coord_gbest = usf[ind]
                                    t += 1
                                else:
                                    if max(value) > fitness:
                                        fitness = max(value)
                                        ind = value.index(fitness)
                                        coord_gbest = usf[ind]
                    self.P.nodes[particle]['D_p']['gbest'] = coord_gbest

    def method_coupled_exploit(self):
        for i, subfleet in enumerate(self.sub_fleets):
            for j, zo in enumerate(self.z_['subfleet%s' % i].keys()):
                zone = self.z_['subfleet%s' % i][zo]
                vehicles = zone['vehicles']
                sensors = zone['sensors'].keys()
                coord = zone['coord']
                for p, particle in enumerate(vehicles):
                    t = 0
                    s_p = self.P.nodes[particle]['S_p'].keys()
                    summatory_mu = list()
                    summatory_sigma = list()
                    for h, sensor in enumerate(s_p):
                        for k, key in enumerate(sensors):
                            if sensor == key:
                                w_value = copy.copy(zone['sensors'][sensor]['w'][particle])
                                mu = list(copy.copy(zone['sensors'][sensor]['mu']['zone']))
                                sigma = list(copy.copy(zone['sensors'][sensor]['sigma']['zone']))
                                if t == 0:
                                    summatory_mu = [data_m * w_value for data_m in mu]
                                    summatory_sigma = [data_s * w_value for data_s in sigma]
                                    t += 1
                                else:
                                    data_mu = [data_m * w_value for data_m in mu]
                                    data_sigma = [data_s * w_value for data_s in sigma]
                                    summatory_mu = list(map(lambda x, y: x + y, summatory_mu, data_mu))
                                    summatory_sigma = list(map(lambda x, y: x + y, summatory_sigma, data_sigma))
                    ind_mu = np.argwhere(summatory_mu == np.amax(summatory_mu))
                    ind_mu = ind_mu.flatten().tolist()
                    indmu = random.randint(0, len(ind_mu) - 1)
                    ind_sigma = np.argwhere(summatory_sigma == np.amax(summatory_sigma))
                    ind_sigma = ind_sigma.flatten().tolist()
                    indsigma = random.randint(0, len(ind_sigma) - 1)
                    prev_mu = self.P.nodes[particle]['D_p']['con_index']
                    prev_sigma = self.P.nodes[particle]['D_p']['un_index']
                    if prev_mu in ind_mu:
                        index_mu = prev_mu
                    else:
                        prev_mu = ind_mu[indmu]
                        index_mu = prev_mu
                        self.P.nodes[particle]['D_p']['con_index'] = prev_mu
                    if prev_sigma in ind_sigma:
                        index_sigma = prev_sigma
                    else:
                        prev_sigma = ind_sigma[indsigma]
                        index_sigma = prev_sigma
                        self.P.nodes[particle]['D_p']['un_index'] = prev_sigma
                    self.P.nodes[particle]['D_p']['con'] = coord[index_mu]
                    self.P.nodes[particle]['D_p']['un'] = coord[index_sigma]
                    # ind_mu = summatory_mu.index(np.max(np.array(summatory_mu)))
                    # ind_sigma = summatory_sigma.index(np.max(np.array(summatory_sigma)))
                    # self.P.nodes[particle]['D_p']['con'] = coord[ind_mu]
                    # self.P.nodes[particle]['D_p']['un'] = coord[ind_sigma]

    def method_decoupled_exploit(self):
        for i, subfleet in enumerate(self.sub_fleets):
            for j, zo in enumerate(self.z_['subfleet%s' % i].keys()):
                zone = self.z_['subfleet%s' % i][zo]
                vehicles = zone['vehicles']
                sensors = zone['sensors'].keys()
                coord = zone['coord']
                for r, sn in enumerate(sensors):
                    zone['sensors'][sn]['mu']['max'], zone['sensors'][sn]['sigma'][
                        'max'] = self.model_max(
                        zone['sensors'][sn]['mu']['zone'], zone['sensors'][sn]['sigma']['zone'], coord)
                for p, particle in enumerate(vehicles):
                    t = 0
                    s_p = self.P.nodes[particle]['S_p'].keys()
                    max_con_value = 0
                    max_un_value = 0
                    max_con = []
                    max_un = []
                    for h, sensor in enumerate(s_p):
                        for k, key in enumerate(sensors):
                            if sensor == key:
                                max_mu, coord_mu = list(copy.copy(zone['sensors'][sensor]['mu']['max']))
                                max_sigma, coord_sigma = list(copy.copy(zone['sensors'][sensor]['sigma']['max']))
                                if t == 0:
                                    max_con = coord_mu
                                    max_con_value = max_mu
                                    max_un = coord_sigma
                                    max_un_value = max_sigma
                                    t += 1
                                else:
                                    if max_con_value < max_mu:
                                        max_con = coord_mu
                                    if max_un_value < max_sigma:
                                        max_un = coord_sigma
                    self.P.nodes[particle]['D_p']['con'] = max_con
                    self.P.nodes[particle]['D_p']['un'] = max_un

    def check_vehicles(self, i, subfleet):
        repeat = False
        assigned = False
        available_vehicles = copy.copy(subfleet)
        available_zones = copy.copy(list(self.z_['subfleet%s' % i].keys()))
        print('subfleet%s' % i, subfleet, available_zones)
        # print('de nuevo')
        check_ = True
        for q, avz in enumerate(self.z_['subfleet%s' % i].keys()):
            self.z_['subfleet%s' % i][avz]['vehicles'] = []
            if len(self.z_['subfleet%s' % i][avz]['sensors'].keys()) == 0:
                check_ = False
                no_assigned = [self.z_['subfleet%s' % i][avz]['number']]
                check = False
            print('sensor', self.z_['subfleet%s' % i][avz]['sensors'].keys())
        if check_:
            while len(available_vehicles) > 0:
                if len(available_zones) > 0:
                    no_assigned = []
                    no_assigned_repeat = []
                    check_repeat = True
                    for z, zo in enumerate(available_zones):
                        vehi_assig = False
                        zone = self.z_['subfleet%s' % i][zo]
                        # print('sensores zone%s' % z, zone['sensors'])
                        nz = zone['number']
                        priority = len(zone['sensors'].keys())
                        zone['priority'] = priority
                        while not vehi_assig:
                            possibility = list()
                            zone['possibility'] = []
                            for v, vehicle in enumerate(available_vehicles):
                                inter = list(zone['sensors'].keys() & self.P.nodes[vehicle]['S_p'].keys())
                                n_p = len(inter)
                                if priority == n_p:
                                    possibility.append(vehicle)
                            if len(possibility) == 0:
                                priority = priority - 1
                                zone['priority'] = priority
                            else:
                                zone['possibility'] = possibility
                                vehi_assig = True
                            if priority <= 0 and len(possibility) == 0:
                                no_assigned.append(nz)
                                no_assigned_repeat.append(zo)
                                vehi_assig = True
                        # print('vehículos', zo, zone['possibility'])
                    if len(no_assigned) == 0:
                        check = True
                        check_repeat = True
                    elif len(no_assigned) > 0 and repeat:
                        check = True
                        check_repeat = False
                    elif len(no_assigned) > 0 and not repeat:
                        check = False
                        break
                    if check:
                        if not check_repeat:
                            for g in range(len(no_assigned_repeat)):
                                # print(available_zones, 'no_assigned', no_assigned_repeat[g])
                                available_zones.remove(no_assigned_repeat[g])
                        if len(available_zones) != 0:
                            j = 0
                            priority_list = []
                            zone_list = []
                            while j < len(available_zones):
                                zone = self.z_['subfleet%s' % i][available_zones[j]]
                                priority_list.append(zone['priority'])
                                zone_list.append(available_zones[j])
                                j += 1
                            index1 = [index for index in range(len(priority_list)) if
                                      priority_list[index] == max(priority_list)]
                            k = 0
                            list_np = []
                            pos_zone = []
                            while k < len(index1):
                                zone = self.z_['subfleet%s' % i][zone_list[index1[k]]]
                                list_np.append(len(zone['possibility']))
                                pos_zone.append(zone_list[index1[k]])
                                # if k == 0:
                                #     min_np = len(zone['possibility'])
                                #     pos_zone = zone_list[index1[k]]
                                # else:
                                #     if min_np > len(zone['possibility']):
                                #         min_np = len(zone['possibility'])
                                #         pos_zone = zone_list[index1[k]]
                                #     if min_np == len(zone['possibility']):
                                k += 1
                            index2 = [index for index in range(len(list_np)) if list_np[index] == min(list_np)]
                            if len(index2) > 1:
                                for w in range(len(index2)):
                                    grids = len(self.z_['subfleet%s' % i][pos_zone[w]]['index'])
                                    if w == 0:
                                        max_grid = grids
                                        large = pos_zone[w]
                                    else:
                                        if max_grid < grids:
                                            max_grid = grids
                                            large = pos_zone[w]
                            else:
                                large = pos_zone[index2[0]]
                            if len(self.z_['subfleet%s' % i][large]['possibility']) > 1:
                                for l in range(len(self.z_['subfleet%s' % i][large]['possibility'])):
                                    asv = self.z_['subfleet%s' % i][large]['possibility'][l]
                                    vz = 0
                                    for t, zon in enumerate(available_zones):
                                        if zon != large:
                                            asv1 = self.z_['subfleet%s' % i][zon]['possibility']
                                            inter = list(set(asv) & set(asv1))
                                            if len(inter) > 0:
                                                vz += 1
                                    if l == 0:
                                        min_v = vz
                                        veh = asv
                                    else:
                                        if min_v > vz:
                                            min_v = vz
                                            veh = asv
                                list_vehicles = copy.copy(self.z_['subfleet%s' % i][large]['vehicles'])
                                list_vehicles.append(veh)
                                # self.P.add_node(Subfleet=i, Zone=large,)
                                self.z_['subfleet%s' % i][large]['vehicles'] = copy.copy(list_vehicles)
                                available_vehicles.remove(veh)
                                available_zones.remove(large)
                            else:
                                list_vehicles = copy.copy(self.z_['subfleet%s' % i][large]['vehicles'])
                                list_vehicles.append(self.z_['subfleet%s' % i][large]['possibility'][0])
                                self.z_['subfleet%s' % i][large]['vehicles'] = copy.copy(list_vehicles)
                                # print(available_vehicles, 'assigned', self.z_['subfleet%s' % i][large]['possibility'][0])
                                available_vehicles.remove(self.z_['subfleet%s' % i][large]['possibility'][0])
                                available_zones.remove(large)
                else:
                    available_zones = copy.copy(list(self.z_['subfleet%s' % i].keys()))
                    repeat = True
            no_assigned = []
            for b, za in enumerate(list(self.z_['subfleet%s' % i].keys())):
                print(za, ':', self.z_['subfleet%s' % i][za]['vehicles'])
                if len(self.z_['subfleet%s' % i][za]['vehicles']) == 0:
                    no_assigned.append(self.z_['subfleet%s' % i][za]['number'])
            if len(no_assigned) > 0:
                assigned = False
                check = False
            else:
                assigned = True
        return check, no_assigned, assigned

    def obtain_zones(self):
        # if self.simulation > 10:
        #     for i, subfleet in enumerate(self.sub_fleets):
        #         sensors = self.s_sf[i]
        #         for s, sensor in enumerate(sensors):
        #             bench = copy.copy(self.dict_benchs_[sensor]['map_created'])
        #             # self.plot.benchmark(bench, sensor)
        #             mu = copy.copy(self.dict_sensors_[sensor]['mu']['data'])
        #             sigma = copy.copy(self.dict_sensors_[sensor]['sigma']['data'])
        #             vehicles = copy.copy(self.dict_sensors_[sensor]['vehicles'])
        #             trajectory = list()
        #             first = True
        #             list_ind = list()
        #             for veh in vehicles:
        #                 list_ind.append(self.P.nodes[veh]['index'])
        #                 if first:
        #                     trajectory = np.array(self.P.nodes[veh]['U_p'])
        #                     first = False
        #                 else:
        #                     new = np.array(self.P.nodes[veh]['U_p'])
        #                     trajectory = np.concatenate((trajectory, new), axis=1)
        #             self.plot.plot_classic(mu, sigma, trajectory, sensor, list_ind)
        for i, subfleet in enumerate(self.sub_fleets):
            sensors = self.s_sf[i]
            for s, sensor in enumerate(sensors):
                radio = 10
                self.dict_sensors_[sensor] = self.detect.areas_levels(self.dict_sensors_[sensor], self.vehicles, radio)
                # if self.simulation > 24:
                # self.plot.action_areas(self.dict_sensors_[sensor]['action_zones'], sensor)
                # self.plot.action_areas(self.dict_benchs_[sensor]['action_zones'], sensor)
        for i, subfleet in enumerate(self.sub_fleets):
            sensors = self.s_sf[i]
            check = True
            self.z_['subfleet%s' % i] = {}
            self.z_['subfleet%s' % i] = self.detect.overlapping_areas(sensors, self.dict_sensors_, check)
            check, no_assigned, assigned = self.check_vehicles(i, subfleet)
            # if self.simulation > 10:
            #   self.plot.zones_plot(self.z_['subfleet%s' % i], len(self.z_['subfleet%s' % i]))
            t = 0
            while not check:
                if t == 0:
                    self.z_['subfleet%s' % i] = {}
                    self.z_['subfleet%s' % i] = self.detect.overlapping_areas(sensors, self.dict_sensors_,
                                                                              check)
                    check, no_assigned, assigned = self.check_vehicles(i, subfleet)
                    t += 1
                    print('in')
                    # self.plot.zones_plot(self.z_['subfleet%s' % i], len(self.z_['subfleet%s' % i]))
                    # if self.simulation > 9:
                    #   self.plot.zones_plot(self.z_['subfleet%s' % i], len(self.z_['subfleet%s' % i]))

                else:
                    # print('no assigned', no_assigned)
                    self.dict_sensors_ = self.detect.re_overlap(self.z_['subfleet%s' % i], no_assigned,
                                                                self.dict_sensors_, sensors)
                    self.z_['subfleet%s' % i] = {}
                    self.z_['subfleet%s' % i] = self.detect.overlapping_areas(sensors, self.dict_sensors_,
                                                                              check)
                    check, no_assigned, assigned = self.check_vehicles(i, subfleet)

                    # check = True
            # self.plot.zones_plot(self.z_['subfleet%s' % i], len(self.z_['subfleet%s' % i]))
            # if self.simulation > 0:
            #     self.plot.zones_plot(self.z_['subfleet%s' % i], len(self.z_['subfleet%s' % i]))
            self.configuration_exploit(i, sensors)

    def calculate_error(self, final):
        if self.type_error == 'all_map_mse':
            mse_simulation = []
            for i, subfleet in enumerate(self.sub_fleets):
                sensors = self.s_sf[i]
                for s, sensor in enumerate(sensors):
                    bench = copy.copy(self.dict_benchs_[sensor]['original'])
                    mu = copy.copy(self.dict_sensors_[sensor]['mu']['data'])
                    mse = mean_squared_error(y_true=bench, y_pred=mu)
                    cant_sensor = self.dict_sensors_[sensor]['cant']
                    # w = self.dict_sensors_[sensor]['w']
                    mse_simulation.append(mse)
                    self.mse_sensor.append(mse)
                    self.sensor_mse.append(sensor)
                    self.cant_sensor_mse.append(cant_sensor)
                    # self.w_mse.append(w)
            mse_simulation = np.array(mse_simulation)
            mse_mean = np.mean(mse_simulation)
            mse_std = np.std(mse_simulation)
            self.mse_data = np.append(self.mse_data, [[mse_mean, np.mean(self.distances)]], axis=0)
            if final:
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
                    # w = self.dict_sensors_[sensor]['w']
                    r2_simulation.append(r2)
                    self.r2_sensor.append(r2)
                    self.sensor.append(sensor)
                    self.cant_sensor.append(cant_sensor)
                    # self.w.append(w)
            r2_simulation = np.array(r2_simulation)
            r2_mean = np.mean(r2_simulation)
            r2_std = np.std(r2_simulation)
            self.r2_data = np.append(self.r2_data, [[r2_mean, np.mean(self.distances)]], axis=0)
            if final:
                self.mean_error.append(r2_mean)
                self.array_r2.append(r2_mean)
                self.conf_error.append(r2_std * 1.96)
        elif self.type_error == 'peaks':
            error_simulation = []
            conf_simulation = []
            for i, subfleet in enumerate(self.sub_fleets):
                sensors = self.s_sf[i]
                for s, sensor in enumerate(sensors):
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
            self.peak_error_data = np.append(self.peak_error_data, [[error_mean, np.mean(self.distances)]], axis=0)
            if final:
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
            self.caz_mse_data = np.append(self.caz_mse_data, [[np.mean(zone_error), np.mean(self.distances)]], axis=0)
            if final:
                self.mean_az_mse.append(np.mean(zone_error))
                self.conf_az_mse.append(np.std(zone_error) * 1.96)

    def first_values(self):

        for part in self.pop:
            self.part_ant, self.distances = self.util.distance_part(self.g, self.n_data, part, self.part_ant,
                                                                    self.distances, self.array_part, dfirst=True)

            self.n_data += 1
            if self.n_data > self.vehicles - 1:
                self.n_data = 0

        self.take_measures()
        self.gp_update()

        self.type_error = 'all_map_r2'
        self.calculate_error(False)
        self.type_error = 'all_map_mse'
        self.calculate_error(False)
        self.type_error = 'peaks'
        self.calculate_error(False)
        self.type_error = 'zones'
        self.calculate_error(False)

        for part in self.pop:
            if self.method_pso == 'coupled':
                self.local_best_coupled(part, dfirst=True)
            elif self.method_pso == 'decoupled':
                self.local_best_decoupled(part, dfirst=True)

        self.u_sf()

        if self.method_pso == 'coupled':
            self.global_best_coupled()
        elif self.method_pso == 'decoupled':
            self.global_best_decoupled()

        if self.method_pso == 'coupled':
            # self.method_coupled()
            self.method_coupled_sp(dfirst=True)
        elif self.method_pso == 'decoupled':
            # self.method_decoupled()
            self.method_decoupled_sp()

    def step_explore(self, action):
        dis_steps = 0
        dist_ant = np.mean(self.distances)
        self.dist_pre = np.mean(self.distances)
        self.n_data = 0

        while dis_steps < 10:

            previous_dist = np.mean(self.distances)

            for part in self.pop:
                if self.weights_b:
                    action = self.P.nodes[part.node]['action_explore']
                else:
                    action = self.action_explore
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

            if (np.mean(self.distances) - self.last_sample) >= (np.mean(self.post_array) * self.lam):
                self.last_sample = np.mean(self.distances)
                self.take_measures()
                self.gp_update()

                self.type_error = 'all_map_r2'
                self.calculate_error(False)
                self.type_error = 'all_map_mse'
                self.calculate_error(False)
                self.type_error = 'peaks'
                self.calculate_error(False)
                self.type_error = 'zones'
                self.calculate_error(False)

                if self.method_pso == 'coupled':
                    # self.method_coupled()
                    self.method_coupled_sp(dfirst=False)
                elif self.method_pso == 'decoupled':
                    # self.method_decoupled()
                    self.method_decoupled_sp()

            dis_steps = np.mean(self.distances) - dist_ant
            if np.mean(self.distances) == previous_dist:
                break
            self.g += 1

        done = False
        # print(self.distances)
        # if self.simulation > 10:
            # for i, subfleet in enumerate(self.sub_fleets):
        #             sensors = self.s_sf[i]
        #             for s, sensor in enumerate(sensors):
        #                 bench = copy.copy(self.dict_benchs_[sensor]['map_created'])
        #                 # self.plot.benchmark(bench, sensor)
        #                 mu = copy.copy(self.dict_sensors_[sensor]['mu']['data'])
        #                 sigma = copy.copy(self.dict_sensors_[sensor]['sigma']['data'])
        #                 vehicles = copy.copy(self.dict_sensors_[sensor]['vehicles'])
        #                 trajectory = list()
        #                 first = True
        #                 list_ind = list()
        #                 for veh in vehicles:
        #                     list_ind.append(self.P.nodes[veh]['index'])
        #                     if first:
        #                         trajectory = np.array(self.P.nodes[veh]['U_p'])
        #                         first = False
        #                     else:
        #                         new = np.array(self.P.nodes[veh]['U_p'])
        #                         trajectory = np.concatenate((trajectory, new), axis=1)
        #                 self.plot.plot_classic(mu, sigma, trajectory, sensor, list_ind)
        #         for part in self.pop:
        #             print(part.node, part, part.speed)
        #             print(self.P.nodes[part.node]['D_p'])

        return done

    def step_exploit(self, action):
        dis_steps = 0
        dist_ant = np.mean(self.distances)
        self.dist_pre = np.mean(self.distances)
        self.n_data = 0

        while dis_steps < 10:

            previous_dist = np.mean(self.distances)

            for part in self.pop:
                reach = self.P.nodes[part.node]['Reach']
                if self.weights_b:
                    action = self.P.nodes[part.node]['action_exploit']
                else:
                    action = self.action_exploit
                # print(reach)
                # if reach:
                self.toolbox.update(part.node, action[0], action[1], action[2], action[3], part)
                # else:
                #     self.toolbox.update(part.node, 0, 0, action[2], action[3], part)

            for part in self.pop:
                if self.method_pso == 'coupled':
                    self.local_best_coupled_exploit(part)
                elif self.method_pso == 'decoupled':
                    self.local_best_decoupled_exploit(part, dfirst=False)

                self.n_data += 1
                if self.n_data > self.vehicles - 1:
                    self.n_data = 0

            self.u_sf_exploit()

            if self.method_pso == 'coupled':
                self.global_best_coupled_exploit()
            elif self.method_pso == 'decoupled':
                self.global_best_decoupled_exploit()

            for part in self.pop:
                self.part_ant, self.distances = self.util.distance_part(self.g, self.n_data, part, self.part_ant,
                                                                        self.distances, self.array_part, dfirst=False)
                self.n_data += 1
                if self.n_data > self.vehicles - 1:
                    self.n_data = 0

            if (np.mean(self.distances) - self.last_sample) >= (np.mean(self.post_array) * self.lam):
                self.last_sample = np.mean(self.distances)
                self.take_measures_exploit()
                self.gp_update_exploit()

                self.gp_update()
                self.type_error = 'all_map_r2'
                self.calculate_error(False)
                self.type_error = 'all_map_mse'
                self.calculate_error(False)
                self.type_error = 'peaks'
                self.calculate_error(False)
                self.type_error = 'zones'
                self.calculate_error(False)

                if self.method_pso == 'coupled':
                    self.method_coupled_exploit()
                elif self.method_pso == 'decoupled':
                    self.method_decoupled_exploit()

            dis_steps = np.mean(self.distances) - dist_ant
            if np.mean(self.distances) == previous_dist:
                break
            self.g += 1

        if (np.mean(self.distances) >= self.exploitation_distance) or np.mean(self.distances) == self.dist_pre:
            done = True
        else:
            done = False

        return done

    def step(self, action):
        if self.stage == "exploration":
            self.explore = True
            done = self.step_explore(action)
            if (self.distances >= self.exploration_distance).any() or np.mean(self.distances) == self.dist_pre:
                # if max(self.max_un) <= 0.5 or np.mean(self.distances) == self.dist_pre or (np.mean(self.distances) >= self.exploitation_distance):
                self.mean_un.append(max(self.max_un))
                self.stage = "exploitation"
                # print('distances:', self.distances)
                # print(self.entropy[self.simulation]['rate'])
        elif self.stage == "exploitation":
            if self.explore:
                self.obtain_zones()
                self.explore = False
            done = self.step_exploit(self.action_exploit)
        elif self.stage == "no_exploitation":
            action = action
            self.exploration_distance = self.exploitation_distance
            done = self.step_explore(action)
            if (self.distances >= self.exploration_distance).any() or np.max(self.distances) == self.dist_pre:
                done = True
        if done:
            self.gp_update()
            self.type_error = 'all_map_r2'
            self.calculate_error(True)
            self.type_error = 'all_map_mse'
            self.calculate_error(True)
            self.type_error = 'peaks'
            self.calculate_error(True)
            self.type_error = 'zones'
            self.calculate_error(True)

            dist = np.arange(0, 210, 10)
            new_mse = np.interp(dist, self.mse_data[:, 1], self.mse_data[:, 0])
            new_r2 = np.interp(dist, self.r2_data[:, 1], self.r2_data[:, 0])
            new_cazmse = np.interp(dist, self.caz_mse_data[:, 1], self.caz_mse_data[:, 0])
            new_peakerror = np.interp(dist, self.peak_error_data[:, 1], self.peak_error_data[:, 0])
            new_peakerror[0] = 1
            new_mse[0] = 0
            new_r2[0] = 0
            new_cazmse[0] = 0
            if self.simulation == 1:
                self.r2_ = np.c_[dist, new_r2]
                self.mse_ = np.c_[dist, new_mse]
                self.caz_mse_ = np.c_[dist, new_cazmse]
                self.peak_error_ = np.c_[dist, new_peakerror]
            else:
                self.r2_ = np.c_[self.r2_, new_r2]
                self.mse_ = np.c_[self.mse_, new_mse]
                self.caz_mse_ = np.c_[self.caz_mse_, new_cazmse]
                self.peak_error_ = np.c_[self.peak_error_, new_peakerror]

            # df1 = {'Sensor': self.sensor, 'R2_sensor': self.r2_sensor, 'MSE_sensor': self.mse_sensor, 'Error_peak_sensor': self.error_peak_sensor,
            #        'Number': self.cant_sensor, 'w': self.w}
            # df1 = pd.DataFrame(data=df1)
            # df1.to_excel('../Test/MC_Sp/Exploit/Sensors_data_' + str(self.seed) + '.xlsx')
            # if self.simulation == 30:
            #     self.error_subfleet_3 = copy.copy(self.array_error)
            #     self.r2_subfleet_3 = copy.copy(self.array_r2)
            # if self.simulation > 2:
            #     for i, subfleet in enumerate(self.sub_fleets):
            #         sensors = self.s_sf[i]
            #         for s, sensor in enumerate(sensors):
            #             bench = copy.copy(self.dict_benchs_[sensor]['map_created'])
            #             self.plot.benchmark(bench, sensor)
            #             mu = copy.copy(self.dict_sensors_[sensor]['mu']['data'])
            #             sigma = copy.copy(self.dict_sensors_[sensor]['sigma']['data'])
            #             vehicles = copy.copy(self.dict_sensors_[sensor]['vehicles'])
            #             trajectory = list()
            #             first = True
            #             list_ind = list()
            #             for veh in vehicles:
            #                 list_ind.append(self.P.nodes[veh]['index'])
            #                 if first:
            #                     trajectory = np.array(self.P.nodes[veh]['U_p'])
            #                     first = False
            #                 else:
            #                     new = np.array(self.P.nodes[veh]['U_p'])
            #                     trajectory = np.concatenate((trajectory, new), axis=1)
            #             self.plot.plot_classic(mu, sigma, trajectory, sensor, list_ind)
            # for i, subfleet in enumerate(self.sub_fleets):
            #     for j, zo in enumerate(self.z_['subfleet%s' % i].keys()):
            #         zone = self.z_['subfleet%s' % i][zo]
            #         vehicles = zone['vehicles']
            #         sensors = zone['sensors'].keys()
            #         for s, sensor in enumerate(sensors):
            #             bench = copy.copy(self.dict_benchs_[sensor]['map_created'])
            #             self.plot.benchmark(bench, sensor)
            #             mu = copy.copy(zone['sensors'][sensor]['mu']['data'])
            #             sigma = copy.copy(zone['sensors'][sensor]['sigma']['data'])
            #             trajectory = list()
            #             first = True
            #             list_ind = list()
            #             for veh in vehicles:
            #                 list_ind.append(self.P.nodes[veh]['index'])
            #                 if first:
            #                     trajectory = np.array(self.P.nodes[veh]['U_p'])
            #                     first = False
            #                 else:
            #                     new = np.array(self.P.nodes[veh]['U_p'])
            #                     trajectory = np.concatenate((trajectory, new), axis=1)
            #             self.plot.plot_classic(mu, sigma, trajectory, sensor, list_ind)
        return done

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
        # self.plot.plot_curves(self.entropy, self.simulation)
        # self.plot.plot_inter(self.entropy, self.simulation)
        # data1 = {'R2': self.mean_error, 'Conf_R2': self.conf_error, 'Mean_Error': self.mean_peak_error, 'Conf_Error': self.conf_peak_error}
        # df = pd.DataFrame(data=data1)
        # df.to_excel('../Test/MC_Sp/Exploit/Main_results.xlsx')
        dist = np.arange(0, 200, 10)
        new_mse = np.interp(dist, self.mse_data[:, 1], self.mse_data[:, 0])
        new_r2 = np.interp(dist, self.r2_data[:, 1], self.r2_data[:, 0])
        new_cazmse = np.interp(dist, self.caz_mse_data[:, 1], self.caz_mse_data[:, 0])
        new_peakerror = np.interp(dist, self.peak_error_data[:, 1], self.peak_error_data[:, 0])
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
        df1 = pd.DataFrame(self.peak_error_)
        df1.to_excel('../Test/Results2/Error/1DCErrorAquaHet.xlsx')
        df2 = pd.DataFrame(self.caz_mse_)
        df2.to_excel('../Test/Results2/MSEAZ/1DCMSEAZAquaHet.xlsx')
        df3 = pd.DataFrame(self.mse_)
        df3.to_excel('../Test/Results2/MSEM/1DCMSEMAquaHet.xlsx')
        df4 = pd.DataFrame(self.r2_)
        df4.to_excel('../Test/Results2/R2M/1DCR2MAquaHet.xlsx')
        # df5 = pd.DataFrame(dist)
        # df5.to_excel('../Test/Results2/DistAquaHet.xlsx')

        print('R2:', np.mean(np.array(self.mean_error)), '+-', np.std(np.array(self.mean_error)) * 1.96)
        print('MSE:', np.mean(np.array(self.mean_mse_error)), '+-', np.std(np.array(self.mean_mse_error) * 1.96))
        print('Error:', np.mean(np.array(self.mean_peak_error)), '+-', np.std(np.array(self.mean_peak_error)) * 1.96)
        print('AZ:', np.mean(np.array(self.mean_az_mse)), '+-', np.std(np.array(self.mean_az_mse)) * 1.96)
