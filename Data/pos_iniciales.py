import warnings
from random import shuffle

import networkx as nx
import numpy as np
from deap import tools, creator, base, algorithms
from scipy.spatial.distance import euclidean as d

from Data.utils import Utils


def init_shuffle(icls, i_size):
    ind = list(range(i_size))
    shuffle(ind)
    return icls(ind)


def configurations(vs, ss):
    def obtain_subfleets():
        ss_sf = []
        cant = 0
        i = 0
        for node_p in set_p.nodes(data=True):
            j = 0
            for node_q in set_p.nodes(data=True):
                j += 1
                if i < j:
                    intersection = sorted(node_p[1]["S_p"].keys() & node_q[1]["S_p"].keys())
                    if node_p != node_q and len(intersection) > 0:
                        if not set_p.has_edge(node_p[0], node_q[0]):
                            set_p.add_edge(node_p[0], node_q[0], S_pq=intersection)
            i += 1
        sub = sorted(nx.connected_components(set_p))
        sub_fleets = [sorted(item) for item in sub]
        for i, sub_fleet in enumerate(sub_fleets):
            sub_fleet = sorted(sub_fleet)
            s_sf = set()
            for particle in sub_fleet:
                s_sf = s_sf | set_p.nodes[particle]['S_p'].keys()
            s_sf = sorted(s_sf)
            for j, sensor in enumerate(s_sf):
                list_vehicles = list()
                for particle in sub_fleet:
                    v_sensors = set_p.nodes[particle]['S_p'].keys()
                    for key in v_sensors:
                        if key == sensor:
                            list_vehicles.append(particle)
            ss_sf.append(s_sf)

            for particle in sub_fleet:
                sensors = set_p.nodes[particle]['S_p'].keys()
                for s, sensor in enumerate(sensors):
                    cant += 1
        return sub_fleets, ss_sf, cant

    set_p = nx.MultiGraph()
    for p_, (part, sen) in enumerate(zip(vs, ss)):
        set_p.add_node(part, S_p=dict.fromkeys(sen, []), index=p_, )

    return set_p, obtain_subfleets()


cantidad_de_subflotas = 1
vehicles, sensores_in_vehicles = Utils.obtain_prefabricated_vehicles()
p, (vsf, ssf, cant) = configurations(vehicles, sensores_in_vehicles)

edge_labels = [(u, v, len(d['S_pq'])) for u, v, d in p.edges(data=True)]

indices = list(p.nodes)
indiv = list(range(len(indices)))
posiciones_p = [[8, 56],
                [37, 16],
                [78, 81],
                [74, 124],
                [20, 40],
                [32, 92],
                [64, 60],
                [52, 10],
                [91, 113],
                [49, 51]]


def evaluate_disp(ind):
    dist = 0
    for vehi, vehj, w in edge_labels:
        id_pos_i = ind[indices.index(vehi)]
        id_pos_j = ind[indices.index(vehj)]
        dist += w*d(posiciones_p[id_pos_i], posiciones_p[id_pos_j])
    return dist,


def similar(ind1, ind2):
    for i in range(len(ind1)):
        if ind1[i] - ind2[i] != 0:
            return False
    return True


IND_SIZE = len(indices)
POP_SIZE = IND_SIZE * 10  # 10
CXPB, MUTPB, NGEN = 0.5, 0.5, IND_SIZE * 10
indmutpb = 0.05

creator.create('FitnessMax', base.Fitness, weights=(1.0,))
creator.create('Individual', list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("individual", init_shuffle, creator.Individual, i_size=len(indices))
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
for index, posicion in enumerate(hof[0]):
    print(f'el vehículo {indices[index]}, debe estar en la posición {posiciones_p[posicion]}')
