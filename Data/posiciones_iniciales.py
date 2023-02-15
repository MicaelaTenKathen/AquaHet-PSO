import warnings
from random import choice
from random import shuffle

import networkx as nx
import numpy as np
from deap import tools, creator, base, algorithms
from scipy.spatial.distance import euclidean as d


def obtain_prefabricated_vehicles(subfleet_number: int = 1):
    """
        Función que retorna una tupla de vehículos y sus sensores dependiendo del numero de subflotas que se requieren
    @param numero_de_subflotas: 1, 2, 3 o 4 subflotas
    @return: (list[str], list[list[str]]): tupla que corresponde a los vehículos (que pueden renombrarse a cualquier
                                           cosa) y a los sensores que tiene dicho vehículo según el numero_de_subflotas
                                           seleccionado.
    """
    if subfleet_number == 1:
        return choice([
            (['v1', 'v2', 'v3', 'v4', 'v5', 'v7', 'v8'],
             [['s1', 's6'], ['s3', 's1'], ['s3'], ['s4'], ['s1'], ['s4', 's3'], ['s6']]),
            (['v1', 'v2', 'v3', 'v5', 'v6', 'v7'],
             [['s1', 's6'], ['s8', 's5'], ['s5'], ['s5', 's1'], ['s6'],
              ['s8', 's6']]),
            (['v3', 'v6', 'v7', 'v8'],
             [['s8', 's4'], ['s7', 's8'], ['s7', 's4', 's2'], ['s8', 's4', 's2']]),
            (['v1', 'v2', 'v3', 'v4', 'v6', 'v7'],
             [['s1'], ['s4', 's1'], ['s2', 's4'], ['s4'], ['s2', 's1'], ['s7', 's1']]),
            (['v1', 'v3', 'v5', 'v6', 'v7'],
             [['s1'], ['s3'], ['s3', 's2', 's1'], ['s7', 's2', 's8'], ['s8', 's1']]),
            (['v2', 'v3', 'v4', 'v5', 'v7', 'v8'],
             [['s2'], ['s8', 's1'], ['s4'], ['s1', 's4'], ['s1', 's8'], ['s2', 's8']]),
            (['v1', 'v2', 'v6', 'v7', 'v8'],
             [['s1', 's3'], ['s5', 's8'], ['s2', 's6'], ['s2', 's5'], ['s8', 's1']]),
            (['v2', 'v4', 'v5', 'v7', 'v8'],
             [['s2', 's6', 's7'], ['s1', 's6'], ['s7'], ['s7', 's5'], ['s8', 's2']]),
            (['v2', 'v3', 'v4', 'v5', 'v6', 'v7', 'v8'],
             [['s4', 's5'], ['s4'], ['s5'], ['s8'], ['s1'], ['s8', 's1'], ['s8', 's4']]),
            (['v1', 'v2', 'v3', 'v5', 'v7', 'v8'],
             [['s7', 's4'], ['s1'], ['s4'], ['s6', 's4'], ['s7', 's6'], ['s1', 's6']])])
    # con 2 subflotas
    elif subfleet_number == 2:
        return choice([
            (['v1', 'v2', 'v3', 'v5', 'v9', 'v10', 'v12', 'v13'],
             [['s9'], ['s4'], ['s8'], ['s9', 's1'], ['s1'], ['s4'], ['s9'], ['s8', 's4']]),
            (['v1', 'v3', 'v4', 'v6', 'v8', 'v10'],
             [['s9'], ['s3', 's6', 's9'], ['s9', 's3'], ['s6'], ['s8'], ['s8', 's2']]),
            (['v2', 'v3', 'v4', 'v5'],
             [['s3', 's4', 's2'], ['s1', 's6', 's7'], ['s4', 's2'], ['s5', 's4']]),
            (['v1', 'v2', 'v3', 'v4', 'v5'], [['s1', 's4'], ['s2'], ['s3', 's2', 's5'], ['s4', 's6'], ['s5', 's7']]),
            (['v1', 'v2', 'v5', 'v7', 'v8'], [['s1'], ['s4', 's7', 's1'], ['s7', 's4'], ['s2', 's8'], ['s8', 's9']]),
            (['v1', 'v2', 'v3', 'v4', 'v5', 'v6', 'v7', 'v8'],
             [['s8'], ['s7'], ['s4'], ['s4', 's8'], ['s4'], ['s4', 's8'], ['s7'], ['s8']]),
            (['v1', 'v2', 'v3', 'v4', 'v5', 'v6', 'v7', 'v8'],
             [['s3'], ['s8'], ['s3'], ['s4', 's7'], ['s4', 's3'], ['s7'], ['s7'], ['s8']]),
            (['v1', 'v2', 'v5', 'v6', 'v7', 'v8'],
             [['s1', 's8'], ['s4', 's8'], ['s1', 's4'], ['s6'], ['s6', 's5'], ['s4']]),
            (['v1', 'v2', 'v4', 'v5', 'v6', 'v7', 'v8'],
             [['s1'], ['s2', 's6'], ['s1'], ['s3', 's4'], ['s2'], ['s3', 's1'], ['s2']]),
            (['v1', 'v2', 'v3', 'v4', 'v6', 'v7', 'v8'],
             [['s1'], ['s3'], ['s3', 's6', 's4'], ['s4'], ['s6'], ['s1'], ['s8', 's1']])])
    # con 3 subflotas
    elif subfleet_number == 3:
        return choice([
            (['v3', 'v4', 'v6', 'v7', 'v8', 'v9', 'v10'],
             [['s3', 's1'], ['s6', 's10'], ['s3'], ['s7'], ['s5', 's3'], ['s3'], ['s3']]),
            (['v1', 'v2', 'v3', 'v5', 'v6', 'v7', 'v9'],
             [['s1'], ['s1', 's5'], ['s10'], ['s5'], ['s6', 's8'], ['s5', 's10'], ['s9']]),
            (['v1', 'v3', 'v4', 'v5', 'v6', 'v7', 'v9', 'v10'],
             [['s2'], ['s10'], ['s7', 's10'], ['s5'], ['s10'], ['s10', 's8'], ['s7'], ['s10']]),
            (['v1', 'v2', 'v3', 'v4', 'v6', 'v7', 'v8', 'v9', 'v10'],
             [['s1'], ['s5'], ['s5'], ['s4'], ['s9'], ['s9'], ['s5'], ['s9', 's5'], ['s9']]),
            (['v1', 'v2', 'v4', 'v5', 'v6', 'v8', 'v9', 'v10'],
             [['s1', 's3'], ['s10'], ['s1'], ['s5'], ['s5'], ['s8', 's1'], ['s10'], ['s10']]),
            (['v1', 'v2', 'v3', 'v5', 'v6', 'v7', 'v9', 'v10'],
             [['s7'], ['s4'], ['s3'], ['s4'], ['s7'], ['s7'], ['s9', 's3'], ['s7', 's1']]),
            (['v1', 'v2', 'v3', 'v4', 'v5', 'v6', 'v7', 'v9', 'v10'],
             [['s5'], ['s10'], ['s5', 's8'], ['s2'], ['s10'], ['s10'], ['s8'], ['s8'], ['s10']]),
            (['v1', 'v2', 'v3', 'v4', 'v7', 'v8', 'v9', 'v10'],
             [['s1', 's2'], ['s2', 's10'], ['s6'], ['s4'], ['s1'], ['s6'], ['s4'], ['s10']]),
            (['v1', 'v2', 'v3', 'v5', 'v6', 'v7', 'v9', 'v10'],
             [['s1'], ['s2'], ['s3'], ['s1', 's6'], ['s4'], ['s4'], ['s4', 's3'], ['s2']]),
            (['v1', 'v2', 'v3', 'v4', 'v5', 'v6', 'v7', 'v8', 'v9', 'v10'],
             [['s7'], ['s7'], ['s8'], ['s9'], ['s9'], ['s8'], ['s8'], ['s8'], ['s8'], ['s7']])])
    # con 4 subflotas
    elif subfleet_number == 4:
        return choice([
            (['v1', 'v2', 'v3', 'v5', 'v6', 'v7', 'v8', 'v9', 'v10'],
             [['s1'], ['s1'], ['s9'], ['s1'], ['s6'], ['s7'], ['s9'], ['s9'], ['s9']]),
            (['v1', 'v4', 'v5', 'v6', 'v7', 'v8', 'v9', 'v10'],
             [['s1'], ['s4'], ['s1'], ['s3'], ['s2'], ['s1', 's5'], ['s10'], ['s10', 's2']]),
            (['v1', 'v2', 'v4', 'v5', 'v7', 'v8', 'v9'],
             [['s1'], ['s2'], ['s4', 's6', 's8'], ['s8'], ['s7'], ['s8'], ['s2']]),
            (['v1', 'v2', 'v3', 'v5', 'v6', 'v7', 'v8', 'v9', 'v10'],
             [['s4'], ['s6'], ['s3'], ['s4'], ['s6'], ['s6'], ['s8'], ['s3'], ['s8']]),
            (['v1', 'v2', 'v3', 'v4', 'v5', 'v6', 'v7', 'v8', 'v10'],
             [['s1'], ['s8', 's6'], ['s10'], ['s10', 's10'], ['s10'], ['s6'], ['s7'], ['s10'], ['s10']]),
            (['v1', 'v2', 'v3', 'v4', 'v6', 'v7', 'v8', 'v9', 'v10'],
             [['s1'], ['s3'], ['s3', 's1'], ['s4'], ['s9'], ['s3'], ['s9'], ['s9'], ['s10']]),
            (['v1', 'v2', 'v3', 'v4', 'v5', 'v6', 'v7', 'v8', 'v9', 'v10'],
             [['s1'], ['s6'], ['s3'], ['s3'], ['s6'], ['s6'], ['s10'], ['s3'], ['s3'], ['s10']]),
            (['v1', 'v2', 'v3', 'v4', 'v5', 'v7', 'v8', 'v9', 'v10'],
             [['s6', 's3'], ['s9'], ['s3'], ['s6'], ['s8'], ['s7'], ['s8'], ['s9'], ['s3']]),
            (['v2', 'v3', 'v4', 'v5', 'v6', 'v7', 'v8', 'v9', 'v10'],
             [['s2'], ['s2'], ['s9'], ['s10'], ['s6'], ['s7'], ['s7'], ['s9'], ['s6', 's7']]),
            (['v1', 'v2', 'v3', 'v4', 'v5', 'v7', 'v8', 'v9', 'v10'],
             [['s6', 's7'], ['s2'], ['s9'], ['s7'], ['s6'], ['s7'], ['s7'], ['s9'], ['s10']])])


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


cantidad_de_subflotas = 2
vehicles, sensores_in_vehicles = obtain_prefabricated_vehicles(cantidad_de_subflotas)
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
        dist += w * d(posiciones_p[id_pos_i], posiciones_p[id_pos_j])
    return dist,


def similar(ind1, ind2):
    return evaluate_disp(ind1) == evaluate_disp(ind2)


IND_SIZE = len(indices)
POP_SIZE = IND_SIZE * 10  # 10
CXPB, MUTPB, NGEN = 0.5, 0.5, 100
indmutpb = 0.05

creator.create('FitnessMax', base.Fitness, weights=(1.0,))
creator.create('Individual', list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("individual", init_shuffle, creator.Individual, i_size=len(posiciones_p))
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

import matplotlib.pyplot as plt

set_p = p
plt.subplot(121)
plt.imshow(np.zeros((150, 100)), cmap='twilight')
pos_vehicles = dict()
for index, posicion in enumerate(hof[0][:len(indices)]):
    print(f'el vehículo {indices[index]}, debe estar en la posición {posiciones_p[posicion]}')
    plt.text(posiciones_p[posicion][0], posiciones_p[posicion][1], indices[index])
    plt.plot(posiciones_p[posicion][0], posiciones_p[posicion][1], 'o')
    p_aux = posiciones_p[posicion].copy()
    p_aux[1] = 150 - p_aux[1]
    pos_vehicles[indices[index]] = np.array(p_aux)

plt.subplot(122)
# Subfleets
sub_fleets = nx.connected_components(set_p)
# Graficamos, para eso necesitamos diferentes colores para diferentes sf
colors = ["gold",
          "red",
          "limegreen",
          "darkorange",
          ]
# recorremos las subflotas
for i, sub_fleet in enumerate(sub_fleets):
    # SOLO PARA GRAFICAR creamos nuevos grafos! SOLO PARA GRAFICAR !!!!!!
    P_sf = set_p.subgraph(sub_fleet)
    # como estamos creamos grafos en cada iteracion, desplazamos los centros de las subflotas con
    # center = []
    # pos_vehicles = nx.circular_layout(P_sf, center=[i * 2.5, 0])
    print(pos_vehicles)
    axis = plt.gca()
    # graficamos y luego además agregamos el color de los nodos con node_color
    nx.draw(P_sf, pos_vehicles, with_labels=True, ax=axis, node_color=colors[i])
    # graficamos los edges
    edge_labels = dict([((u, v,), d['S_pq'])
                        for u, v, d in P_sf.edges(data=True)])
    nx.draw_networkx_edge_labels(P_sf, pos_vehicles, edge_labels=edge_labels, label_pos=0.2, font_size=12)
    plt.tight_layout()

plt.show(block=True)
