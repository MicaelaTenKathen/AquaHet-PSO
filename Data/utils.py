import math
import random

import numpy as np
import openpyxl


class Utils:
    def __init__(self, vehicles):
        self.MSE_data = list()
        self.it = list()
        self.vehicles = vehicles

    def distance_part(self, g, n_data, part, part_ant, distances, array_part, dfirst=False):

        """
        Calculates the distance between (x,y)^t and (x,y)^(t-1).
        """
        if dfirst:
            part_ant[0, 2 * n_data] = part[0]
            part_ant[0, 2 * n_data + 1] = part[1]
            # if n_data == 1:
            #    part_ant[0, 0] = part[0]
            #   part_ant[0, 1] = part[1]
            # if n_data == 2:
            #   part_ant[0, 2] = part[0]
            #  part_ant[0, 3] = part[1]
            # if n_data == 3:
            #   part_ant[0, 4] = part[0]
            #  part_ant[0, 5] = part[1]
            # if n_data == 4:
            #   part_ant[0, 6] = part[0]
            #  part_ant[0, 7] = part[1]
        else:
            array_part[0, 2 * n_data] = part[0]
            array_part[0, 2 * n_data + 1] = part[1]
            distances[n_data] = math.sqrt(
                (array_part[0, 2 * n_data] - part_ant[g, 2 * n_data]) ** 2 + (
                        array_part[0, 2 * n_data + 1] - part_ant[g, 2 * n_data + 1])
                ** 2) + distances[n_data]

            # if n_data == 1:
            #   array_part[0, 0] = part[0]
            #  array_part[0, 1] = part[1]
            # distances[0] = math.sqrt(
            #    (array_part[0, 0] - part_ant[g, 0]) ** 2 + (array_part[0, 1] - part_ant[g, 1])
            #   ** 2) + distances[0]
            # elif n_data == 2:
            #   array_part[0, 2] = part[0]
            #  array_part[0, 3] = part[1]
            # distances[1] = math.sqrt(
            #    (array_part[0, 2] - part_ant[g, 2]) ** 2 + (array_part[0, 3] - part_ant[g, 3])
            #   ** 2) + distances[1]
            # elif n_data == 3:
            #   array_part[0, 4] = part[0]
            #  array_part[0, 5] = part[1]
            # distances[2] = math.sqrt(
            #    (array_part[0, 4] - part_ant[g, 4]) ** 2 + (array_part[0, 5] - part_ant[g, 5])
            #   ** 2) + distances[2]
            # elif n_data == 4:
            #   array_part[0, 6] = part[0]
            #  array_part[0, 7] = part[1]
            # distances[3] = math.sqrt(
            #    (array_part[0, 6] - part_ant[g, 6]) ** 2 + (array_part[0, 7] - part_ant[g, 7])
            #   ** 2) + distances[3]
            if n_data == self.vehicles - 1:
                part_ant = np.append(part_ant, array_part, axis=0)

        return part_ant, distances

    def mse(self, g, y_data, mu_data):

        """
        Calculates the mean square error (MSE) between the collected data and the estimated data.
        """

        total_suma = 0
        y_array = np.array(y_data)
        mu_array = np.array(mu_data)
        for i in range(len(mu_array)):
            total_suma = ((float(y_array[i]) - float(mu_array[i])) ** 2) + total_suma
        MSE = total_suma / y_data.shape[0]
        self.MSE_data.append(MSE)
        self.it.append(g)
        return self.MSE_data, self.it

    def savexlsx(self, sigma_data, mu_data, seed, file):

        """
        Save the data in excel documents.
        """

        for i in range(len(mu_data)):
            mu_data[i] = float(mu_data[i])

        wb1 = openpyxl.Workbook()
        hoja1 = wb1.active
        hoja1.append(self.MSE_data)
        wb1.save('Test/' + file + '/ALLCONError' + str(seed[0]) + '.xlsx')

        wb2 = openpyxl.Workbook()
        hoja2 = wb2.active
        hoja2.append(sigma_data)
        wb2.save('Test/' + file + '/ALLCONSigma' + str(seed[0]) + '.xlsx')

        wb3 = openpyxl.Workbook()
        hoja3 = wb3.active
        hoja3.append(mu_data)
        wb3.save('Test/' + file + '/ALLCONMu' + str(seed[0]) + '.xlsx')

        wb4 = openpyxl.Workbook()
        hoja4 = wb4.active
        hoja4.append(list(self.distances))
        wb4.save('Test/' + file + '/ALLCONDistance' + str(seed[0]) + '.xlsx')

        wb5 = openpyxl.Workbook()
        hoja5 = wb5.active
        hoja5.append(self.it)
        wb5.save('Test/' + file + '/ALLCONData' + str(seed[0]) + '.xlsx')


def obtener_vehiculos_prefabricados(numero_de_subflotas: int = 1):
    """
        Función que retorna una tupla de vehículos y sus sensores dependiendo del numero de subflotas que se requieren

    @param numero_de_subflotas: 1, 2, 3 o 4 subflotas
    @return: (list[str], list[list[str]]): tupla que corresponde a los vehículos (que pueden renombrarse a cualquier
                                           cosa) y a los sensores que tiene dicho vehículo según el numero_de_subflotas
                                           seleccionado.
    """
    if numero_de_subflotas == 1:
        return random.choice([
            (['v1', 'v2', 'v3', 'v4', 'v5', 'v7', 'v8'],
             [['s1', 's6'], ['s3', 's1'], ['s3'], ['s4'], ['s6'], ['s4', 's3'], ['s6']]),
            (['v1', 'v2', 'v3', 'v5', 'v6', 'v7'],
             [['s1', 's6', 's6'], ['s8', 's5'], ['s5'], ['s5', 's1', 's6', 's6'], ['s6'],
              ['s6', 's6']]),
            (['v3', 'v6', 'v7', 'v8'],
             [['s8', 's4'], ['s7'], ['s7', 's8', 's4', 's2'], ['s8', 's4', 's2']]),
            (['v1', 'v2', 'v3', 'v4', 'v6', 'v7'],
             [['s1'], ['s4', 's1'], ['s2', 's4', 's2', 's2'], ['s4'], ['s7', 's1'], ['s7', 's1']]),
            (['v1', 'v3', 'v5', 'v6', 'v7'],
             [['s1'], ['s3'], ['s3', 's8', 's2', 's3'], ['s1', 's7', 's2', 's8'], ['s3']]),
            (['v2', 'v3', 'v4', 'v5', 'v7', 'v8'],
             [['s2'], ['s8', 's1', 's8'], ['s2'], ['s1', 's4'], ['s1', 's8'], ['s2', 's4', 's4']]),
            (['v1', 'v2', 'v6', 'v7', 'v8'],
             [['s1', 's3'], ['s5', 's5'], ['s8', 's2', 's6'], ['s2', 's5'], ['s8', 's1']]),
            (['v2', 'v4', 'v5', 'v7', 'v8'],
             [['s2', 's6', 's7'], ['s1', 's6'], ['s7'], ['s7', 's5'], ['s8', 's2']]),
            (['v2', 'v3', 'v4', 'v5', 'v6', 'v7', 'v8'],
             [['s4', 's4'], ['s4'], ['s5'], ['s4'], ['s5'], ['s8', 's1', 's5'], ['s8', 's4']]),
            (['v1', 'v2', 'v3', 'v5', 'v7', 'v8'],
             [['s7', 's6', 's6', 's6'], ['s4'], ['s4'], ['s6', 's4'], ['s7', 's6'], ['s7', 's6']])])
    # con 2 subflotas
    elif numero_de_subflotas == 2:
        return random.choice([
            (['v1', 'v2', 'v3', 'v5', 'v9', 'v10', 'v12', 'v13'],
             [['s9'], ['s4'], ['s9'], ['s9'], ['s9'], ['s9'], ['s9'], ['s8', 's4']]),
            (['v1', 'v3', 'v4', 'v6', 'v8', 'v10'],
             [['s9'], ['s3', 's9', 's9'], ['s9', 's3'], ['s6'], ['s8'], ['s3', 's3', 's6']]),
            (['v2', 'v3', 'v4', 'v5'],
             [['s2', 's2', 's4', 's2'], ['s1'], ['s4', 's2', 's3', 's4'], ['s5', 's4', 's2', 's3', 's4']]),
            (['v1', 'v2', 'v3', 'v4', 'v5'], [['s1', 's4'], ['s2'], ['s3', 's2', 's5'], ['s4', 's6'], ['s5', 's7']]),
            (['v1', 'v2', 'v5', 'v7', 'v8'], [['s1'], ['s4', 's7', 's1'], ['s7', 's7', 's4'], ['s7', 's4'], ['s8']]),
            (['v1', 'v2', 'v3', 'v4', 'v5', 'v6', 'v7', 'v8'],
             [['s4'], ['s4'], ['s4'], ['s4', 's8', 's4'], ['s4'], ['s4', 's8'], ['s7'], ['s8']]),
            (['v1', 'v2', 'v3', 'v4', 'v5', 'v6', 'v7', 'v8'],
             [['s3'], ['s8'], ['s3'], ['s4', 's7'], ['s4', 's3'], ['s7'], ['s7'], ['s8']]),
            (['v1', 'v2', 'v5', 'v6', 'v7', 'v8'], [['s1'], ['s4'], ['s1', 's4'], ['s6'], ['s6', 's5'], ['s4']]),
            (['v1', 'v2', 'v4', 'v5', 'v6', 'v7', 'v8'],
             [['s1'], ['s2', 's6'], ['s1'], ['s3'], ['s2'], ['s3', 's1'], ['s2']]),
            (['v1', 'v2', 'v3', 'v4', 'v6', 'v7', 'v8'],
             [['s1'], ['s3'], ['s3', 's6', 's4'], ['s4'], ['s6'], ['s1'], ['s8', 's1']])])
    # con 3 subflotas
    elif numero_de_subflotas == 3:
        return random.choice([
            (['v3', 'v4', 'v6', 'v7', 'v8', 'v9', 'v10'],
             [['s3', 's1'], ['s6', 's10'], ['s3'], ['s7'], ['s5', 's3'], ['s3'], ['s3']]),
            (['v1', 'v2', 'v3', 'v5', 'v6', 'v7', 'v9'],
             [['s1'], ['s1', 's5'], ['s10'], ['s5'], ['s6'], ['s5', 's10'], ['s9']]),
            (['v1', 'v3', 'v4', 'v5', 'v6', 'v7', 'v9', 'v10'],
             [['s2'], ['s10'], ['s7', 's10'], ['s5'], ['s10'], ['s10', 's8'], ['s7'], ['s10']]),
            (['v1', 'v2', 'v3', 'v4', 'v6', 'v7', 'v8', 'v9', 'v10'],
             [['s1'], ['s5'], ['s5'], ['s4'], ['s9'], ['s9'], ['s5'], ['s9', 's5'], ['s9']]),
            (['v1', 'v2', 'v4', 'v5', 'v6', 'v8', 'v9', 'v10'],
             [['s1', 's3'], ['s10'], ['s1'], ['s5'], ['s5'], ['s8', 's1'], ['s10'], ['s10']]),
            (['v1', 'v2', 'v3', 'v5', 'v6', 'v7', 'v9', 'v10'],
             [['s7'], ['s4'], ['s3'], ['s4'], ['s7'], ['s7'], ['s9', 's3'], ['s7']]),
            (['v1', 'v2', 'v3', 'v4', 'v5', 'v6', 'v7', 'v9', 'v10'],
             [['s5'], ['s10'], ['s5', 's8'], ['s2'], ['s10'], ['s10'], ['s8'], ['s8'], ['s10']]),
            (['v1', 'v2', 'v3', 'v4', 'v7', 'v8', 'v9', 'v10'],
             [['s1', 's2'], ['s2', 's10'], ['s6'], ['s4'], ['s1'], ['s6'], ['s4'], ['s10']]),
            (['v1', 'v2', 'v3', 'v5', 'v6', 'v7', 'v9', 'v10'],
             [['s1'], ['s2'], ['s3'], ['s1'], ['s4'], ['s4'], ['s4', 's3'], ['s2']]),
            (['v1', 'v2', 'v3', 'v4', 'v5', 'v6', 'v7', 'v8', 'v9', 'v10'],
             [['s8'], ['s8'], ['s8'], ['s8'], ['s9'], ['s8'], ['s8'], ['s8', 's8'], ['s8'], ['s7']])])
    # con 4 subflotas
    elif numero_de_subflotas == 4:
        return random.choice([
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


if __name__ == '__main__':
    import networkx as nx
    import matplotlib.pyplot as plt

    cantidad_de_subflotas = 4


    def test_configuration(vs, ss):
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
        for p, (part, sen) in enumerate(zip(vs, ss)):
            set_p.add_node(part, S_p=dict.fromkeys(sen, []), index=p, )

        return set_p, obtain_subfleets()


    vehicles, sensores_in_vehicles = obtener_vehiculos_prefabricados(cantidad_de_subflotas)

    p, (vsf, ssf, cant) = test_configuration(vehicles, sensores_in_vehicles)

    print(f'{vehicles=}')
    print(f'{sensores_in_vehicles=}')
    print(f'numero_total_de_sensores={cant}')

    sub_fleets = nx.connected_components(p)
    # Graficamos, para eso necesitamos diferentes colores para diferentes sf
    colors = ["gold",
              "red",
              "limegreen",
              "darkorange",
              ]

    # recorremos las subflotas
    for i, sub_fleet in enumerate(sub_fleets):
        # SOLO PARA GRAFICAR creamos nuevos grafos! SOLO PARA GRAFICAR !!!!!!
        P_sf = p.subgraph(sub_fleet)

        all_S_p = nx.get_node_attributes(P_sf, 'S_p')
        labels = dict()
        for vehicle in all_S_p:
            labels[vehicle] = f"{vehicle} \n S({vehicle}): {list(all_S_p[vehicle].keys())}"
        # como estamos creando grafos en cada iteracion, desplazamos los centros de las subflotas con
        # center = []
        pos_vehicles = nx.circular_layout(P_sf, center=[i * 2.5, 0])
        axis = plt.gca()
        # graficamos y luego además agregamos el color de los nodos con node_color
        nx.draw(P_sf, pos_vehicles, labels=labels, ax=axis, node_color=colors[i], node_size=5000)
        # graficamos los edges
        edge_labels = dict([((u, v,), d['S_pq'])
                            for u, v, d in P_sf.edges(data=True)])
        nx.draw_networkx_edge_labels(P_sf, pos_vehicles, edge_labels=edge_labels, label_pos=0.3, font_size=15)
        plt.tight_layout()
    plt.show()
