import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def error_plot():
    lawn = pd.read_excel('../Test/Results/Error/ErrorLawnmower.xlsx')
    lawn = lawn.to_numpy()
    lawn = lawn[:, 1]
    way = pd.read_excel('../Test/Results/Error/ErrorWaypoints.xlsx')
    way = way.to_numpy()
    way = way[:, 1]
    grid = pd.read_excel('../Test/Results/Error/ErrorGridPath.xlsx')
    grid = grid.to_numpy()
    grid = grid[:, 1]
    bo = pd.read_excel('../Test/Results/Error/ErrorBO.xlsx')
    bo = bo.to_numpy()
    bo = bo[:, 1]
    cpso = pd.read_excel('../Test/Results/Error/ErrorClassic.xlsx')
    cpso = cpso.to_numpy()
    cpso = cpso[:, 1]
    explore = pd.read_excel('../Test/Results/Error/ErrorExplore.xlsx')
    explore = explore.to_numpy()
    explore = explore[:, 1]
    exploit = pd.read_excel('../Test/Results/Error/ErrorExploit.xlsx')
    exploit = exploit.to_numpy()
    exploit = exploit[:, 1]
    epsilon = pd.read_excel('../Test/Results/Error/ErrorEpsilon.xlsx')
    epsilon = epsilon.to_numpy()
    epsilon = epsilon[:, 1]
    aqua = pd.read_excel('../Test/Results/Error/ErrorAquaFeL.xlsx')
    aqua = aqua.to_numpy()
    aqua = aqua[:, 1]

    my_dict_error = {'Lawnmower': lawn,
                     'Random Waypoints': way,
                     'Random Grip Path': grid,
                     'BO for M-ASVs': bo,
                     'Classic PSO': cpso,
                     'EGPSO Exploration': explore,
                     'EGPSO Exploitation': exploit,
                     'Epsilon Greedy': epsilon,
                     'AquaFeL-PSO': aqua}

    return my_dict_error


def mseaz_plot():
    lawn = pd.read_excel('../Test/Results/MSEAZ/MSEAZLawnmower.xlsx')
    lawn = lawn.to_numpy()
    lawn = lawn[:, 1]
    way = pd.read_excel('../Test/Results/MSEAZ/MSEAZWaypoints.xlsx')
    way = way.to_numpy()
    way = way[:, 1]
    grid = pd.read_excel('../Test/Results/MSEAZ/MSEAZGridPath.xlsx')
    grid = grid.to_numpy()
    grid = grid[:, 1]
    bo = pd.read_excel('../Test/Results/MSEAZ/MSEAZBO.xlsx')
    bo = bo.to_numpy()
    bo = bo[:, 1]
    cpso = pd.read_excel('../Test/Results/MSEAZ/MSEAZClassic.xlsx')
    cpso = cpso.to_numpy()
    cpso = cpso[:, 1]
    explore = pd.read_excel('../Test/Results/MSEAZ/MSEAZExplore.xlsx')
    explore = explore.to_numpy()
    explore = explore[:, 1]
    exploit = pd.read_excel('../Test/Results/MSEAZ/MSEAZExploit.xlsx')
    exploit = exploit.to_numpy()
    exploit = exploit[:, 1]
    epsilon = pd.read_excel('../Test/Results/MSEAZ/MSEAZEpsilon.xlsx')
    epsilon = epsilon.to_numpy()
    epsilon = epsilon[:, 1]
    aqua = pd.read_excel('../Test/Results/MSEAZ/MSEAZAquaFeL.xlsx')
    aqua = aqua.to_numpy()
    aqua = aqua[:, 1]

    my_dict_mseaz = {'Lawnmower': lawn,
                     'Random Waypoints': way,
                     'Random Grip Path': grid,
                     'BO for M-ASVs': bo,
                     'Classic PSO': cpso,
                     'EGPSO Exploration': explore,
                     'EGPSO Exploitation': exploit,
                     'Epsilon Greedy': epsilon,
                     'AquaFeL-PSO': aqua}

    return my_dict_mseaz


def msem_plot():
    lawn = pd.read_excel('../Test/Results/MSEM/MSEMLawnmower.xlsx')
    lawn = lawn.to_numpy()
    lawn = lawn[:, 1]
    way = pd.read_excel('../Test/Results/MSEM/MSEMWaypoints.xlsx')
    way = way.to_numpy()
    way = way[:, 1]
    grid = pd.read_excel('../Test/Results/MSEM/MSEMGridPath.xlsx')
    grid = grid.to_numpy()
    grid = grid[:, 1]
    bo = pd.read_excel('../Test/Results/MSEM/MSEMBO.xlsx')
    bo = bo.to_numpy()
    bo = bo[:, 1]
    cpso = pd.read_excel('../Test/Results/MSEM/MSEMClassic.xlsx')
    cpso = cpso.to_numpy()
    cpso = cpso[:, 1]
    explore = pd.read_excel('../Test/Results/MSEM/MSEMExplore.xlsx')
    explore = explore.to_numpy()
    explore = explore[:, 1]
    exploit = pd.read_excel('../Test/Results/MSEM/MSEMExploit.xlsx')
    exploit = exploit.to_numpy()
    exploit = exploit[:, 1]
    epsilon = pd.read_excel('../Test/Results/MSEM/MSEMEpsilon.xlsx')
    epsilon = epsilon.to_numpy()
    epsilon = epsilon[:, 1]
    aqua = pd.read_excel('../Test/Results/MSEM/MSEMAquaFeL.xlsx')
    aqua = aqua.to_numpy()
    aqua = aqua[:, 1]

    my_dict_msem = {'Lawnmower': lawn,
                     'Random Waypoints': way,
                     'Random Grip Path': grid,
                     'BO for M-ASVs': bo,
                     'Classic PSO': cpso,
                     'EGPSO Exploration': explore,
                     'EGPSO Exploitation': exploit,
                     'Epsilon Greedy': epsilon,
                     'AquaFeL-PSO': aqua}

    return my_dict_msem


if __name__ == '__main__':
    error = error_plot()
    map = msem_plot()
    az = mseaz_plot()
    n = 3
    x = np.arange(n)

    data = {'error': error,
            'az': az,
            'map': map}

    data_name = ['az', 'error', 'map']


    title = ['MSE AZ', 'Peak Error', 'MSE Map']

    for n in range(len(title)):
        fig, ax = plt.subplots()
        ax.boxplot(data[data_name[n]].values())
        ax.margins(y=0.05)
        ax.set_xticklabels(data[data_name[n]].keys(), rotation=30, ha='right', rotation_mode='anchor')
        ax.set_title(title[n])
