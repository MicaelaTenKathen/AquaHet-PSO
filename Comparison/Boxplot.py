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


method_ = True
sensor_ = False
method = ['CC', 'CS', 'DC', 'DS', 'LM']
dr2 = []
dmse = []
dmseca = []
derrpeak = []
sensor = [2, 3, 4, 5, 6, 1]


def data(direct):
    aqua1 = pd.read_excel(direct)
    dist = aqua1[0].to_numpy()
    aqua1 = aqua1.drop([0], axis=1).to_numpy()
    new_array = np.zeros([aqua1.shape[0], aqua1.shape[1] - 1])
    mean_ = np.zeros(dist.shape)
    std_ = np.zeros(dist.shape)
    for i in range(len(aqua1)):
        array = aqua1[i, :]
        b = np.delete(array, 0)
        mean_[i] = np.mean(b)
        std_[i] = np.std(b)
    return mean_, std_, dist


if method_ and not sensor_:
    for i in range(len(method)):
        dr2.append('../Test/Results2/R2M/1' + method[i] + 'R2MAquaHet.xlsx')
        dmse.append('../Test/Results2/MSEM/1' + method[i] + 'MSEMAquaHet.xlsx')
        dmseca.append('../Test/Results2/MSEAZ/1' + method[i] + 'MSEAZAquaHet.xlsx')
        derrpeak.append('../Test/Results2/Error/1' + method[i] + 'ErrorAquaHet.xlsx')
        label_ = ['AquaHet-PSO-C-GA', 'AquaHet-PSO-C', 'AquaHet-PSO-D-GA',
                  'AquaHet-PSO-D', 'Lawnmower']
        name = ['R2 Score - 4 or more sensors', 'MSE Map - 4 or more sensors', 'MSE CAZ - 4 or more sensors',
                'Error Peaks - 4 or more sensors']

elif not method_ and sensor_:
    for i in range(len(sensor)):
        label_ = ['2 sensors', '3 sensors', '4 sensors', '5 sensors', '6 sensors', '4 or more sensors']
        dr2.append('../Test/Results2/R2M/' + str(sensor[i]) + 'CCR2MAquaHet.xlsx')
        dmse.append('../Test/Results2/MSEM/' + str(sensor[i]) + 'CCMSEMAquaHet.xlsx')
        dmseca.append('../Test/Results2/MSEAZ/' + str(sensor[i]) + 'CCMSEAZAquaHet.xlsx')
        derrpeak.append('../Test/Results2/Error/' + str(sensor[i]) + 'CCErrorAquaHet.xlsx')
        name = ['R2 Score - AquaHet-PSO Coupled with GA', 'MSE Map - AquaHet-PSO Coupled with GA',
                'MSE CAZ - AquaHet-PSO Coupled with GA',
                'Error Peaks - AquaHet-PSO Coupled with GA']

if __name__ == '__main__':
    # error = error_plot()
    # map = msem_plot()
    # az = mseaz_plot()
    # n = 3
    # x = np.arange(n)
    adr = [dr2, dmse, derrpeak, dmseca]
    #
    # data = {'error': error,
    #         'az': az,
    #         'map': map}

    # data_name = ['az', 'error', 'map']

    title = ['MSE AZ', 'Peak Error', 'MSE Map']
    for j in range(len(adr)):
        direct = adr[j]
        fig, ax = plt.subplots()
        ax.set_title(name[j])
        dict_ = {}
        # with sns.axes_style("darkgrid"):
        for i in range(len(label_)):
            dict_[label_[i]] = data(direct[i])[0]

        print(dict_.values())
        ax.boxplot(dict_.values())
        ax.margins(y=0.05)
        ax.set_xticklabels(dict_.keys(), rotation=30, ha='right', rotation_mode='anchor')
            # ax.set_title(name[])
