import openpyxl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns


def data_val(n):
    wb2 = openpyxl.load_workbook(n)
    data = list()
    h = 0
    hoja2 = wb2.active
    while True:
        h += 1
        cel2 = hoja2.cell(row=1, column=h)
        l = cel2.value
        if l is None:
            del (data[-1])
            break
        else:
            data.append(cel2.value)
    mean = np.mean(data)
    std = np.std(data)
    conf = std * 1.96
    return mean, std


# dr2 = [ '../Test/Results2/R2M/2R2MAquaHet.xlsx', '../Test/Results2/R2M/3R2MAquaHet.xlsx', '../Test/Results2/R2M/4R2MAquaHet.xlsx',
#           '../Test/Results2/R2M/5R2MAquaHet.xlsx', '../Test/Results2/R2M/6R2MAquaHet.xlsx', '../Test/Results2/R2M/1R2MAquaHet.xlsx']
#
# dmse = [ '../Test/Results2/MSEM/2MSEMAquaHet.xlsx', '../Test/Results2/MSEM/3MSEMAquaHet.xlsx', '../Test/Results2/MSEM/4MSEMAquaHet.xlsx',
#           '../Test/Results2/MSEM/5MSEMAquaHet.xlsx', '../Test/Results2/MSEM/6MSEMAquaHet.xlsx', '../Test/Results2/MSEM/1MSEMAquaHet.xlsx']
#
# dmseca = [ '../Test/Results2/MSEAZ/2MSEAZAquaHet.xlsx', '../Test/Results2/MSEAZ/3MSEAZAquaHet.xlsx', '../Test/Results2/MSEAZ/4MSEAZAquaHet.xlsx',
#           '../Test/Results2/MSEAZ/5MSEAZAquaHet.xlsx', '../Test/Results2/MSEAZ/6MSEAZAquaHet.xlsx', '../Test/Results2/MSEAZ/1MSEAZAquaHet.xlsx']
#
# derrpeak = [ '../Test/Results2/Error/2ErrorAquaHet.xlsx', '../Test/Results2/Error/3ErrorAquaHet.xlsx', '../Test/Results2/Error/4ErrorAquaHet.xlsx',
#           '../Test/Results2/Error/5ErrorAquaHet.xlsx', '../Test/Results2/Error/6ErrorAquaHet.xlsx', '../Test/Results2/Error/1ErrorAquaHet.xlsx']


method = ['CC', 'CS', 'DC', 'DS', 'LM']



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

# name = ['AquaHet-PSO Coupled with GA - R2 Map', 'AquaHet-PSO Coupled with GA - MSE Map', 'AquaHet-PSO Coupled with GA - MSE CAZ', 'AquaHet-PSO Coupled with GA - Error Peaks']
sensor = [2, 3, 4, 5, 6, 1]
name = ['R2 Map (4 or more sensors)', 'MSE Map (4 or more sensors)', 'MSE CAZ (4 or more sensors)', 'Error Peaks (4 or more sensors)']


dr2 = []
dmse = []
dmseca = []
derrpeak = []

method_ = True
sensor_ = False

if method_ and not sensor_:
    for i in range(len(method)):
        dr2.append('../Test/Results2/R2M/1' + method[i] + 'R2MAquaHet.xlsx')
        dmse.append('../Test/Results2/MSEM/1' + method[i] + 'MSEMAquaHet.xlsx')
        dmseca.append('../Test/Results2/MSEAZ/1' + method[i] + 'MSEAZAquaHet.xlsx')
        derrpeak.append('../Test/Results2/Error/1' + method[i] + 'ErrorAquaHet.xlsx')
        label_ = ['AquaHet-PSO Coupled with GA', 'AquaHet-PSO Coupled without GA', 'AquaHet-PSO Decoupled with GA',
                  'AquaHet-PSO Decoupled without GA', 'Lawnmower']
elif not method_ and sensor_:
    for i in range(len(sensor)):
        label_ = ['2 sensors', '3 sensors', '4 sensors', '5 sensors', '6 sensors', '4 or more sensors']
        dr2.append('../Test/Results2/R2M/' + str(sensor[i]) + 'CCR2MAquaHet.xlsx')
        dmse.append('../Test/Results2/MSEM/' + str(sensor[i]) + 'CCMSEMAquaHet.xlsx')
        dmseca.append('../Test/Results2/MSEAZ/' + str(sensor[i]) + 'CCMSEAZAquaHet.xlsx')
        derrpeak.append('../Test/Results2/Error/' + str(sensor[i]) + 'CCErrorAquaHet.xlsx')


adr = [dr2, dmse, dmseca, derrpeak]

for j in range(len(adr)):
    direct = adr[j]
    fig, ax = plt.subplots()
    ax.set_title(name[j])
    clrs = sns.color_palette("husl", 6)
    # with sns.axes_style("darkgrid"):
    for i in range(len(label_)):
        dirc = data(direct[i])
        meanst = np.array(dirc[0], dtype=np.float64)
        sdt = np.array(dirc[1], dtype=np.float64)
        ax.plot(dirc[2], meanst, label=label_[i])
        ax.fill_between(dirc[2], meanst - sdt, meanst + sdt, alpha=0.3, facecolor=clrs[i])
        # ax.set_ylim([0, 10^-1])
        ax.set_xlim([10, 200])
        ticks_x = ticker.FuncFormatter(lambda x, pos: format(int(x * 100), ','))
        ax.xaxis.set_major_formatter(ticks_x)
        ax.set_xlabel("meters")
        ax.legend()
    # ticks_y = ticker.FuncFormatter(lambda x, pos: format(int(x * 100), ','))
    # ax.yaxis.set_major_formatter(ticks_y)

# ax.set_yscale('log')
