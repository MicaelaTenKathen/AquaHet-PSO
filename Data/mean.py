import openpyxl
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


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


def comparison(meanbl, confbl, meanbg, confbg, meanun, confun, meancon, confcon, meancla, confcla, meanexr, confexr, meanext, confext, meanep, confep, mult):
    width = 0.2

    fig, ax = plt.subplots()
    ax.bar(mult - 4 * width, meanbl, width, color='mediumvioletred', yerr=confbl, label='Local Best Method', alpha=0.8)
    ax.bar(mult - 3 * width, meanbg, width, color='darkcyan', yerr=confbg, label='Global Best Method', alpha=1)
    ax.bar(mult - 2 * width, meanun, width, color='orange', yerr=confun, label='Uncertainty Method', alpha=1)
    ax.bar(mult - 1 * width, meancon, width, color='blueviolet', yerr=confcon, label='Contamination Method',
           alpha=0.9)
    ax.bar(mult, meancla, width, color='deepskyblue', yerr=confcla, label='Classic PSO',
           alpha=0.9)
    ax.bar(mult + 1 * width, meanexr, width, color='red', yerr=confexr, label='Enhanced GP-based PSO (Exploration)',
           alpha=0.7)
    ax.bar(mult + 2 * width, meanext, width, color='goldenrod', yerr=confext, label='Enhanced GP-based PSO (Exploitation)',
           alpha=0.8)
    ax.bar(mult + 3 * width, meanep, width, color='royalblue', yerr=confep, label='Epsilon Greedy Method',
           alpha=0.9)

    ax.set_ylabel('MSE', fontsize=12)
    ax.set_xlabel('Distance traveled (m)', fontsize=12)
    ticks_x = ticker.FuncFormatter(lambda x, pos: format(int(x * 1000), ','))
    ax.xaxis.set_major_formatter(ticks_x)

    #ax.legend(loc=1, fontsize=6)
    ax.grid(True)
    #plt.savefig("../Image/Error/MSE.png")

mse_mean_bl = list()
mse_conf_bl = list()

mse_mean_bg = list()
mse_conf_bg = list()

mse_mean_un = list()
mse_conf_un = list()

mse_mean_con = list()
mse_conf_con = list()

mse_mean_cla = list()
mse_conf_cla = list()

mse_mean_exr = list()
mse_conf_exr = list()

mse_mean_ext = list()
mse_conf_ext = list()

mse_mean_ep = list()
mse_conf_ep = list()

save = [25, 50, 75, 100, 125, 150]
# save = [125, 150]

for i in range(len(save)):
    mean, conf = data_val('../Test/Chapter/LocalBest/ALLCONError_'+str(save[i]) + '.xlsx')
    mse_mean_bl.append(mean)
    mse_conf_bl.append(conf)

    mean, conf = data_val('../Test/Chapter/GlobalBest/ALLCONError_' + str(save[i]) + '.xlsx')
    mse_mean_bg.append(mean)
    mse_conf_bg.append(conf)

    mean, conf = data_val('../Test/Chapter/Uncertainty/ALLCONError_' + str(save[i]) + '.xlsx')
    mse_mean_un.append(mean)
    mse_conf_un.append(conf)

    mean, conf = data_val('../Test/Chapter/Contamination/ALLCONError_' + str(save[i]) + '.xlsx')
    mse_mean_con.append(mean)
    mse_conf_con.append(conf)

    mean, conf = data_val('../Test/Chapter/ClassicPSO/ALLCONError_' + str(save[i]) + '.xlsx')
    mse_mean_cla.append(mean)
    mse_conf_cla.append(conf)

    mean, conf = data_val('../Test/Chapter/Rome/ALLCONError_' + str(save[i]) + '.xlsx')
    mse_mean_exr.append(mean)
    mse_conf_exr.append(conf)

    mean, conf = data_val('../Test/Chapter/Syracuse/ALLCONError_' + str(save[i]) + '.xlsx')
    mse_mean_ext.append(mean)
    mse_conf_ext.append(conf)

    mean, conf = data_val('../Test/Chapter/Epsilon/ALLCONError_' + str(save[i]) + '.xlsx')
    mse_mean_ep.append(mean)
    mse_conf_ep.append(conf)

mult = np.array([2.5, 5.0, 7.5, 10.0, 12.5, 15.0])
# mult= np.array([12.5, 15.0])

comparison(mse_mean_bl, mse_conf_bl, mse_mean_bg, mse_conf_bg, mse_mean_un, mse_conf_un, mse_mean_con, mse_conf_con,
           mse_mean_cla, mse_conf_cla, mse_mean_exr, mse_conf_exr, mse_mean_ext, mse_conf_ext, mse_mean_ep, mse_conf_ep, mult)

plt.show()