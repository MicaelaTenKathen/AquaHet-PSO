import openpyxl
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.patches import Rectangle


def zoom_outside(srcax, roi, dstax, label, color="red", linewidth=2, roiKwargs={}, arrowKwargs={}):
    '''Create a zoomed subplot outside the original subplot

    srcax: matplotlib.axes
        Source axis where locates the original chart
    dstax: matplotlib.axes
        Destination axis in which the zoomed chart will be plotted
    roi: list
        Region Of Interest is a rectangle defined by [xmin, ymin, xmax, ymax],
        all coordinates are expressed in the coordinate system of data
    roiKwargs: dict (optional)
        Properties for matplotlib.patches.Rectangle given by keywords
    arrowKwargs: dict (optional)
        Properties used to draw a FancyArrowPatch arrow in annotation
    '''
    roiKwargs = dict([("fill", False), ("linestyle", "dashed"),
                      ("color", color), ("label", label), ("linewidth", linewidth)]
                     + list(roiKwargs.items()))
    arrowKwargs = dict([("arrowstyle", "-"), ("color", color),
                        ("linewidth", linewidth)]
                       + list(arrowKwargs.items()))
    # draw a rectangle on original chart
    srcax.add_patch(Rectangle([roi[0], roi[1]], roi[2] - roi[0], roi[3] - roi[1],
                              **roiKwargs))
    # get coordinates of corners
    srcCorners = [[roi[0], roi[1]], [roi[0], roi[3]],
                  [roi[2], roi[1]], [roi[2], roi[3]]]


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


def comparison(meanbl, confbl, meanbg, confbg, meanun, confun, meancon, confcon):
               #meancla, confcla, meanexr, confexr, meanext, confext, meanep, confep, mult):
    width = 0.5

    fig, ax = plt.subplots(1, 2)
    ax[0].bar(mult - 1.5 * width, meanbl, width, color='dimgrey', yerr=confbl, label='Exploration 5km, '
                                                                                        'Exploitation 15km', alpha=0.8)
    ax[0].bar(mult - 0.5 * width, meanbg, width, color='mediumvioletred', yerr=confbg, label='Exploration 10km, Exploitation 10km', alpha=1)
    ax[0].bar(mult + 0.5 * width, meanun, width, color='darkgoldenrod', yerr=confun, label='Exploration 15km, Exploitation 5km', alpha=1)
    ax[0].bar(mult + 1.5 * width, meancon, width, color='teal', yerr=confcon, label='Only Exploration (Exploration 20km)',
           alpha=0.9)


    zoom_outside(ax[0], [13.8, -0.001, 21.2, 0.006], ax[1], 'Zoom')


#ax.bar(mult, meancla, width, color='deepskyblue', yerr=confcla, label='Classic PSO',
     #      alpha=0.9)
    #ax.bar(mult + 1 * width, meanexr, width, color='red', yerr=confexr, label='Enhanced GP-based PSO (Exploration)',
     #      alpha=0.7)
    #ax.bar(mult + 2 * width, meanext, width, color='goldenrod', yerr=confext, label='Enhanced GP-based PSO (Exploitation)',
     #      alpha=0.8)
    #ax.bar(mult + 3 * width, meanep, width, color='royalblue', yerr=confep, label='Epsilon Greedy Method',
     #      alpha=0.9)

    ax[0].set_ylabel('MSE', fontsize=12)
    ax[0].set_xlabel('Distance traveled (m)', fontsize=12)
    ticks_x = ticker.FuncFormatter(lambda x, pos: format(int(x * 1000), ','))
    ax[0].xaxis.set_major_formatter(ticks_x)

    ax[0].legend(loc=1, fontsize=10)
    ax[0].grid(True)
    ax[1].set_xlabel('Distance traveled (m)', fontsize=12)
    ticks_x = ticker.FuncFormatter(lambda x, pos: format(int(x * 1000), ','))
    ax[1].xaxis.set_major_formatter(ticks_x)

    ax[1].legend(loc=1, fontsize=10)
    ax[1].grid(True)
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

save = [25, 50, 75, 100, 125, 150, 175, 200]
# save = [125, 150]

for i in range(len(save)):
    #if save[i] <= 100:
    mean, conf = data_val('../Test/MultiPSO/4Vehicles/Explore50Exploit200/ALLCONError_'+str(save[i]) + '.xlsx')
    mse_mean_bl.append(mean)
    mse_conf_bl.append(conf)
    #else:
     #   mean, conf = data_val('../Test/MultiPSO/Explore50Exploit200/ALLCONError_100.xlsx')
      #  mse_mean_bl.append(mean)
       # mse_conf_bl.append(conf)

    mean, conf = data_val('../Test/MultiPSO/4Vehicles/Explore100Exploit300/ALLCONError_' + str(save[i]) + '.xlsx')
    mse_mean_bg.append(mean)
    mse_conf_bg.append(conf)

    mean, conf = data_val('../Test/MultiPSO/4Vehicles/Explore150Exploit300/ALLCONError_' + str(save[i]) + '.xlsx')
    mse_mean_un.append(mean)
    mse_conf_un.append(conf)

    mean, conf = data_val('../Test/MultiPSO/4Vehicles/Explore200Exploit300/ALLCONError_' + str(save[i]) + '.xlsx')
    mse_mean_con.append(mean)
    mse_conf_con.append(conf)

    #mean, conf = data_val('../Test/Chapter/ClassicPSO/ALLCONError_' + str(save[i]) + '.xlsx')
    #mse_mean_cla.append(mean)
    #mse_conf_cla.append(conf)

    #mean, conf = data_val('../Test/Chapter/Rome/ALLCONError_' + str(save[i]) + '.xlsx')
    #mse_mean_exr.append(mean)
    #mse_conf_exr.append(conf)

    #mean, conf = data_val('../Test/Chapter/Syracuse/ALLCONError_' + str(save[i]) + '.xlsx')
    #mse_mean_ext.append(mean)
    #mse_conf_ext.append(conf)

    #mean, conf = data_val('../Test/Chapter/Epsilon/ALLCONError_' + str(save[i]) + '.xlsx')
    #mse_mean_ep.append(mean)
    #mse_conf_ep.append(conf)

mult = np.array([2.5, 5.0, 7.5, 10.0, 12.5, 15.0, 17.5, 20])
# mult= np.array([12.5, 15.0])

comparison(mse_mean_bl, mse_conf_bl, mse_mean_bg, mse_conf_bg, mse_mean_un, mse_conf_un, mse_mean_con, mse_conf_con)
           #mse_mean_cla, mse_conf_cla, mse_mean_exr, mse_conf_exr, mse_mean_ext, mse_conf_ext, mse_mean_ep, mse_conf_ep, mult)

plt.show()