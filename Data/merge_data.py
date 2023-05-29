import numpy as np
import pandas as pd
import math
from functools import reduce
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from bioinfokit.analys import stat
import matplotlib.pyplot as plt


def error():
    lawn1 = pd.read_excel('../Test/Results/Error/ErrorLawnmower.xlsx')
    lawn = lawn1.to_numpy()
    lawn = lawn[:, 1]
    # way1 = pd.read_excel('../Test/Results/Error/ErrorWaypoints.xlsx')
    # way = way1.to_numpy()
    # way = way[:, 1]
    # grid1 = pd.read_excel('../Test/Results/Error/ErrorGridPath.xlsx')
    # grid = grid1.to_numpy()
    # grid = grid[:, 1]
    # bo1 = pd.read_excel('../Test/Results/Error/ErrorBO.xlsx')
    # bo = bo1.to_numpy()
    # bo = bo[:, 1]
    # cpso1 = pd.read_excel('../Test/Results/Error/ErrorClassic.xlsx')
    # cpso = cpso1.to_numpy()
    # cpso = cpso[:, 1]
    # explore1 = pd.read_excel('../Test/Results/Error/ErrorExplore.xlsx')
    # explore = explore1.to_numpy()
    # explore = explore[:, 1]
    # exploit1 = pd.read_excel('../Test/Results/Error/ErrorExploit.xlsx')
    # exploit = exploit1.to_numpy()
    # exploit = exploit[:, 1]
    # epsilon1 = pd.read_excel('../Test/Results/Error/ErrorEpsilon.xlsx')
    # epsilon = epsilon1.to_numpy()
    # epsilon = epsilon[:, 1]
    aqua1 = pd.read_excel('../Test/Results/Error/ErrorAquaHet.xlsx')
    aqua = aqua1.to_numpy()
    aqua = aqua[:, 1]
    return aqua1, lawn1#, way1, grid1, bo1, cpso1, explore1, exploit1, epsilon1


def mseaz():
    lawn1 = pd.read_excel('../Test/Results/MSEAZ/MSEAZLawnmower.xlsx')
    lawn = lawn1.to_numpy()
    # lawn = lawn[:, 1]
    # way1 = pd.read_excel('../Test/Results/MSEAZ/MSEAZWaypoints.xlsx')
    # way = way1.to_numpy()
    # way = way[:, 1]
    # grid1 = pd.read_excel('../Test/Results/MSEAZ/MSEAZGridPath.xlsx')
    # grid = grid1.to_numpy()
    # grid = grid[:, 1]
    # bo1 = pd.read_excel('../Test/Results/MSEAZ/MSEAZBO.xlsx')
    # bo = bo1.to_numpy()
    # bo = bo[:, 1]
    # cpso1 = pd.read_excel('../Test/Results/MSEAZ/MSEAZClassic.xlsx')
    # cpso = cpso1.to_numpy()
    # cpso = cpso[:, 1]
    # explore1 = pd.read_excel('../Test/Results/MSEAZ/MSEAZExplore.xlsx')
    # explore = explore1.to_numpy()
    # explore = explore[:, 1]
    # exploit1 = pd.read_excel('../Test/Results/MSEAZ/MSEAZExploit.xlsx')
    # exploit = exploit1.to_numpy()
    # exploit = exploit[:, 1]
    # epsilon1 = pd.read_excel('../Test/Results/MSEAZ/MSEAZEpsilon.xlsx')
    # epsilon = epsilon1.to_numpy()
    # epsilon = epsilon[:, 1]
    aqua1 = pd.read_excel('../Test/Results/MSEAZ/MSEAZAquaHet.xlsx')
    aqua = aqua1.to_numpy()
    aqua = aqua[:, 1]
    return aqua1, lawn1#, way1, grid1, bo1, cpso1, explore1, exploit1, epsilon1


def msem():
    lawn1= pd.read_excel('../Test/Results/MSEM/MSEMLawnmower.xlsx')
    lawn = lawn1.to_numpy()
    lawn = lawn[:, 1]
    # way1 = pd.read_excel('../Test/Results/MSEM/MSEMWaypoints.xlsx')
    # way = way1.to_numpy()
    # way = way[:, 1]
    # grid1 = pd.read_excel('../Test/Results/MSEM/MSEMGridPath.xlsx')
    # grid = grid1.to_numpy()
    # grid = grid[:, 1]
    # bo1 = pd.read_excel('../Test/Results/MSEM/MSEMBO.xlsx')
    # bo = bo1.to_numpy()
    # bo = bo[:, 1]
    # cpso1 = pd.read_excel('../Test/Results/MSEM/MSEMClassic.xlsx')
    # cpso = cpso1.to_numpy()
    # cpso = cpso[:, 1]
    # explore1 = pd.read_excel('../Test/Results/MSEM/MSEMExplore.xlsx')
    # explore = explore1.to_numpy()
    # explore = explore[:, 1]
    # exploit1 = pd.read_excel('../Test/Results/MSEM/MSEMExploit.xlsx')
    # exploit = exploit1.to_numpy()
    # exploit = exploit[:, 1]
    # epsilon1 = pd.read_excel('../Test/Results/MSEM/MSEMEpsilon.xlsx')
    # epsilon = epsilon1.to_numpy()
    # epsilon = epsilon[:, 1]
    aqua1 = pd.read_excel('../Test/Results/MSEM/MSEMAquaHet.xlsx')
    aqua = aqua1.to_numpy()
    aqua = aqua[:, 1]
    return aqua1, lawn1#, way1, grid1, bo1, cpso1, explore1, exploit1, epsilon1

def r2m():
    lawn1= pd.read_excel('../Test/Results/R2M/R2MLawnmower.xlsx')
    lawn = lawn1.to_numpy()
    lawn = lawn[:, 1]
    # way1 = pd.read_excel('../Test/Results/R2M/R2MWaypoints.xlsx')
    # way = way1.to_numpy()
    # way = way[:, 1]
    # grid1 = pd.read_excel('../Test/Results/R2M/R2MGridPath.xlsx')
    # grid = grid1.to_numpy()
    # grid = grid[:, 1]
    # bo1 = pd.read_excel('../Test/Results/R2M/R2MBO.xlsx')
    # bo = bo1.to_numpy()
    # bo = bo[:, 1]
    # cpso1 = pd.read_excel('../Test/Results/R2M/R2MClassic.xlsx')
    # cpso = cpso1.to_numpy()
    # cpso = cpso[:, 1]
    # explore1 = pd.read_excel('../Test/Results/R2M/R2MExplore.xlsx')
    # explore = explore1.to_numpy()
    # explore = explore[:, 1]
    # exploit1 = pd.read_excel('../Test/Results/R2M/R2MExploit.xlsx')
    # exploit = exploit1.to_numpy()
    # exploit = exploit[:, 1]
    # epsilon1 = pd.read_excel('../Test/Results/R2M/R2MEpsilon.xlsx')
    # epsilon = epsilon1.to_numpy()
    # epsilon = epsilon[:, 1]
    aqua1 = pd.read_excel('../Test/Results/R2M/R2MAquaHet.xlsx')
    aqua = aqua1.to_numpy()
    aqua = aqua[:, 1]
    return aqua1, lawn1#, way1, grid1, bo1, cpso1, explore1, exploit1, epsilon1


def table_data(dfs):
    nan_value = 0
    data = reduce(lambda df_left, df_right: pd.merge(df_left, df_right, left_index=True, right_index=True, how='outer'),
                      dfs).fillna(nan_value)

    data.columns = ['A', 'AFeL', 'A', 'LM']#, 'A', 'WAY', 'A', 'GRID', 'A', 'BO', 'A', 'CPSO', 'A', 'ERPSO', 'A', 'EXPSO', 'A', 'EPS']
    # data_frame = pd.DataFrame(data, columns=['LM', 'WAY', 'GRID', 'CPSO', 'ERPSO', 'EXPSO', 'EPS', 'AFeL'])
    data = data.drop(['A'], axis=1)
    df_melt = pd.melt(data.reset_index(), id_vars=['index'], value_vars=['AFeL', 'LM'])#, 'WAY', 'GRID', 'BO', 'CPSO', 'ERPSO', 'EXPSO', 'EPS'])
    df_melt.columns = ['index', 'treatments', 'value']
    return data, df_melt


def f_value(data, dfn, dfd):
    fvalue, pvalue = stats.f_oneway(data['AFeL'], data['LM'])#, data['WAY'], data['GRID'], data['BO'], data['CPSO'], data['ERPSO'], data['EXPSO'], data['EPS'])
    fcritical = stats.f.ppf(q=1-.05, dfn=dfn, dfd=dfd)

    return fvalue, pvalue, fcritical


def t_value(data):
    a = data.to_numpy()
    return stats.ttest_1samp(a=a, popmean=np.mean(np.array(data)))


def t_value2(data):
    res = stat()
    res.ttest(df=data, test_type=1, res='WAY',  mu=np.mean(np.array(data['WAY'])))
    return res.summary


def anova(df_melt):
    model = ols('value ~ C(treatments)', data=df_melt).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    return model, anova_table


def anova2(df_melt, title_):
    res = stat()
    res.anova_stat(df=df_melt, res_var='value', anova_model='value ~ C(treatments)')
    sm.qqplot(res.anova_std_residuals, line='45')
    plt.xlabel("Theoretical Quantiles")
    plt.ylabel("Standardized Residuals")
    plt.title(title_)
    fig, axs = plt.subplots(1, 1)
    axs.hist(res.anova_model_out.resid, bins='auto', histtype='bar', ec='k')
    axs.set_xlabel("Residuals")
    axs.set_ylabel('Frequency')
    axs.set_title(title_)
    plt.show()
    return res.anova_summary


def tukey_hsd_test(df_melt):
    res = stat()
    res.tukey_hsd(df=df_melt, res_var='value', xfac_var='treatments', anova_model='value ~ C(treatments)')
    return res.tukey_summary


def shapiro(model):
    return stats.shapiro(model.resid)


def bartletts(data, df_melt):
    w, pvalue = stats.bartlett(data['AFeL'], data['LM'])#, data['WAY'], data['GRID'], data['BO'], data['CPSO'], data['ERPSO'], data['EXPSO'], data['EPS'])
    res = stat()
    res.bartlett(df=df_melt, res_var='value', xfac_var='treatments')
    return w, pvalue, res.bartlett_summary


def levene(df_melt):
    res = stat()
    res.levene(df=df_melt, res_var='value', xfac_var='treatments')
    return res.levene_summary


if __name__ == '__main__':

    data_name = [error(), mseaz(), msem(), r2m()]

    title = ['Peak Error', 'MSE AZ', 'MSE Map', 'R2 Map']

    for n in range(len(data_name)):
        print(title[n])
        aqua1, lawn1 = data_name[n]
        # aqua1, lawn1, way1, grid1, bo1, cpso1, explore1, exploit1, epsilon1 = data_name[n]
        dfs = [aqua1, lawn1]#, way1, grid1, bo1, cpso1, explore1, exploit1, epsilon1]
        dfn = len(dfs) - 1
        dfd = len(aqua1) * len(dfs) - len(dfs)
        print('dfn', dfn, 'dfd', dfd)

        data, df_melt = table_data(dfs)
        fvalue, pvalue, fcritical = f_value(data, dfn, dfd)
        # model, anova_table = anova(df_melt)
        # anova_summary = anova2(df_melt, title[n])
        # tukey_summary = tukey_hsd_test(df_melt)
        # w_b, pvalue_b, bartletts_summary = bartletts(data, df_melt)
        #
        # print('t', t_value2(data))
        print('fvalue:', fvalue, ', fcritical:', fcritical, ', pvalue:', pvalue)
        # print('anova', anova_summary)
        # print('tukey', tukey_summary)
        # print('w_shapiro', 'pvalue_shapiro', shapiro(model))
        # print('w_bartletts', w_b, 'pvalue_barletts', pvalue_b, 'bartletts', bartletts_summary)
        # print('levene', levene(df_melt))