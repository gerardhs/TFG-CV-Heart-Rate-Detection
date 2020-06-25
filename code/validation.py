'''Validation of Heart Rate Detector'''

import glob
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
import numpy as np
import scipy.stats as st

PATH = 'data/experiment*.csv'
DATAFRAMES = []
sns.set_context('paper')
sns.set_style('darkgrid')
sns.set(font_scale=1.25)

for fname in glob.glob(PATH):

    fig, axs = plt.subplots(ncols=2, nrows=2, figsize=(18.5, 12))
    gs = axs[1, 0].get_gridspec()
    for ax in axs[0, 0:]:
        ax.remove()
    axbig = fig.add_subplot(gs[0, 0:])
    fig.suptitle('Resultats ' + fname[5:-4])
    data = pd.read_csv(fname)
    DATAFRAMES.append(data)
    data['Error Absolut'] = abs(data['Sistema'] - data['Pulsioxímetre'])
    data['Error Relatiu'] = (data['Error Absolut'] / data['Pulsioxímetre'])
    data['Error Percentual'] = data['Error Relatiu'] * 100

    data.to_csv('data/resultats_'+fname[5:], index=False, header=True)

    sns.lineplot(x=data['#'], y=data['Sistema'], ax=axbig, color='red')
    sns.lineplot(x=data['#'], y=data['Pulsioxímetre'], ax=axbig,
                 color='blue')
    axbig.set_xlabel('Lectura')
    axbig.set_ylabel('Pulsacions per minut')
    axbig.axhline(data['Sistema'].mean(), color='red', linewidth=0.5,
                  linestyle='dashed')
    axbig.axhline(data['Pulsioxímetre'].mean(), color='blue',
                  linewidth=0.5, linestyle='dashed')
    axbig.text(49, data['Sistema'].mean()+0.05,
               "Mean: {}".format(round(data['Sistema'].mean(), 2)),
               color='red')
    axbig.text(49, data['Pulsioxímetre'].mean()+0.3,
               "Mean: {}".format(round(data['Pulsioxímetre'].mean(), 2)),
               color='blue')
    axbig.yaxis.set_major_locator(ticker.MultipleLocator(1))
    axbig.yaxis.set_major_formatter(ticker.ScalarFormatter())
    axbig.legend(labels=['Sistema', 'Pulsioxímetre'], loc='best',
                 facecolor='lightgreen')
    axbig.tick_params(labelsize=7)
    axbig.set_title('(a) Mesuraments Realitzats')

    mean = data['Error Absolut'].mean()
    stdev = data['Error Absolut'].std()
    sns.boxplot(data=data['Error Absolut'], ax=axs[1][0], color='salmon',
                width=0.3, linewidth=2.5)
    sns.swarmplot(data=data['Error Absolut'], ax=axs[1][0], color='darkred')
    axs[1][0].yaxis.set_major_locator(ticker.MultipleLocator(1))
    axs[1][0].yaxis.set_major_formatter(ticker.ScalarFormatter())
    axs[1][0].set_ylabel('Error Absolut')
    axs[1][0].axhline(mean, color='blue', linewidth=0.5, linestyle='dashed')
    axs[1][0].axhline(mean+stdev, color='black', linewidth=0.5,
                      linestyle='dashed')
    axs[1][0].axhline(mean-stdev, color='black', linewidth=0.5,
                      linestyle='dashed')
    axs[1][0].text(0.25, mean+0.1, "Mean: {}".format(round(mean, 3)),
                   color='blue')
    axs[1][0].text(0.25, mean+0.1+stdev,
                   "Mean+StDev: {}+{}".format(round(mean, 3),
                                              round(stdev, 3)))
    axs[1][0].text(0.25, mean+0.1-stdev,
                   "Mean-StDev: {}-{}".format(round(mean, 3),
                                              round(stdev, 3)))
    axs[1][0].tick_params(labelsize=7)
    axs[1][0].set_title('(b) Error Absolut')

    mean = data['Error Percentual'].mean()
    stdev = data['Error Percentual'].std()
    sns.boxplot(data=data['Error Percentual'], ax=axs[1][1], color='salmon',
                width=0.3, linewidth=2.5)
    sns.swarmplot(data=data['Error Percentual'], ax=axs[1][1], color='darkred')
    axs[1][1].yaxis.set_major_locator(ticker.MultipleLocator(0.5))
    axs[1][1].yaxis.set_major_formatter(ticker.ScalarFormatter())
    axs[1][1].set_ylabel('Error Percentual (%)')
    axs[1][1].axhline(mean, color='blue', linewidth=0.5, linestyle='dashed')
    axs[1][1].axhline(mean+stdev, color='black', linewidth=0.5,
                      linestyle='dashed')
    axs[1][1].axhline(mean-stdev, color='black', linewidth=0.5,
                      linestyle='dashed')
    axs[1][1].text(0.16, mean+0.1, "Mean: {}%".format(round(mean, 3)),
                   color='blue')
    axs[1][1].text(0.16, mean+0.1+stdev,
                   "Mean+StDev: {}+{}%".format(round(mean, 3),
                                               round(stdev, 3)))
    axs[1][1].text(0.16, mean+0.1-stdev,
                   "Mean-StDev: {}-{}%".format(round(mean, 3),
                                               round(stdev, 3)))
    axs[1][1].tick_params(labelsize=7)
    axs[1][1].set_title('(c) Error Percentual')

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig('data/plot_'+fname[5:-4]+'.png')

    data = data.drop('#', axis=1)
    SEM = {'Sistema': data['Sistema'].sem(),
           'Pulsioxímetre': data['Pulsioxímetre'].sem(),
           'Error Absolut': data['Error Absolut'].sem(),
           'Error Relatiu': data['Error Relatiu'].sem(),
           'Error Percentual': data['Error Percentual'].sem()}
    MOE_ci95 = {'Sistema': 1.96*data['Sistema'].sem(),
                'Pulsioxímetre': 1.96*data['Pulsioxímetre'].sem(),
                'Error Absolut': 1.96*data['Error Absolut'].sem(),
                'Error Relatiu': 1.96*data['Error Relatiu'].sem(),
                'Error Percentual': 1.96*data['Error Percentual'].sem()}
    DESCRIPTION = data.describe()
    DESCRIPTION.loc['sem'] = SEM
    DESCRIPTION.loc['MOE_ci95'] = MOE_ci95

    file1 = open('data/tex_'+fname[5:-4]+'.txt', 'a')
    file1.write(data.to_latex(index=False))
    file1.write('\n')
    file1.write(DESCRIPTION.to_latex(index=True))
    file1.close()


data_total = pd.concat(DATAFRAMES)

fig, axs2 = plt.subplots(ncols=2, nrows=1, figsize=(18.5, 12))

mean = data_total['Error Absolut'].mean()
stdev = data_total['Error Absolut'].std()
sns.boxplot(y=data_total['Error Absolut'], ax=axs2[0], color='salmon',
            width=0.5, linewidth=2.5)
sns.swarmplot(y=data_total['Error Absolut'], ax=axs2[0], color='darkred')
axs2[0].yaxis.set_major_locator(ticker.MultipleLocator(1))
axs2[0].yaxis.set_major_formatter(ticker.ScalarFormatter())
axs2[0].set_ylabel('Error Absolut')
axs2[0].axhline(mean, color='blue', linewidth=0.5, linestyle='dashed')
axs2[0].axhline(mean+stdev, color='black', linewidth=0.5, linestyle='dashed')
axs2[0].axhline(mean-stdev, color='black', linewidth=0.5, linestyle='dashed')
axs2[0].text(0.25, mean+0.1, "Mean: {}".format(round(mean, 3)), color='blue')
axs2[0].text(0.25, mean+0.1+stdev, "Mean+StDev: {}+{}".format(round(mean, 3),
                                                              round(stdev, 3)))
axs2[0].text(0.25, mean+0.1-stdev, "Mean-StDev: {}-{}".format(round(mean, 3),
                                                              round(stdev, 3)))
axs2[0].set_title('(a) Error Absolut')

mean = data_total['Error Percentual'].mean()
stdev = data_total['Error Percentual'].std()
sns.boxplot(y=data_total['Error Percentual'], ax=axs2[1], color='salmon',
            width=0.5, linewidth=2.5)
sns.swarmplot(y=data_total['Error Percentual'], ax=axs2[1],
              color='darkred')
axs2[1].yaxis.set_major_locator(ticker.MultipleLocator(0.5))
axs2[1].yaxis.set_major_formatter(ticker.ScalarFormatter())
axs2[1].set_ylabel('Error Percentual (%)')
axs2[1].axhline(mean, color='blue', linewidth=0.5, linestyle='dashed')
axs2[1].axhline(mean+stdev, color='black', linewidth=0.5, linestyle='dashed')
axs2[1].axhline(mean-stdev, color='black', linewidth=0.5, linestyle='dashed')
axs2[1].text(0.25, mean+0.1, "Mean: {}%".format(round(mean, 3)), color='blue')
axs2[1].text(0.25, mean+0.1+stdev,
             "Mean+StDev: {}+{}%".format(round(mean, 3), round(stdev, 3)))
axs2[1].text(0.25, mean+0.1-stdev,
             "Mean-StDev: {}-{}%".format(round(mean, 3),
                                         round(stdev, 3)))
axs2[1].set_title('(b) Error Percentual')
fig.suptitle('Resultats acumulats')

fig.tight_layout(rect=[0, 0.03, 1, 0.95])
fig.savefig('data/plot_all_experiments.png')

data_total = data_total.drop('#', axis=1)
SEM = {'Sistema': data_total['Sistema'].sem(),
       'Pulsioxímetre': data_total['Pulsioxímetre'].sem(),
       'Error Absolut': data_total['Error Absolut'].sem(),
       'Error Relatiu': data_total['Error Relatiu'].sem(),
       'Error Percentual': data_total['Error Percentual'].sem()}

MOE_ci95 = {'Sistema': 1.96*data_total['Sistema'].sem(),
            'Pulsioxímetre': 1.96*data_total['Pulsioxímetre'].sem(),
            'Error Absolut': 1.96*data_total['Error Absolut'].sem(),
            'Error Relatiu': 1.96*data_total['Error Relatiu'].sem(),
            'Error Percentual': 1.96*data_total['Error Percentual'].sem()}
DESCRIPTION = data_total.describe()
DESCRIPTION.loc['sem'] = SEM
DESCRIPTION.loc['MOE_ci95'] = MOE_ci95

file1 = open('data/tex_all_experiments.txt', 'a')
file1.write(DESCRIPTION.to_latex(index=True))
file1.close()


plt.show()
