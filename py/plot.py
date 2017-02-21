import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import itertools
from pandas import DataFrame


def add_margin(ax,x=0.05,y=0.05):
    # This will, by default, add 5% to the x and y margins. You
    # can customise this using the x and y arguments when you call it.

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    xmargin = (xlim[1]-xlim[0])*x
    ymargin = (ylim[1]-ylim[0])*y

    ax.set_xlim(xlim[0]-xmargin,xlim[1]+xmargin)
    ax.set_ylim(ylim[0]-ymargin,ylim[1]+ymargin)


y_base_eng = 0.769946066579877
y_base_tur = 0.634961439589
label_size = 17
tick_size = 16
legend_size = 14

sns.set(style='whitegrid', context='paper', font='Times New Roman')
# sns.set_style({'axes.grid' : True})
all_data = np.asarray([0.757400856, 0.772168652, 0.796366939, 0.799342706, 0.796431173, 0.798608568, 0.796431173, 0.798608568, 0.796431173, 0.798608568])
all_rounds = np.asarray([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0])
# all_rounds = np.asarray([0, 1.0, 2.0, 3.0, 4] * 2)
data = {'round': all_rounds, 'F1': all_data, 'color': ['un', 'lp'] * 5}
data = DataFrame(data=data)

plt.rcParams['axes.linewidth'] = 5
plt.rcParams['lines.linewidth'] = 2
ax1 = sns.pointplot(x='round', y='F1', data=data, markers='')
plt.setp(ax1.lines, zorder=2)
ax2 = sns.pointplot(x='round', y='F1', hue='color', data=data, markers=['o', '^'])
handles, labels = ax2.get_legend_handles_labels()
plt.setp(ax2.lines, zorder=1)
plt.hlines(y_base_eng, ax2.get_xlim()[0], ax2.get_xlim()[1], linestyle='--', color='k', zorder=0)

ax1.legend(handles, ['Contrastive Estimation', 'ILP'], title='Stage', loc='upper left', prop={'size':legend_size})
ax1.get_legend().get_title().set_fontsize(legend_size + 1) 
ax1.set_ylim([0.75, 0.815])



for tick in ax1.xaxis.get_major_ticks():
    tick.label.set_fontsize(tick_size)
for tick in ax1.yaxis.get_major_ticks():
    tick.label.set_fontsize(tick_size)
ax1.set_ylabel("F1 score", fontsize=label_size)
ax1.set_xlabel("Round", fontsize=label_size)


sns.despine()

x, y = plt.gcf().get_size_inches()
plt.gcf().set_size_inches(x, 1.2 * y)

# sns.plt.show()
plt.savefig("round.png", dpi=500)
plt.clf()

plt.gcf().set_size_inches(x, y)

############################################

label_size = 16
tick_size = 12
color_tick_size = 12

my_dpi = 500.0

fig, axn = plt.subplots(2, 1, sharex=False, sharey=False)
beta_tur = [0.0, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0005, 0.0005, 0.0005, 0.0005, 0.0005, 0.0005, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.0, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.0, 0.0, 0.0, 0.0, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
gamma_tur = [0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 2.0, 3.0, 4.0, 5.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
F1_tur = [0.629190515, 0.659717205, 0.665405298, 0.655923261, 0.639136905, 0.624617048, 0.613459351, 0.598823122, 0.639142433, 0.655966081, 0.649899589, 0.643811303, 0.628699151, 0.590484967, 0.611385029, 0.644935733, 0.656150198, 0.646240075, 0.644979555, 0.508920663, 0.560273649, 0.596322045, 0.604406233, 0.613230519, 0.622243714, 0.631878931, 0.460763521, 0.500566822, 0.520163226, 0.551849744, 0.575053763, 0.594719871, 0.626185958, 0.616074698, 0.601969307, 0.59872944, 0, 0, 0.104526046, 0.354730184, 0.388638992, 0.408103131, 0, 0, 0, 0.066241587, 0, 0.076197134]


my_palette = sns.cubehelix_palette(10, start=2.9, as_cmap=True)
data_tur = {'beta_tur': beta_tur, 'gamma_tur': gamma_tur, 'F1_tur': F1_tur}
data_tur = DataFrame(data=data_tur)

data_tur = data_tur.pivot('gamma_tur', 'beta_tur', 'F1_tur')
ax = sns.heatmap(data_tur, vmin=0.56, vmax=0.66, robust=True, annot=True, annot_kws={"size": 12}, fmt='1.3f', ax=axn.flat[1])
cax = plt.gcf().axes[-1]
cax.tick_params(labelsize=color_tick_size)
ax.invert_yaxis()
ax.set_xlabel('Turkish' + ': ' + r'$\alpha$', fontsize=label_size)
ax.set_ylabel(r'$\beta$', fontsize=label_size)
# label_size = 8
# tick_size = 14
for tick in ax.xaxis.get_major_ticks():
    tick.label.set_fontsize(tick_size)
for tick in ax.yaxis.get_major_ticks():
    tick.label.set_fontsize(tick_size)

beta_eng = [0.0, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0005, 0.0005, 0.0005, 0.0005, 0.0005, 0.0005, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.0, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.0, 0.0, 0.0, 0.0, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
gamma_eng = [0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 2.0, 3.0, 4.0, 5.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
F1_eng = [0.747766584, 0.783392226, 0.782004516, 0.771755856, 0.76026135, 0.74868594, 0.731925784, 0.799781182, 0.798302987, 0.786641287, 0.777964394, 0.766964581, 0.74364676, 0.798257246, 0.798608568, 0.804573439, 0.784575835, 0.770097795, 0.751937984, 0.763553906, 0.779654202, 0.804606526, 0.806096217, 0.8, 0.792771084, 0.756827048, 0.75893984, 0.753058262, 0.782956591, 0.785811733, 0.796385098, 0.788043478, 0.756396396, 0.752064664, 0.741814461, 0.72796554, 0.69448183, 0.714003945, 0.726843911, 0.749176277, 0.771286142, 0.76929165, 0.69448183, 0.700133274, 0.70709282, 0.720835999, 0.721470837, 0.731017266]
data_eng = {'beta_eng': beta_eng, 'gamma_eng': gamma_eng, 'F1_eng': F1_eng}
data_eng = DataFrame(data=data_eng)
data_eng = data_eng.pivot('gamma_eng', 'beta_eng', 'F1_eng')
ax = sns.heatmap(data_eng, vmin=0.7, vmax=0.80, robust=True, annot=True, annot_kws={"size": 12}, fmt='1.3f', ax=axn.flat[0])
cax = plt.gcf().axes[-1]
cax.tick_params(labelsize=color_tick_size)
ax.set_xlabel('English' + ': ' + r'$\alpha$', fontsize=label_size)
ax.set_ylabel(r'$\beta$', fontsize=label_size)
# plt.savefig('hyper_eng.png', dpi=500)
# plt.show()
x, y = plt.gcf().get_size_inches()
plt.gcf().set_size_inches(x, 2 * y)
ax.invert_yaxis()

for tick in ax.xaxis.get_major_ticks():
    tick.label.set_fontsize(tick_size)
for tick in ax.yaxis.get_major_ticks():
    tick.label.set_fontsize(tick_size)
plt.tight_layout()
plt.savefig('hyper.png', bbox_inches='tight', dpi=500)
plt.clf()

############################################
label_size = 18
tick_size = 17


plt.gcf().set_size_inches(x, y)

fig, axn = plt.subplots(2, 1, sharex=False, sharey=False, gridspec_kw={'height_ratios':[1, 1]})

plt.rcParams['axes.linewidth'] = 8
plt.rcParams['lines.linewidth'] = 4
plt.rcParams['lines.markersize'] = 11

size = [10000, 20000, 30000, 40000, 50000] * 2
big0_eng = [0.757400856, 0.745736731, 0.757142857, 0.762108539, 0.729221436]
big1_eng = [0.798608568, 0.808392127739, 0.82258359218, 0.830003632401, 0.827335514358]
big_eng = big0_eng + big1_eng
method = ['un'] * 5 + ['alt'] * 5

data_eng = {'size': size, 'F1': big_eng, 'method': method}
data_eng = DataFrame(data=data_eng)


big0_tur = [0.6330537942, 0.62122646471, 0.582186666667, 0.545640095249, 0.541026516445]
big1_tur = [0.656150198, 0.660282544799, 0.627410597266, 0.640521622399, 0.635569596729]
big_tur = big0_tur + big1_tur
method = ['un'] * 5 + ['alt'] * 5

data_tur = {'size': size, 'F1': big_tur, 'method': method}
data_tur = DataFrame(data=data_tur)

axn.flat[0].xaxis.grid(False)
axn.flat[1].xaxis.grid(False)


ax = sns.tsplot(data=data_eng, time='size', value='F1', condition='method', unit='method', marker='o', ax=axn.flat[0])
add_margin(ax)
ax.hlines(y_base_eng, ax.get_xlim()[0], ax.get_xlim()[1], linestyle='--', color='k')
ax.set_xlabel("English: " + r'$K$' + " (training set size)", fontsize=label_size)
ax.set_ylabel('F1 score', fontsize=label_size)
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, [r'$NBJ$' + '-' + r'$Imp$', r'$Our model$'], title='Method', loc='upper left', prop={'size':legend_size})
ax.set_ylim([0.70, 0.85])
d = ax.get_legend().get_title().set_fontsize(legend_size + 1) 
for tick in ax.xaxis.get_major_ticks():
    tick.label.set_fontsize(tick_size)
for tick in ax.yaxis.get_major_ticks():
    tick.label.set_fontsize(tick_size)

sns.despine()

ax = sns.tsplot(data=data_tur, time='size', value='F1', condition='method', unit='method', marker='o', ax=axn.flat[1])
add_margin(ax)
plt.hlines(y_base_tur, ax.get_xlim()[0], ax.get_xlim()[1], linestyle='--', color='k')
ax.set_xlabel("Turkish: " r'$K$' + " (training set size)", fontsize=label_size)
ax.set_ylabel('F1 score', fontsize=label_size)
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, [r'$NBJ$' + '-' + r'$Imp$', r'$Our model$'], title='Method', loc='upper left', prop={'size':legend_size})
ax.set_ylim([0.50, 0.70])
d = ax.get_legend().get_title().set_fontsize(legend_size + 1) 
for tick in ax.xaxis.get_major_ticks():
    tick.label.set_fontsize(tick_size)
for tick in ax.yaxis.get_major_ticks():
    tick.label.set_fontsize(tick_size)
sns.despine()

x, y = plt.gcf().get_size_inches()
plt.gcf().set_size_inches(x, 2.75 * y)
plt.tight_layout()
plt.savefig('big.png', dpi=500)
