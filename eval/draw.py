import os
import numpy as np
import matplotlib.pyplot as plt

linestyle = '-'
linewidth = 2
marker = 'o'
markersize = 3
markersize = 5

# recon quantity maicity
x = np.array([10, 15, 20, 25, 30, 35, 40, 45, 50])
chamfer_loss = np.array([1.740137573,1.652615321,1.626417989,1.598619474,1.580639635,1.57094054,1.568409511,1.559219683,1.562328578])
f1_score = np.array([98.747425629,98.884420986,98.911179736,98.923287838,98.95279367,98.942050761,98.924957865,98.928133584,98.923456473])
timing = np.array([85.01,128.591,163.551,214.483,252.910333333,298.99,332.655,368.883,394.071])

a = 4
fig, ax = plt.subplots(figsize=(8, 5))
plt.rcParams['ytick.direction'] = 'in'  # 刻度线显示在内部
plt.rcParams['xtick.direction'] = 'in'  # 刻度线显示在内部
plt.subplots_adjust(left=None, bottom=None, right=0.8,
                    top=0.95, wspace=0.3, hspace=0.3)


ax.plot(x, chamfer_loss, label="C-l1",
        linestyle=linestyle, linewidth=linewidth, marker=marker, markersize=markersize, color='black')
ax.set_ylim(ymin=1.5, ymax=1.75)
ax.set_ylabel("Chamfer-L1(cm)", fontdict={
    'family': 'Times New Roman', 'size': 16})
ax.set_xlabel("Reconstruction Iter", fontdict={
    'family': 'Times New Roman', 'size': 16})
ax.spines['right'].set_visible(False)  # ax右轴隐藏
legend_font = {"family": "Times New Roman", "size": 11}
# ax.legend(prop=legend_font)

bx_color = 'red'
bx = ax.twinx()
bx.plot(x, f1_score, 's--',  color=bx_color)
bx.set_ylim(ymin=98.7, ymax=99)
bx.set_ylabel("F1-Score(%)", fontdict={
    'family': 'Times New Roman', 'size': 16}, color=bx_color)
bx.tick_params(length=6, width=2, color=bx_color, labelcolor=bx_color)
bx.spines['right'].set(color=bx_color, linewidth=2.0, linestyle=':')
# bx.legend(prop=legend_font)

cx_color = 'blue'
cx = ax.twinx()
cx.plot(x, timing, 'o-.',  color=cx_color)
cx.set_ylim(ymin=50, ymax=450)
cx.set_ylabel("Timing(ms)", fontdict={
    'family': 'Times New Roman', 'size': 16}, color=cx_color)
cx.tick_params(length=6, width=2, color=cx_color, labelcolor=cx_color)
cx.spines['right'].set(color=cx_color, linewidth=2.0, linestyle='-.')
cx.spines['right'].set_position(('axes', 1.15))
# cx.legend(prop=legend_font)

# plt.plot(x, f1_score, label="F1-Score",
#          linestyle=linestyle, linewidth=linewidth, marker=marker, markersize=markersize)

# plt.xlim(0, 11)
# plt.xticks(np.linspace(0, 11, 12, endpoint=True))
# plt.ylim(0.0, 1.0)
# plt.yticks(np.linspace(0, 1, 5, endpoint=True))


# plt.title('Surface Coverage Evaluation')

plt.show()
