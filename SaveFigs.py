import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
from pathlib import Path
import os
import tempfile


DIR_Q_TABLE_CHARTS = os.path.join(tempfile.gettempdir(), 'qtable_charts')
print(f"Q Table Charts Directory: {DIR_Q_TABLE_CHARTS}")
Path(DIR_Q_TABLE_CHARTS).mkdir(parents=True, exist_ok=True)

DIR_Q_TABLES = 'qtables'
Path(DIR_Q_TABLES).mkdir(parents=True, exist_ok=True)

style.use('ggplot')


def get_q_color(value, vals):
    if value == max(vals):
        return "green", 1.0
    else:
        return "red", 0.3


fig = plt.figure(figsize=(12, 9))

for i in range(10, 50001, 10):
    print(i)
    ax1 = fig.add_subplot(311)
    ax2 = fig.add_subplot(312)
    ax3 = fig.add_subplot(313)

    q_table = np.load(f"{DIR_Q_TABLES}/{i}-qtable.npy")

    for x, x_vals in enumerate(q_table):
        for y, y_vals in enumerate(x_vals):
            ax1.scatter(x, y, c=get_q_color(y_vals[0], y_vals)[0], marker="o", alpha=get_q_color(y_vals[0], y_vals)[1])
            ax2.scatter(x, y, c=get_q_color(y_vals[1], y_vals)[0], marker="o", alpha=get_q_color(y_vals[1], y_vals)[1])
            ax3.scatter(x, y, c=get_q_color(y_vals[2], y_vals)[0], marker="o", alpha=get_q_color(y_vals[2], y_vals)[1])

            ax1.set_ylabel("Action 0")
            ax2.set_ylabel("Action 1")
            ax3.set_ylabel("Action 2")

    # plt.show()
    plt.savefig(f"{DIR_Q_TABLE_CHARTS}/{i}.png")
    plt.clf()
