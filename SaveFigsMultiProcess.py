import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
from pathlib import Path
import os
import tempfile
import multiprocessing as mp


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


def create_save_fig(i):
    """
    
    Creates the figure using values and saves it to given file

    :param i: episode number
    """
    
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
    plt.savefig(os.path.join(DIR_Q_TABLE_CHARTS, f"{i}.png"))
    plt.clf()
    

fig = plt.figure(figsize=(12, 9))


pending_files = []
for i in range(10, 500001, 10):
    file_name = os.path.join(DIR_Q_TABLE_CHARTS, f"{i}.png")
    if not os.path.isfile(file_name):
        pending_files.append(i)
        print(i)

pool = mp.Pool(processes=4)
# _ = pool.map(create_save_fig, range(10, 50001, 10))

_ = pool.map(create_save_fig, pending_files)

print("Task Completed!")    
