import tempfile
import pickle
import os
from pathlib import Path


DIR_Q_TABLE_CHARTS = os.path.join(tempfile.gettempdir(), 'qtable_charts')
print(f'Q Table Charts Directory: {DIR_Q_TABLE_CHARTS}')
Path(DIR_Q_TABLE_CHARTS).mkdir(parents=True, exist_ok=True)


pending_files = []
for i in range(10, 50001, 10):
    file_name = os.path.join(DIR_Q_TABLE_CHARTS, f"{i}.png")
    if not os.path.isfile(file_name):
        pending_files.append(i)
        print(i)


with open("pending_files.pickle", 'wb') as pkf:
    pickle.dump(pending_files, pkf)
