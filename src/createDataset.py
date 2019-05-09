import numpy as np
import pandas as pd
import os
from PCP import PCP
CtoN = {'a': 0, 'am': 1, 'bm': 2, 'c': 3, 'd': 4,
        'dm': 5, 'e': 6, 'em': 7, 'f': 8, 'g': 9}
dir = './'
folders = [os.path.join(dir, i) for i in os.listdir(dir) if len(i) < 4]
files = []
for folder in folders:
    files += [os.path.join(folder, i) for i in os.listdir(folder)]
data = []
for i in range(len(files)):
    print(files[i])
    crp = PCP('/'.join(files[i].rsplit('/')[:-1]), '/' + files[i].rsplit('/')[-1])
    crp = np.array(crp).reshape(-1,1)
    print(i, files[i].rsplit('/')[-1])

    for j in range(crp.shape[1]):
        data.append(list(crp[:, j]) + [CtoN[files[i].rsplit('/')[-2][:-1]]])
data = np.array(data)
Data = pd.DataFrame(data)
Data.to_csv('Guitar Training Dataset PCP.csv')
