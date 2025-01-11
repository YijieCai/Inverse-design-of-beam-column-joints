import os

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
dir=r'F:\Python_venv\Deep_Learning\pythonProject\MNIST-CYJ\02 OS_Calculate_Beam\Train'

outarray=np.array([])
list = os.listdir(fr'F:\Python_venv\Deep_Learning\pythonProject\MNIST-CYJ\02 OS_Calculate_Beam\Train')

for i in tqdm(list):
# for i in tqdm(range(10000)):
    disp=np.loadtxt(f'{dir}\\{i}\\disp.txt')
    displength=disp.shape[0]
    if displength<5200:
        outarray=np.append(outarray,i)
        print(outarray,i)
np.savetxt(fr'F:\Python_venv\Deep_Learning\pythonProject\MNIST-CYJ\02 OS_Calculate_Beam\Exclusion_Train_Cal.txt',outarray,fmt="%d")
