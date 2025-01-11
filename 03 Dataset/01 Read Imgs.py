import os

import numpy as np
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm

Set = 'Test'
set = 'test'
train=np.loadtxt(f'F:\\Python_venv\\Deep_Learning\\pythonProject\\MNIST-CYJ\\mnist_{set}.csv',delimiter=',')
dir=rf'F:\Python_venv\Deep_Learning\pythonProject\MNIST-CYJ\02 OS_Calculate_Beam\{Set}'

if not os.path.exists(f'F:\\Python_venv\\Deep_Learning\\pythonProject\\MNIST-CYJ\\03 Dataset\\{Set}\\Imgs'):
    os.mkdir(fr'F:\Python_venv\Deep_Learning\pythonProject\MNIST-CYJ\03 Dataset\{Set}\Imgs')
Exclusion=np.loadtxt(rf'F:\Python_venv\Deep_Learning\pythonProject\MNIST-CYJ\03 Dataset\Exclusion_{Set}.txt')
print(train.shape[0])
for i in tqdm(range(train.shape[0])):
    if i not in Exclusion:
        # 柱子
        img=train[i,1:].reshape(28,28)
        img = cv2.resize(img, (40,40), interpolation=cv2.INTER_LINEAR)
        img[img<125] = 0
        img[img>=125] = 1
        # 梁
        if Set=='Train':
            # print((i+10000)%60000)
            img2=train[((i+10000)%60000),1:].reshape(28,28)
        else:
            img2=train[((i+5000)%10000),1:].reshape(28,28)
        img2 = cv2.resize(img2, (40,40), interpolation=cv2.INTER_LINEAR)
        img2[img2<125] = 0
        img2[img2>=125] = 1
    if not os.path.exists(f'F:\\Python_venv\\Deep_Learning\\pythonProject\\MNIST-CYJ\\03 Dataset\\{Set}\\Imgs\\{i}'):
        os.mkdir(f'F:\\Python_venv\\Deep_Learning\\pythonProject\\MNIST-CYJ\\03 Dataset\\{Set}\\Imgs\\{i}')
    np.savetxt(f'F:\\Python_venv\\Deep_Learning\\pythonProject\\MNIST-CYJ\\03 Dataset\\{Set}\\Imgs\\{i}\\Column.csv',img,delimiter=',',fmt="%d")
    np.savetxt(f'F:\\Python_venv\\Deep_Learning\\pythonProject\\MNIST-CYJ\\03 Dataset\\{Set}\\Imgs\\{i}\\Beam.csv',img2,delimiter=',',fmt="%d")
