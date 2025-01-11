import h5py
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from tqdm import tqdm

Set = 'Train'
set = 'train'

work_dir=fr'F:\Python_venv\Deep_Learning\pythonProject\MNIST-CYJ\03 Dataset'
def make_data(data,label,data_dir):
    savepath=os.path.join('..', os.path.join('checkpoint', data_dir, 'Dataset_exceed.h5'))
    if not os.path.exists(os.path.join('..', os.path.join('checkpoint', data_dir))):
        os.makedirs(os.path.join('..', os.path.join('checkpoint', data_dir)))
    with h5py.File(savepath,'w') as hf:
        hf.create_dataset('feature',data=data)
        hf.create_dataset('label',data=label)

input_features=[]
input_labels=[]

features_Column = np.loadtxt(f'{work_dir}\\Test\\Imgs_2D\\5\\Column.csv',delimiter=',')
features_Beam = np.loadtxt(f'{work_dir}\\Test\\Imgs_2D\\5\\Beam.csv',delimiter=',')

# 截面1 的label*3
labels =np.loadtxt(f'F:\Python_venv\Deep_Learning\pythonProject\MNIST-CYJ\\04 Mehanical Property\\Train\\5.txt')
# labels[:,1]=labels[:,1]*1.30668395
labels[:,1]=labels[:,1]*1.30668395*2


input = np.hstack((features_Column,features_Beam))
input_features.append(input)
input_labels.append(labels)

ar_features = np.asarray(input_features)
ar_labels = np.asarray(input_labels)
make_data(ar_features,ar_labels,"F:\\Python_venv\\Deep_Learning\\pythonProject\\MNIST-CYJ\\14 Exceed Target")

# labels=np.loadtxt(f'F:\\Python_venv\\Deep_Learning\\pythonProject\\MNIST-CYJ\\04 Mehanical Property\\Test\\{index}.txt')
