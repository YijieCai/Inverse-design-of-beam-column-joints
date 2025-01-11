import h5py
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from tqdm import tqdm

Set = 'Train'
set = 'train'
dir='F:\\Python_venv\\Deep_Learning\\pythonProject\\MNIST-CYJ\\02 OS_Calculate_Beam\\Test'
Exclusion=np.loadtxt(rf'F:\Python_venv\Deep_Learning\pythonProject\MNIST-CYJ\03 Dataset\Exclusion_Test.txt')
Exclusion2=np.loadtxt(rf'F:\Python_venv\Deep_Learning\pythonProject\MNIST-CYJ\03 Dataset\Exclusion_unweak_Test.txt')
work_dir=fr'F:\Python_venv\Deep_Learning\pythonProject\MNIST-CYJ\03 Dataset'
def make_data(data,label,data_dir):
    savepath=os.path.join('..', os.path.join('checkpoint', data_dir, 'Dataset_Test_unweak_2D.h5'))
    if not os.path.exists(os.path.join('..', os.path.join('checkpoint', data_dir))):
        os.makedirs(os.path.join('..', os.path.join('checkpoint', data_dir)))
    with h5py.File(savepath,'w') as hf:
        hf.create_dataset('feature',data=data)
        hf.create_dataset('label',data=label)

input_features=[]
input_labels=[]
for index in tqdm(range(10000)):
    if index not in Exclusion:
        if index not in Exclusion2:
            features_Column = np.loadtxt(f'{work_dir}\\Test\\Imgs_2D\\{index}\\Column.csv',delimiter=',')
            features_Beam = np.loadtxt(f'{work_dir}\\Test\\Imgs_2D\\{index}\\Beam.csv',delimiter=',')
            labels=np.loadtxt(f'F:\\Python_venv\\Deep_Learning\\pythonProject\\MNIST-CYJ\\04 Mehanical Property\\Test\\{index}.txt')

            input = np.hstack((features_Column,features_Beam))
            input_features.append(input)
            # print(input_features.shape)
            # input_features.append(features_Beam)
            input_labels.append(labels)
ar_features = np.asarray(input_features)
print(ar_features.shape)
# print(ar_features.shape)
ar_labels = np.asarray(input_labels)
make_data(ar_features,ar_labels,"F:\\Python_venv\\Deep_Learning\\pythonProject\\MNIST-CYJ\\03 Dataset\\Test")
