import torch
import torch.nn as nn
from torch.utils.data import Dataset
import h5py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from tqdm.auto import tqdm
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DiffusionDataset(Dataset):
    def __init__(self,file):
        self.file_object=h5py.File(file,"r")
        self.feature=self.file_object.get('feature')
        self.label=self.file_object.get('label')
        pass
    def __len__(self):
        return len(self.feature)
    def __getitem__(self, index):
        if (index >= len(self.feature)):
            raise IndexError()
        feature = np.array(self.feature[index],dtype=np.int32)
        label = np.array(self.label[index],dtype=np.float32)
        feature=torch.Tensor(feature).type(torch.float32).unsqueeze(0)
        # print(feature)
        mask=torch.Tensor(label).type(torch.float32)
        return (feature,mask)
    def plot_image(self,index):
        fig,(ax)=plt.subplots(nrows=1,ncols=1)
        # print(self.feature[index])
        # print(self.feature[index].shape)
        ax.imshow(np.array(self.feature[index]),cmap='bone',alpha=0.5)
        # plt.show()
        # plt.close()
        # ax.imshow(np.array(self.feature[index][1]),cmap='bone',alpha=0.5)
        plt.show()
        pass
    pass

if __name__=='__main__':
    diffusiondataset = DiffusionDataset(r"F:\Python_venv\Deep_Learning\pythonProject\MNIST-CYJ\03 Dataset\Train\Dataset_Train_unweak_2D.h5")
    print('Dataset[0][0]Shape:',diffusiondataset[0][0].shape,'Dataset[0][1]Shape:',diffusiondataset[0][1].shape)
    # print(diffusiondataset[0][1])
    diffusiondataset.plot_image(3)