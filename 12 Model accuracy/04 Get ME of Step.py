from PIL import Image, ImageSequence
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from Dataset import DiffusionDataset
from tqdm import tqdm

import os

import matplotlib.pyplot as plt
import numpy as np

import scipy.signal as sg


#字体设置
from pylab import *
import seaborn as sns
from matplotlib.ticker import MultipleLocator

font_Times_New_Roman={"family":"Times New Roman",
                      # "style": "italic",
                    #   "weight":"heavy",
                      "size":30}

font_Times_New_Roman_label={"family":"Times New Roman",
                      # "style": "italic",
                    #   "weight":"heavy",
                      "size":12}

font_Song={"family":"SimSun",
           "style":"italic",
        #    "weight":"heavy",
           "size":15}
#坐标轴字体

plt.rcParams['font.sans-serif']=['Times New Roman']
plt.rc('axes', unicode_minus=False)


def skeleton_curve(disp,reaction):
    disp_sc=np.array([0])
    reaction_sc=np.array([0])
    zero_points=[]
    zero_points=[0, 399, 799, 1199, 1599, 1999, 2399, 2799, 3199, 3599, 3999, 4399,4799,5199]
    for i in range(len(zero_points)-1):
        disp_loop=disp[zero_points[i]:zero_points[i+1]]
        reaction_loop = reaction[zero_points[i]:zero_points[i+1]]
        disp_max = np.max(disp_loop)
        reaction_max = reaction_loop[np.argmax(disp_loop)]
        disp_sc = np.append(disp_sc, disp_max)
        reaction_sc = np.append(reaction_sc, reaction_max)
    return disp_sc,reaction_sc



ME_Step_list = []
diffusiondataset = DiffusionDataset(
    "F:/Python_venv/Deep_Learning/pythonProject/MNIST-CYJ/03 Dataset/Test/Dataset_Test_unweak_2D.h5")
ME_Step = []
for i in range(1000):

    x1=diffusiondataset[i][0]
    y1=diffusiondataset[i][1]

    disp1=np.loadtxt(f'./03Result/Disp_{i}_{999}.txt',usecols=1)
    reaction1=np.loadtxt(f'./03Result/Reaction_{i}_{999}.txt',usecols=1)

    x1_, y1_ = skeleton_curve(disp1, reaction1)
    y1 = np.array(y1[1:, 1])
    y1_ = y1_[1:]

    mse = np.mean(abs(y1_ - y1) / y1)

    ME_Step.append([mse,i])
    # print([mse,i])
ME_Step_list.append(ME_Step)
ME_Step=np.array(ME_Step)
np.savetxt('ME.txt',ME_Step)
# print(ME_Step)
# plt.plot(ME_Step[:,1],ME_Step[:,0])
# plt.show()