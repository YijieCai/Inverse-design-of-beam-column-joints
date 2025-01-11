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

for number in range(0,50):
    choice=[0,3000,6000,9000]
    diffusiondataset = DiffusionDataset("F:/Python_venv/Deep_Learning/pythonProject/MNIST-CYJ/03 Dataset/Test/Dataset_Test_unweak_2D.h5")

    x1=diffusiondataset[choice[0]][0]
    y1=diffusiondataset[choice[0]][1]
    y2=diffusiondataset[choice[1]][1]
    y3=diffusiondataset[choice[2]][1]
    y4=diffusiondataset[choice[3]][1]
    disp1=np.loadtxt(f'../08 Result of each model/{number}/Disp_1.txt',usecols=1)
    reaction1=np.loadtxt(f'../08 Result of each model/{number}/Reaction_1.txt',usecols=1)
    disp2=np.loadtxt(f'../08 Result of each model/{number}/Disp_2.txt',usecols=1)
    reaction2=np.loadtxt(f'../08 Result of each model/{number}/Reaction_2.txt',usecols=1)
    disp3=np.loadtxt(f'../08 Result of each model/{number}/Disp_3.txt',usecols=1)
    reaction3=np.loadtxt(f'../08 Result of each model/{number}/Reaction_3.txt',usecols=1)
    disp4=np.loadtxt(f'../08 Result of each model/{number}/Disp_4.txt',usecols=1)
    reaction4=np.loadtxt(f'../08 Result of each model/{number}/Reaction_4.txt',usecols=1)

    x1_,y1_=skeleton_curve(disp1,reaction1)
    x2_,y2_=skeleton_curve(disp2,reaction2)
    x3_,y3_=skeleton_curve(disp3,reaction3)
    x4_,y4_=skeleton_curve(disp4,reaction4)
    # print(np.array(y1[:,1]))
    y1=np.array(y1[1:,1])
    y2=np.array(y2[1:,1])
    y3=np.array(y3[1:,1])
    y4=np.array(y4[1:,1])
    y1_=y1_[1:]
    y2_=y2_[1:]
    y3_=y3_[1:]
    y4_=y4_[1:]
    # print(abs(y1_-y1)/y1)
    mse=np.mean([np.mean(abs(y1_-y1)/y1),np.mean(abs(y2_-y2)/y2),np.mean(abs(y3_-y3)/y3),np.mean(abs(y4_-y4)/y4)])
    print(mse,number)