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

for number in tqdm(range(50)):
    choice=[0,3000,6000,9000]
    diffusiondataset = DiffusionDataset("F:/Python_venv/Deep_Learning/pythonProject/MNIST-CYJ/03 Dataset/Test/Dataset_Test_unweak_2D.h5")
    x1=diffusiondataset[choice[0]][0]
    y1=diffusiondataset[choice[0]][1]


    disp1=np.loadtxt(f'../08 Result of each model/{number}/Disp_1.txt',usecols=1)
    reaction1=np.loadtxt(f'../08 Result of each model/{number}/Reaction_1.txt',usecols=1)

    fig, (ax) = plt.subplots(1, 1)
    ax.grid(True, which="major", linestyle="--", color="lightgray", linewidth=1, zorder=-1,alpha=0.3)
    ax.grid(True, which="minor", linestyle="--", color="lightgray", linewidth=1, zorder=-1,alpha=0.3)
    ax.scatter(y1[:,0],y1[:,1]/1000,color='red',label='Target',zorder=1)
    ax.plot(y1[:,0],y1[:,1]/1000,color='red',label='Target',zorder=1)
    ax.plot(disp1,reaction1/1000,color='blue',label='DDPM Generate',zorder=-1)


    ax.set_xlabel("Displacement (mm)", fontproperties=font_Times_New_Roman)  # 添加x轴名
    ax.set_ylabel("Force (kN)", fontproperties=font_Times_New_Roman)
    ax.set_xlim(-100, 100)  # 设置x轴的刻度范围
    ax.set_ylim(-300, 300)

    # 主刻度
    x_major_locator = MultipleLocator(50)
    ax.xaxis.set_major_locator(x_major_locator)

    y_major_locator = MultipleLocator(150)
    ax.yaxis.set_major_locator(y_major_locator)

    # 副刻度
    x_minor_locator = MultipleLocator(25)
    ax.xaxis.set_minor_locator(x_minor_locator)

    y_minor_locator = MultipleLocator(75)
    ax.yaxis.set_minor_locator(y_minor_locator)

    bwith = 2  # 边框宽度设置为2
    ax = plt.gca()  # 获取边框
    ax.spines['top'].set_color('k')  # 设置上‘脊梁’为红色
    ax.spines['right'].set_color('k')  # 设置上‘脊梁’为无色
    ax.spines['bottom'].set_linewidth(bwith)
    ax.spines['left'].set_linewidth(bwith)
    ax.spines['top'].set_linewidth(bwith)
    ax.spines['right'].set_linewidth(bwith)

    ax.tick_params(axis='x', which='major', direction='in', labelsize=30, length=10, width=2,
                   pad=5)  # 主刻度x：朝向、长短、大小
    ax.tick_params(axis='x', which='minor', direction='in', color='k', labelsize=30, length=5, width=1,
                   pad=5)  # 副刻度x
    ax.tick_params(axis='y', which='major', direction='in', labelsize=30, length=10, width=2,
                   pad=5)  # 主刻度x
    ax.tick_params(axis='y', which='minor', direction='in', color='k', labelsize=30, length=5, width=1,
                   pad=5)  # 副刻度y

    # ax.axis('off')


    plt.savefig(f'./02result/{number}.png', bbox_inches='tight', transparent=True)
    plt.close()
