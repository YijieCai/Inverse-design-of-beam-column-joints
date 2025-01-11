import numpy as np
import matplotlib.pyplot as plt
from Dataset import DiffusionDataset
from tqdm import tqdm

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

error_list=[]
for number in tqdm(range(500)):
    choice=[0+number,3000+number,6000+number,9000+number]
    diffusiondataset = DiffusionDataset("F:/Python_venv/Deep_Learning/pythonProject/MNIST-CYJ/03 Dataset/Test/Dataset_Test_unweak_2D.h5")
    x1=diffusiondataset[choice[0]][0]
    x2=diffusiondataset[choice[1]][0]
    x3=diffusiondataset[choice[2]][0]
    x4=diffusiondataset[choice[3]][0]

    y1=diffusiondataset[choice[0]][1]
    y2=diffusiondataset[choice[1]][1]
    y3=diffusiondataset[choice[2]][1]
    y4=diffusiondataset[choice[3]][1]

    disp1=np.loadtxt(f'./01Img/{number}/Disp_1.txt',usecols=1)
    reaction1=np.loadtxt(f'./01Img/{number}/Reaction_1.txt',usecols=1)
    disp2=np.loadtxt(f'./01Img/{number}/Disp_2.txt',usecols=1)
    reaction2=np.loadtxt(f'./01Img/{number}/Reaction_2.txt',usecols=1)
    disp3=np.loadtxt(f'./01Img/{number}/Disp_3.txt',usecols=1)
    reaction3=np.loadtxt(f'./01Img/{number}/Reaction_3.txt',usecols=1)
    disp4=np.loadtxt(f'./01Img/{number}/Disp_4.txt',usecols=1)
    reaction4=np.loadtxt(f'./01Img/{number}/Reaction_4.txt',usecols=1)

    fig, (ax) = plt.subplots(2, 2)
    ax[0, 0].scatter(y1[:,0],y1[:,1],color='red',zorder=1)
    ax[0, 0].plot(disp1,reaction1,color='blue',zorder=-1)

    ax[0, 1].scatter(y2[:,0],y2[:,1],color='red',zorder=1)
    ax[0, 1].plot(disp2,reaction2,color='blue',zorder=-1)

    ax[1, 0].scatter(y3[:,0],y3[:,1],color='red',zorder=1)
    ax[1, 0].plot(disp3,reaction3,color='blue',zorder=-1)

    ax[1, 1].scatter(y4[:,0],y4[:,1],color='red',zorder=1)
    ax[1, 1].plot(disp4,reaction4,color='blue',zorder=-1)

    ske_c1=skeleton_curve(disp1,reaction1)
    ske_c2=skeleton_curve(disp2,reaction2)
    ske_c3=skeleton_curve(disp3,reaction3)
    ske_c4=skeleton_curve(disp4,reaction4)
    # print(ske_c1[1])
    # print(np.array(y1[:,1]))
    error1=np.mean((ske_c1[1][1:]-np.array(y1[1:,1]))/np.array(y1[1:,1]))
    error2=np.mean((ske_c2[1][1:]-np.array(y2[1:,1]))/np.array(y2[1:,1]))
    error3=np.mean((ske_c3[1][1:]-np.array(y3[1:,1]))/np.array(y3[1:,1]))
    error4=np.mean((ske_c4[1][1:]-np.array(y4[1:,1]))/np.array(y4[1:,1]))

    error_list.append(error1)
    error_list.append(error2)
    error_list.append(error3)
    error_list.append(error4)
    # 求 单组数据中 平均误差 最大误差 最小误差
    # print(ske_c1)
    plt.savefig(f'./result/{number}.png', bbox_inches='tight', transparent=True)
    plt.close()
np.savetxt('./Error.txt',error_list)