import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

dir='F:\\Python_venv\\Deep_Learning\\pythonProject\\MNIST-CYJ\\02 OS_Calculate_Beam\\Train'
Exclusion=np.loadtxt(rf'F:\Python_venv\Deep_Learning\pythonProject\MNIST-CYJ\03 Dataset\Exclusion_Train.txt')

print(Exclusion)

def skeleton_curve(disp,reaction):
    disp_sc=np.array([0])
    reaction_sc=np.array([0])
    zero_points=[]
    # for i in range(disp.shape[0]-1):
    #     if disp[i]*disp[i+1]<0 and disp[i]>0:
    #         zero_points.append(i)
    # zero_points.insert(0,0)
    # print(zero_points)
    # for j in range(len(zero_points)-1):
    #     disp_loop=disp[zero_points[j]:zero_points[j+1]]
    #     reaction_loop=reaction[zero_points[j]:zero_points[j+1]]
    #     disp_max=np.max(disp_loop)
    #     reaction_max=reaction_loop[np.argmax(disp_loop)]
    #     disp_sc=np.append(disp_sc,disp_max)
    #     reaction_sc=np.append(reaction_sc,reaction_max)

    zero_points=[0, 399, 799, 1199, 1599, 1999, 2399, 2799, 3199, 3599, 3999, 4399,4799,5199]
    for i in range(len(zero_points)-1):
        disp_loop=disp[zero_points[i]:zero_points[i+1]]
        reaction_loop = reaction[zero_points[i]:zero_points[i+1]]
        disp_max = np.max(disp_loop)
        reaction_max = reaction_loop[np.argmax(disp_loop)]
        disp_sc = np.append(disp_sc, disp_max)
        reaction_sc = np.append(reaction_sc, reaction_max)
    return disp_sc,reaction_sc

for i in tqdm(range(60000)):
    if i not in Exclusion:
        disp=np.loadtxt(f'{dir}\\{i}\\disp.txt',usecols=1)
        disp=disp
        reaction=np.loadtxt(f'{dir}\\{i}\\reaction.txt',usecols=1)
        disp_sc,reaction_sc=skeleton_curve(disp,reaction)
        skeleton_curves=np.vstack([disp_sc,reaction_sc]).T
        np.savetxt(f'F:\\Python_venv\\Deep_Learning\\pythonProject\MNIST-CYJ\\04 Mehanical Property\\Train\\{i}.txt',skeleton_curves)
