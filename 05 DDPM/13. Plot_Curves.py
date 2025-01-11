import numpy as np
import matplotlib.pyplot as plt
from Dataset import DiffusionDataset

choice=[0,3000,6000,9000]
# choice=[1000,3000,6000,8000]

diffusiondataset = DiffusionDataset("F:/Python_venv/Deep_Learning/pythonProject/MNIST-CYJ/03 Dataset/Test/Dataset_Test_unweak_2D.h5")
x1=diffusiondataset[choice[0]][0]
x2=diffusiondataset[choice[1]][0]
x3=diffusiondataset[choice[2]][0]
x4=diffusiondataset[choice[3]][0]

y1=diffusiondataset[choice[0]][1]
y2=diffusiondataset[choice[1]][1]
y3=diffusiondataset[choice[2]][1]
y4=diffusiondataset[choice[3]][1]

disp1=np.loadtxt(f'F:/Python_venv/Deep_Learning/pythonProject/MNIST-CYJ/05 DDPM/Val/skeleton_curve/Disp_1.txt',usecols=1)
reaction1=np.loadtxt(f'F:/Python_venv/Deep_Learning/pythonProject/MNIST-CYJ/05 DDPM/Val/skeleton_curve/Reaction_1.txt',usecols=1)
disp2=np.loadtxt(f'F:/Python_venv/Deep_Learning/pythonProject/MNIST-CYJ/05 DDPM/Val/skeleton_curve/Disp_2.txt',usecols=1)
reaction2=np.loadtxt(f'F:/Python_venv/Deep_Learning/pythonProject/MNIST-CYJ/05 DDPM/Val/skeleton_curve/Reaction_2.txt',usecols=1)
disp3=np.loadtxt(f'F:/Python_venv/Deep_Learning/pythonProject/MNIST-CYJ/05 DDPM/Val/skeleton_curve/Disp_3.txt',usecols=1)
reaction3=np.loadtxt(f'F:/Python_venv/Deep_Learning/pythonProject/MNIST-CYJ/05 DDPM/Val/skeleton_curve/Reaction_3.txt',usecols=1)
disp4=np.loadtxt(f'F:/Python_venv/Deep_Learning/pythonProject/MNIST-CYJ/05 DDPM/Val/skeleton_curve/Disp_4.txt',usecols=1)
reaction4=np.loadtxt(f'F:/Python_venv/Deep_Learning/pythonProject/MNIST-CYJ/05 DDPM/Val/skeleton_curve/Reaction_4.txt',usecols=1)

fig, (ax) = plt.subplots(2, 2)
ax[0, 0].scatter(y1[:,0],y1[:,1],color='red',zorder=1)
ax[0, 0].plot(disp1,reaction1,color='blue',zorder=-1)

ax[0, 1].scatter(y2[:,0],y2[:,1],color='red',zorder=1)
ax[0, 1].plot(disp2,reaction2,color='blue',zorder=-1)

ax[1, 0].scatter(y3[:,0],y3[:,1],color='red',zorder=1)
ax[1, 0].plot(disp3,reaction3,color='blue',zorder=-1)

ax[1, 1].scatter(y4[:,0],y4[:,1],color='red',zorder=1)
ax[1, 1].plot(disp4,reaction4,color='blue',zorder=-1)

plt.show()