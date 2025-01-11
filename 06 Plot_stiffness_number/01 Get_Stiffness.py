import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# dir=rf'F:\\Python_venv\\Deep_Learning\\pythonProject\\MNIST-CYJ\\02 OS_Calculate_Beam\\Test'
Exclusion=np.loadtxt(rf'F:\Python_venv\Deep_Learning\pythonProject\MNIST-CYJ\03 Dataset\Exclusion_Train.txt')

work_dir=rf'F:\\Python_venv\\Deep_Learning\\pythonProject\\MNIST-CYJ\\04 Mehanical Property\\Train'
Img_dir=fr'F:\Python_venv\Deep_Learning\pythonProject\MNIST-CYJ\03 Dataset\Train\Imgs'
Mnist=np.loadtxt(rf'F:\\Python_venv\\Deep_Learning\\pythonProject\\MNIST-CYJ\\mnist_train.csv',delimiter=',')

Data=[]
for i in tqdm(range(60000)):
    if i not in Exclusion:
        skeleton_curve=np.loadtxt(fr'{work_dir}/{i}.txt')
        # 计算刚度
        strength_residual = skeleton_curve[-1,1]
        strength_initial = max(skeleton_curve[:,1])
        degradation_rate = strength_residual/strength_initial
        # print('strength_residual=',strength_residual)
        # print('strength_initial=',strength_initial)
        # print('degradation_rate=',degradation_rate)


        # 梁和柱子的数字形式
        number_column=Mnist[i][0]
        j=(i+10000)%60000
        number_beam=Mnist[j][0]
        # 计算柱子钢材面积占比
        Img_Column=np.loadtxt(fr'{Img_dir}\{i}\Column.csv',delimiter=',')
        num_Column_1=np.count_nonzero(Img_Column)
        Area_Column=num_Column_1/1600
        # print('Area_Colum=',Area_Column)

        # 计算梁的钢材面积占比
        Img_Beam=np.loadtxt(fr'{Img_dir}\{i}\Beam.csv',delimiter=',')
        num_Beam_1=np.count_nonzero(Img_Beam)
        Area_Beam=num_Beam_1/1600
        # print('Area_Beam=',Area_Beam)

        Data.append([number_column,number_beam,Area_Column,Area_Beam,strength_initial,strength_residual,degradation_rate])

np.savetxt(f'F:\\Python_venv\\Deep_Learning\\pythonProject\MNIST-CYJ\\06 Plot_stiffness_number\\Train.txt',
           Data)


