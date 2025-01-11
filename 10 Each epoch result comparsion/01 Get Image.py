import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

for num in tqdm(range(0,50)):
    Img_Column = np.loadtxt(fr'F:/Python_venv/Deep_Learning/pythonProject/MNIST-CYJ/08 Result of each model/{num}/x1.csv',delimiter=',')
    Img_Beam = np.loadtxt(fr'F:/Python_venv/Deep_Learning/pythonProject/MNIST-CYJ/08 Result of each model/{num}/x1_.csv',delimiter=',')

    color_list=['grey','red']
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(4,8),dpi=200)
    for i in range(0,len(Img_Column)):
        for j in range(0,len(Img_Column[0])):
            # print(i,j,Img_Column[i][j])
            plt.scatter(i,j,color=color_list[int(Img_Column[i][j])],marker='s',alpha=0.5,s=25)
            plt.scatter(i,j+40,color=color_list[int(Img_Beam[i][j])],marker='s',alpha=0.5,s=25)

    ax.axis('off')
    # plt.show()
    plt.savefig(f'./01Img/{num}.png',dpi=400,bbox_inches='tight',transparent=True)
    plt.close()



