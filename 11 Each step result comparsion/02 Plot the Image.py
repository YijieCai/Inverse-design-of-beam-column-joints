import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import openseespy.opensees as ops
import numpy as np
from multiprocessing import cpu_count    # 查看cpu核心数
from multiprocessing import Pool         # 并行处理必备，进程池
import os
from tqdm import tqdm


work_dir = fr'F:/Python_venv/Deep_Learning/pythonProject/MNIST-CYJ/11 Each step result comparsion/01Img'
save_dir = fr'F:/Python_venv/Deep_Learning/pythonProject/MNIST-CYJ/11 Each step result comparsion/02Section'
for num in tqdm(range(100)):
    for section_num in range(1,5):
        Img_Column = np.loadtxt(fr'{work_dir}/DDPM_Section_{num*10}_{section_num}.csv',delimiter=',')

        color_list=['grey','red']
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8,4),dpi=200)
        steel=[]
        concrete=[]
        for i in range(0,len(Img_Column)):
            for j in range(0,len(Img_Column[0])):
                if int(Img_Column[i][j])==1:
                    steel.append([j,-i])
                else:
                    concrete.append([j,-i])
        steel=np.array(steel)
        concrete=np.array(concrete)

        if len(steel) != 0:
            plt.scatter(steel[:,0], steel[:,1], color=color_list[1], marker='s', alpha=0.5, s=25)
        plt.scatter(concrete[:,0], concrete[:,1], color=color_list[0], marker='s', alpha=0.5, s=25)
        # for i in range(0,len(Img_Column)):
        #     for j in range(0,len(Img_Column[0])):
        #         # print(i,j,Img_Column[i][j])
        #         plt.scatter(j,-i,color=color_list[int(Img_Column[i][j])],marker='s',alpha=0.5,s=25)

        ax.axis('off')
        # plt.show()
        plt.savefig(f'{save_dir}/{num}_{section_num}.png',dpi=400,bbox_inches='tight',transparent=False)
        plt.close()

# if __name__ == "__main__":
    # List_imgs = range(100)
    # Len_imgs = len(List_imgs)
    # # num_cores = cpu_count()
    # num_cores = 8
    # subset1 = List_imgs[:Len_imgs // 8]
    # subset2 = List_imgs[Len_imgs // 8: Len_imgs // 4]
    # subset3 = List_imgs[Len_imgs // 4: (Len_imgs * 3) // 8]
    # subset4 = List_imgs[(Len_imgs * 3) // 8: Len_imgs // 2]
    # subset5 = List_imgs[Len_imgs // 2: (Len_imgs * 5) // 8]
    # subset6 = List_imgs[(Len_imgs * 5) // 8: (Len_imgs * 6) // 8]
    # subset7 = List_imgs[(Len_imgs * 6) // 8: (Len_imgs * 7) // 8]
    # subset8 = List_imgs[(Len_imgs * 7) // 8:]
    # List_subsets = [subset1, subset2, subset3, subset4, subset5, subset6, subset7, subset8]
    # p = Pool(num_cores)
    # single_worker(List_subsets[0])
    # # for i in range(num_cores):
    # #     p.apply_async(single_worker, args=(List_subsets[i]))
    # #
    # # # 当进程完成时，关闭进程池
    # # # 以下两行代码不需要改动
    # # p.close()
    # # p.join()

