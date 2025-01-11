import shutil
from PIL import Image, ImageSequence
from Dataset import DiffusionDataset
import os
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

from PIL.Image import Resampling
from matplotlib import pyplot as plt
plt.rcParams['font.sans-serif']=['Times New Roman']
plt.rc('axes', unicode_minus=False)
num=[0,1,2,3]
section_num_list=[0,1,2,3]
Lim=[300,500,500,300]
for target in range(4):
    fig, (ax) = plt.subplots(2, 4,figsize=(20,8),gridspec_kw={'height_ratios': [1, 2]})
    for number in range(4):

        choice=[0,1000,2000,3000]
        diffusiondataset = DiffusionDataset("F:/Python_venv/Deep_Learning/pythonProject/MNIST-CYJ/03 Dataset/Test/Dataset_Test_unweak_2D.h5")
        x1=diffusiondataset[choice[target]][0]
        y1=diffusiondataset[choice[target]][1]

        disp1=np.loadtxt(f'./03Result/{target}/Disp_{number+target*100}_999.txt',usecols=1)
        reaction1=np.loadtxt(f'./03Result/{target}/Reaction_{number+target*100}_999.txt',usecols=1)

        work_dir = fr'F:/Python_venv/Deep_Learning/pythonProject/MNIST-CYJ/13 different demands/01Img'
        Img_Column = np.loadtxt(fr'{work_dir}/{target}/DDPM_Section_{number+target*100}_999.csv',delimiter=',')
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
        color_list=['grey','red']
        ax[0,number].scatter(steel[:, 0], steel[:, 1], color=color_list[1], marker='s', alpha=0.5, s=25)
        ax[0,number].scatter(concrete[:,0], concrete[:,1], color=color_list[0], marker='s', alpha=0.5, s=25)
        ax[0, number].axis('off')



    

        ax[1,number].grid(True, which="major", linestyle="--", color="lightgray", linewidth=1, zorder=-1,alpha=0.3)
        # ax.grid(True, which="minor", linestyle="--", color="lightgray", linewidth=1, zorder=-1,alpha=0.3)
        ax[1,number].scatter(y1[:,0],y1[:,1]/1000,color='red',label='Target',zorder=1)
        ax[1,number].plot(y1[:,0],y1[:,1]/1000,color='red',label='Target',zorder=1)
        ax[1,number].plot(disp1,reaction1/1000,color='blue',label='DDPM Generate',zorder=-1)

        # ax[1,number].scatter(y1[:,0],y1[:,1]/1000,color='red',label='Target',zorder=1)
        # ax[1,number].plot(y1[:,0],y1[:,1]/1000,color='red',label='Target',zorder=1)
        # ax[1,number].plot(disp1,reaction1/1000,color='blue',label='DDPM Generate',zorder=-1)

        x1_, y1_ = skeleton_curve(disp1, reaction1)
        y1 = np.array(y1[1:, 1])
        y1_ = y1_[1:]
        mse = np.mean(abs(y1_ - y1) / y1)
        ax[1,number].text(-90,400,f'ME={mse*100:.2f}%',font=font_Times_New_Roman)


        # ax.set_xlabel("Displacement (mm)", fontproperties=font_Times_New_Roman)  # 添加x轴名
        # ax.set_ylabel("Force (kN)", fontproperties=font_Times_New_Roman)
        ax[1,number].set_xlim(-100, 100)  # 设置x轴的刻度范围
        ax[1,number].set_ylim(-500, 500)

        # 主刻度
        x_major_locator = MultipleLocator(50)
        ax[1,number].xaxis.set_major_locator(x_major_locator)

        y_major_locator = MultipleLocator(250)
        ax[1,number].yaxis.set_major_locator(y_major_locator)

        # 副刻度
        x_minor_locator = MultipleLocator(25)
        ax[1,number].xaxis.set_minor_locator(x_minor_locator)

        y_minor_locator = MultipleLocator(125)
        ax[1,number].yaxis.set_minor_locator(y_minor_locator)


        ax[1,number].tick_params(axis='x', which='major', direction='in', labelsize=0, length=5, width=1,
                       pad=5)  # 主刻度x：朝向、长短、大小
        ax[1,number].tick_params(axis='x', which='minor', direction='in', color='k', labelsize=0, length=2.5, width=0.5,
                       pad=5)  # 副刻度x
        ax[1,number].tick_params(axis='y', which='major', direction='in', labelsize=0, length=5, width=1,
                       pad=5)  # 主刻度x
        ax[1,number].tick_params(axis='y', which='minor', direction='in', color='k', labelsize=0, length=2.5, width=0.5,
                       pad=5)  # 副刻度y



        # ax.axis('off')

    plt.subplots_adjust(hspace=0.02, wspace=0.02)
    plt.tight_layout()
    plt.savefig(fr'F:\Python_venv\Deep_Learning\pythonProject\MNIST-CYJ\13 different demands\04Plot\{target}.png')
    plt.close()

    # for i in range(len(num)):
    #     source_file_path=fr'F:\Python_venv\Deep_Learning\pythonProject\MNIST-CYJ\11 Each step result comparsion\02Section'
    #     destination_folder=fr'F:\Python_venv\Deep_Learning\pythonProject\MNIST-CYJ\11 Each step result comparsion\06Plot'
    #     shutil.copy(f'{source_file_path}/{int(num[i]/10)}_{section_num}.png',
    #                 f'{destination_folder}/A_{i}.png')
    #     # source_file_path=fr'F:\Python_venv\Deep_Learning\pythonProject\MNIST-CYJ\11 Each step result comparsion\04Result'
    #     # shutil.copy(f'{source_file_path}/{int(num[i]/10)}/{section_num}.png',
    #     #             f'{destination_folder}/B_{i}.png')
    #
    # # 定义输入和输出文件夹路径
    # input_folder = r'F:\Python_venv\Deep_Learning\pythonProject\MNIST-CYJ\11 Each step result comparsion\06Plot'
    # output_file = fr'F:\Python_venv\Deep_Learning\pythonProject\MNIST-CYJ\11 Each step result comparsion\image_{section_num}.jpg'
    #
    #
    # # 图像类别及其对应的目标尺寸
    # image_classes = {
    #     'A': {'target_size': (1200, 600)},
    #     'B': {'target_size': (1200, 900)}
    # }
    #
    # # 加载并调整图像大小
    # images_A = []
    # images_B = []
    #
    # for class_name in image_classes.keys():
    #     for i in range(len(num)):
    #         file_path = f"{input_folder}\{class_name}_{i}.png"
    #         if not os.path.exists(file_path):
    #             raise FileNotFoundError(f"File {file_path} does not exist.")
    #
    #         img = Image.open(file_path)
    #         if class_name == 'B':
    #             resized_img = img.resize(image_classes[class_name]['target_size'], Resampling.LANCZOS)
    #             images_B.append(resized_img)
    #         else:
    #             resized_img = img.resize(image_classes[class_name]['target_size'], Resampling.LANCZOS)
    #             images_A.append(resized_img)
    #
    # # 计算画布的宽度和高度
    # arranged_width = max(len(images_A) * images_A[0].width, len(images_B) * images_B[0].width)
    # arranged_height = images_A[0].height + images_B[0].height
    #
    # # 创建一个新的空白图像用于排列所有图片
    # arranged_image = Image.new('RGB', (arranged_width, arranged_height))
    #
    # # 将A类图片粘贴到第一行
    # x_offset = 0
    # y_offset = 0
    # for img in images_A:
    #     arranged_image.paste(img, (x_offset, y_offset))
    #     x_offset += img.width
    #
    # # 将B类图片粘贴到第二行
    # x_offset = 0
    # y_offset = images_A[0].height
    # for img in images_B:
    #     arranged_image.paste(img, (x_offset, y_offset))
    #     x_offset += img.width
    #
    # # 保存结果图像
    # arranged_image.save(output_file)