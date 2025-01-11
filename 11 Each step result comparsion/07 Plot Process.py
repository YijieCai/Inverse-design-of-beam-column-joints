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
from PIL.Image import Resampling
from matplotlib import pyplot as plt
plt.rcParams['font.sans-serif']=['Times New Roman']
plt.rc('axes', unicode_minus=False)
num=[0,40,90,190,390,990]
section_num_list=[1,2,3,4]
Lim=[300,500,500,300]
for section_num in section_num_list:
    for i in range(len(num)):
        iii=section_num
        choice=[0,3000,6000,9000]
        diffusiondataset = DiffusionDataset("F:/Python_venv/Deep_Learning/pythonProject/MNIST-CYJ/03 Dataset/Test/Dataset_Test_unweak_2D.h5")
        x1=diffusiondataset[choice[iii-1]][0]
        y1=diffusiondataset[choice[iii-1]][1]

        disp1=np.loadtxt(f'./04Result/{int(num[i]/10)}/Disp_{iii}.txt',usecols=1)
        reaction1=np.loadtxt(f'./04Result/{int(num[i]/10)}/Reaction_{iii}.txt',usecols=1)
        # print(y1)

        fig, (ax) = plt.subplots(1, 1)
        ax.grid(True, which="major", linestyle="--", color="lightgray", linewidth=1, zorder=-1,alpha=0.3)
        ax.grid(True, which="minor", linestyle="--", color="lightgray", linewidth=1, zorder=-1,alpha=0.3)
        ax.scatter(y1[:,0],y1[:,1]/1000,color='red',label='Target',zorder=1)
        ax.plot(y1[:,0],y1[:,1]/1000,color='red',label='Target',zorder=1)
        ax.plot(disp1,reaction1/1000,color='blue',label='DDPM Generate',zorder=-1)


        # ax.set_xlabel("Displacement (mm)", fontproperties=font_Times_New_Roman)  # 添加x轴名
        # ax.set_ylabel("Force (kN)", fontproperties=font_Times_New_Roman)
        ax.set_xlim(-100, 100)  # 设置x轴的刻度范围
        ax.set_ylim(-Lim[section_num-1], Lim[section_num-1])

        # 主刻度
        x_major_locator = MultipleLocator(50)
        ax.xaxis.set_major_locator(x_major_locator)

        y_major_locator = MultipleLocator(Lim[section_num-1]/2)
        ax.yaxis.set_major_locator(y_major_locator)

        # 副刻度
        x_minor_locator = MultipleLocator(25)
        ax.xaxis.set_minor_locator(x_minor_locator)

        y_minor_locator = MultipleLocator(Lim[section_num-1]/4)
        ax.yaxis.set_minor_locator(y_minor_locator)

        bwith = 2  # 边框宽度设置为2
        ax = plt.gca()  # 获取边框
        ax.spines['top'].set_color('k')  # 设置上‘脊梁’为红色
        ax.spines['right'].set_color('k')  # 设置上‘脊梁’为无色
        ax.spines['bottom'].set_linewidth(bwith)
        ax.spines['left'].set_linewidth(bwith)
        ax.spines['top'].set_linewidth(bwith)
        ax.spines['right'].set_linewidth(bwith)

        ax.tick_params(axis='x', which='major', direction='in', labelsize=0, length=10, width=2,
                       pad=5)  # 主刻度x：朝向、长短、大小
        ax.tick_params(axis='x', which='minor', direction='in', color='k', labelsize=0, length=5, width=1,
                       pad=5)  # 副刻度x
        ax.tick_params(axis='y', which='major', direction='in', labelsize=0, length=10, width=2,
                       pad=5)  # 主刻度x
        ax.tick_params(axis='y', which='minor', direction='in', color='k', labelsize=0, length=5, width=1,
                       pad=5)  # 副刻度y

        # ax.axis('off')


        plt.savefig(fr'F:\Python_venv\Deep_Learning\pythonProject\MNIST-CYJ\11 Each step result comparsion\06Plot\B_{i}.png', bbox_inches='tight', transparent=False)
        plt.close()


    for i in range(len(num)):
        source_file_path=fr'F:\Python_venv\Deep_Learning\pythonProject\MNIST-CYJ\11 Each step result comparsion\02Section'
        destination_folder=fr'F:\Python_venv\Deep_Learning\pythonProject\MNIST-CYJ\11 Each step result comparsion\06Plot'
        shutil.copy(f'{source_file_path}/{int(num[i]/10)}_{section_num}.png',
                    f'{destination_folder}/A_{i}.png')
        # source_file_path=fr'F:\Python_venv\Deep_Learning\pythonProject\MNIST-CYJ\11 Each step result comparsion\04Result'
        # shutil.copy(f'{source_file_path}/{int(num[i]/10)}/{section_num}.png',
        #             f'{destination_folder}/B_{i}.png')

    # 定义输入和输出文件夹路径
    input_folder = r'F:\Python_venv\Deep_Learning\pythonProject\MNIST-CYJ\11 Each step result comparsion\06Plot'
    output_file = fr'F:\Python_venv\Deep_Learning\pythonProject\MNIST-CYJ\11 Each step result comparsion\image_{section_num}.jpg'


    # 图像类别及其对应的目标尺寸
    image_classes = {
        'A': {'target_size': (1200, 600)},
        'B': {'target_size': (1200, 900)}
    }

    # 加载并调整图像大小
    images_A = []
    images_B = []

    for class_name in image_classes.keys():
        for i in range(len(num)):
            file_path = f"{input_folder}\{class_name}_{i}.png"
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File {file_path} does not exist.")

            img = Image.open(file_path)
            if class_name == 'B':
                resized_img = img.resize(image_classes[class_name]['target_size'], Resampling.LANCZOS)
                images_B.append(resized_img)
            else:
                resized_img = img.resize(image_classes[class_name]['target_size'], Resampling.LANCZOS)
                images_A.append(resized_img)

    # 计算画布的宽度和高度
    arranged_width = max(len(images_A) * images_A[0].width, len(images_B) * images_B[0].width)
    arranged_height = images_A[0].height + images_B[0].height

    # 创建一个新的空白图像用于排列所有图片
    arranged_image = Image.new('RGB', (arranged_width, arranged_height))

    # 将A类图片粘贴到第一行
    x_offset = 0
    y_offset = 0
    for img in images_A:
        arranged_image.paste(img, (x_offset, y_offset))
        x_offset += img.width

    # 将B类图片粘贴到第二行
    x_offset = 0
    y_offset = images_A[0].height
    for img in images_B:
        arranged_image.paste(img, (x_offset, y_offset))
        x_offset += img.width

    # 保存结果图像
    arranged_image.save(output_file)