from matplotlib.ticker import MultipleLocator
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

font_Times_New_Roman={"family":"Times New Roman",
                      # "style": "italic",
                    #   "weight":"heavy",
                      "size":15}

font_Times_New_Roman_label={"family":"Times New Roman",
                      # "style": "italic",
                    #   "weight":"heavy",
                      "size":15}

font_Song={"family":"SimSun",
           "style":"italic",
        #    "weight":"heavy",
           "size":15}
#坐标轴字体
plt.rcParams['font.sans-serif']=['Times New Roman']
plt.rc('axes', unicode_minus=False)



color_list=['C0','C1','C2','C3','C4','C5','C6','C7','C8','C9']

work_dir=r'F:\\Python_venv\\Deep_Learning\\pythonProject\MNIST-CYJ\\06 Plot_stiffness_number'
Data=np.loadtxt(fr'{work_dir}\\Train.txt')

Length=int(len(Data))
for num in range(0,10):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    for i in tqdm(range(Length)):
        if Data[i,1]==num:
            ax.scatter(Data[i,3]*100, Data[i,6]*100, c=color_list[int(Data[i,0])], marker='o',alpha=0.2, s=15)
    # for color_legend in range(10):
    #     ax.scatter(-100,-100,label=f'{color_legend}-Column', c=color_list[color_legend], marker='o', alpha=0.2, s=15)

    ax.set_xlabel("Area", fontproperties=font_Times_New_Roman)  # 添加x轴名
    ax.set_ylabel("Degradation", fontproperties=font_Times_New_Roman)

    ax.set_xlim(0, 40)  # 设置x轴的刻度范围
    ax.set_ylim(60, 120)

    x_major_locator = MultipleLocator(10)
    ax.xaxis.set_major_locator(x_major_locator)

    y_major_locator = MultipleLocator(20)
    ax.yaxis.set_major_locator(y_major_locator)

    x_minor_locator = MultipleLocator(5)
    ax.xaxis.set_minor_locator(x_minor_locator)

    y_minor_locator = MultipleLocator(10)
    ax.yaxis.set_minor_locator(y_minor_locator)

    # ax.legend(frameon=True, facecolor='White', edgecolor="None", fancybox=False, labelspacing=0.2,
    #           borderpad=0.05, handlelength=0.3, prop=font_Times_New_Roman_label, loc='lower right',
    #           ncol=3)

    bwith = 2  # 边框宽度设置为2
    ax = plt.gca()  # 获取边框
    ax.spines['top'].set_color('k')  # 设置上‘脊梁’为红色
    ax.spines['right'].set_color('k')  # 设置上‘脊梁’为无色
    ax.spines['bottom'].set_linewidth(bwith)
    ax.spines['left'].set_linewidth(bwith)
    ax.spines['top'].set_linewidth(bwith)
    ax.spines['right'].set_linewidth(bwith)

    ax.tick_params(axis='x', which='major', direction='in', labelsize=15, length=10, width=2,
                   pad=5)  # 主刻度x：朝向、长短、大小
    ax.tick_params(axis='x', which='minor', direction='in', color='k', labelsize=15, length=5, width=1,
                   pad=5)  # 副刻度x
    ax.tick_params(axis='y', which='major', direction='in', labelsize=15, length=10, width=2,
                   pad=5)  # 主刻度x
    ax.tick_params(axis='y', which='minor', direction='in', color='k', labelsize=15, length=5, width=1,
                   pad=5)  # 副刻度y

    # ax.grid(True, which="major", linestyle="--", color="lightgray", linewidth=1, zorder=-1)
    # ax.grid(True, which="minor", linestyle=":", color="lightgray", linewidth=1, zorder=-1)

    plt.savefig(f'{work_dir}/{num}.PNG',dpi=300)
    plt.close()

