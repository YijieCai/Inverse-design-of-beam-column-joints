import openseespy.opensees as ops
import numpy as np
from multiprocessing import cpu_count    # 查看cpu核心数
from multiprocessing import Pool         # 并行处理必备，进程池
import os
from tqdm import tqdm

def single_worker(subset,dir):
    length_subset=len(subset)
    for iii in tqdm(range(length_subset)):
        file=subset[iii][:-4]
        if not os.path.exists(f'{dir}\\{file}'):
            os.makedirs(f'{dir}\\{file}')

        ops.wipe() # 初始清空
        ops.model('basic', '-ndm', 2, '-ndf', 3)  # frame 2D

        IDcoreC=1
        fpc_cover=-29.39
        epsc0_cover=-0.0025
        fpcu_cover=-10.36
        epsU_cover=-0.015
        lambda_cover=0.1
        ft_cover=2.8
        Ets_cover=3000
        ops.uniaxialMaterial('Concrete02', IDcoreC, fpc_cover, epsc0_cover, fpcu_cover, epsU_cover,lambda_cover,ft_cover,Ets_cover)

        IDSteel = 2
        Fy_Steel = 400
        E0_Steel = 210000
        bs_Steel = 0.001
        R0 = 16
        cR1 = 0.925
        cR2 = 0.15
        ops.uniaxialMaterial('Steel02', IDSteel, Fy_Steel, E0_Steel, bs_Steel,R0,cR1,cR2)

        OpenSees_Import=np.loadtxt(fr'F:\Python_venv\Deep_Learning\pythonProject\MNIST-CYJ\01 Create Dataset\\Train\\{file}.txt')
        fiber_column_section=1
        ops.section('Fiber', fiber_column_section)
        for fiber in OpenSees_Import:
            y=fiber[0]
            z=fiber[1]
            A=fiber[2]
            mat=int(fiber[3])+1
            ops.fiber(z,y,A,mat)

        file_int=(int(file)+10000)%60000
        # file_int=(int(file)+5000)%10000

        OpenSees_Import=np.loadtxt(rf'F:\Python_venv\Deep_Learning\pythonProject\MNIST-CYJ\01 Create Dataset\\Train\\{file_int}.txt')
        fiber_beam_section=2

        ops.section('Fiber', fiber_beam_section)
        for fiber in OpenSees_Import:
            y=fiber[0]/2
            z=fiber[1]
            A=fiber[2]/2
            mat=int(fiber[3])+1
            ops.fiber(z,y,A,mat)

        # 节点坐标(x,y)
        ops.node(1, 0, 0)
        ops.node(2, 0, 700)
        ops.node(3, 0, 1400)
        ops.node(4, 800, 700)
        ops.node(5, -800, 700)
        ops.node(6, 1650, 700)
        ops.node(7, -1650, 700)

        #1节点的x平动,y平动,转动固定
        ops.fix(1, 1, 1, 0)
        ops.fix(3, 1, 0, 0)

        coordTransf = "PDelta"  # Linear, PDelta, Corotational
        IDColumnTransf=1
        ops.geomTransf(coordTransf, IDColumnTransf)
        numIntgrPts=5
        IDColumnIntegration = 1
        ops.beamIntegration('Lobatto', IDColumnIntegration, fiber_column_section, numIntgrPts)

        IDBeamIntegration = 2
        ops.beamIntegration('Lobatto', IDBeamIntegration, fiber_beam_section, numIntgrPts)

        ops.element('dispBeamColumn', 1, 1, 2, IDColumnTransf, IDColumnIntegration)
        ops.element('dispBeamColumn', 2, 2, 3, IDColumnTransf, IDColumnIntegration)
        ops.element('dispBeamColumn', 3, 2, 4, IDColumnTransf, IDBeamIntegration)
        ops.element('dispBeamColumn', 4, 2, 5, IDColumnTransf, IDBeamIntegration)
        ops.element('dispBeamColumn', 5, 4, 6, IDColumnTransf, IDBeamIntegration)
        ops.element('dispBeamColumn', 6, 5, 7, IDColumnTransf, IDBeamIntegration)

        ops.timeSeries('Linear', 11)
        ops.pattern("Plain", 100, 11)
        ops.load(3, 0, -398000, 0)
        ops.constraints("Penalty", 1e20, 1e20)
        ops.numberer("RCM")
        ops.system("BandGeneral")
        ops.test('NormDispIncr', 1e-4, 2000)
        ops.algorithm("KrylovNewton")
        ops.integrator("LoadControl", 0.1)
        ops.analysis("Static")
        ops.analyze(10)
        ops.loadConst("-time",0.0)
        ops.recorder('Node', '-file', f"{dir}\\{file}\\Disp.txt","-time",'-node', 6, '-dof',2, 'disp')
        ops.recorder('Node', '-file', f"{dir}\\{file}\\Reaction.txt","-time",'-node', 6, '-dof',2, 'reaction')

        ops.timeSeries('Linear', 22)
        ops.pattern("Plain", 200, 22)
        ops.sp(6, 2, 1)
        ops.sp(7, 2, -1)
        ops.test('NormDispIncr', 1e-4, 2000)
        increments = [0.06, -0.06, -0.06, 0.06,
                      0.12, -0.12, -0.12, 0.12,
                      0.18, -0.18, -0.18, 0.18,
                      0.24, -0.24, -0.24, 0.24,
                      0.32, -0.32, -0.32, 0.32,
                      0.40, -0.40, -0.40, 0.40,
                      0.48, -0.48, -0.48, 0.48,
                      0.56, -0.56, -0.56, 0.56,
                      0.64, -0.64, -0.64, 0.64,
                      0.72, -0.72, -0.72, 0.72,
                      0.80, -0.80, -0.80, 0.80,
                      0.88, -0.88, -0.88, 0.88,
                      0.96, -0.96, -0.96, 0.96]

        # increments = [0.06136, -0.06136, -0.0488, 0.0488,
        #               0.12376, -0.12376, -0.11752, 0.11752,
        #               0.18616, -0.18616, -0.17992, 0.17992,
        #               0.24232, -0.24232, -0.24856, 0.24856,
        #               0.25064, -0.25064, -0.24024, 0.24024,
        #               0.35256, -0.35256, -0.3588, 0.3588,
        #               0.35256, -0.35256, -0.36296, 0.36296,
        #               0.43784, -0.43784, -0.4836, 0.4836,
        #               0.55016, -0.55016, -0.57096, 0.57096,
        #               0.52936, -0.52936, -0.56264, 0.56264,
        #               0.63752, -0.63752, -0.66456, 0.66456,
        #               0.6396, -0.6396, -0.66664, 0.66664,
        #               0.7644, -0.7644, -0.77064, 0.77064,
        #               0.75608, -0.75608, -0.78728, 0.78728,
        #               0.90168, -0.90168, -0.90168, 0.90168,
        #               0.90168, -0.90168, -0.90168, 0.90168,
        #               1.05168, -1.05168, -1.05168, 1.05168,
        #               1.05168, -1.05168, -1.05168, 1.05168]
        #
        for s in range(52):
            ops.integrator("DisplacementControl", 6, 2, increments[s])
            ops.integrator("DisplacementControl", 7, 2, increments[s])
            ops.analysis("Static")
            ops.analyze(100)

        ops.remove('recorders')
        ops.reset()
        ops.remove('loadPattern', 100)
        ops.reset()
        ops.remove('loadPattern', 200)
        ops.reset()
        ops.remove('timeSeries', 11)
        ops.reset()
        ops.remove('timeSeries', 22)
        ops.reset()
        ops.wipeAnalysis()

if __name__ == "__main__":
    # dirdataset = 'F:\\Python_venv\\Deep_Learning\\pythonProject\\MNIST\\OpenSees_Section\\Test'
    dirdataset = r'F:\Python_venv\Deep_Learning\pythonProject\MNIST-CYJ\01 Create Dataset\Train'

    # 将该文件夹下的所有文件名存入列表
    List_imgs = os.listdir(dirdataset)

    # List_imgs-List_exist+exclusion
    List_exist=os.listdir(fr'F:\Python_venv\Deep_Learning\pythonProject\MNIST-CYJ\02 OS_Calculate_Beam\Train')
    exclusion=['12929','16689','1682','20435','24189','27931','31694','35444','39175','42939','46689','50434','54192','5436','57905','9195']
    for i in List_exist:
        List_imgs.remove(f'{i}.txt')
    for i in exclusion:
        List_imgs.append(f'{i}.txt')
    print(len(List_imgs))
    # 自定义排序键函数，从文件名中提取数字部分
    def extract_number(filename):
        return int(filename.split('.')[0])

    # 使用sorted函数进行排序，并使用自定义排序键
    List_imgs = sorted(List_imgs, key=extract_number)

    # 获取列表的长度
    Len_imgs = len(List_imgs)  # 数据集长度
    num_cores = cpu_count()  # cpu核心数
    if num_cores == 2:  # 双核，将所有数据集分成两个子数据集
        subset1 = List_imgs[:Len_imgs // 2]
        subset2 = List_imgs[Len_imgs // 2:]

        List_subsets = [subset1, subset2]

    elif num_cores == 4:  # 四核，将所有数据集分成四个子数据集
        subset1 = List_imgs[:Len_imgs // 4]
        subset2 = List_imgs[Len_imgs // 4: Len_imgs // 2]
        subset3 = List_imgs[Len_imgs // 2: (Len_imgs * 3) // 4]
        subset4 = List_imgs[(Len_imgs * 3) // 4:]

        List_subsets = [subset1, subset2, subset3, subset4]

    elif num_cores == 8:  # 八核以上，将所有数据集分成八个子数据集
        num_cores = 8
        subset1 = List_imgs[:Len_imgs // 8]
        subset2 = List_imgs[Len_imgs // 8: Len_imgs // 4]
        subset3 = List_imgs[Len_imgs // 4: (Len_imgs * 3) // 8]
        subset4 = List_imgs[(Len_imgs * 3) // 8: Len_imgs // 2]
        subset5 = List_imgs[Len_imgs // 2: (Len_imgs * 5) // 8]
        subset6 = List_imgs[(Len_imgs * 5) // 8: (Len_imgs * 6) // 8]
        subset7 = List_imgs[(Len_imgs * 6) // 8: (Len_imgs * 7) // 8]
        subset8 = List_imgs[(Len_imgs * 7) // 8:]

        List_subsets = [subset1, subset2, subset3, subset4, subset5, subset6, subset7, subset8]

    elif num_cores >= 16:  # 16核以上，将所有数据集分成16个子数据集
        num_cores = 16
        subset1 = List_imgs[: (Len_imgs * 1) // 16]
        subset2 = List_imgs[(Len_imgs * 1) // 16: (Len_imgs * 2) // 16]
        subset3 = List_imgs[(Len_imgs * 2) // 16: (Len_imgs * 3) // 16]
        subset4 = List_imgs[(Len_imgs * 3) // 16: (Len_imgs * 4) // 16]
        subset5 = List_imgs[(Len_imgs * 4) // 16: (Len_imgs * 5) // 16]
        subset6 = List_imgs[(Len_imgs * 5) // 16: (Len_imgs * 6) // 16]
        subset7 = List_imgs[(Len_imgs * 6) // 16: (Len_imgs * 7) // 16]
        subset8 = List_imgs[(Len_imgs * 7) // 16: (Len_imgs * 8) // 16]
        subset9 = List_imgs[(Len_imgs * 8) // 16: (Len_imgs * 9) // 16]
        subset10 = List_imgs[(Len_imgs * 9) // 16: (Len_imgs * 10) // 16]
        subset11 = List_imgs[(Len_imgs * 10) // 16: (Len_imgs * 11) // 16]
        subset12 = List_imgs[(Len_imgs * 11) // 16: (Len_imgs * 12) // 16]
        subset13 = List_imgs[(Len_imgs * 12) // 16: (Len_imgs * 13) // 16]
        subset14 = List_imgs[(Len_imgs * 13) // 16: (Len_imgs * 14) // 16]
        subset15 = List_imgs[(Len_imgs * 14) // 16: (Len_imgs * 15) // 16]
        subset16 = List_imgs[(Len_imgs * 15) // 16:]

        List_subsets = [subset1, subset2, subset3, subset4, subset5, subset6, subset7, subset8, subset9, subset10, subset11, subset12, subset13, subset14, subset15, subset16]


    dir=r'F:\Python_venv\Deep_Learning\pythonProject\MNIST-CYJ\02 OS_Calculate_Beam\Train'
    # 开辟进程池，不需要改动
    # num_cores为cpu核心数，也就是开启的进程数
    p = Pool(num_cores)

    # 对每个进程分配工作
    for i in range(num_cores):
        # 格式：p.apply_async(task, args=(...))
        # task：当前进程需要进行的任务/函数，只需要填写函数名
        # args：task函数中所需要传入的参数
        # 注意看 List_subsets[i] 就是传入不同的数据子集
        p.apply_async(single_worker, args=(List_subsets[i], dir))

    # 当进程完成时，关闭进程池
    # 以下两行代码不需要改动
    p.close()
    p.join()