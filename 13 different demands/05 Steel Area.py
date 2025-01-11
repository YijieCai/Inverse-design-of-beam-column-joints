import numpy as np

work_dir=fr'F:\Python_venv\Deep_Learning\pythonProject\MNIST-CYJ\13 different demands\01Img'

Area_list=[]
for target in range(4):
    for number in range(4):
        # 柱子
        column_data=np.loadtxt(fr'{work_dir}/{target}/x_{number+target*100}_999.csv',delimiter=',')
        beam_data = np.loadtxt(fr'{work_dir}/{target}/x_{number+target*100}_999_.csv',delimiter=',')

        column_counts=np.sum(column_data == 1)
        beam_counts = np.sum(beam_data == 1)

        Area_Column=column_counts/1600
        Area_beam = beam_counts/1600
        Area_list.append([Area_Column,Area_beam,number+target*100])
    Section_data = np.loadtxt(fr'{work_dir}/{target}/Target_{target * 100}.csv', delimiter=',')
    column_data=Section_data[:,:40]
    beam_data=Section_data[:,40:]
    column_counts = np.sum(column_data == 1)
    beam_counts = np.sum(beam_data == 1)

    Area_Column = column_counts / 1600
    Area_beam = beam_counts / 1600
    Area_list.append([Area_Column, Area_beam, 4+target * 100])
np.savetxt('Area_txt',Area_list,fmt='%.4f')


