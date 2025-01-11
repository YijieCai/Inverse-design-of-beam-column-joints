import numpy as np
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm

Element_Abaqus_Load=np.loadtxt('Elements.txt',skiprows=1,delimiter=',')
Node_Abaqus_Load=np.loadtxt('Nodes.txt',skiprows=1,delimiter=',')

def PolyArea(x,y):
    return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))
number_section=1000
# number 是第number个截面
step_list=[0,999]
for number in tqdm(range(number_section)):
    for i in step_list:
        img=np.loadtxt(f'./01Img/x_{number}_{i}_.csv',delimiter=',')
        img=img.ravel()
        OpenSees_Import=np.empty([0,4])
        for index,Element in enumerate(Element_Abaqus_Load):
            Temp_Import=np.zeros([1,4])
            ElementTag=int(Element[0])
            Element1=int(Element[1])
            Element2=int(Element[2])
            Element3=int(Element[3])
            Element4=int(Element[4])
            Element1_Coordinate=Node_Abaqus_Load[Element1-1,[1,2]]
            Element2_Coordinate=Node_Abaqus_Load[Element2-1,[1,2]]
            Element3_Coordinate=Node_Abaqus_Load[Element3-1,[1,2]]
            Element4_Coordinate=Node_Abaqus_Load[Element4-1,[1,2]]
            Element_Coordinate=(Element1_Coordinate+Element2_Coordinate+Element3_Coordinate+Element4_Coordinate)/4
            x = [Element1_Coordinate[0],Element2_Coordinate[0],Element3_Coordinate[0],Element4_Coordinate[0]]
            y = [-Element1_Coordinate[1],-Element2_Coordinate[1],-Element3_Coordinate[1],-Element4_Coordinate[1]]
            Element_Area = PolyArea(x,y)
            Temp_Import[0,0]=Element_Coordinate[0]
            Temp_Import[0,1]=-Element_Coordinate[1]
            Temp_Import[0,2]=Element_Area
            Temp_Import[0,3]=int(img[index])
            OpenSees_Import = np.append(OpenSees_Import, Temp_Import, axis=0)
        np.savetxt(f'./02Section/x_{number}_{i}_.txt',OpenSees_Import)

for number in tqdm(range(number_section)):
    for i in step_list:
        img=np.loadtxt(f'./01Img/x_{number}_{i}.csv',delimiter=',')
        img=img.ravel()
        OpenSees_Import=np.empty([0,4])
        for index,Element in enumerate(Element_Abaqus_Load):
            Temp_Import=np.zeros([1,4])
            ElementTag=int(Element[0])
            Element1=int(Element[1])
            Element2=int(Element[2])
            Element3=int(Element[3])
            Element4=int(Element[4])
            Element1_Coordinate=Node_Abaqus_Load[Element1-1,[1,2]]
            Element2_Coordinate=Node_Abaqus_Load[Element2-1,[1,2]]
            Element3_Coordinate=Node_Abaqus_Load[Element3-1,[1,2]]
            Element4_Coordinate=Node_Abaqus_Load[Element4-1,[1,2]]
            Element_Coordinate=(Element1_Coordinate+Element2_Coordinate+Element3_Coordinate+Element4_Coordinate)/4
            x = [Element1_Coordinate[0],Element2_Coordinate[0],Element3_Coordinate[0],Element4_Coordinate[0]]
            y = [-Element1_Coordinate[1],-Element2_Coordinate[1],-Element3_Coordinate[1],-Element4_Coordinate[1]]
            Element_Area = PolyArea(x,y)
            Temp_Import[0,0]=Element_Coordinate[0]
            Temp_Import[0,1]=-Element_Coordinate[1]
            Temp_Import[0,2]=Element_Area
            Temp_Import[0,3]=int(img[index])
            OpenSees_Import = np.append(OpenSees_Import, Temp_Import, axis=0)
        np.savetxt(f'./02Section/x_{number}_{i}.txt',OpenSees_Import)