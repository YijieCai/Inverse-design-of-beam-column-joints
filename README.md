<div align=center>
  
# Inverse Design of Joints Section
# 梁柱节点截面逆向设计
</div> 


* *If you need more information. Please contact with caiyijiehehe@gmail.com*
* 如果你需要更多信息，请联系这个邮箱caiyijiehehe@gmail.com
<div align=center>
  <img width="200" src="Chart/Curve_1.gif"/>
  <img width="200" src="Chart/Curve_2.gif"/>
  <img width="200" src="Chart/Curve_3.gif"/>
  <img width="200" src="Chart/Curve_4.gif"/>
  <img width="200" src="Chart/Section_1.gif"/>
  <img width="200" src="Chart/Section_2.gif"/>
  <img width="200" src="Chart/Section_3.gif"/>
  <img width="200" src="Chart/Section_4.gif"/>
  <img width="200" src="Chart/Section_DDPM.gif"/>
   <div align=center><strong>Results of the inverse design of Joints section</strong></div>
   <div align=center><strong>节点梁柱截面逆向设计动态结果</strong></div>
</div><br>    

<div align=center>
  <img width="700" src="Chart/40x80 Diffusion model architecture.png"/>
   <div align=center><strong>2D-40x80 Diffusion model architecture</strong></div>
   <div align=center><strong>2D-40x80 模型框架</strong></div>
</div><br>   

<div align=center>
  <img width="500" src="Chart/Dataset_Performance.png"/>
  <img width="200" src="Chart/All_Performance.PNG"/>
   <div align=center><strong>Data cleaning(数据清洗)</strong></div>
</div><br>   


* ## ⚛️ **_Datasets & Weights_**  
* ## ⚛️ **_数据集和权重文件_**  
[**🔗The MNIST-Section dataset(数据集文件)**](https://github.com/YijieCai/Inverse-design-of-beam-column-joints/releases/tag/Dataset)     
[**🔗The Weights of the DDPM(权重文件)**](https://github.com/YijieCai/Inverse-design-of-beam-column-joints/releases/tag/Weight)



* The structure of the folder is as follows:
* 流程图如下:
```
  |--Main folder
          |--Diffusion_Dataset
                |--Test
                      |--Dataset_Test_unweak_2D.h5
                      |--Dataset_Test_unweak_3D.h5 (Poor effect)
                |--Train
                      |--Dataset_Train_unweak_2D.h5
                      |--Dataset_Train_unweak_3D.h5 (Poor effect)
          |--DDPM
                |--ConditionDiffusionModel.py
                |--Dataset.py
                |--Train_Diffusion.py
          |--Analysis
                |--Result of each model
                |--Each epoch result comparsion
                |--Each step result comparison
                |--Model accuracy
                |--Different demands
                |--Exceed Target
```
