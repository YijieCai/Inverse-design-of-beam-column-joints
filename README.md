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
</div><br>    


<div align=center>
  <img width="200" src="Chart/Section_1.gif"/>
  <img width="200" src="Chart/Section_2.gif"/>
  <img width="200" src="Chart/Section_3.gif"/>
  <img width="200" src="Chart/Section_4.gif"/>
</div><br>   


<div align=center>
  <img width="200" src="Chart/Section_DDPM.gif"/>
   <div align=center><strong>Results of the inverse design of Joints section</strong></div>
   <div align=center><strong>节点梁柱截面逆向设计动态结果</strong></div>
</div><br>   


<div align=center>
  <img width="700" src="Chart/40x80 Diffusion model architecture.png"/>
   <div align=center><strong>2D-40x80 Diffusion model architecture</strong></div>
   <div align=center><strong>2D-40x80 模型框架</strong></div>
</div><br>   

<!-- 数据集 -->
* ## **_The MNIST-Section Dataset_**

The digit portions were mapped to steel, while the remaining portions were mapped to concrete. Figure illustrates the relationship between the proportion of steel in the cross-section and the strength degradation of the columns after loading, categorized into ten classes corresponding to the digits 0 through 9. Weak column and strong beam lead to serious weakening of joint performance, which is not conducive to the generation of accuracy section. Therefore, the data set is cleaned as shown in Figure(red is eliminated).
</div><br>数字部分映射到钢，而其余部分映射到混凝土。图说明了横截面中的钢比例与加载后柱强度退化之间的关系，分为十类，对应于数字0到9。弱柱和强梁导致节点性能严重弱化，不利于模型生成准确有效的截面。因此，对数据集进行清洗（红色为被消除的）。
</div><br>

<div align=center>
  <img width="375" src="Chart/Dataset_Performance.png"/>
  <img width="325" src="Chart/All_Performance.PNG"/>
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
          |--Make Dataset based on MNIST
                |--Create Section Based on Mnist (Features)
                |--Calculate Using OpenSeespy (Label)
                |--Data Cleaning
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
