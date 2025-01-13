<div align=center>
  
# Inverse Design of Joints Section
  
</div> 


* *If you need the .pt file. Please contact with caiyijiehehe@gmail.com*

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
</div><br>    


* ## ⚛️ **_Datasets & Pre-trained models_**    
[**🔗The MNIST-Section dataset**](https://github.com/YijieCai/Inverse-design-of-beam-column-joints/releases/tag/Dataset)     
[**🔗The weights of the DDPM**](https://github.com/YijieCai/Inverse-design-of-beam-column-joints/releases/tag/Weight)



* The structure of the folder is as follows:
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
