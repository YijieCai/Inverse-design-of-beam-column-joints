import torch
import torchvision
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from diffusers import DDPMScheduler,DDIMScheduler,UNet2DModel
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import numpy as np
from torch.utils.data import Dataset
import h5py
from diffusers import DDPMPipeline
from Dataset import DiffusionDataset

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device=torch.device('cpu')
noise_scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule='squaredcos_cap_v2')
model = torch.load(r'F:/Python_venv/Deep_Learning/pythonProject/MNIST-CYJ/05 DDPM/model98.pt')
model=model.to(device)
model.eval()

for jjj in range(10):
    number_section=100*jjj
    sample=torch.randn(100,1,40,80).to(device)

    choice=[]
    for i in range(number_section,number_section+100):
        choice.append(i)
    # choice=[0,0,0,0]

    diffusiondataset = DiffusionDataset("F:/Python_venv/Deep_Learning/pythonProject/MNIST-CYJ/03 Dataset/Test/Dataset_Test_unweak_2D.h5")
    x_list=[]
    y_list=[]
    for i in range(number_section,number_section+100):
        x_list.append(diffusiondataset[choice[i]][0])
        y_list.append(diffusiondataset[choice[i]][1].view(-1,28))

    Y=torch.cat(y_list,dim=0)

    xx_list=[]
    for i in range(number_section,number_section+100):
        xx_list.append(x_list[i][0].detach().cpu().numpy())

    work_dir= fr'F:/Python_venv/Deep_Learning/pythonProject/MNIST-CYJ/12 Model accuracy/01Img'
    for i in range(number_section,number_section+100):
        np.savetxt(f'{work_dir}/Target_{i}.csv', xx_list[i], delimiter=',', fmt="%d")

    for i, t in enumerate(noise_scheduler.timesteps):
        model_input = noise_scheduler.scale_model_input(sample, t)
        with torch.no_grad():
            noise_pred = model(sample, t, Y.to(device))
        scheduler_output = noise_scheduler.step(noise_pred, t, sample)

        sample = scheduler_output.prev_sample

        pred_x0 = scheduler_output.pred_original_sample
        a = pred_x0[0, 0, :, :].detach().cpu().numpy().clip(-1, 1) * 0.5 + 0.5

        if i == 0 or i==999:
            xxxx_list=[]
            limit=0.1
            for iii in range(number_section,number_section+100):
                xxxx_list.append(pred_x0[iii, 0, :, :].detach().cpu().numpy().clip(-1, 1) * 0.5 + 0.5)
                xxxx_list[iii][xxxx_list[iii]<limit]=0
                xxxx_list[iii][xxxx_list[iii] > limit] = 1
                np.savetxt(f'{work_dir}/DDPM_Section_{iii}_{i}.csv', xxxx_list[iii], delimiter=',', fmt="%d")
            x_list=[]
            column_x_list=[]
            beam_x_list=[]
            for iii in range(number_section,number_section+100):
                x_list.append(pred_x0[iii, 0, :, :].detach().cpu().numpy().clip(-1, 1) * 0.5 + 0.5)
                x_list[iii][x_list[iii] > limit] = 1
                x_list[iii][x_list[iii] < limit] = 0

                column_x_list.append(x_list[iii][:, :40])
                beam_x_list.append(x_list[iii][:, 40:])
                np.savetxt(f'./01Img/x_{iii}_{i}.csv' , column_x_list[iii], delimiter=',', fmt="%d")
                np.savetxt(f'./01Img/x_{iii}_{i}_.csv', beam_x_list[iii]  , delimiter=',', fmt="%d")
