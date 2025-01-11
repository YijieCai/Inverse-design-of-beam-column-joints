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


for jjj in range(3,4):
    number_section=4
    sample=torch.randn(number_section,1,40,80).to(device)

    choice=[]
    for i in range(number_section):
        choice.append(jjj*1000)
    # choice=[0,0,0,0]

    diffusiondataset = DiffusionDataset("F:/Python_venv/Deep_Learning/pythonProject/MNIST-CYJ/03 Dataset/Test/Dataset_Test_unweak_2D.h5")
    x_list=[]
    y_list=[]
    for i in range(number_section):
        x_list.append(diffusiondataset[choice[i]][0])
        y_list.append(diffusiondataset[choice[i]][1].view(-1,28))

    Y=torch.cat(y_list,dim=0)

    xx_list=[]
    for i in range(number_section):
        xx_list.append(x_list[i][0].detach().cpu().numpy())


    work_dir=fr'F:\Python_venv\Deep_Learning\pythonProject\MNIST-CYJ\13 different demands\01Img'

    fig, (ax) = plt.subplots(2, 2)
    for i in range(number_section):
        np.savetxt(f'{work_dir}/Target_{i+jjj*100}.csv', xx_list[i], delimiter=',', fmt="%d")


    # ax.imshow(x1[0].detach().cpu().numpy(), cmap='bone',alpha=0.75)
    # ax.axis('off')

    ax[0, 0].imshow(x_list[0][0].detach().cpu().numpy(), cmap='bone', alpha=0.75)
    ax[0, 1].imshow(x_list[1][0].detach().cpu().numpy(), cmap='bone', alpha=0.75)
    ax[1, 0].imshow(x_list[2][0].detach().cpu().numpy(), cmap='bone', alpha=0.75)
    ax[1, 1].imshow(x_list[3][0].detach().cpu().numpy(), cmap='bone', alpha=0.75)
    #
    ax[0, 0].axis('off')
    ax[0, 1].axis('off')
    ax[1, 0].axis('off')
    ax[1, 1].axis('off')
    plt.savefig(f'{work_dir}/Target_section_{jjj*100}.png', bbox_inches='tight', transparent=True)
    plt.close(fig)

    for i, t in enumerate(noise_scheduler.timesteps):
        model_input = noise_scheduler.scale_model_input(sample, t)
        with torch.no_grad():
            noise_pred = model(sample, t, Y.to(device))
        scheduler_output = noise_scheduler.step(noise_pred, t, sample)

        sample = scheduler_output.prev_sample

        pred_x0 = scheduler_output.pred_original_sample
        a = pred_x0[0, 0, :, :].detach().cpu().numpy().clip(-1, 1) * 0.5 + 0.5

        # if i == 0 or (i+1)%100==0:
        if i == 0 or i==999:
            xxxx_list=[]
            limit=0.1

            fig, (ax) = plt.subplots(2, 2)
            for iii in range(number_section):
                xxxx_list.append(pred_x0[iii, 0, :, :].detach().cpu().numpy().clip(-1, 1) * 0.5 + 0.5)
                xxxx_list[iii][xxxx_list[iii]<limit]=0
                xxxx_list[iii][xxxx_list[iii] > limit] = 1
                np.savetxt(f'{work_dir}/DDPM_Section_{iii+jjj*100}_{i}.csv', xxxx_list[iii], delimiter=',', fmt="%d")
                # ax.imshow(pred_x0[0, 0, :, :].detach().cpu().numpy().clip(-1, 1) * 0.5 + 0.5, cmap='bone',alpha=0.75)
                # ax.axis('off')
            ax[0, 0].imshow(pred_x0[0, 0, :, :].detach().cpu().numpy().clip(-1, 1) * 0.5 + 0.5, cmap='bone',
                            alpha=0.75)
            ax[0, 1].imshow(pred_x0[1, 0, :, :].detach().cpu().numpy().clip(-1, 1) * 0.5 + 0.5, cmap='bone',
                            alpha=0.75)
            ax[1, 0].imshow(pred_x0[2, 0, :, :].detach().cpu().numpy().clip(-1, 1) * 0.5 + 0.5, cmap='bone',
                            alpha=0.75)
            ax[1, 1].imshow(pred_x0[3, 0, :, :].detach().cpu().numpy().clip(-1, 1) * 0.5 + 0.5, cmap='bone',
                            alpha=0.75)
            ax[0, 0].axis('off')
            ax[0, 1].axis('off')
            ax[1, 0].axis('off')
            ax[1, 1].axis('off')
            plt.savefig(f'{work_dir}/DDPM_Section_{jjj*100}_{i}.png', bbox_inches='tight', transparent=True)
            plt.close(fig)

            x_list=[]
            column_x_list=[]
            beam_x_list=[]
            for iii in range(number_section):
                x_list.append(pred_x0[iii, 0, :, :].detach().cpu().numpy().clip(-1, 1) * 0.5 + 0.5)
                x_list[iii][x_list[iii] > limit] = 1
                x_list[iii][x_list[iii] < limit] = 0

                column_x_list.append(x_list[iii][:, :40])
                beam_x_list.append(x_list[iii][:, 40:])
                np.savetxt(f'./01Img/x_{iii+jjj*100}_{i}.csv' , column_x_list[iii], delimiter=',', fmt="%d")
                np.savetxt(f'./01Img/x_{iii+jjj*100}_{i}_.csv', beam_x_list[iii]  , delimiter=',', fmt="%d")
# ***_第iii个截面 第i步骤

# x1=pred_x0[0, 0, :, :].detach().cpu().numpy().clip(-1, 1) * 0.5 + 0.5
# x1_=pred_x0[0, 1, :, :].detach().cpu().numpy().clip(-1, 1) * 0.5 + 0.5
# x2=pred_x0[1, 0, :, :].detach().cpu().numpy().clip(-1, 1) * 0.5 + 0.5
# x2_=pred_x0[1, 1, :, :].detach().cpu().numpy().clip(-1, 1) * 0.5 + 0.5
# x3=pred_x0[2, 0, :, :].detach().cpu().numpy().clip(-1, 1) * 0.5 + 0.5
# x3_=pred_x0[2, 1, :, :].detach().cpu().numpy().clip(-1, 1) * 0.5 + 0.5
# x4=pred_x0[3, 0, :, :].detach().cpu().numpy().clip(-1, 1) * 0.5 + 0.5
# x4_=pred_x0[3, 1, :, :].detach().cpu().numpy().clip(-1, 1) * 0.5 + 0.5
#
# x1[x1<0.5]=0
# x1[x1>0.5]=1
# x2[x2<0.5]=0
# x2[x2>0.5]=1
# x3[x3<0.5]=0
# x3[x3>0.5]=1
# x4[x4<0.5]=0
# x4[x4>0.5]=1
#
# x1_[x1_<0.5]=0
# x1_[x1_>0.5]=1
# x2_[x2_<0.5]=0
# x2_[x2_>0.5]=1
# x3_[x3_<0.5]=0
# x3_[x3_>0.5]=1
# x4_[x4_<0.5]=0
# x4_[x4_>0.5]=1
#
# np.savetxt(f'F:/Python_venv/Deep_Learning/pythonProject/MNIST-CYJ/05 DDPM/Val/skeleton_curve/x1.csv', x1, delimiter=',', fmt="%d")
# np.savetxt(f'F:/Python_venv/Deep_Learning/pythonProject/MNIST-CYJ/05 DDPM/Val/skeleton_curve/x2.csv', x2, delimiter=',', fmt="%d")
# np.savetxt(f'F:/Python_venv/Deep_Learning/pythonProject/MNIST-CYJ/05 DDPM/Val/skeleton_curve/x3.csv', x3, delimiter=',', fmt="%d")
# np.savetxt(f'F:/Python_venv/Deep_Learning/pythonProject/MNIST-CYJ/05 DDPM/Val/skeleton_curve/x4.csv', x4, delimiter=',', fmt="%d")
#
# np.savetxt(f'F:/Python_venv/Deep_Learning/pythonProject/MNIST-CYJ/05 DDPM/Val/skeleton_curve/x1_.csv', x1_, delimiter=',', fmt="%d")
# np.savetxt(f'F:/Python_venv/Deep_Learning/pythonProject/MNIST-CYJ/05 DDPM/Val/skeleton_curve/x2_.csv', x2_, delimiter=',', fmt="%d")
# np.savetxt(f'F:/Python_venv/Deep_Learning/pythonProject/MNIST-CYJ/05 DDPM/Val/skeleton_curve/x3_.csv', x3_, delimiter=',', fmt="%d")
# np.savetxt(f'F:/Python_venv/Deep_Learning/pythonProject/MNIST-CYJ/05 DDPM/Val/skeleton_curve/x4_.csv', x4_, delimiter=',', fmt="%d")
