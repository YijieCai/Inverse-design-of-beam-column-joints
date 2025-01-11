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
sample=torch.randn(4,1,40,80).to(device)

choice=[0,3000,6000,9000]
# choice=[0,0,0,0]

diffusiondataset = DiffusionDataset("F:/Python_venv/Deep_Learning/pythonProject/MNIST-CYJ/03 Dataset/Test/Dataset_Test_unweak_2D.h5")
x1=diffusiondataset[choice[0]][0]
x2=diffusiondataset[choice[1]][0]
x3=diffusiondataset[choice[2]][0]
x4=diffusiondataset[choice[3]][0]

y1=diffusiondataset[choice[0]][1].view(-1,28)
y2=diffusiondataset[choice[1]][1].view(-1,28)
y3=diffusiondataset[choice[2]][1].view(-1,28)
y4=diffusiondataset[choice[3]][1].view(-1,28)


Y=torch.cat([y1,y2,y3,y4],dim=0)

xx1=x1[0].detach().cpu().numpy()
xx2=x2[0].detach().cpu().numpy()
xx3=x3[0].detach().cpu().numpy()
xx4=x4[0].detach().cpu().numpy()

work_dir=fr'F:\Python_venv\Deep_Learning\pythonProject\MNIST-CYJ\11 Each step result comparsion\01Img'
np.savetxt(f'{work_dir}/Target_1.csv',xx1,delimiter=',',fmt="%d")
np.savetxt(f'{work_dir}/Target_2.csv',xx2,delimiter=',',fmt="%d")
np.savetxt(f'{work_dir}/Target_3.csv',xx3,delimiter=',',fmt="%d")
np.savetxt(f'{work_dir}/Target_4.csv',xx4,delimiter=',',fmt="%d")

fig, (ax) = plt.subplots(2, 2)
# ax.imshow(x1[0].detach().cpu().numpy(), cmap='bone',alpha=0.75)
# ax.axis('off')

ax[0, 0].imshow(x1[0].detach().cpu().numpy(), cmap='bone',alpha=0.75)
ax[0, 1].imshow(x2[0].detach().cpu().numpy(), cmap='bone',alpha=0.75)
ax[1, 0].imshow(x3[0].detach().cpu().numpy(), cmap='bone',alpha=0.75)
ax[1, 1].imshow(x4[0].detach().cpu().numpy(), cmap='bone',alpha=0.75)
#
ax[0,0].axis('off')
ax[0,1].axis('off')
ax[1,0].axis('off')
ax[1,1].axis('off')
plt.savefig(f'{work_dir}/Target_section.png', bbox_inches='tight', transparent=True)
plt.close(fig)
#

#
for i, t in enumerate(noise_scheduler.timesteps):
    model_input = noise_scheduler.scale_model_input(sample, t)
    print(sample)
    with torch.no_grad():
        noise_pred = model(sample, t, Y.to(device))
    scheduler_output = noise_scheduler.step(noise_pred, t, sample)

    sample = scheduler_output.prev_sample

    pred_x0 = scheduler_output.pred_original_sample
    a = pred_x0[0, 0, :, :].detach().cpu().numpy().clip(-1, 1) * 0.5 + 0.5

    if i % 10 == 0:
        fig, (ax) = plt.subplots(2, 2)
        # ax.imshow(pred_x0[0, 0, :, :].detach().cpu().numpy().clip(-1, 1) * 0.5 + 0.5, cmap='bone',alpha=0.75)
        # ax.axis('off')

        ax[0, 0].imshow(pred_x0[0, 0, :, :].detach().cpu().numpy().clip(-1, 1) * 0.5 + 0.5, cmap='bone',alpha=0.75)
        ax[0, 1].imshow(pred_x0[1, 0, :, :].detach().cpu().numpy().clip(-1, 1) * 0.5 + 0.5, cmap='bone',alpha=0.75)
        ax[1, 0].imshow(pred_x0[2, 0, :, :].detach().cpu().numpy().clip(-1, 1) * 0.5 + 0.5, cmap='bone',alpha=0.75)
        ax[1, 1].imshow(pred_x0[3, 0, :, :].detach().cpu().numpy().clip(-1, 1) * 0.5 + 0.5, cmap='bone',alpha=0.75)
        xxxx1=pred_x0[0, 0, :, :].detach().cpu().numpy().clip(-1, 1) * 0.5 + 0.5
        xxxx2=pred_x0[1, 0, :, :].detach().cpu().numpy().clip(-1, 1) * 0.5 + 0.5
        xxxx3=pred_x0[2, 0, :, :].detach().cpu().numpy().clip(-1, 1) * 0.5 + 0.5
        xxxx4=pred_x0[3, 0, :, :].detach().cpu().numpy().clip(-1, 1) * 0.5 + 0.5
        limit=0.1
        xxxx1[xxxx1 < limit] = 0
        xxxx1[xxxx1 > limit] = 1
        xxxx2[xxxx2 < limit] = 0
        xxxx2[xxxx2 > limit] = 1
        xxxx3[xxxx3 < limit] = 0
        xxxx3[xxxx3 > limit] = 1
        xxxx4[xxxx4 < limit] = 0
        xxxx4[xxxx4 > limit] = 1
        np.savetxt(f'{work_dir}/DDPM_Section_{i}_1.csv', xxxx1, delimiter=',', fmt="%d")
        np.savetxt(f'{work_dir}/DDPM_Section_{i}_2.csv', xxxx2, delimiter=',', fmt="%d")
        np.savetxt(f'{work_dir}/DDPM_Section_{i}_3.csv', xxxx3, delimiter=',', fmt="%d")
        np.savetxt(f'{work_dir}/DDPM_Section_{i}_4.csv', xxxx4, delimiter=',', fmt="%d")
        ax[0, 0].axis('off')
        ax[0, 1].axis('off')
        ax[1, 0].axis('off')
        ax[1, 1].axis('off')
        plt.savefig(f'{work_dir}/DDPM_Section_{i}.png', bbox_inches='tight', transparent=True)
        plt.close(fig)



        x1 = pred_x0[0, 0, :, :].detach().cpu().numpy().clip(-1, 1) * 0.5 + 0.5
        x2 = pred_x0[1, 0, :, :].detach().cpu().numpy().clip(-1, 1) * 0.5 + 0.5
        x3 = pred_x0[2, 0, :, :].detach().cpu().numpy().clip(-1, 1) * 0.5 + 0.5
        x4 = pred_x0[3, 0, :, :].detach().cpu().numpy().clip(-1, 1) * 0.5 + 0.5


        x1[x1 < limit] = 0
        x1[x1 > limit] = 1
        x2[x2 < limit] = 0
        x2[x2 > limit] = 1
        x3[x3 < limit] = 0
        x3[x3 > limit] = 1
        x4[x4 < limit] = 0
        x4[x4 > limit] = 1

        column_x1 = x1[:, :40]
        column_x2 = x2[:, :40]
        column_x3 = x3[:, :40]
        column_x4 = x4[:, :40]

        beam_x1 = x1[:, 40:]
        beam_x2 = x2[:, 40:]
        beam_x3 = x3[:, 40:]
        beam_x4 = x4[:, 40:]

        x1 = column_x1
        x2 = column_x2
        x3 = column_x3
        x4 = column_x4

        x1_ = beam_x1
        x2_ = beam_x2
        x3_ = beam_x3
        x4_ = beam_x4
        np.savetxt(f'./01Img/x_{i}_1.csv', x1, delimiter=',', fmt="%d")
        np.savetxt(f'./01Img/x_{i}_2.csv', x2, delimiter=',', fmt="%d")
        np.savetxt(f'./01Img/x_{i}_3.csv', x3, delimiter=',', fmt="%d")
        np.savetxt(f'./01Img/x_{i}_4.csv', x4, delimiter=',', fmt="%d")

        np.savetxt(f'./01Img/x_{i}_1_.csv', x1_, delimiter=',', fmt="%d")
        np.savetxt(f'./01Img/x_{i}_2_.csv', x2_, delimiter=',', fmt="%d")
        np.savetxt(f'./01Img/x_{i}_3_.csv', x3_, delimiter=',', fmt="%d")
        np.savetxt(f'./01Img/x_{i}_4_.csv', x4_, delimiter=',', fmt="%d")


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
