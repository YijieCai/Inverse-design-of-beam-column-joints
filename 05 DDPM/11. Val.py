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

choice=[10,3100,6100,9100]
# print(choice[1])
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
# np.savetxt(f'F:/Python_venv/Deep_Learning/pythonProject/MNIST/DDPM/Val/skeleton_curve/x1.csv',xx1,delimiter=',',fmt="%d")
# np.savetxt(f'F:/Python_venv/Deep_Learning/pythonProject/MNIST/DDPM/Val/skeleton_curve/x2.csv',xx2,delimiter=',',fmt="%d")
# np.savetxt(f'F:/Python_venv/Deep_Learning/pythonProject/MNIST/DDPM/Val/skeleton_curve/x3.csv',xx3,delimiter=',',fmt="%d")
# np.savetxt(f'F:/Python_venv/Deep_Learning/pythonProject/MNIST/DDPM/Val/skeleton_curve/x4.csv',xx4,delimiter=',',fmt="%d")

fig, (ax) = plt.subplots(2, 2)
ax[0, 0].imshow(x1[0].detach().cpu().numpy(), cmap='bone')
ax[0, 1].imshow(x2[0].detach().cpu().numpy(), cmap='bone')
ax[1, 0].imshow(x3[0].detach().cpu().numpy(), cmap='bone')
ax[1, 1].imshow(x4[0].detach().cpu().numpy(), cmap='bone')
plt.savefig(f'./Val/Observed.png', bbox_inches='tight', transparent=True)
plt.close(fig)

for i, t in enumerate(noise_scheduler.timesteps):
    model_input = noise_scheduler.scale_model_input(sample, t)
    with torch.no_grad():
        noise_pred = model(sample, t, Y.to(device))
    scheduler_output = noise_scheduler.step(noise_pred, t, sample)

    sample = scheduler_output.prev_sample

    pred_x0 = scheduler_output.pred_original_sample
    a = pred_x0[0, 0, :, :].detach().cpu().numpy().clip(-1, 1) * 0.5 + 0.5

    if i % 10 == 0:
        fig, (ax) = plt.subplots(2, 2)
        ax[0, 0].imshow(pred_x0[0, 0, :, :].detach().cpu().numpy().clip(-1, 1) * 0.5 + 0.5, cmap='bone')
        ax[0, 1].imshow(pred_x0[1, 0, :, :].detach().cpu().numpy().clip(-1, 1) * 0.5 + 0.5, cmap='bone')
        ax[1, 0].imshow(pred_x0[2, 0, :, :].detach().cpu().numpy().clip(-1, 1) * 0.5 + 0.5, cmap='bone')
        ax[1, 1].imshow(pred_x0[3, 0, :, :].detach().cpu().numpy().clip(-1, 1) * 0.5 + 0.5, cmap='bone')
        plt.savefig(f'./Val/Output{i}.png', bbox_inches='tight', transparent=True)
        plt.close(fig)

x1=pred_x0[0, 0, :, :].detach().cpu().numpy().clip(-1, 1) * 0.5 + 0.5
x2=pred_x0[1, 0, :, :].detach().cpu().numpy().clip(-1, 1) * 0.5 + 0.5
x3=pred_x0[2, 0, :, :].detach().cpu().numpy().clip(-1, 1) * 0.5 + 0.5
x4=pred_x0[3, 0, :, :].detach().cpu().numpy().clip(-1, 1) * 0.5 + 0.5

x1[x1<0.5]=0
x1[x1>0.5]=1
x2[x2<0.5]=0
x2[x2>0.5]=1
x3[x3<0.5]=0
x3[x3>0.5]=1
x4[x4<0.5]=0
x4[x4>0.5]=1
print(x1)
print(x1.shape)
print(x1[:,40:])
print(x1[0:40].shape)
column_x1=x1[:,:40]
column_x2=x2[:,:40]
column_x3=x3[:,:40]
column_x4=x4[:,:40]
np.savetxt(f'F:/Python_venv/Deep_Learning/pythonProject/MNIST-CYJ/05 DDPM/Val/skeleton_curve/x1.csv',column_x1,delimiter=',',fmt="%d")
np.savetxt(f'F:/Python_venv/Deep_Learning/pythonProject/MNIST-CYJ/05 DDPM/Val/skeleton_curve/x2.csv',column_x2,delimiter=',',fmt="%d")
np.savetxt(f'F:/Python_venv/Deep_Learning/pythonProject/MNIST-CYJ/05 DDPM/Val/skeleton_curve/x3.csv',column_x3,delimiter=',',fmt="%d")
np.savetxt(f'F:/Python_venv/Deep_Learning/pythonProject/MNIST-CYJ/05 DDPM/Val/skeleton_curve/x4.csv',column_x4,delimiter=',',fmt="%d")

beam_x1=x1[:,40:]
beam_x2=x2[:,40:]
beam_x3=x3[:,40:]
beam_x4=x4[:,40:]
np.savetxt(f'F:/Python_venv/Deep_Learning/pythonProject/MNIST-CYJ/05 DDPM/Val/skeleton_curve/x1_.csv',beam_x1,delimiter=',',fmt="%d")
np.savetxt(f'F:/Python_venv/Deep_Learning/pythonProject/MNIST-CYJ/05 DDPM/Val/skeleton_curve/x2_.csv',beam_x2,delimiter=',',fmt="%d")
np.savetxt(f'F:/Python_venv/Deep_Learning/pythonProject/MNIST-CYJ/05 DDPM/Val/skeleton_curve/x3_.csv',beam_x3,delimiter=',',fmt="%d")
np.savetxt(f'F:/Python_venv/Deep_Learning/pythonProject/MNIST-CYJ/05 DDPM/Val/skeleton_curve/x4_.csv',beam_x4,delimiter=',',fmt="%d")

# np.savetxt(f'F:/Python_venv/Deep_Learning/pythonProject/MNIST-CYJ/05 DDPM/Val/skeleton_curve/x1.csv',x1,delimiter=',',fmt="%d")
# np.savetxt(f'F:/Python_venv/Deep_Learning/pythonProject/MNIST-CYJ/05 DDPM/Val/skeleton_curve/x2.csv',x2,delimiter=',',fmt="%d")
# np.savetxt(f'F:/Python_venv/Deep_Learning/pythonProject/MNIST-CYJ/05 DDPM/Val/skeleton_curve/x3.csv',x3,delimiter=',',fmt="%d")
# np.savetxt(f'F:/Python_venv/Deep_Learning/pythonProject/MNIST-CYJ/05 DDPM/Val/skeleton_curve/x4.csv',x4,delimiter=',',fmt="%d")

