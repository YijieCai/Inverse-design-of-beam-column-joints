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

from ConditionDiffusionModel import ClassConditionedUnet
import torch.optim as optim

# 文件路径
filepath='F:/Python_venv/Deep_Learning/pythonProject/MNIST-CYJ'

# 加载数据集
from Dataset import DiffusionDataset
# 数据集格式是纤维截面分布 0是混凝土纤维 1是钢筋纤维
diffusiondataset = DiffusionDataset(filepath+"/03 Dataset/Train/Dataset_Train_unweak_2D.h5")

# 训练
# 加载'数据集：diffusiondataset' 每次加载’64‘批，且打乱顺序
train_loader = torch.utils.data.DataLoader(dataset=diffusiondataset,batch_size=64,shuffle=True)

# 训练的设备
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 创建一个调度器
# 调度器类型是 DDPMScheduler 训练时的总步数是1000 选'squaredcos_cap_v2'这种β调度策略
noise_scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule='squaredcos_cap_v2')
# noise_scheduler = DDIMScheduler(num_train_timesteps=50)

# 训练100轮
n_epochs = 100
# 记录损失函数
loss_fn = nn.MSELoss()
# Unet架构的神经网络
net = ClassConditionedUnet().to(device)

# model_state_dict = torch.load('F:\Python_venv\Deep_Learning\pythonProject\Infilled_Wall_Diffusion_5500\Train\model_pretrain2.pt')
# net.load_state_dict(model_state_dict.state_dict())

# Define optimizer
#使用Adamw优化算法，对神经网络进行优化，学习率为1e-5.学习率太大会导致学习不稳定，学习率太小导致计算太慢
optimizer = optim.AdamW(net.parameters(), lr=1e-5)

# Define learning rate scheduler
# optimizer 前面已经定义了。
#ExponentialLR:是optim.lr_scheduler模块中的一个类
# 用于实现指数衰减的学习率调度策略。在这种策略中，学习率在每个epoch（或指定的step）后按照一个固定的衰减率（gamma）进行指数衰减。
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)

# 训练开始
for epoch in range(n_epochs):
    with tqdm(train_loader) as pbar:
        for i, (x, y) in enumerate(pbar):
            # 获取数据并添加噪声
            x = x.to(device)*2-1  # 数据被归一化到区间(-1, 1)c
            y = y.to(device).view(-1,28)

            # 生成与x相同的噪声
            noise = torch.randn_like(x)

            # 随机生成时间步
            timesteps = torch.randint(0, 999, (x.shape[0],)).long().to(device)

            # print(x.shape)
            # print(timesteps.shape)
            # print(noise.shape)
            # 使用噪声调度器 给x增加 timesteps 次的 noise噪声
            noisy_x = noise_scheduler.add_noise(x, noise, timesteps)


            # 将结果传递给之前设置的神经网络net得到预测结果pred
            pred = net(noisy_x, timesteps,y)  # 注意这里输入了类别信息

            # 计算损失值
            loss = loss_fn(pred, noise)

            # 清理之前的梯度
            optimizer.zero_grad()

            # 计算损失关于模型参数的梯度
            loss.backward()

            # 根据梯度更新模型参数
            optimizer.step()

            # 设置进度条
            pbar.set_postfix(Epoch=epoch+1,loss=loss.item(), lr=optimizer.param_groups[0]['lr'])

    scheduler.step()
    if epoch % 2==0:
        torch.save(net,f"./model{epoch}.pt")

torch.save(net, r"./diffusionmodel.pt")