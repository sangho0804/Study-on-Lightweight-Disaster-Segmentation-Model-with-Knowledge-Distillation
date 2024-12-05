import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from lpcvc.loader import get_loader
from torch.utils import data
from tqdm.notebook import tqdm
from lpcvc.augmentations import get_composed_augmentations
# 이미지의 RGB 채널별 통계량 확인 함수

def print_stats(dataset):
    imgs = np.array([img.numpy() for img, _ in dataset])
    print(f'shape: {imgs.shape}')
    
    min_r = np.min(imgs, axis=(2, 3))[:, 0].min()
    min_g = np.min(imgs, axis=(2, 3))[:, 1].min()
    min_b = np.min(imgs, axis=(2, 3))[:, 2].min()

    max_r = np.max(imgs, axis=(2, 3))[:, 0].max()
    max_g = np.max(imgs, axis=(2, 3))[:, 1].max()
    max_b = np.max(imgs, axis=(2, 3))[:, 2].max()

    mean_r = np.mean(imgs, axis=(2, 3))[:, 0].mean()
    mean_g = np.mean(imgs, axis=(2, 3))[:, 1].mean()
    mean_b = np.mean(imgs, axis=(2, 3))[:, 2].mean()

    std_r = np.std(imgs, axis=(2, 3))[:, 0].std()
    std_g = np.std(imgs, axis=(2, 3))[:, 1].std()
    std_b = np.std(imgs, axis=(2, 3))[:, 2].std()
    
    print(f'min: {min_r, min_g, min_b}')
    print(f'max: {max_r, max_g, max_b}')
    print(f'mean: {mean_r, mean_g, mean_b}')
    print(f'std: {std_r, std_g, std_b}')

def calculate_norm(dataset):
    # dataset의 axis=1, 2에 대한 평균 산출
    mean_ = np.array([np.mean(x.numpy(), axis=(1, 2)) for x, _ in dataset])

    # r, g, b 채널에 대한 각각의 평균 산출
    mean_r = mean_[:, 0].mean()
    mean_g = mean_[:, 1].mean()
    mean_b = mean_[:, 2].mean()

    # dataset의 axis=1, 2에 대한 표준편차 산출
    std_ = np.array([np.std(x.numpy(), axis=(1, 2)) for x, _ in dataset])
    # r, g, b 채널에 대한 각각의 표준편차 산출
    std_r = std_[:, 0].mean()
    std_g = std_[:, 1].mean()
    std_b = std_[:, 2].mean()
    
    return (mean_r, mean_g, mean_b), (std_r, std_g, std_b)

test = {"norm" : 255.0 ,
        "cnorm" : [[.485, .456, .406],[.229, .224, .225]]}
train_augmentations = test
t_data_aug = get_composed_augmentations(train_augmentations)

data_loader = get_loader("lpcvc")
data_path = "/home/guest/sangho/2023LPCVC_SampleSolution/dataset/LPCVC/"
t_loader = data_loader(data_path,split="train")
t_loader = data_loader(data_path,split="train", augmentations=t_data_aug)
#v_loader = data_loader(data_path,split="val",augmentations=t_data_aug)

trainloader = data.DataLoader(t_loader, shuffle=False,num_workers=os.cpu_count())



mean_, std_ = calculate_norm(t_loader)
print(f'평균(R,G,B): {mean_}\n표준편차(R,G,B): {std_}')

print_stats(t_loader)
print('==='*10)