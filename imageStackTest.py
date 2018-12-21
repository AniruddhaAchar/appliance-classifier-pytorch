import torch
from torch.utils.data import  DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from PIL import Image
import numpy as np

rootDir = "D:\\Work\\Progarmming\\Datasets\\data-release"
lables = pd.read_csv(os.path.join(rootDir,"train_labels.csv"))
train_lables, valid_lables = train_test_split(lables, test_size = 0.2, random_state=42)

class AppDataset(Dataset):
    def __init__(self, dataset, imageDir):
        self.ids = np.array(dataset.iloc[:,0])
        self.lables = np.array(dataset.iloc[:,1])
        self.imageDir = imageDir
    
    def __len__(self):
        return len(self.ids)
    
    def __getitem__(self, index):
        cimage = os.path.join(self.imageDir+"\\current", str(self.ids[index])+"_c.png")
        vimage = os.path.join(self.imageDir+"\\volt", str(self.ids[index])+"_v.png")
        cimage = Image.open(cimage)
        vimage = Image.open(vimage)
        transformer = transforms.ToTensor()
        cimage = transformer(cimage)
        vimage = transformer(vimage)
        lable = self.lables[index]
        return cimage, vimage, lable

def get_loaders():
    train_set = AppDataset(train_lables, "D:\\Work\\Progarmming\\Datasets\\data-release\\train_processed_unsplit")
    valid_set = AppDataset(valid_lables, "D:\\Work\\Progarmming\\Datasets\\data-release\\train_processed_unsplit")
    train_loader = DataLoader(train_set,batch_size=32,shuffle=True,num_workers=4)
    valid_loader = DataLoader(valid_set,batch_size=32,shuffle=False,num_workers=4)
    return train_loader, valid_loader
