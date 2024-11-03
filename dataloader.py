import numpy as np
import os
from torch.utils.data import Dataset,DataLoader
import scipy.io as scio

class MyDataset(Dataset):
    def __init__(self,data_path, transform=None):
        self.data_path = data_path
        self.labels = sorted([ label for label in os.listdir(data_path)])
        self.label_to_idx = {label: idx for idx, label in enumerate(self.labels)}
        self.sample = self.GetData()
        self.transform = transform

    def GetData(self):
        sample = []
        for label in self.labels:
            path_data = os.path.join(self.data_path,label)
            for mat_path in os.listdir(path_data):
                path_total = os.path.join(path_data,mat_path)  #每个数据的绝对路径
                sample.append((path_total, self.label_to_idx[label]))
        return sample
    def __getitem__(self, index):
        DataPath ,label= self.sample[index]
        data = scio.loadmat(DataPath)
        features = data['eeg_final'].astype(np.float32)  # Convert to float32
        if self.transform:
            features = self.transform(features)
        return features, label

    def Myprint(self):
        print(self.label_to_idx)
    def __len__(self):
        return len(self.sample)
class testDataset(Dataset):
    def __init__(self,data_path, transform=None):
        self.data_path = data_path
        self.labels = sorted([ label for label in os.listdir(data_path)])
        self.label_to_idx = {label: idx for idx, label in enumerate(self.labels)}
        self.sample = self.GetData()
        self.transform = transform

    def GetData(self):
        sample = []
        for label in self.labels:
            path_data = os.path.join(self.data_path,label)
            for mat_path in os.listdir(path_data):
                path_total = os.path.join(path_data,mat_path)  #每个数据的绝对路径
                sample.append((path_total, self.label_to_idx[label]))
        return sample
    def __getitem__(self, index):
        DataPath ,label= self.sample[index]
        data = scio.loadmat(DataPath)
        features = data['eeg_fina'].astype(np.float32)  # Convert to float32
        if self.transform:
            features = self.transform(features)
        return features, label

    def Myprint(self):
        print(self.label_to_idx)
    def __len__(self):
        return len(self.sample)
