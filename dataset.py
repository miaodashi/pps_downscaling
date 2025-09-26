import os
import glob

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


class Dataset_Train(Dataset):
    """
    Dataset for training
    """
    def __init__(self, datadir: str, labeldir: str, transform=None):
        """
        Args:
            datadir (str): data path
            labeldir (str): label path
        """
        assert os.path.exists(datadir), f"{datadir} not exists!"
        assert os.path.exists(labeldir), f"{labeldir} not exists!"
        
        img_path = sorted(glob.glob(os.path.join(datadir, "*.npy")))
        lab_path = sorted(glob.glob(os.path.join(labeldir, "*.npy")))
        
        if len(img_path) != len(lab_path):
            raise ValueError("The number of data and label is not equal!")
        
        self.img = []
        self.label = []
        
        for i, j in zip(img_path, lab_path):
            img_data = np.load(i)
            label_data = np.load(j)
            
            # 检查并处理NaN值
            if np.isnan(img_data).any():
                print(f"警告：发现输入数据中存在NaN值：{i}")
                img_data = np.nan_to_num(img_data, nan=0.0)
                
            if np.isnan(label_data).any():
                print(f"警告：发现标签数据中存在NaN值：{j}")
                label_data = np.nan_to_num(label_data, nan=0.0)
                
            self.img.append(img_data)
            self.label.append(label_data)
        
        self.transform = transform

    def __getitem__(self, index):
        # image = np.load(self.img_path[index])
        # label = np.load(self.lab_path[index])
        image = self.img[index]
        label = self.label[index]
        
        if self.transform:
            image = self.transform(image)
            label = self.transform(label)

        # 将numpy数组转换为torch张量
        image = torch.from_numpy(image).float()
        label = torch.from_numpy(label).float()

        return image, label

    def __len__(self):
        return len(self.img)



if __name__ == "__main__":
    img_path = rf"D:\SoilErosion\PPS_Downscaling\500m\Environment\npy_patches"
    soil_path = rf"D:\SoilErosion\PPS_Downscaling\500m\Soil\npy_patches"

        
    soil_name = ["Fenli", "Nianli", "Shali", "SOC"]
    for i in soil_name:
        lab_path = os.path.join(soil_path, i)
    
        dataset = Dataset_Train(img_path, lab_path)
        DataLoader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True)
        
        for j, (data, label) in tqdm(enumerate(DataLoader), total=len(DataLoader), desc=f"处理{i}批次"):
            data = data.cuda()
            label = label.cuda()
            print(data.shape, label.shape)