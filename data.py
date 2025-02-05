from torch.utils.data import Dataset
import torch
import torchvision.transforms as transforms
from skimage.io import imread
from skimage.color import gray2rgb

class ChallengeDataset(Dataset):
    def __init__(self, data, mode="train"):
        self.data = data
        self.mode = mode.lower().strip()

        # 定义基础转换（对所有模式均适用）
        base_transforms = [
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.59685254] * 3, std=[0.16043035] * 3)
        ]

        if self.mode == "train":
            # 训练模式先进行数据增强，再执行基础预处理
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=5),  # 旋转角度减小到10度
                transforms.ColorJitter(
                    brightness=0.2,  # 调低亮度扰动
                    contrast=0.2,    # 调低对比度扰动
                    saturation=0.2,  # 调低饱和度扰动
                    hue=0.05       # 调低色调扰动
                ),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # 平移范围减小
                transforms.ToTensor(),
                transforms.RandomApply([transforms.GaussianBlur(3)], p=0.2),  # 添加噪声增强
                transforms.Normalize(mean=[0.59685254] * 3, std=[0.16043035] * 3)
            ])
        else:
            # 非训练模式只执行基础预处理
            self.transform = transforms.Compose(base_transforms)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_path = self.data.iloc[index, 0]
        image = imread(img_path)
        if image.ndim == 2:
            image = gray2rgb(image)
        
        image = self.transform(image)
        
        label_values = self.data.iloc[index, 1:].values.astype(float)
        label = torch.tensor(label_values, dtype=torch.float32)
        
        return image, label