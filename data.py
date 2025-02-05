from torch.utils.data import Dataset  # 导入 PyTorch 数据集基类
import torch
import torchvision.transforms as transforms  # 导入数据预处理和数据增强模块
from skimage.io import imread            # 导入图像读取函数
from skimage.color import gray2rgb        # 导入灰度图转换为RGB图的函数

# 自定义数据集类，继承自 PyTorch 的 Dataset
class ChallengeDataset(Dataset):
    def __init__(self, data, mode="train"):
        """
        初始化 ChallengeDataset 数据集
        :param data: 包含数据路径及标签的 DataFrame
        :param mode: 数据集使用模式，"train" 表示训练模式，其它模式一般为验证或测试
        """
        self.data = data
        # 将模式字符串转换为小写并去除首尾空格，确保一致性
        self.mode = mode.lower().strip()

        # 定义基础预处理转换（适用于所有模式）
        base_transforms = [
            transforms.ToPILImage(),  # 将 NumPy 数组或张量转换为 PIL 图像
            transforms.ToTensor(),      # 将 PIL 图像转换为张量，并将像素值归一化到 [0, 1]
            # 标准化图像数据，均值和标准差均为 3 个通道相同的值
            transforms.Normalize(mean=[0.59685254] * 3, std=[0.16043035] * 3)
        ]

        # 根据数据集模式（训练/验证/测试）设置不同的转换流程
        if self.mode == "train":
            # 在训练模式下，添加数据增强以扩充数据集和提高模型泛化能力
            self.transform = transforms.Compose([
                transforms.ToPILImage(),             # 转换为 PIL 图像
                transforms.RandomHorizontalFlip(p=0.5),  # 随机水平翻转，概率 50%
                transforms.RandomRotation(degrees=5),      # 随机旋转，角度范围为 ±5 度
                transforms.ColorJitter(
                    brightness=0.2,  # 随机调整亮度
                    contrast=0.2,    # 随机调整对比度
                    saturation=0.2,  # 随机调整饱和度
                    hue=0.05       # 随机调整色调
                ),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # 随机平移，平移比例不超过 10%
                transforms.ToTensor(),                # 转换为张量
                transforms.RandomApply([transforms.GaussianBlur(3)], p=0.2),  # 随机应用高斯模糊，内核大小为 3，概率 20%
                transforms.Normalize(mean=[0.59685254] * 3, std=[0.16043035] * 3)  # 标准化图像数据
            ])
        else:
            # 非训练模式下（如验证、测试），只执行基础预处理
            self.transform = transforms.Compose(base_transforms)

    def __len__(self):
        """
        返回数据集的样本总数
        """
        return len(self.data)

    def __getitem__(self, index):
        """
        根据索引返回数据集中的图像和标签
        :param index: 数据索引
        :return: 图像张量和对应的标签张量
        """
        # 从 DataFrame 中获取第 index 行的图像路径，假设图像路径在第一列
        img_path = self.data.iloc[index, 0]
        # 使用 skimage 的 imread 读取图像
        image = imread(img_path)
        
        # 检查图像是否为灰度图（即只有二维），若是则转换为 RGB 图像
        if image.ndim == 2:
            image = gray2rgb(image)
        
        # 对图像进行预处理转换（数据增强、标准化等）
        image = self.transform(image)
        
        # 获取标签数据，假设标签数据从第二列开始，转换为浮点数
        label_values = self.data.iloc[index, 1:].values.astype(float)
        # 将标签转换为 PyTorch 浮点张量
        label = torch.tensor(label_values, dtype=torch.float32)
        
        # 返回预处理后的图像和对应的标签
        return image, label
