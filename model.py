import torch
import torch.nn as nn
from torchsummary import summary  # 导入用于打印模型结构的工具

# 定义基本残差块（BasicBlock），用于构建 ResNet 模型中的残差连接
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        """
        初始化 BasicBlock 模块
        :param in_channels: 输入通道数
        :param out_channels: 输出通道数
        :param stride: 卷积步长，控制特征图尺寸
        """
        super(BasicBlock, self).__init__()
        # 第一个卷积层，使用 3x3 卷积，步长可调，padding=1 保持尺寸
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        # 第一个批归一化层，用于加速收敛和稳定训练
        self.batch_norm1 = nn.BatchNorm2d(out_channels)
        # 第一个 ReLU 激活函数
        self.relu_1 = nn.ReLU()
        # 第二个卷积层，使用 3x3 卷积，padding=1 保持尺寸
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        # 第二个批归一化层
        self.batch_norm2 = nn.BatchNorm2d(out_channels)
        
        # 判断是否需要对输入进行 1x1 卷积（当输入和输出尺寸或通道数不一致时使用）
        self.residual_conv = True
        self.conv1X1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
        if in_channels == out_channels and stride == 1:
            self.residual_conv = False
        else:
            self.residual_conv = True

        # 对经过 1x1 卷积后的残差进行 BN 和激活处理
        self.batch_norm3 = nn.BatchNorm2d(out_channels)
        self.relu_3 = nn.ReLU()
        # 将前两个卷积层、批归一化和激活函数组合成一个 Sequential 模块
        self.seq = nn.Sequential(self.conv1, self.batch_norm1, self.relu_1, self.conv2, self.batch_norm2)
        # 初始化残差属性，用于保存输入
        self.residual = None

    def forward(self, input_tensor):
        """
        前向传播函数
        :param input_tensor: 输入张量
        :return: 输出张量，经过残差连接和激活处理
        """
        # 保存输入作为残差
        self.residual = input_tensor
        # 通过两层卷积操作
        output_tensor = self.seq(input_tensor)
        
        # 如果需要调整残差尺寸，则使用 1x1 卷积调整输入尺寸
        if self.residual_conv:
            self.residual = self.conv1X1(self.residual)
        # 对调整后的残差进行批归一化
        self.residual = self.batch_norm3(self.residual)
        # 将卷积结果和残差相加，实现残差连接
        output_tensor += self.residual
        # 对相加后的结果进行激活
        output_tensor = self.relu_3(output_tensor)
        return output_tensor


# 定义 ResNet 模型
class ResNet(nn.Module):
    def __init__(self, num_classes=2):
        """
        初始化 ResNet 模型
        :param num_classes: 输出类别数，默认为 2
        """
        super(ResNet, self).__init__()
        # 第一部分：初始卷积层，包含大卷积核和池化操作，用于提取初步特征
        self.seq1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2),  # 使用 7x7 卷积提取特征
            nn.BatchNorm2d(64),                         # 批归一化
            nn.ReLU(),                                  # 激活函数
            nn.MaxPool2d(kernel_size=3, stride=2)         # 最大池化，降低特征图尺寸
        )
        # 第二部分：残差块堆叠，通过 BasicBlock 构建，逐步加深网络
        self.seq2 = nn.Sequential(
            BasicBlock(64, 64, stride=1),   # 第一块保持尺寸不变
            BasicBlock(64, 128, stride=2),  # 第二块下采样，输出通道数翻倍
            BasicBlock(128, 256, stride=2), # 第三块继续下采样
            BasicBlock(256, 512, stride=2), # 第四块进一步下采样
        )
        # 第三部分：分类器，包含池化、扁平化和全连接层
        self.seq3 = nn.Sequential(
            nn.AvgPool2d(10),                           # 平均池化，将特征图降至固定大小（假设输入尺寸匹配）
            nn.Flatten(),               # 将多维张量展平成一维
            nn.Dropout(0.5),  # 添加 Dropout
            nn.Linear(in_features=512, out_features=num_classes),  # 全连接层，将特征映射到类别数
            nn.Sigmoid()                                # Sigmoid 激活函数，输出范围为 (0, 1)，适用于二分类
        )

    def forward(self, input_tensor):
        """
        前向传播函数
        :param input_tensor: 输入张量
        :return: 模型输出张量
        """
        # 依次通过三个序列模块：初始卷积、残差块堆叠、分类器
        output_tensor = self.seq1(input_tensor)
        output_tensor = self.seq2(output_tensor)
        output_tensor = self.seq3(output_tensor)
        return output_tensor


# 测试模型部分
if __name__ == "__main__":
    # 设置设备：如果 GPU 可用则使用 GPU，否则使用 CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 实例化 ResNet 模型，并指定类别数为 2，然后移动到设备上
    model = ResNet(num_classes=2).to(device)

    print("✅ Model initialized on", device)
    
    # 使用 torchsummary 显示模型结构和参数量，输入尺寸为 (3, 300, 300)
    summary(model, input_size=(3, 300, 300))
    
    # 测试随机输入：生成 50 个 3x300x300 的随机张量，并移动到设备上
    dummy_input = torch.randn(50, 3, 300, 300).to(device)
    output = model(dummy_input)
    # 输出预测结果的形状，期望形状为 [50, 2]
    print("Output shape:", output.shape)  # 期望输出: torch.Size([50, 2])
