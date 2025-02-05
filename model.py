import torch
import torch.nn as nn
from torchsummary import summary

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.batch_norm1 = nn.BatchNorm2d(out_channels)
        self.relu_1 = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.batch_norm2 = nn.BatchNorm2d(out_channels)
        self.residual_conv = True
        self.conv1X1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
        if in_channels == out_channels and stride == 1:
            self.residual_conv = False
        else:
            self.residual_conv = True

        self.batch_norm3 = nn.BatchNorm2d(out_channels)
        self.relu_3 = nn.ReLU()
        self.seq = nn.Sequential(self.conv1, self.batch_norm1, self.relu_1, self.conv2, self.batch_norm2)
        self.residual = None
        self.cnt = 0

    def forward(self, input_tensor):
        self.residual = input_tensor  # 移除非必要的self.residual属性
        output_tensor = self.seq(input_tensor)
        
        if self.residual_conv:
            self.residual = self.conv1X1(self.residual)  # 仅当使用1x1卷积时应用BN
        self.residual = self.batch_norm3(self.residual)
        output_tensor += self.residual
        output_tensor = self.relu_3(output_tensor)
        return output_tensor


class ResNet(nn.Module):
    def __init__(self, num_classes=2):
        super(ResNet, self).__init__()
        self.seq1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2)
            )
        self.seq2 = nn.Sequential(
            BasicBlock(64, 64, stride=1),
            BasicBlock(64, 128, stride=2),
            BasicBlock(128, 256, stride=2),
            BasicBlock(256, 512, stride=2),
            )
        
        self.seq3 = nn.Sequential(
            nn.AvgPool2d(10),
            nn.Flatten(),
            nn.Linear(in_features=512, out_features=num_classes),
            nn.Sigmoid()
        )

        
    def forward(self, input_tensor):
        output_tensor = self.seq1(input_tensor)
        output_tensor = self.seq2(output_tensor)
        output_tensor = self.seq3(output_tensor)

        return output_tensor


# 测试模型
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNet(num_classes=2).to(device)

    print("✅ Model initialized on", device)
    
    # 显示模型结构
    summary(model, input_size=(3, 300, 300))
    
    # 测试随机输入
    dummy_input = torch.randn(50, 3, 300, 300).to(device)
    output = model(dummy_input)
    print("Output shape:", output.shape)  # 期望输出: torch.Size([50, 2])
