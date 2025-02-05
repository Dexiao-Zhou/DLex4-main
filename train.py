import torch as t
from data import ChallengeDataset  # 导入自定义的数据集类
from trainer import Trainer         # 导入自定义的训练器类
from matplotlib import pyplot as plt
import numpy as np
import model                      # 导入模型相关代码
import pandas as pd
from sklearn.model_selection import train_test_split  # 导入数据集拆分工具
from torch.optim.lr_scheduler import ReduceLROnPlateau  # 导入学习率调度器

def main():
    # 设置设备：若 GPU 可用则使用 GPU，否则使用 CPU
    device = t.device("cuda" if t.cuda.is_available() else "cpu")

    # 加载 CSV 数据，并指定分隔符为分号
    csv_path = 'data.csv'
    tab = pd.read_csv(csv_path, sep=';')

    # 划分数据集为训练集和验证集，其中验证集占 20%，并使用 stratify 保证类别比例相同
    train_tab, val_tab = train_test_split(
        tab,
        test_size=0.2,
        random_state=31,
        shuffle=True,
        stratify=tab.iloc[:, 1]  # 假设第二列是类别标签
    )

    # 创建训练集的 DataLoader，优化数据加载
    train_dl = t.utils.data.DataLoader(
        ChallengeDataset(train_tab, 'train'),  # 使用自定义数据集
        batch_size=256,         # 设置批次大小，增大 batch_size 可提高 GPU 利用率
        shuffle=True,           # 训练数据随机打乱
        num_workers=14,         # 使用多进程数据加载，加快数据准备速度
        pin_memory=True,        # 将数据锁页，提高 CPU-GPU 数据传输速度
        pin_memory_device="cuda",  # PyTorch 2.0+ 可直接将内存固定到 GPU
        persistent_workers=True,   # 持久化 worker 进程，避免每个 epoch 重新启动
        prefetch_factor=4       # 每个 worker 预取数据的批次数量
    )

    # 创建验证集的 DataLoader
    val_dl = t.utils.data.DataLoader(
        ChallengeDataset(val_tab, 'val'),  # 使用自定义数据集，模式为验证
        batch_size=256,
        num_workers=14,
        pin_memory=True,
        pin_memory_device="cuda",  # 固定 GPU 内存
        persistent_workers=True,
        prefetch_factor=4
    )

    # 实例化模型，并将模型移动到指定设备（GPU 或 CPU）
    model_instance = model.ResNet().to(device)

    # 定义损失函数，这里使用二分类交叉熵损失函数（BCELoss）
    crit = t.nn.BCELoss()

    # 定义优化器，使用 Adam 优化器，并设置学习率和权重衰减
    optimizer = t.optim.Adam(model_instance.parameters(), lr=1e-3, weight_decay=1e-5)

    # 设置学习率调度器：当验证集损失不再下降时降低学习率
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',        # 监控指标越小越好
        factor=0.5,        # 学习率下降的因子
        patience=10,       # 当验证损失在 10 个 epoch 内没有下降时调整学习率
        threshold=0.0001,  # 调整阈值
        cooldown=2         # 调整后等待2个 epoch再开始监控
    )

    # 实例化训练器，传入模型、损失函数、优化器、训练集和验证集 DataLoader，并指定使用 GPU
    trainer = Trainer(model_instance, crit, optimizer, train_dl, val_dl, cuda=True)

    # 开始训练模型，训练 300 个 epoch，同时将学习率调度器传递给训练器
    res = trainer.fit(epochs=300, scheduler=scheduler)

    # 绘制训练和验证的 Loss 曲线
    plt.plot(np.arange(len(res[0])), res[0], label='train loss') 
    plt.plot(np.arange(len(res[1])), res[1], label='val loss')  
    plt.yscale('log')  
    plt.legend()   
    plt.savefig('losses.png') 
    plt.show()             

# 程序入口
if __name__ == '__main__':
    main()
