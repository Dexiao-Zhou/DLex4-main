import torch as t
from data import ChallengeDataset
from trainer import Trainer
from matplotlib import pyplot as plt
import numpy as np
import model
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import ReduceLROnPlateau

def main():
    # 设置设备
    device = t.device("cuda" if t.cuda.is_available() else "cpu")

    # 加载数据
    csv_path = 'data.csv'
    tab = pd.read_csv(csv_path, sep=';')

    train_tab, val_tab = train_test_split(tab, test_size=0.2, random_state=31, shuffle=True, stratify=tab.iloc[:, 1])

    # 创建 DataLoader（优化数据加载）
    train_dl = t.utils.data.DataLoader(
        ChallengeDataset(train_tab, 'train'),
        batch_size=256,  # 增大 batch_size 以提高 GPU 利用率
        shuffle=True,
        num_workers=14,   # 启用多线程数据加载
        pin_memory=True,  # 提高 CPU-GPU 数据传输速度
        pin_memory_device="cuda",  # PyTorch 2.0+ 直接固定 GPU 内存
        persistent_workers=True,
        prefetch_factor=4
    )

    val_dl = t.utils.data.DataLoader(
        ChallengeDataset(val_tab, 'val'),
        batch_size=256,
        num_workers=14,
        pin_memory=True,
        pin_memory_device="cuda",  # PyTorch 2.0+ 直接固定 GPU 内存
        persistent_workers=True,
        prefetch_factor=4
    )

    # 创建模型并移动到 GPU
    model_instance = model.ResNet().to(device)

    # 设置损失函数和优化器
    crit = t.nn.BCELoss()
    optimizer = t.optim.Adam(model_instance.parameters(), lr=1e-3, weight_decay=1e-5)

    # 创建学习率调度器
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=10,  # 延长响应周期
        threshold=0.0001,
        cooldown=2
    )

    # 创建 Trainer（确保 Trainer 代码内部正确使用 GPU）
    trainer = Trainer(model_instance, crit, optimizer, train_dl, val_dl, cuda=True)

    # 训练模型
    res = trainer.fit(epochs=300, scheduler=scheduler)  # 将 scheduler 传递给 fit 方法

    # 画出 Loss 曲线
    plt.plot(np.arange(len(res[0])), res[0], label='train loss')
    plt.plot(np.arange(len(res[1])), res[1], label='val loss')
    plt.yscale('log')
    plt.legend()
    plt.savefig('losses.png')
    plt.show()

if __name__ == '__main__':
    main()