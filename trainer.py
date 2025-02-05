import torch as t
from torch.utils.tensorboard import SummaryWriter  # 导入 TensorBoard 日志记录工具
from tqdm.autonotebook import tqdm              # 导入带进度条的循环工具，便于可视化训练过程
import os
from sklearn.metrics import f1_score              # 导入 f1_score 用于计算 F1 分数

class Trainer:
    def __init__(self,
                 model,                        # 训练的模型
                 crit,                         # 损失函数
                 optim=None,                   # 优化器
                 train_dl=None,                # 训练数据集的 DataLoader
                 val_test_dl=None,             # 验证/测试数据集的 DataLoader
                 cuda=True,                    # 是否使用 GPU 加速
                 early_stopping_patience=10,   # 早停策略的耐心值（即连续多少个 epoch 没有改善时停止训练）
                 log_dir="runs/experiment"):   # TensorBoard 日志目录
        # 保存传入的参数
        self._model = model
        self._crit = crit
        self._optim = optim
        self._train_dl = train_dl
        self._val_test_dl = val_test_dl
        self._cuda = cuda
        self._early_stopping_patience = early_stopping_patience

        # 如果使用 GPU，则将模型和损失函数移动到 GPU
        if cuda:
            self._model = model.cuda()
            self._crit = crit.cuda()

        # 确保 TensorBoard 日志目录存在，不存在则创建
        os.makedirs(log_dir, exist_ok=True)
        # 初始化 SummaryWriter，用于记录训练过程中的指标
        self.writer = SummaryWriter(log_dir)

    def save_checkpoint(self, epoch):
        """
        每 10 个 epoch 保存一次当前模型的检查点
        :param epoch: 当前训练的 epoch 序号，用于生成检查点文件名
        """
        if epoch % 10 == 0:  # 只有当 epoch 为 10 的倍数时才保存
            os.makedirs("checkpoints", exist_ok=True)  # 确保检查点目录存在
            checkpoint_path = f'checkpoints/checkpoint_{epoch:03d}.ckp'
            t.save({'state_dict': self._model.state_dict()}, checkpoint_path)
            print(f"✅ Checkpoint saved at {checkpoint_path}")


    def restore_checkpoint(self, epoch_n):
        """
        从指定 epoch 的检查点中恢复模型状态
        :param epoch_n: 要恢复的检查点对应的 epoch 序号
        """
        ckp = t.load(f'checkpoints/checkpoint_{epoch_n:03d}.ckp',
                     map_location='cuda' if self._cuda else 'cpu')
        self._model.load_state_dict(ckp['state_dict'])

    def save_onnx(self, fn):
        """
        将模型导出为 ONNX 格式文件
        :param fn: 导出的文件名或路径
        """
        # 先将模型移动到 CPU，并设置为评估模式
        m = self._model.cpu()
        m.eval()
        # 构造一个假输入数据，用于导出时的模型跟踪，尺寸为 (1, 3, 300, 300)
        x = t.randn(1, 3, 300, 300, requires_grad=True)
        y = self._model(x)
        # 使用 PyTorch 的 onnx.export 导出模型
        t.onnx.export(m,                # 待导出的模型
                      x,                # 模型的输入（单个张量或元组）
                      fn,               # 导出文件保存路径
                      export_params=True,  # 将训练好的参数一起导出
                      opset_version=10,    # ONNX 版本
                      do_constant_folding=True,  # 是否执行常量折叠优化
                      input_names=['input'],     # 定义模型输入名称
                      output_names=['output'],   # 定义模型输出名称
                      dynamic_axes={'input': {0: 'batch_size'},  # 指定输入的动态轴
                                    'output': {0: 'batch_size'}})

    def train_step(self, x, y):
        """
        单步训练过程：前向传播、计算损失、反向传播和更新模型参数
        :param x: 输入数据
        :param y: 目标标签
        :return: 当前 batch 的损失值（标量）
        """
        self._model.train()           # 设置模型为训练模式
        self._optim.zero_grad()       # 清除上一步的梯度信息
        pred = self._model(x)         # 前向传播获得预测结果
        loss = self._crit(pred, y)    # 计算损失
        loss.backward()               # 反向传播，计算梯度
        
        # 对梯度进行裁剪，防止梯度爆炸，控制最大范数为 2.0
        t.nn.utils.clip_grad_norm_(self._model.parameters(), max_norm=2.0)
        
        self._optim.step()           # 根据梯度更新模型参数
        return loss.item()           # 返回当前 batch 的损失值（数值型）

    def val_test_step(self, x, y):
        """
        单步验证/测试过程：前向传播计算损失（不更新参数）
        :param x: 输入数据
        :param y: 目标标签
        :return: 当前 batch 的损失值和预测结果
        """
        self._model.eval()           # 设置模型为评估模式
        with t.no_grad():            # 关闭梯度计算，节省内存和计算
            pred = self._model(x)    # 前向传播获得预测结果
            loss = self._crit(pred, y)  # 计算损失
        return loss.item(), pred

    def train_epoch(self, epoch):
        """
        执行一个 epoch 的训练过程
        :param epoch: 当前 epoch 序号
        :return: 当前 epoch 的平均训练损失
        """
        self._model.train()   # 确保模型处于训练模式
        total_loss = 0.0      # 累计损失初始化
        count = 0             # batch 数计数器
        # 遍历训练集的每个 batch，使用 tqdm 显示进度条
        for x, y in tqdm(self._train_dl, desc=f"Training Epoch {epoch+1}"):
            if self._cuda:
                x, y = x.cuda(), y.cuda()  # 将数据移动到 GPU
            loss = self.train_step(x, y)    # 执行单步训练
            total_loss += loss            # 累加当前 batch 的损失
            count += 1                    # batch 数加 1
        # 计算平均训练损失
        avg_loss = total_loss / count if count > 0 else 0.0
        # 使用 TensorBoard 记录当前 epoch 的训练损失
        self.writer.add_scalar("Loss/Train", avg_loss, epoch)
        return avg_loss

    def val_test(self, epoch):
        """
        执行一个 epoch 的验证/测试过程，计算平均损失和 F1 分数
        :param epoch: 当前 epoch 序号
        :return: 当前 epoch 的平均验证损失和 F1 分数
        """
        self._model.eval()  # 设置模型为评估模式
        total_loss = 0.0    # 累计损失初始化
        count = 0           # batch 数计数器
        
        # 用于存储所有 batch 的预测结果和真实标签
        all_preds = []
        all_labels = []
        
        with t.no_grad():
            # 遍历验证/测试集的每个 batch
            for x, y in tqdm(self._val_test_dl, desc=f"Validating Epoch {epoch+1}"):
                if self._cuda:
                    x, y = x.cuda(), y.cuda()  # 将数据移动到 GPU
                loss, pred = self.val_test_step(x, y)  # 执行单步验证/测试
                total_loss += loss          # 累加损失
                count += 1                  # batch 数加 1
                
                # 将当前 batch 的预测和标签保存起来
                # 这里假设模型输出为概率值，使用 round() 四舍五入得到类别标签（如 0 或 1）
                all_preds.append(pred.cpu().round())
                all_labels.append(y.cpu())
                
        # 计算平均验证损失
        avg_loss = total_loss / count if count > 0 else 0.0
        
        # 将所有 batch 的预测和标签拼接成一个完整的张量
        all_preds = t.cat(all_preds, dim=0).squeeze()
        all_labels = t.cat(all_labels, dim=0).squeeze()
        
        # 转换张量为 numpy 数组，并计算加权 F1 分数
        f1 = f1_score(all_labels.numpy(), all_preds.numpy(), average='weighted')
        # 使用 TensorBoard 记录验证损失和 F1 分数
        self.writer.add_scalar("F1/Validation", f1, epoch)
        self.writer.add_scalar("Loss/Validation", avg_loss, epoch)
        
        print(f"Validation Epoch {epoch+1}: Avg Loss: {avg_loss:.4f}, F1 Score: {f1:.4f}")
        return avg_loss, f1

    def fit(self, epochs=10, scheduler=None):
        """
        开始模型训练，执行多个 epoch，并使用早停策略
        :param epochs: 最大训练的 epoch 数量
        :param scheduler: 学习率调度器，根据验证损失调整学习率
        :return: 训练和验证过程中记录的训练损失、验证损失和 F1 分数列表
        """
        best_val_loss = float('inf')   # 初始化最佳验证损失为无穷大
        patience_counter = 0           # 早停计数器
        train_losses, val_losses, f1_scores = [], [], []  # 用于记录每个 epoch 的指标
        
        for epoch in range(epochs):
            # 训练一个 epoch
            train_loss = self.train_epoch(epoch)
            
            # 验证一个 epoch
            val_loss, f1 = self.val_test(epoch)
            
            # 记录当前 epoch 的训练和验证指标
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            f1_scores.append(f1)
            
            # 若提供学习率调度器，则根据验证损失调整学习率
            if scheduler is not None:
                scheduler.step(val_loss)
            
            # 保存当前 epoch 的模型检查点
            self.save_checkpoint(epoch)
            
            # 打印当前 epoch 的训练结果
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f} - F1 Score: {f1:.4f}")
            
            # 如果验证损失有明显下降（降低至少 0.5%），则重置早停计数器，否则累加计数
            if val_loss < best_val_loss * 0.995:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                # 当连续多个 epoch 验证损失无改善时，触发早停策略
                if patience_counter >= 20:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        # 训练结束后关闭 TensorBoard 记录器
        self.writer.close()
        
        # 返回训练过程中的训练损失、验证损失和 F1 分数记录
        return train_losses, val_losses, f1_scores
