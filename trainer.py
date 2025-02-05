import torch as t
from torch.utils.tensorboard import SummaryWriter
from tqdm.autonotebook import tqdm
import os
from sklearn.metrics import f1_score  # 引入 f1_score 计算指标

class Trainer:
    def __init__(self,
                 model,                        # 训练的模型
                 crit,                         # 损失函数
                 optim=None,                   # 优化器
                 train_dl=None,                # 训练数据集
                 val_test_dl=None,             # 验证/测试数据集
                 cuda=True,                    # 是否使用 GPU
                 early_stopping_patience=10,   # 早停策略
                 log_dir="runs/experiment"):   # TensorBoard 日志目录
        
        self._model = model
        self._crit = crit
        self._optim = optim
        self._train_dl = train_dl
        self._val_test_dl = val_test_dl
        self._cuda = cuda
        self._early_stopping_patience = early_stopping_patience

        if cuda:
            self._model = model.cuda()
            self._crit = crit.cuda()

        # 🔥 确保 TensorBoard 日志目录存在
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir)

    def save_checkpoint(self, epoch):
        os.makedirs("checkpoints", exist_ok=True)  # 确保检查点目录存在
        t.save({'state_dict': self._model.state_dict()}, 
               f'checkpoints/checkpoint_{epoch:03d}.ckp')

    def restore_checkpoint(self, epoch_n):
        ckp = t.load(f'checkpoints/checkpoint_{epoch_n:03d}.ckp',
                     map_location='cuda' if self._cuda else 'cpu')
        self._model.load_state_dict(ckp['state_dict'])

    def save_onnx(self, fn):
        m = self._model.cpu()
        m.eval()
        x = t.randn(1, 3, 300, 300, requires_grad=True)
        y = self._model(x)
        t.onnx.export(m,  # model being run
                      x,  # model input (or a tuple for multiple inputs)
                      fn,  # where to save the model (can be a file or file-like object)
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=10,  # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names=['input'],  # the model's input names
                      output_names=['output'],  # the model's output names
                      dynamic_axes={'input': {0: 'batch_size'},  # variable lenght axes
                                    'output': {0: 'batch_size'}})

    def train_step(self, x, y):
        self._model.train()
        self._optim.zero_grad()
        pred = self._model(x)
        loss = self._crit(pred, y)
        loss.backward()
        
        # 添加梯度裁剪（控制梯度最大范数为1.0）
        t.nn.utils.clip_grad_norm_(self._model.parameters(), max_norm=2.0)
        
        self._optim.step()
        return loss.item()

    def val_test_step(self, x, y):
        self._model.eval()
        with t.no_grad():
            pred = self._model(x)
            loss = self._crit(pred, y)
        return loss.item(), pred

    def train_epoch(self, epoch):
        self._model.train()
        total_loss = 0.0
        count = 0
        for x, y in tqdm(self._train_dl, desc=f"Training Epoch {epoch+1}"):
            if self._cuda:
                x, y = x.cuda(), y.cuda()
            loss = self.train_step(x, y)
            total_loss += loss
            count += 1
        avg_loss = total_loss / count if count > 0 else 0.0
        self.writer.add_scalar("Loss/Train", avg_loss, epoch)  # ✅ 记录训练损失
        return avg_loss

    def val_test(self, epoch):
        self._model.eval()
        total_loss = 0.0
        count = 0
        
        # 用于存储所有预测和标签
        all_preds = []
        all_labels = []
        
        with t.no_grad():
            for x, y in tqdm(self._val_test_dl, desc=f"Validating Epoch {epoch+1}"):
                if self._cuda:
                    x, y = x.cuda(), y.cuda()
                loss, pred = self.val_test_step(x, y)
                total_loss += loss
                count += 1
                
                # 将当前 batch 的预测和标签保存起来，预测结果四舍五入得到 0 或 1
                # 注意：这里假设模型输出形状适合直接 round() 得到类别（如二分类问题）
                all_preds.append(pred.cpu().round())
                all_labels.append(y.cpu())
                
        avg_loss = total_loss / count if count > 0 else 0.0
        
        # 拼接所有 batch 的预测和标签
        all_preds = t.cat(all_preds, dim=0).squeeze()
        all_labels = t.cat(all_labels, dim=0).squeeze()
        
        # 将 tensor 转为 numpy 数组并计算 F1 Score
        f1 = f1_score(all_labels.numpy(), all_preds.numpy(), average='weighted')
        self.writer.add_scalar("F1/Validation", f1, epoch)  # ✅ 记录验证集 F1 分数
        self.writer.add_scalar("Loss/Validation", avg_loss, epoch)  # ✅ 记录验证损失
        
        print(f"Validation Epoch {epoch+1}: Avg Loss: {avg_loss:.4f}, F1 Score: {f1:.4f}")
        return avg_loss, f1

    def fit(self, epochs=10, scheduler=None):
        best_val_loss = float('inf')
        patience_counter = 0
        train_losses, val_losses, f1_scores = [], [], []
        for epoch in range(epochs):
            # 训练一个 epoch
            train_loss = self.train_epoch(epoch)
            
            # 验证一个 epoch
            val_loss, f1 = self.val_test(epoch)
            
            # 记录损失和 F1 分数
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            f1_scores.append(f1)
            
            # 调用学习率调度器（如果提供了 scheduler）
            if scheduler is not None:
                scheduler.step(val_loss)  # 根据验证损失调整学习率
            
            # 保存模型检查点
            self.save_checkpoint(epoch)
            
            # 打印当前 epoch 的结果
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f} - F1 Score: {f1:.4f}")
            if val_loss < best_val_loss*0.995:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter +=1
                if patience_counter >= 20:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        # 训练结束后关闭 TensorBoard
        self.writer.close()
        
        # 返回训练和验证的损失以及 F1 分数
        return train_losses, val_losses, f1_scores