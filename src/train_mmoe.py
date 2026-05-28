import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score
import numpy as np
import pandas as pd
import time
import os

# 导入我们之前写的 MMoE 模型
from mmoe_model import MMoE

class AliCCPDataset(Dataset):
    """
    针对 Ali-CCP (阿里点击与转化数据集) 的 PyTorch Dataset。
    在真实场景中，你会从 CSV 或 TFRecord 中读取数据。
    这里为了保证代码能直接跑通，我们加入了一个自动生成 Mock 数据的逻辑。
    """
    def __init__(self, csv_path=None, num_samples=10000, feature_dim=16):
        self.feature_dim = feature_dim
        
        if csv_path and os.path.exists(csv_path):
            print(f"Loading real data from {csv_path}...")
            # 真实场景读取代码：
            # df = pd.read_csv(csv_path)
            # self.features = torch.tensor(df.drop(['click', 'conversion'], axis=1).values, dtype=torch.float32)
            # self.labels_ctr = torch.tensor(df['click'].values, dtype=torch.float32)
            # self.labels_cvr = torch.tensor(df['conversion'].values, dtype=torch.float32)
            pass
        else:
            print("No real CSV provided. Generating Mock Ali-CCP data for training pipeline verification...")
            # 模拟特征输入 (User Profile + Item Features + LLM Features)
            self.features = torch.randn(num_samples, feature_dim)
            
            # 模拟标签
            # 1. 模拟 CTR (点击): 约 10% 的点击率
            self.labels_ctr = (torch.rand(num_samples) > 0.9).float()
            
            # 2. 模拟 CVR (转化): 只有点击了才有可能转化，且转化率更低 (约点击后的 20%)
            self.labels_cvr = (self.labels_ctr * (torch.rand(num_samples) > 0.8)).float()

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels_ctr[idx], self.labels_cvr[idx]

def train_and_evaluate():
    # 1. 超参数设置 (Hyperparameters)
    FEATURE_DIM = 16
    BATCH_SIZE = 512
    EPOCHS = 5
    LEARNING_RATE = 0.001
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    # 2. 准备数据
    # 提示：在 AutoDL 上，你可以把 csv_path 替换为你下载的真实 Ali-CCP 数据路径
    train_dataset = AliCCPDataset(num_samples=50000, feature_dim=FEATURE_DIM)
    val_dataset = AliCCPDataset(num_samples=10000, feature_dim=FEATURE_DIM)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 3. 初始化模型、Loss 和优化器
    model = MMoE(feature_dim=FEATURE_DIM, num_experts=3, expert_hidden_units=64).to(device)
    
    # 因为是 0/1 二分类问题，使用 Binary Cross Entropy Loss
    criterion_ctr = nn.BCELoss()
    criterion_cvr = nn.BCELoss()
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)

    # 4. 训练循环 (Training Loop)
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        start_time = time.time()
        
        for batch_features, batch_ctr, batch_cvr in train_loader:
            batch_features = batch_features.to(device)
            batch_ctr = batch_ctr.to(device).unsqueeze(1)
            batch_cvr = batch_cvr.to(device).unsqueeze(1)

            optimizer.zero_grad()

            # 前向传播 (Forward pass)
            pred_ctr, pred_cvr = model(batch_features)

            # 计算损失 (Calculate Loss)
            loss_ctr = criterion_ctr(pred_ctr, batch_ctr)
            loss_cvr = criterion_cvr(pred_cvr, batch_cvr)
            
            # 联合 Loss: 这里给了 CTR 和 CVR 相同的权重。真实业务中由于 CVR 样本少，可能会调大 CVR 的权重
            loss = loss_ctr + loss_cvr

            # 反向传播 (Backward pass)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        
        # 5. 验证环节 (Evaluation)
        model.eval()
        val_ctr_preds, val_cvr_preds = [], []
        val_ctr_trues, val_cvr_trues = [], []
        
        with torch.no_grad():
            for batch_features, batch_ctr, batch_cvr in val_loader:
                batch_features = batch_features.to(device)
                pred_ctr, pred_cvr = model(batch_features)
                
                val_ctr_preds.extend(pred_ctr.cpu().numpy().flatten())
                val_cvr_preds.extend(pred_cvr.cpu().numpy().flatten())
                val_ctr_trues.extend(batch_ctr.numpy().flatten())
                val_cvr_trues.extend(batch_cvr.numpy().flatten())
        
        # 计算 AUC
        try:
            auc_ctr = roc_auc_score(val_ctr_trues, val_ctr_preds)
            auc_cvr = roc_auc_score(val_cvr_trues, val_cvr_preds)
        except ValueError:
            # 如果 batch 里面只有一类标签，AUC 算不出来会报错，这里做个保护
            auc_ctr, auc_cvr = 0.5, 0.5

        epoch_time = time.time() - start_time
        print(f"Epoch {epoch+1}/{EPOCHS} | Time: {epoch_time:.2f}s | Train Loss: {avg_train_loss:.4f} | Val CTR AUC: {auc_ctr:.4f} | Val CVR AUC: {auc_cvr:.4f}")

    # 6. 保存模型 (供导出 ONNX 使用)
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/mmoe_trained.pt")
    print("Training Complete! Model saved to models/mmoe_trained.pt")

if __name__ == "__main__":
    train_and_evaluate()
