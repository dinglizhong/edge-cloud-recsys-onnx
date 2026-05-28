import torch
import torch.nn as nn
import faiss
import numpy as np
import pandas as pd
import os

# --- 1. 双塔模型定义 ---
class TwoTowerModel(nn.Module):
    """
    极简双塔模型 (召回层)
    """
    def __init__(self, user_dim=16, item_dim=16, embedding_dim=64):
        super(TwoTowerModel, self).__init__()
        # User 塔
        self.user_tower = nn.Sequential(
            nn.Linear(user_dim, 128),
            nn.ReLU(),
            nn.Linear(128, embedding_dim)
        )
        # Item 塔
        self.item_tower = nn.Sequential(
            nn.Linear(item_dim, 128),
            nn.ReLU(),
            nn.Linear(128, embedding_dim)
        )

    def forward(self, user_x, item_x):
        user_emb = self.user_tower(user_x)
        item_emb = self.item_tower(item_x)
        # 归一化，方便做余弦相似度计算
        user_emb = nn.functional.normalize(user_emb, p=2, dim=1)
        item_emb = nn.functional.normalize(item_emb, p=2, dim=1)
        return user_emb, item_emb
    
    def get_user_embedding(self, user_x):
        emb = self.user_tower(user_x)
        return nn.functional.normalize(emb, p=2, dim=1)
        
    def get_item_embedding(self, item_x):
        emb = self.item_tower(item_x)
        return nn.functional.normalize(emb, p=2, dim=1)

# --- 2. 离线构建 Faiss 向量库 ---
def build_faiss_index(csv_path="data/app_features_enriched.csv", index_path="models/app_faiss.index"):
    print("Building Faiss Index for Recall Layer...")
    os.makedirs("models", exist_ok=True)
    
    # 1. 加载所有 App 数据
    df = pd.read_csv(csv_path)
    num_apps = len(df)
    
    # 2. 初始化未训练的双塔模型 (实际工程中应该是加载训练好的权重)
    # 这里我们用随机权重模拟，保证代码全链路通畅
    model = TwoTowerModel(item_dim=16, embedding_dim=64)
    model.eval()
    
    # 3. 模拟把 50 个 App 的特征喂给 Item 塔
    # 真实场景中，这里的输入应该是 CSV 里的各种标签转成稠密特征
    dummy_item_features = torch.randn(num_apps, 16) 
    
    with torch.no_grad():
        item_embeddings = model.get_item_embedding(dummy_item_features).numpy()
    
    # 4. 构建 Faiss 索引 (使用内积 IndexFlatIP，因为之前做了归一化，所以等价于余弦相似度)
    embedding_dim = 64
    index = faiss.IndexFlatIP(embedding_dim)
    
    # 为了演示 Faiss 的 ID 映射功能，我们必须用 IndexIDMap
    # 这样查出来的时候才知道是哪个 App_ID (0~49)
    index = faiss.IndexIDMap(index)
    app_ids = np.arange(num_apps).astype(np.int64)
    
    index.add_with_ids(item_embeddings, app_ids)
    
    # 5. 保存索引到硬盘
    faiss.write_index(index, index_path)
    print(f"Faiss Index built successfully with {index.ntotal} items. Saved to {index_path}")

# --- 3. 线上实时检索函数 ---
class FaissRetriever:
    def __init__(self, index_path="models/app_faiss.index"):
        self.index = faiss.read_index(index_path)
        self.model = TwoTowerModel(user_dim=16, embedding_dim=64)
        self.model.eval()
        
    def recall(self, user_features_tensor, top_k=30):
        """
        根据用户特征，从 Faiss 库中召回 Top-K 个 App
        """
        # 1. 实时过 User 塔
        with torch.no_grad():
            user_emb = self.model.get_user_embedding(user_features_tensor).numpy()
            
        # 2. Faiss 极速检索
        # D 是相似度分数，I 是召回的 App 索引 (0~49)
        D, I = self.index.search(user_emb, top_k)
        
        # 返回第一条请求（因为 batch_size=1）的召回结果索引列表
        return I[0].tolist()

if __name__ == "__main__":
    # 测试代码
    build_faiss_index()
    
    # 测试检索
    retriever = FaissRetriever()
    dummy_user = torch.randn(1, 16)
    recalled_indices = retriever.recall(dummy_user, top_k=5)
    print(f"Test Recall Top 5 App Indices: {recalled_indices}")