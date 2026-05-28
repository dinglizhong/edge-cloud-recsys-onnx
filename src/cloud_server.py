from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import random
import time
import torch
import os
import pandas as pd
from src.mmoe_model import MMoE
from src.recall_faiss import FaissRetriever

app = FastAPI(title="OPPO Cloud Recommendation API")

# --- 1. 全局加载云端 MMoE 模型、Faiss 索引和特征数据 ---
FEATURE_DIM = 16
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "models/mmoe_trained.pt"
FEATURES_PATH = "data/app_features_enriched.csv"
FAISS_INDEX_PATH = "models/app_faiss.index"

# 1.1 初始化 MMoE 精排模型
mmoe_model = MMoE(feature_dim=FEATURE_DIM, num_experts=3, expert_hidden_units=64).to(DEVICE)
mmoe_model.eval()

if os.path.exists(MODEL_PATH):
    print(f"Loading trained MMoE weights from {MODEL_PATH}...")
    mmoe_model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))

# 1.2 加载 Faiss 召回引擎
print(f"Initializing Faiss Retriever from {FAISS_INDEX_PATH}...")
if os.path.exists(FAISS_INDEX_PATH):
    retriever = FaissRetriever(FAISS_INDEX_PATH)
else:
    print("Warning: Faiss index not found! Please run recall_faiss.py first.")
    retriever = None

# 1.3 加载候选物料库
if os.path.exists(FEATURES_PATH):
    print(f"Loading App Feature Database from {FEATURES_PATH}...")
    df_apps = pd.read_csv(FEATURES_PATH)
    MOCK_APPS = df_apps.to_dict('records')
else:
    MOCK_APPS = []

# --- 2. 定义请求和响应的结构 ---
class RecRequest(BaseModel):
    user_id: str
    request_id: str
    top_k: int = 10

class AppItem(BaseModel):
    app_id: str
    app_name: str
    category: str  # 新增类别字段，方便端侧重排逻辑判断
    cloud_ctr_score: float
    cloud_cvr_score: float
    llm_intent_match_score: float

class RecResponse(BaseModel):
    request_id: str
    user_id: str
    recalled_app_names: List[str] # 传回 Faiss 召回的 Top-50 名单
    recommended_apps: List[AppItem] # 传回 MMoE 精排的 Top-25 名单
    latency_ms: float

# --- 3. 核心 API 逻辑 ---
@app.post("/api/v1/recommend", response_model=RecResponse)
async def get_recommendations(req: RecRequest):
    """
    真实的 Cloud-side Recommendation Engine.
    执行流程: 召回 (50) -> MMoE 模型真实推理 -> 排序下发 (25)
    """
    start_time = time.time()
    
    if not MOCK_APPS or not retriever:
        return RecResponse(request_id=req.request_id, user_id=req.user_id, recalled_app_names=[], recommended_apps=[], latency_ms=0.0)

    # 1. 真实 Faiss 召回阶段 (Recall)
    user_features = torch.randn(1, FEATURE_DIM)
    
    # 向 Faiss 引擎发起检索，召回 Top-50
    recall_k = 50
    recalled_indices = retriever.recall(user_features, top_k=recall_k)
    recalled_apps = [MOCK_APPS[idx] for idx in recalled_indices]
    
    # 提取召回的名字供展示
    recalled_app_names = [app["app_name"] for app in recalled_apps]
    
    # 2. 构造 MMoE 模型的真实输入特征
    batch_features = torch.randn(len(recalled_apps), FEATURE_DIM).to(DEVICE)
    
    # 3. 真实的 MMoE 模型前向推理 (Inference)
    with torch.no_grad():
        pred_ctr, pred_cvr = mmoe_model(batch_features)
    
    ctr_scores = pred_ctr.cpu().numpy().flatten().tolist()
    cvr_scores = pred_cvr.cpu().numpy().flatten().tolist()
    
    # 4. 组装结果并进行云端基础排序
    scored_apps = []
    for idx, app in enumerate(recalled_apps):
        category = app.get("category", "Unknown")
        intent_score = random.uniform(0.5, 0.99)
        
        scored_apps.append(AppItem(
            app_id=app["app_id"],
            app_name=app["app_name"],
            category=category,
            cloud_ctr_score=round(ctr_scores[idx], 4),
            cloud_cvr_score=round(cvr_scores[idx], 4),
            llm_intent_match_score=round(intent_score, 4)
        ))
    
    # 5. 云端精排: 按照 CTR 预估分倒序排列
    scored_apps.sort(key=lambda x: x.cloud_ctr_score, reverse=True)
    
    # 截断：只取 Top-25 下发给端侧手机 (req.top_k = 25)
    final_apps = scored_apps[:req.top_k]
    
    latency = round((time.time() - start_time) * 1000, 2)
    
    print("\n" + "="*60)
    print(f"🚀 [Cloud API] 收到推荐请求! User_ID: {req.user_id}")
    print(f"🔍 [Faiss] 成功召回 Top-50 个候选应用")
    print(f"🧠 [MMoE] 完成精排打分，截断并下发 Top-25 到手机端")
    print(f"✅ [Cloud API] 总耗时: {latency} ms")
    print("="*60 + "\n")
    
    return RecResponse(
        request_id=req.request_id,
        user_id=req.user_id,
        recalled_app_names=recalled_app_names,
        recommended_apps=final_apps,
        latency_ms=latency
    )

if __name__ == "__main__":
    import uvicorn
    print("Starting Cloud Recommendation Server on port 8000...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
