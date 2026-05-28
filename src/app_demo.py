import streamlit as st
import pandas as pd
import numpy as np
import requests
import onnxruntime as ort
import time
import os

# Configurations
# 由于你使用的是 SSH 隧道转发，并且 AutoDL 工具要求端口不能同名
CLOUD_API_URL = os.environ.get("CLOUD_API_URL", "http://127.0.0.1:16006/api/v1/recommend")
ONNX_MODEL_PATH = "models/edge_reranker.onnx"

# Set page config
st.set_page_config(page_title="OPPO 端云协同推荐 Demo", page_icon="📱", layout="wide")

# Custom CSS for better UI
st.markdown("""
<style>
    .app-card {
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 10px;
        background-color: #f9f9f9;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.05);
    }
    .score-badge {
        background-color: #ff6a00;
        color: white;
        padding: 3px 8px;
        border-radius: 5px;
        font-weight: bold;
        font-size: 0.9em;
    }
    .delta-positive { color: #00a65a; font-weight: bold; }
    .delta-negative { color: #e53935; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

st.title("📱 OPPO 面试 Demo：端云协同推荐系统")
st.markdown("通过调整左侧的**手机端实时特征**，观察右侧推荐列表如何被 **ONNX 端侧模型** 瞬间重排。")

# --- Sidebar: Edge Features ---
st.sidebar.header("⚙️ 手机端实时特征 (Edge Features)")
st.sidebar.markdown("*(这些数据高度隐私，不上报云端)*")

battery_level = st.sidebar.slider("🔋 当前电量 (%)", min_value=1, max_value=100, value=15) / 100.0
network_type = st.sidebar.selectbox("📶 网络状态", ["4G", "WIFI"])
is_wifi = 1.0 if network_type == "WIFI" else 0.0

st.sidebar.markdown("---")
st.sidebar.subheader("最近 3 分钟点击流")
recent_intent = st.sidebar.selectbox("用户刚点击了什么？", [
    "无明显偏好", 
    "休闲/消除类 (欢乐消消消...)", 
    "重度游戏类 (王者荣耀, 原神...)", 
    "社交/视频类 (微信, 抖音, B站...)",
    "工具/出行类 (高德, 夸克, 清理...)"
])

# --- Main Logic ---

@st.cache_data(ttl=60)
def fetch_cloud_recommendations():
    """Fetch base recommendations from Cloud MMoE"""
    try:
        req_data = {
            "user_id": "OPPO_Demo_User",
            "request_id": "demo_req_1",
            "top_k": 25  # MMoE 精选出 25 个
        }
        res = requests.post(CLOUD_API_URL, json=req_data)
        res.raise_for_status()
        data = res.json()
        return data["recalled_app_names"], data["recommended_apps"], data["latency_ms"]
    except Exception as e:
        st.error(f"❌ 无法连接到云端 API。请确保在终端运行了 `python src/cloud_server.py`\n\nError: {e}")
        return [], [], 0

def run_edge_onnx(cloud_apps, battery, wifi, intent):
    """Run ONNX Re-ranker"""
    try:
        session = ort.InferenceSession(ONNX_MODEL_PATH, providers=['CPUExecutionProvider'])
        input_name = session.get_inputs()[0].name
        
        reranked_apps = []
        for app in cloud_apps:
            cloud_score = app["cloud_ctr_score"]
            cat = app.get("category", "")
            name = app["app_name"]
            
            # 判断是否为重度应用 (游戏、长视频)
            # 休闲游戏(Games类别中的消除、塔防等)不属于重度耗电应用
            is_heavy = 1.0 if cat in ["Gaming", "Adventure", "Video"] or name in ["王者荣耀", "原神", "和平精英", "三角洲行动", "爱奇艺", "腾讯视频"] else 0.0
            
            # 判断意图匹配
            recent_match = 0.0
            if "休闲" in intent and (cat in ["Games", "Strategy"] or "消" in name or name in ["愤怒的小鸟", "保卫萝卜", "部落冲突", "王国保卫战"]):
                recent_match = 1.0
            elif "重度游戏" in intent and (cat in ["Gaming", "Adventure"] or name in ["王者荣耀", "原神", "和平精英", "三角洲行动"]):
                recent_match = 1.0
            elif "社交/视频" in intent and cat in ["Social", "Social Media", "Video"]:
                recent_match = 1.0
            elif "工具/出行" in intent and cat in ["Utilities", "Tool", "Navigation", "Travel", "Search"]:
                recent_match = 1.0
                
            # 构造交叉特征: 只有当是重度应用，且环境恶劣时，惩罚项才会变大！
            heavy_battery_penalty = is_heavy * (1.0 - battery)
            heavy_wifi_penalty = is_heavy * (1.0 - wifi)
            light_battery_bonus = (1.0 - is_heavy) * (1.0 - battery)
            
            input_data = np.array([[cloud_score, recent_match, heavy_battery_penalty, heavy_wifi_penalty, light_battery_bonus]], dtype=np.float32)
            
            start_t = time.time()
            outputs = session.run(None, {input_name: input_data})
            inf_time = (time.time() - start_t) * 1000
            
            final_score = float(outputs[0][0][0])
            
            app_dict = dict(app)
            app_dict["edge_final_score"] = final_score
            app_dict["cloud_base_score"] = cloud_score
            app_dict["inf_time"] = inf_time
            app_dict["is_heavy"] = is_heavy
            # 记录原始云端排名 (索引 + 1)
            app_dict["original_cloud_rank"] = cloud_apps.index(app) + 1 
            reranked_apps.append(app_dict)
            
        reranked_apps.sort(key=lambda x: x["edge_final_score"], reverse=True)
        return reranked_apps
    except Exception as e:
        st.error(f"ONNX 模型加载失败: {e}")
        return cloud_apps

# Layout Columns
col1, col2, col3 = st.columns([1, 1, 1.2])

# 1. Fetch Cloud Data
recalled_names, cloud_apps, cloud_latency = fetch_cloud_recommendations()

if cloud_apps:
    with col1:
        st.subheader("🔍 双塔+Faiss 召回")
        st.caption("从 200 个物料库中极速召回 **Top-50**")
        with st.container(height=500):
            for i, name in enumerate(recalled_names):
                st.markdown(f"**{i+1}.** {name}")

    with col2:
        st.subheader("☁️ MMoE 精排下发")
        st.caption(f"网络延迟+推理耗时: **{cloud_latency} ms**")
        st.info("从左侧 50 个中精挑细选出 **Top-25** 下发给手机")
        
        with st.container(height=500):
            for i, app in enumerate(cloud_apps):
                st.markdown(f"""
                <div class="app-card">
                    <h5 style="margin:0;">#{i+1} {app['app_name']}</h5>
                    <p style="margin:0; font-size:0.8em;">云端预估分: <span class="score-badge">{app['cloud_ctr_score']:.4f}</span></p>
                </div>
                """, unsafe_allow_html=True)

    # 2. Run Edge Re-rank
    final_apps = run_edge_onnx(cloud_apps, battery_level, is_wifi, recent_intent)
    
    with col3:
        st.subheader("📱 ONNX 端侧最终展示")
        total_inf_time = sum(a.get("inf_time", 0) for a in final_apps)
        st.caption(f"提取本地隐私特征+推理耗时: **{total_inf_time:.2f} ms**")
        st.success("手机端重排中部的 25 个 App，最终渲染 **Top-5** 给用户")
        
        # 严格截断：最终只渲染 Top-5
        for i, app in enumerate(final_apps[:5]):
            delta = app['edge_final_score'] - app['cloud_base_score']
            delta_color = "delta-positive" if delta > 0 else "delta-negative"
            delta_symbol = "↑" if delta > 0 else "↓"
            
            # 排名变化提示
            orig_rank = app['original_cloud_rank']
            rank_change = orig_rank - (i+1)
            rank_text = f"持平"
            if rank_change > 0:
                rank_text = f"上升 {rank_change} 位 (原排第{orig_rank})"
            elif rank_change < 0:
                rank_text = f"下降 {abs(rank_change)} 位 (原排第{orig_rank})"
                
            # 重度/轻量标签视觉透出
            app_type_badge = "<span style='color:#e53935; border:1px solid #e53935; border-radius:3px; padding:0 3px; font-size:0.7em;'>[重度耗电]</span>" if app.get('is_heavy') == 1.0 else "<span style='color:#00a65a; border:1px solid #00a65a; border-radius:3px; padding:0 3px; font-size:0.7em;'>[轻量省电]</span>"
                
            st.markdown(f"""
            <div class="app-card" style="border-left: 4px solid {'#00a65a' if delta>0 else '#e53935'}; padding: 10px;">
                <h4 style="margin:0;">#{i+1} {app['app_name']} {app_type_badge}</h4>
                <p style="margin:0; font-size:0.8em; color:gray;">排名: {rank_text}</p>
                <p style="margin-top:5px; margin-bottom:0;">综合得分: <span class="score-badge">{app['edge_final_score']:.4f}</span> 
                   <span class="{delta_color}">({delta_symbol} {abs(delta):.4f})</span>
                </p>
            </div>
            """, unsafe_allow_html=True)
            
st.markdown("---")
st.markdown("**项目核心亮点**：通过将重排模型导出为 ONNX，在手机端利用 CPU 进行极速推理，既保护了用户隐私（电量、极短期行为不上云），又完美解决了云端推荐在极端场景下不准的问题。")