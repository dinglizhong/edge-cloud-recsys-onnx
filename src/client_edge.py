import requests
import json
import uuid
import onnxruntime as ort
import numpy as np
import os

# Simulate the Edge Device (e.g., OPPO Phone via local laptop)

# 允许通过环境变量动态设置云端 API 地址
# 由于你使用的是 SSH 隧道转发，并且 AutoDL 工具要求端口不能同名
# 假设你将“代理到本地端口”设置为了 8888，那么这里就是 8888
CLOUD_API_URL = os.environ.get("CLOUD_API_URL", "http://127.0.0.1:16006/api/v1/recommend")
ONNX_MODEL_PATH = "models/edge_reranker.onnx"

def get_realtime_edge_features():
    """
    Simulate highly private and real-time features collected on the device.
    These features NEVER leave the phone.
    """
    return {
        "battery_level": 0.15, # 15% battery, user might prefer lightweight apps
        "is_wifi": 0.0,        # 0.0 for 4G, 1.0 for WIFI
        "last_3_mins_clicks": ["A001", "A011"], # Recently clicked casual games and lifestyle
    }

def edge_rerank_with_onnx(cloud_apps, edge_features):
    """
    Use the exported ONNX model to perform real-time re-ranking on the Edge CPU.
    """
    print(f"\n[Edge CPU] Extracting local features: {edge_features}")
    print(f"[Edge CPU] Loading ONNX model from {ONNX_MODEL_PATH} ...")
    
    # Initialize ONNX Runtime Session (CPU)
    session = ort.InferenceSession(ONNX_MODEL_PATH, providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name
    
    reranked_apps = []
    
    for app in cloud_apps:
        # Base score from Cloud
        cloud_score = app["cloud_ctr_score"]
        
        # Calculate a simple match score for simulation
        recent_match = 1.0 if ("消" in app["app_name"] or "休闲" in app.get("category", "")) else 0.0
        
        # Construct input vector for ONNX model
        # Shape: [batch_size=1, features=4] 
        # Features: [cloud_score, battery_level, is_wifi, recent_match]
        input_data = np.array([[
            cloud_score,
            edge_features["battery_level"],
            edge_features["is_wifi"],
            recent_match
        ]], dtype=np.float32)
        
        # Run ONNX Inference
        outputs = session.run(None, {input_name: input_data})
        final_score = outputs[0][0][0] # Extract the scalar value
        
        # Save results
        app_dict = dict(app)
        app_dict["edge_final_score"] = float(final_score)
        app_dict["cloud_base_score"] = cloud_score
        reranked_apps.append(app_dict)
        
    # Sort by Edge Final Score
    reranked_apps.sort(key=lambda x: x["edge_final_score"], reverse=True)
    return reranked_apps

def simulate_edge_request():
    """
    Main loop for the Edge device.
    """
    req_data = {
        "user_id": "OPPO_U_998877",
        "request_id": str(uuid.uuid4()),
        "top_k": 8
    }
    
    print(">>> [Edge] Sending request to Cloud Server...")
    try:
        response = requests.post(CLOUD_API_URL, json=req_data)
        response.raise_for_status()
        res_json = response.json()
        
        cloud_apps = res_json["recommended_apps"]
        print(f"<<< [Edge] Received {len(cloud_apps)} candidates from Cloud in {res_json['latency_ms']}ms.")
        
        # Print Cloud Top 3
        print("\n--- Cloud Top 3 (Before Edge Rerank) ---")
        for i, app in enumerate(cloud_apps[:3]):
            print(f"{i+1}. {app['app_name']} (Cloud CTR: {app['cloud_ctr_score']})")
            
        # Execute Edge Re-ranking
        edge_features = get_realtime_edge_features()
        final_apps = edge_rerank_with_onnx(cloud_apps, edge_features)
        
        # Print Edge Final Top 3
        print("\n--- Edge Final Top 3 (Displayed to User) ---")
        for i, app in enumerate(final_apps[:3]):
            print(f"{i+1}. {app['app_name']} (Final Score: {app['edge_final_score']} | Cloud Base: {app['cloud_base_score']})")
            if app['edge_final_score'] < app['cloud_base_score']:
                 print(f"   -> Score decreased due to local Edge rules (e.g. low battery/no WIFI)")
            elif app['edge_final_score'] > app['cloud_base_score']:
                 print(f"   -> Score increased due to real-time intent match")
                 
    except Exception as e:
        print(f"[Error] Failed to connect to Cloud API: {e}")
        print("Make sure 'python src/cloud_server.py' is running.")

if __name__ == "__main__":
    simulate_edge_request()
