import torch
import torch.nn as nn
import onnx
import onnxruntime as ort
import numpy as np

class EdgeReRanker(nn.Module):
    """
    A highly lightweight model intended to run on edge devices (smartphones).
    Input Features (Dim=5):
        0. cloud_score: 云端基础分
        1. recent_match: 实时意图匹配度 (1.0 强匹配, 0.0 不匹配)
        2. heavy_battery_penalty: 重度应用低电量交叉惩罚 = is_heavy * (1.0 - battery)
        3. heavy_wifi_penalty: 重度应用差网络交叉惩罚 = is_heavy * (1.0 - is_wifi)
        4. light_battery_bonus: 轻量应用低电量交叉奖励 = (1.0 - is_heavy) * (1.0 - battery)
    Output:
        - final_score: 1 dim (Sigmoid 输出 0~1 的最终得分)
    """
    def __init__(self):
        super(EdgeReRanker, self).__init__()
        # Total input dim = 5
        self.fc = nn.Linear(5, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.fc(x)
        return self.sigmoid(out)

def export_to_onnx(model_path="models/edge_reranker.onnx"):
    import os
    os.makedirs("models", exist_ok=True)
    
    model = EdgeReRanker()
    
    # 2. 手动注入强逻辑权重，让交叉特征产生巨大的影响！
    with torch.no_grad():
        # w0 (cloud_score): +3.0 (保留云端排序的底色)
        # w1 (recent_match): +8.0 (一旦命中用户刚点的意图，瞬间霸榜！)
        # w2 (heavy_battery_penalty): -12.0 (如果是大游戏且快没电了，直接扣成负分滚出列表！)
        # w3 (heavy_wifi_penalty): -10.0 (如果是大游戏且没WIFI，同样强力扣分)
        # w4 (light_battery_bonus): +10.0 (如果没电了，轻量级应用疯狂加分，强行逆袭)
        model.fc.weight.copy_(torch.tensor([[3.0, 8.0, -12.0, -10.0, 10.0]]))
        model.fc.bias.copy_(torch.tensor([-3.0])) # 整体压低基础分，凸显匹配项
        
    model.eval()
    
    # 3. Create a dummy input
    dummy_input = torch.randn(1, 5, requires_grad=True)
    
    # 4. Export the model
    print(f"Exporting PyTorch model to ONNX format: {model_path} ...")
    torch.onnx.export(
        model,               
        dummy_input,         
        model_path,          
        export_params=True,  
        opset_version=17,    
        do_constant_folding=True,  
        input_names = ['input_features'],   
        output_names = ['final_score'], 
        dynamic_axes={'input_features' : {0 : 'batch_size'},    
                      'final_score' : {0 : 'batch_size'}}
    )
    print("Export complete!")

if __name__ == "__main__":
    export_to_onnx()
