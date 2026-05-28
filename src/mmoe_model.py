import torch
import torch.nn as nn
import torch.nn.functional as F

class MMoE(nn.Module):
    """
    Multi-gate Mixture-of-Experts (MMoE) model for CTR and CVR prediction.
    Suitable for deployment as a lightweight cloud-side model.
    """
    def __init__(self, feature_dim, num_experts=3, expert_hidden_units=64, num_tasks=2, task_hidden_units=32):
        super(MMoE, self).__init__()
        
        self.num_experts = num_experts
        self.num_tasks = num_tasks
        
        # 1. Experts: Shared across all tasks
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(feature_dim, expert_hidden_units),
                nn.ReLU(),
                nn.Linear(expert_hidden_units, expert_hidden_units),
                nn.ReLU()
            ) for _ in range(self.num_experts)
        ])
        
        # 2. Gates: One gate per task to assign weights to experts
        self.gates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(feature_dim, num_experts),
                nn.Softmax(dim=-1)
            ) for _ in range(self.num_tasks)
        ])
        
        # 3. Task Towers: Specific layers for each task (e.g., CTR, CVR)
        self.task_towers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(expert_hidden_units, task_hidden_units),
                nn.ReLU(),
                nn.Linear(task_hidden_units, 1),
                nn.Sigmoid() # Output probability [0, 1]
            ) for _ in range(self.num_tasks)
        ])

    def forward(self, x):
        """
        x: [batch_size, feature_dim]
        """
        # Step 1: Compute expert outputs
        # [batch_size, num_experts, expert_hidden_units]
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)
        
        task_outputs = []
        # Step 2: Compute gate weights and final prediction for each task
        for i in range(self.num_tasks):
            # gate_weights: [batch_size, num_experts]
            gate_weights = self.gates[i](x)
            
            # Reshape for broadcasting: [batch_size, num_experts, 1]
            gate_weights = gate_weights.unsqueeze(-1)
            
            # Weighted sum of experts: [batch_size, expert_hidden_units]
            tower_input = torch.sum(expert_outputs * gate_weights, dim=1)
            
            # Task prediction: [batch_size, 1]
            output = self.task_towers[i](tower_input)
            task_outputs.append(output)
            
        # Returns [CTR_pred, CVR_pred]
        return task_outputs

# Quick test script
if __name__ == "__main__":
    # Mock some data (batch_size=4, feature_dim=16)
    # The 16 dims can represent [User_Embedding, LLM_App_Embedding, Context_Features]
    mock_input = torch.randn(4, 16)
    
    # Initialize lightweight model
    model = MMoE(feature_dim=16, num_experts=3, expert_hidden_units=32)
    
    print("Forward Pass Test:")
    outputs = model(mock_input)
    print(f"CTR Predictions: \n{outputs[0].detach().numpy()}")
    print(f"CVR Predictions: \n{outputs[1].detach().numpy()}")
    
    # Print model size
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params}")
    print("This is a highly lightweight model, very cheap to fine-tune on AutoDL!")
