import torch
import torch.nn as nn
import json
import numpy as np

def run():
    torch.manual_seed(42)
    np.random.seed(42)
    
    BATCH_SIZE = 4
    INPUT_SIZE = 4
    OUTPUT_SIZE = 2
    HIDDEN_SIZE = 4
    
    # 1. Init Data
    data = torch.randn(BATCH_SIZE, INPUT_SIZE)
    target = torch.randn(BATCH_SIZE, OUTPUT_SIZE)
    
    # 2. Init Model
    model = nn.Sequential(
        nn.Linear(INPUT_SIZE, HIDDEN_SIZE),
        nn.Sigmoid(),
        nn.Linear(HIDDEN_SIZE, OUTPUT_SIZE)
    )
    
    # Manually init weights for deterministic float values
    # FC1
    fc1_w = torch.tensor([[0.1, -0.2, 0.3, -0.4], 
                          [-0.1, 0.2, -0.3, 0.4],
                          [0.5, -0.5, 0.5, -0.5],
                          [-0.5, 0.5, -0.5, 0.5]]) # [Out, In]
    fc1_b = torch.tensor([0.01, -0.01, 0.02, -0.02])
    
    model[0].weight.data.copy_(fc1_w)
    model[0].bias.data.copy_(fc1_b)
    
    # FC2
    fc2_w = torch.tensor([[0.1, 0.2, 0.3, 0.4],
                          [-0.4, -0.3, -0.2, -0.1]])
    fc2_b = torch.tensor([0.1, -0.1])
    
    model[2].weight.data.copy_(fc2_w)
    model[2].bias.data.copy_(fc2_b)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1, betas=(0.9, 0.999), eps=1e-8)
    criterion = nn.MSELoss()
    
    # 3. Step 0 (Forward + Backward + Update)
    optimizer.zero_grad()
    output = model(data)
    loss = criterion(output, target)
    loss.backward()
    
    # Capture Gradients
    g_fc1_w = model[0].weight.grad.clone()
    g_fc1_b = model[0].bias.grad.clone()
    g_fc2_w = model[2].weight.grad.clone()
    g_fc2_b = model[2].bias.grad.clone()
    
    optimizer.step()
    
    # Capture Updated Weights
    u_fc1_w = model[0].weight.data.clone()
    u_fc1_b = model[0].bias.data.clone()
    u_fc2_w = model[2].weight.data.clone()
    u_fc2_b = model[2].bias.data.clone()
    
    dump = {
        "input": data.tolist(),
        "target": target.tolist(),
        "init": {
            "fc1_w": fc1_w.tolist(),
            "fc1_b": fc1_b.tolist(),
            "fc2_w": fc2_w.tolist(),
            "fc2_b": fc2_b.tolist()
        },
        "step0": {
            "output": output.tolist(),
            "loss": loss.item(),
            "grad": {
                "fc1_w": g_fc1_w.tolist(),
                "fc1_b": g_fc1_b.tolist(),
                "fc2_w": g_fc2_w.tolist(),
                "fc2_b": g_fc2_b.tolist()
            },
            "updated": {
                "fc1_w": u_fc1_w.tolist(),
                "fc1_b": u_fc1_b.tolist(),
                "fc2_w": u_fc2_w.tolist(),
                "fc2_b": u_fc2_b.tolist()
            }
        }
    }
    
    with open("debug_golden.json", "w") as f:
        json.dump(dump, f, indent=2)
        
    print("Golden data generated: debug_golden.json")

if __name__ == "__main__":
    run()
