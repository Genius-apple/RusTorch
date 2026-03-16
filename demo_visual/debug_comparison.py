import torch
import torch.nn as nn
import json
import copy

def run():
    torch.manual_seed(42)
    
    # Setup
    # Input: [2, 3]
    x = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], requires_grad=False)
    
    # Linear: 3 -> 2
    linear = nn.Linear(3, 2)
    # Manually set weights for determinism
    # Weights in PyTorch are [Out, In]
    with torch.no_grad():
        linear.weight.copy_(torch.tensor([[0.1, -0.1, 0.2], [-0.2, 0.3, 0.4]]))
        linear.bias.copy_(torch.tensor([0.01, -0.01]))
    
    init_weight = linear.weight.clone().tolist()
    init_bias = linear.bias.clone().tolist()

    # Optimizer
    optimizer = torch.optim.Adam(linear.parameters(), lr=0.1, betas=(0.9, 0.999), eps=1e-8)
    
    # Forward
    optimizer.zero_grad()
    y = linear(x)
    
    # Loss (Mean Squared Error against target)
    target = torch.tensor([[0.5, -0.5], [0.1, 0.2]])
    loss = nn.MSELoss()(y, target)
    
    # Backward
    loss.backward()
    
    grad_weight = linear.weight.grad.clone().tolist()
    grad_bias = linear.bias.grad.clone().tolist()

    # Step
    optimizer.step()
    
    # Collect Data
    data = {
        "input": x.tolist(),
        "target": target.tolist(),
        "init_weight": init_weight,
        "init_bias": init_bias,
        "output": y.tolist(),
        "loss": loss.item(),
        "grad_weight": grad_weight,
        "grad_bias": grad_bias,
        "updated_weight": linear.weight.tolist(),
        "updated_bias": linear.bias.tolist(),
    }
    
    print(json.dumps(data, indent=2))

if __name__ == "__main__":
    run()
