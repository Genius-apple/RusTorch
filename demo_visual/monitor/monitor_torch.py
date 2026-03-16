
import torch
import torch.nn as nn
import torch.optim as optim
import json
import os

# Configuration
CONFIG = {
    "input_size": 10,
    "hidden_size": 20,
    "output_size": 1,
    "batch_size": 32,
    "learning_rate": 0.01,
    "epochs": 100,
    "seed": 42
}

class Monitor:
    def __init__(self, filename):
        self.filename = filename
        if os.path.exists(filename):
            os.remove(filename)
        self.file = open(filename, 'w')

    def log(self, step, data):
        entry = {"step": step, **data}
        self.file.write(json.dumps(entry) + '\n')
        self.file.flush()

    def close(self):
        self.file.close()

def run():
    torch.manual_seed(CONFIG["seed"])
    
    # 1. Model Definition (Linear -> ReLU -> Linear)
    model = nn.Sequential(
        nn.Linear(CONFIG["input_size"], CONFIG["hidden_size"]),
        nn.ReLU(),
        nn.Linear(CONFIG["hidden_size"], CONFIG["output_size"])
    )
    
    # 2. Data Generation
    X = torch.randn(CONFIG["batch_size"], CONFIG["input_size"])
    Y = torch.randn(CONFIG["batch_size"], CONFIG["output_size"])
    
    # 3. Save Initialization Data for Rust
    init_data = {
        "config": CONFIG,
        "X": X.tolist(),
        "Y": Y.tolist(),
        "fc1_weight": model[0].weight.tolist(),
        "fc1_bias": model[0].bias.tolist(),
        "fc2_weight": model[2].weight.tolist(),
        "fc2_bias": model[2].bias.tolist()
    }
    
    with open("demo_visual/monitor/init_data.json", "w") as f:
        json.dump(init_data, f, indent=2)
    
    print("Saved initialization data to init_data.json")

    # 4. Training Loop
    optimizer = optim.SGD(model.parameters(), lr=CONFIG["learning_rate"])
    criterion = nn.MSELoss()
    
    monitor = Monitor("demo_visual/monitor/torch_log.jsonl")
    
    # Hooks to capture activations and gradients
    activations = {}
    gradients = {}

    def get_activation(name):
        def hook(model, input, output):
            activations[name] = output.detach().tolist()
        return hook

    def get_gradient(name):
        def hook(grad):
            gradients[name] = grad.tolist()
        return hook

    model[0].register_forward_hook(get_activation('fc1_out'))
    model[1].register_forward_hook(get_activation('relu_out'))
    model[2].register_forward_hook(get_activation('fc2_out'))

    # Gradient hooks on parameters
    # Note: PyTorch accumulates gradients in .grad, but we can hook the tensor
    
    print("Starting PyTorch training...")
    
    for epoch in range(CONFIG["epochs"]):
        optimizer.zero_grad()
        
        # Forward
        output = model(X)
        loss = criterion(output, Y)
        
        # Backward
        loss.backward()
        
        # Capture gradients before step
        grads = {
            "fc1_w_grad": model[0].weight.grad.tolist(),
            "fc1_b_grad": model[0].bias.grad.tolist(),
            "fc2_w_grad": model[2].weight.grad.tolist(),
            "fc2_b_grad": model[2].bias.grad.tolist(),
        }
        
        # Capture weights before update
        weights_before = {
            "fc1_w": model[0].weight.tolist(),
            "fc1_b": model[0].bias.tolist(),
            "fc2_w": model[2].weight.tolist(),
            "fc2_b": model[2].bias.tolist(),
        }

        optimizer.step()
        
        # Log everything
        log_data = {
            "epoch": epoch,
            "loss": loss.item(),
            "activations": activations.copy(),
            "gradients": grads,
            "weights": weights_before # Log weights used for this step (before update)
        }
        monitor.log(epoch, log_data)
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.6f}")

    monitor.close()
    print("PyTorch training complete. Log saved to torch_log.jsonl")

if __name__ == "__main__":
    run()
