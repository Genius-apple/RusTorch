import torch
import torch.nn as nn
import torch.optim as optim
import time
import json
import sys

# Configuration
BATCH_SIZE = 1024
INPUT_SIZE = 784
OUTPUT_SIZE = 10
LEARNING_RATE = 0.001
EPOCHS = 50  # Short benchmark
STEPS_PER_EPOCH = 20


MASK64 = (1 << 64) - 1


def deterministic_unit(idx: int, seed: int) -> float:
    x = (idx + (seed * 0x9E3779B97F4A7C15)) & MASK64
    x ^= (x >> 30)
    x = (x * 0xBF58476D1CE4E5B9) & MASK64
    x ^= (x >> 27)
    x = (x * 0x94D049BB133111EB) & MASK64
    x ^= (x >> 31)
    return float(x & 0xFFFFFFFF) / float(0xFFFFFFFF)

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc = nn.Linear(INPUT_SIZE, OUTPUT_SIZE)

    def forward(self, x):
        return self.fc(x)

def run_benchmark():
    torch.manual_seed(42)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.backends.mkldnn.enabled = False
    # Use CUDA if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", file=sys.stderr)
    
    model = MLP().to(device)
    with torch.no_grad():
        model.fc.weight.zero_()
        model.fc.bias.zero_()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()

    # Synthetic Data
    # Use Uniform[-0.5, 0.5] to match Rust implementation
    # Pre-generate data on device
    data_arr = [deterministic_unit(i, 42) - 0.5 for i in range(BATCH_SIZE * INPUT_SIZE)]
    w_arr = [deterministic_unit(i, 20260315) * 0.2 - 0.1 for i in range(INPUT_SIZE * OUTPUT_SIZE)]
    b_arr = [deterministic_unit(i, 20260401) * 0.2 - 0.1 for i in range(OUTPUT_SIZE)]
    data = torch.tensor(data_arr, dtype=torch.float32, device=device).view(BATCH_SIZE, INPUT_SIZE)
    teacher_w = torch.tensor(w_arr, dtype=torch.float32, device=device).view(INPUT_SIZE, OUTPUT_SIZE)
    teacher_b = torch.tensor(b_arr, dtype=torch.float32, device=device)
    target = data @ teacher_w + teacher_b
    baseline_mse = float((target * target).mean().item())
    with torch.no_grad():
        baseline_acc = float((torch.abs(target) < 0.1).float().mean().item())

    # Run warmup
    for _ in range(5):
        _ = model(data)
    
    print(json.dumps({"type": "init", "framework": "PyTorch", "device": str(device), "baseline_mse": baseline_mse}), flush=True)
    print(
        json.dumps(
            {
                "type": "update",
                "framework": "PyTorch",
                "epoch": 0,
                "loss": baseline_mse,
                "speed": 0.0,
                "accuracy": baseline_acc,
                "grad_norm": 0.0,
                "timestamp": time.time(),
            }
        ),
        flush=True,
    )

    start_time = time.time()
    total_samples = 0
    last_loss = None
    last_acc = None

    for epoch in range(1, EPOCHS + 1):
        epoch_start = time.time()
        epoch_loss = 0.0
        
        for step in range(STEPS_PER_EPOCH):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            total_samples += BATCH_SIZE

        epoch_end = time.time()
        duration = epoch_end - epoch_start
        speed = (BATCH_SIZE * STEPS_PER_EPOCH) / duration
        
        # Calculate Gradient Norm
        total_norm_sq = 0.0
        for p in model.parameters():
            if p.grad is not None:
                total_norm_sq += p.grad.norm(2).item() ** 2
        grad_norm = total_norm_sq ** 0.5
        
        # Calculate Accuracy (Regression tolerance < 0.1)
        # abs(pred - target) < 0.1
        with torch.no_grad():
            output = model(data)
            diff = torch.abs(output - target)
            correct = (diff < 0.1).sum().item()
            accuracy = correct / (BATCH_SIZE * OUTPUT_SIZE)

        # Emit metrics
        metrics = {
            "type": "update",
            "framework": "PyTorch",
            "epoch": epoch,
            "loss": epoch_loss / STEPS_PER_EPOCH,
            "speed": speed, # samples/sec
            "accuracy": accuracy,
            "grad_norm": grad_norm,
            "timestamp": time.time()
        }
        last_loss = metrics["loss"]
        last_acc = metrics["accuracy"]
        print(json.dumps(metrics), flush=True)
        # Small sleep to simulate real-time stream if too fast, but we want raw speed.
        # time.sleep(0.01) 

    total_time = time.time() - start_time
    print(json.dumps({
        "type": "finish", 
        "framework": "PyTorch", 
        "total_time": total_time,
        "avg_speed": total_samples / total_time,
        "final_loss": last_loss if last_loss is not None else 0.0,
        "final_accuracy": last_acc if last_acc is not None else 0.0
    }), flush=True)

if __name__ == "__main__":
    run_benchmark()
