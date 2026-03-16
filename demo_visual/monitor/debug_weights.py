
import json
import numpy as np

def check():
    with open('demo_visual/monitor/init_data.json') as f:
        init = json.load(f)
    
    with open('demo_visual/monitor/rust_log.jsonl') as f:
        rust = json.loads(f.readline())

    with open('demo_visual/monitor/torch_log.jsonl') as f:
        torch_log = json.loads(f.readline())

    # 1. Reconstruct grad_fc2_out manually from Rust activations
    fc2_out = np.array(rust['activations']['fc2_out']).reshape(32, 1) # [32, 1]
    y = np.array(init['Y']).reshape(32, 1) # [32, 1]
    
    # MSE Derivative: 2/N * (output - target)
    N = 32
    grad_fc2_out_manual = (2.0 / N) * (fc2_out - y)
    
    # 2. Reconstruct fc2_w_grad manually
    relu_out = np.array(rust['activations']['relu_out']).reshape(32, 20) # [32, 20]
    # dL/dW = X.T @ dL/dY
    # relu_out.T @ grad_fc2_out
    # [20, 32] @ [32, 1] -> [20, 1]
    fc2_w_grad_manual = relu_out.T @ grad_fc2_out_manual
    fc2_w_grad_manual = fc2_w_grad_manual.T # [1, 20] (PyTorch/Rust weights are [Out, In])
    
    # 3. Compare with Rust log
    fc1_w_grad_rust = np.array(rust['gradients']['fc1_w_grad']).reshape(20, 10)
    
    # 4. Compare with PyTorch log
    fc1_w_grad_torch = np.array(torch_log['gradients']['fc1_w_grad']).reshape(20, 10)
    diff_torch = np.max(np.abs(fc1_w_grad_rust - fc1_w_grad_torch))
    print(f"Rust vs Torch Diff (fc1_w_grad): {diff_torch}")
    
    if diff_torch > 1e-6:
        print("Torch Mismatch!")
        print("Rust[:5]: ", fc1_w_grad_rust.flatten()[:5])
        print("Torch[:5]:", fc1_w_grad_torch.flatten()[:5])

if __name__ == "__main__":
    check()
