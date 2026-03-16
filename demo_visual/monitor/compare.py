
import json
import numpy as np
import sys
import matplotlib.pyplot as plt

def load_log(filename):
    data = []
    with open(filename, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def compare_tensors(name, t1, t2, threshold=1e-5):
    a1 = np.array(t1).flatten()
    a2 = np.array(t2).flatten()
    
    if a1.shape != a2.shape:
        return False, f"Shape mismatch after flatten: {a1.shape} vs {a2.shape}", 0.0

    diff = np.abs(a1 - a2)
    max_diff = np.max(diff)
    
    if max_diff > threshold:
         print(f"DEBUG: {name} Mismatch. a1[:5]={a1[:5]}, a2[:5]={a2[:5]}")
    
    # Relative error handling for small numbers
    denominator = np.maximum(np.abs(a1), np.abs(a2))
    # Avoid division by zero
    mask = denominator > 1e-7
    rel_diff = np.zeros_like(diff)
    rel_diff[mask] = diff[mask] / denominator[mask]
    max_rel_diff = np.max(rel_diff)

    passed = max_diff < threshold or max_rel_diff < threshold
    
    msg = f"Max Diff: {max_diff:.2e}, Max Rel Diff: {max_rel_diff:.2e}"
    return passed, msg, max_diff

def calculate_smoothness(loss_history):
    loss = np.array(loss_history)
    diffs = np.diff(loss)
    variance = np.var(diffs)
    return variance

def main():
    print("Loading logs...")
    try:
        rust_data = load_log('demo_visual/monitor/rust_log.jsonl')
        torch_data = load_log('demo_visual/monitor/torch_log.jsonl')
    except FileNotFoundError as e:
        print(f"Error loading logs: {e}")
        sys.exit(1)

    print(f"Loaded {len(rust_data)} Rust steps and {len(torch_data)} PyTorch steps.")
    
    min_steps = min(len(rust_data), len(torch_data))
    
    first_failure = None
    
    rust_losses = []
    torch_losses = []
    
    for i in range(min_steps):
        r = rust_data[i]
        t = torch_data[i]
        
        rust_losses.append(r['loss'])
        torch_losses.append(t['loss'])
        
        print(f"--- Epoch {i} ---")
        
        # Compare Loss
        passed, msg, diff = compare_tensors("loss", r['loss'], t['loss'])
        print(f"Loss: {msg}")
        if not passed and first_failure is None:
            first_failure = (i, "loss")
            
        # Compare Activations
        for key in r['activations']:
            passed, msg, diff = compare_tensors(f"act_{key}", r['activations'][key], t['activations'][key])
            if not passed:
                print(f"  Activation {key}: FAILED. {msg}")
                if first_failure is None: first_failure = (i, f"act_{key}")
            # else:
            #     print(f"  Activation {key}: OK. {msg}")

        # Compare Gradients
        for key in r['gradients']:
            passed, msg, diff = compare_tensors(f"grad_{key}", r['gradients'][key], t['gradients'][key])
            if not passed:
                print(f"  Gradient {key}: FAILED. {msg}")
                if first_failure is None: first_failure = (i, f"grad_{key}")
            # else:
            #     print(f"  Gradient {key}: OK. {msg}")

        # Compare Weights
        for key in r['weights']:
            passed, msg, diff = compare_tensors(f"weight_{key}", r['weights'][key], t['weights'][key])
            if not passed:
                print(f"  Weight {key}: FAILED. {msg}")
                if first_failure is None: first_failure = (i, f"weight_{key}")

        if first_failure:
            print(f"\nFirst divergence detected at Epoch {first_failure[0]} in {first_failure[1]}")
            # Continue to analyze smoothness even if divergent
            # break 

    # Smoothness Analysis
    rust_smoothness = calculate_smoothness(rust_losses)
    torch_smoothness = calculate_smoothness(torch_losses)
    
    print("\n--- Smoothness Analysis ---")
    print(f"Rust Loss Variance of Diff: {rust_smoothness:.2e}")
    print(f"Torch Loss Variance of Diff: {torch_smoothness:.2e}")
    
    if torch_smoothness > 0:
        improvement = (torch_smoothness - rust_smoothness) / torch_smoothness * 100
        print(f"Improvement: {improvement:.2f}%")
        if improvement >= 20:
            print("SUCCESS: Smoothness target met.")
        else:
            print("FAILURE: Smoothness target not met.")
    else:
        print("Torch smoothness is 0 (constant loss?), cannot calculate improvement.")

if __name__ == "__main__":
    main()
