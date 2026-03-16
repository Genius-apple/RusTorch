
import torch
import time

def main():
    m = 64
    k = 784
    n = 128
    print(f"Benchmarking PyTorch Small MatMul [{m}, {k}] x [{k}, {n}] ...")
    
    a = torch.ones(m, k)
    b = torch.ones(k, n)
    
    # Warmup
    _ = torch.matmul(a, b)
    
    start = time.time()
    iters = 1000
    for _ in range(iters):
        _ = torch.matmul(a, b)
    end = time.time()
    
    print(f"PyTorch Average Time: {(end - start) / iters:.6f} s")

if __name__ == "__main__":
    main()
