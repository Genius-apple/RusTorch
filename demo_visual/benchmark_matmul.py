
import torch
import time

def main():
    size = 1024
    print(f"Benchmarking PyTorch MatMul [{size}, {size}] ...")
    
    a = torch.randn(size, size)
    b = torch.randn(size, size)
    
    # Warmup
    _ = torch.matmul(a, b)
    
    start = time.time()
    for _ in range(10):
        _ = torch.matmul(a, b)
    end = time.time()
    
    print(f"PyTorch Average Time: {(end - start) / 10.0:.4f} s")

if __name__ == "__main__":
    main()
