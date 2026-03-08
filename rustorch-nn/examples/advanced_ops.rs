use std::sync::{Arc, Mutex};
use rustorch_core::Tensor;
use rustorch_nn::{Conv2d, MaxPool2d, BatchNorm2d, Linear, ReLU, Module};
use rustorch_nn::optim::{Optimizer, SGD};

fn main() {
    println!("--- Testing MaxPool2d and BatchNorm2d ---");
    
    // (N=1, C=1, H=4, W=4)
    let input = Tensor::new(&[
        1.0, 2.0, 3.0, 4.0,
        5.0, 6.0, 7.0, 8.0,
        9.0, 10.0, 11.0, 12.0,
        13.0, 14.0, 15.0, 16.0
    ], &[1, 1, 4, 4]).set_requires_grad(true);
    
    println!("Input: {:?}", input);
    
    // MaxPool
    let pool = MaxPool2d::new((2, 2), Some((2, 2)), (0, 0));
    let pooled = pool.forward(&input);
    println!("Pooled: {:?}", pooled);
    // Should be 2x2: [[6, 8], [14, 16]]
    
    // BatchNorm
    let bn = BatchNorm2d::new(1);
    let normalized = bn.forward(&pooled);
    println!("Normalized: {:?}", normalized);
    
    // Loss (sum)
    // let loss = normalized.sum();
    
    // Simulating loss.backward() from a scalar loss = sum(normalized)
    // dLoss/dx = 1.0 for all x
    let grad = Tensor::ones(normalized.shape());
    
    println!("Backward pass...");
    normalized.backward_with_grad(grad);
    
    println!("Input Grad: {:?}", input.grad().unwrap());
    println!("BN Weight Grad: {:?}", bn.weight.as_ref().unwrap().grad().unwrap());
    
    // Running stats
    println!("Running Mean: {:?}", bn.running_mean.lock().unwrap());
    println!("Running Var: {:?}", bn.running_var.lock().unwrap());
}
