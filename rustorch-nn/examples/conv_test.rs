use rustorch_core::Tensor;
use rustorch_nn::{Conv2d, Module};
use rustorch_nn::optim::{Optimizer, SGD};

fn main() {
    // 1. Data
    // (N=1, C=1, H=4, W=4)
    let n = 1;
    let c_in = 1;
    let h = 4;
    let w = 4;
    
    let mut x_data = vec![0.0; n * c_in * h * w];
    // Fill with some data
    for i in 0..x_data.len() {
        x_data[i] = i as f32;
    }
    
    let x = Tensor::new(&x_data, &[n, c_in, h, w]);
    
    // 2. Model
    // 1 input channel, 1 output channel, 3x3 kernel
    let conv = Conv2d::new(c_in, 1, (3, 3), (1, 1), (0, 0));
    
    // 3. Optimizer
    let mut optimizer = SGD::new(conv.parameters(), 0.01, 0.0);
    
    // 4. Forward
    println!("Forward pass...");
    let output = conv.forward(&x);
    println!("Output shape: {:?}", output.shape());
    println!("Output data: {:?}", &output.data()[..]);
    
    // 5. Backward
    println!("Backward pass...");
    let grad = Tensor::ones(output.shape());
    output.backward_with_grad(grad);
    
    // Check gradients
    let w_grad = conv.weight.grad().unwrap();
    println!("Weight Grad shape: {:?}", w_grad.shape());
    
    // 6. Step
    optimizer.step();
    println!("Step done.");
}
