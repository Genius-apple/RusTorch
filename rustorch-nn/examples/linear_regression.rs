use rustorch_core::Tensor;
use rustorch_nn::{Linear, Module};
use rustorch_nn::optim::{Optimizer, SGD};

fn main() {
    // 1. Data
    // y = 2x + 1
    let x_data = vec![0.0, 1.0, 2.0, 3.0];
    let y_data = vec![1.0, 3.0, 5.0, 7.0];
    
    let x = Tensor::new(&x_data, &[4, 1]);
    let y = Tensor::new(&y_data, &[4, 1]);
    
    // 2. Model
    let model = Linear::new(1, 1);
    
    // 3. Optimizer
    let mut optimizer = SGD::new(model.parameters(), 0.01, 0.9);
    
    for epoch in 0..100 {
        optimizer.zero_grad();
        
        // Forward
        let output = model.forward(&x);
        
        // Loss (MSE) for printing: (output - y)^2 / N
        // We use sub but compute square manually
        let diff = output.sub(&y);
        
        {
            let diff_data = diff.data();
            let loss_val: f32 = diff_data.iter().map(|v| v * v).sum::<f32>() / 4.0;
            
            if epoch % 10 == 0 {
                println!("Epoch {}: Loss = {:.4}", epoch, loss_val);
            }
        }
        
        // Manual Gradient for MSE Loss
        // dL/dOutput = 2 * (Output - Y) / N
        // grad = 2/4 * diff
        let grad_scale = 0.5;
        let diff_data = diff.data();
        let grad_data: Vec<f32> = diff_data.iter().map(|v| v * grad_scale).collect();
        let grad_output = Tensor::new(&grad_data, output.shape());
        
        // Backward
        output.accumulate_grad(&grad_output);
        output.backward_step();
        
        // Optimize
        optimizer.step();
    }
    
    // Check result
    let final_out = model.forward(&x);
    println!("Final Output: {:?}", final_out);
    println!("Target: {:?}", y);
}
