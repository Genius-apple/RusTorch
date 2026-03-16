use rustorch_core::Tensor;
use rustorch_nn::{
    optim::{Adam, Optimizer},
    Linear, Module,
};
// use std::sync::Arc;

fn main() {
    println!("Running Numerical Verification...");

    // Setup
    // Input: [2, 3]
    let x_data = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6];
    let x = Tensor::new(&x_data, &[2, 3]);

    // Linear: 3 -> 2
    let linear = Linear::new(3, 2);

    // Manually set weights
    // Rustorch Linear weights are [Out, In] (same as PyTorch)
    // W: [[0.1, -0.1, 0.2], [-0.2, 0.3, 0.4]]
    let w_data = vec![0.1, -0.1, 0.2, -0.2, 0.3, 0.4];
    linear.weight.copy_from_slice(&w_data);

    // Bias: [0.01, -0.01]
    // Rustorch bias shape is [1, Out]
    let b_data = vec![0.01, -0.01];
    if let Some(bias) = &linear.bias {
        bias.copy_from_slice(&b_data);
    }

    // Optimizer
    let mut optimizer = Adam::new(linear.parameters(), 0.1);
    // Note: PyTorch default betas=(0.9, 0.999), eps=1e-8.
    // Need to ensure RusTorch defaults match or set them.
    // RusTorch Adam::new signature: (params, lr).
    // It uses default betas=0.9, 0.999, eps=1e-8 inside.

    // Forward
    optimizer.zero_grad();
    let y = linear.forward(&x);

    println!("Output: {:?}", y.data());

    // Loss (MSE)
    let target_data = vec![0.5, -0.5, 0.1, 0.2];
    let target = Tensor::new(&target_data, &[2, 2]);

    let diff = y.sub(&target);
    let sq_diff = diff.clone() * diff.clone();
    let numel = sq_diff.shape().iter().product::<usize>() as f32; // 4.0

    // Mean reduction manually (since we want to be sure about the graph)
    // loss = sum(sq_diff) / 4
    // RusTorch backward:
    // If we call .mean().backward(), it should work.
    // But let's replicate the manual backward step from demo to be safe,
    // or use mean() if implemented.
    // Let's use manual accumulation to match demo pattern first.

    let loss_val = sq_diff.data().iter().sum::<f32>() / numel;
    println!("Loss: {}", loss_val);

    // Backward
    // dLoss/dy = 2(y-t)/N * dy/d...
    // Actually, dLoss/dOutput = 2(y-t)/N = 0.5(y-t)
    // If we start backward from `sq_diff`, grad is 1/N.
    // d(mean)/d(sq_diff) = 1/N.

    let grad_loss = Tensor::full(sq_diff.shape(), 1.0 / numel);
    sq_diff.accumulate_grad(&grad_loss);
    sq_diff.backward_step();

    // Check Gradients
    if let Some(grad) = linear.weight.grad() {
        println!("Grad Weight: {:?}", grad.data());
    } else {
        println!("Grad Weight: None");
    }

    if let Some(bias) = &linear.bias {
        if let Some(grad) = bias.grad() {
            println!("Grad Bias: {:?}", grad.data());
        } else {
            println!("Grad Bias: None");
        }
    }

    // Step
    optimizer.step();

    // Check Updated Weights
    println!("Updated Weight: {:?}", linear.weight.data());
    if let Some(bias) = &linear.bias {
        println!("Updated Bias: {:?}", bias.data());
    }

    // Additional Activation Verification
    println!("\nVerifying Activations...");
    let mut t_in = Tensor::new(&[0.5, -0.5, 0.0], &[3]);
    t_in.set_requires_grad_mut(true);

    // Sigmoid
    let t_out = rustorch_core::ops::activations::sigmoid(&t_in);
    println!("Sigmoid({:?}) = {:?}", t_in.data(), t_out.data());

    // Grad check: dSigmoid(0) = 0.5 * 0.5 = 0.25
    let grad_out = Tensor::ones(&[3]);
    t_out.accumulate_grad(&grad_out);
    t_out.backward_step();

    if let Some(grad) = t_in.grad() {
        println!("Sigmoid Grad: {:?}", grad.data());
    }
}
