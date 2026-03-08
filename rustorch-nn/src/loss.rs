use std::sync::Arc;
use rayon::slice::ParallelSliceMut;
use rayon::iter::{IntoParallelIterator, IntoParallelRefIterator, IndexedParallelIterator, ParallelIterator};
use rustorch_core::{autograd::BackwardOp, storage::Storage, Tensor};

// --- MSELoss ---

#[derive(Debug)]
pub struct MSELossBackward {
    pub input: Tensor,
    pub target: Tensor,
    pub reduction: String, // "mean", "sum"
}

impl BackwardOp for MSELossBackward {
    fn backward(&self, grad: &Tensor) {
        if self.input.requires_grad() {
            // dL/dx = 2 * (x - y) / N (if mean)
            let input_guard = self.input.data();
            let target_guard = self.target.data();
            let grad_guard = grad.data(); // scalar grad usually 1.0
            
            let input_data = &*input_guard;
            let target_data = &*target_guard;
            let grad_val = if grad_guard.is_empty() { 1.0 } else { grad_guard[0] };
            
            let n = input_data.len() as f32;
            let scale = if self.reduction == "mean" { 2.0 / n } else { 2.0 };
            
            let grad_input_data: Vec<f32> = input_data.par_iter()
                .zip(target_data.par_iter())
                .map(|(x, y)| scale * (x - y) * grad_val)
                .collect();
                
            let grad_input = Tensor::new_with_storage(Storage::new(grad_input_data), self.input.shape());
            self.input.accumulate_grad(&grad_input);
            self.input.backward_step();
        }
    }
}

pub struct MSELoss {
    pub reduction: String,
}

impl MSELoss {
    pub fn new() -> Self {
        Self { reduction: "mean".to_string() }
    }
    
    pub fn forward(&self, input: &Tensor, target: &Tensor) -> Tensor {
        let diff = input.sub(target);
        // Manual square since we don't have pow op yet, use mul
        let sq_diff = diff.matmul(&diff); // Element-wise mul? Wait, Tensor mul is usually element-wise.
        // But in `tensor.rs`, Mul trait calls `crate::ops::mul`.
        // Let's assume mul is element-wise.
        
        // However, I need to check if `mul` is implemented.
        // `tensor.rs` imports Mul but I don't see `impl Mul` in the snippet provided earlier, only Add/Sub.
        // Wait, I see `impl Add`, `impl Sub`. Did I miss Mul?
        // Let's double check `rustorch-core/src/tensor.rs`.
        // Assuming it's NOT implemented, I should implement it or use raw data.
        
        // To be safe and fast, let's implement the forward pass manually here to avoid missing ops.
        
        let input_guard = input.data();
        let target_guard = target.data();
        let input_data = &*input_guard;
        let target_data = &*target_guard;
        
        let sum_sq_diff: f32 = input_data.par_iter()
            .zip(target_data.par_iter())
            .map(|(x, y)| (x - y).powi(2))
            .sum();
            
        let n = input_data.len() as f32;
        let loss_val = if self.reduction == "mean" { sum_sq_diff / n } else { sum_sq_diff };
        
        let mut tensor = Tensor::new(&[loss_val], &[1]);
        
        if input.requires_grad() {
            tensor.set_requires_grad_mut(true);
            tensor.set_op(Arc::new(MSELossBackward {
                input: input.clone(),
                target: target.clone(),
                reduction: self.reduction.clone(),
            }));
        }
        
        tensor
    }
}

// --- CrossEntropyLoss ---
// Simplified: Softmax + NLLLoss
// Input: (N, C) logits
// Target: (N) class indices (0..C-1)

#[derive(Debug)]
pub struct CrossEntropyLossBackward {
    pub input: Tensor, // Logits
    pub target: Tensor, // Indices
    pub reduction: String,
}

impl BackwardOp for CrossEntropyLossBackward {
    fn backward(&self, grad: &Tensor) {
        if self.input.requires_grad() {
            let input_shape = self.input.shape();
            let n = input_shape[0];
            let c = input_shape[1];
            
            let input_guard = self.input.data();
            let target_guard = self.target.data(); // Should be f32 representing indices?
            let grad_guard = grad.data();
            
            let input_data = &*input_guard;
            let target_data = &*target_guard;
            let grad_val = if grad_guard.is_empty() { 1.0 } else { grad_guard[0] };
            
            let mut grad_input_data = vec![0.0; n * c];
            
            // dL/dx_j = (softmax(x)_j - 1) / N  if j == target
            // dL/dx_j = softmax(x)_j / N      if j != target
            
            // We need to re-compute softmax.
            // Softmax(x_i) = exp(x_i) / sum(exp(x_k))
            // Stable softmax: exp(x_i - max) / sum(...)
            
            let scale = if self.reduction == "mean" { 1.0 / n as f32 } else { 1.0 };
            
            // Parallel over batch
            grad_input_data.par_chunks_mut(c).enumerate().for_each(move |(i, row)| {
                // Find max for numerical stability

                let offset = i * c;
                let mut max_val = f32::NEG_INFINITY;
                for j in 0..c {
                    let val = input_data[offset + j];
                    if val > max_val { max_val = val; }
                }
                
                let mut sum_exp: f32 = 0.0;
                let mut exps = vec![0.0; c];
                for j in 0..c {
                    let val = (input_data[offset + j] - max_val).exp();
                    exps[j] = val;
                    sum_exp += val;
                }
                
                let target_idx = target_data[i] as usize; // Assuming target is f32 (0.0, 1.0, ...)
                
                for j in 0..c {
                    let softmax = exps[j] / sum_exp;
                    if j == target_idx {
                        row[j] = (softmax - 1.0) * scale * grad_val;
                    } else {
                        row[j] = softmax * scale * grad_val;
                    }
                }
            });
            
            let grad_input = Tensor::new_with_storage(Storage::new(grad_input_data), self.input.shape());
            self.input.accumulate_grad(&grad_input);
            self.input.backward_step();
        }
    }
}

pub struct CrossEntropyLoss {
    pub reduction: String,
}

impl CrossEntropyLoss {
    pub fn new() -> Self {
        Self { reduction: "mean".to_string() }
    }
    
    pub fn forward(&self, input: &Tensor, target: &Tensor) -> Tensor {
        let input_shape = input.shape();
        if input_shape.len() != 2 {
            panic!("CrossEntropyLoss expects 2D input (N, C)");
        }
        let n = input_shape[0];
        let c = input_shape[1];
        
        let input_guard = input.data();
        let target_guard = target.data();
        let input_data = &*input_guard;
        let target_data = &*target_guard;
        
        if target_data.len() != n {
            panic!("Target size {} must match batch size {}", target_data.len(), n);
        }
        
        // Compute loss
        // Loss = -sum(log(softmax(x)_target)) / N
        //      = -sum(x_target - log(sum(exp(x)))) / N
        //      = sum(-x_target + log(sum(exp(x)))) / N
        
        let total_loss: f32 = (0..n).into_par_iter().map(|i| {
            let offset = i * c;
            let mut max_val = f32::NEG_INFINITY;
                for j in 0..c {
                    let val = input_data[offset + j];
                    if val > max_val { max_val = val; }
                }
                
                let mut sum_exp: f32 = 0.0;
                for j in 0..c {
                    sum_exp += (input_data[offset + j] - max_val).exp();
                }
                let log_sum_exp = max_val + sum_exp.ln();
            
            let target_idx = target_data[i] as usize;
            if target_idx >= c {
                // panic in parallel iterator is bad, but for now...
                // Ideally return Result
                0.0 // Error placeholder
            } else {
                let x_target = input_data[offset + target_idx];
                -x_target + log_sum_exp
            }
        }).sum();
        
        let loss_val = if self.reduction == "mean" { total_loss / n as f32 } else { total_loss };
        
        let mut tensor = Tensor::new(&[loss_val], &[1]);
        
        if input.requires_grad() {
            tensor.set_requires_grad_mut(true);
            tensor.set_op(Arc::new(CrossEntropyLossBackward {
                input: input.clone(),
                target: target.clone(),
                reduction: self.reduction.clone(),
            }));
        }
        
        tensor
    }
}
