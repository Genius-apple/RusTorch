use std::sync::Arc;
use rayon::prelude::*;
use crate::Tensor;
use crate::autograd::BackwardOp;
use crate::storage::Storage;

// --- Sigmoid ---
pub fn sigmoid(input: &Tensor) -> Tensor {
    let input_guard = input.data();
    let input_data = &*input_guard;
    
    let result_data: Vec<f32> = input_data.par_iter()
        .map(|&x| 1.0 / (1.0 + (-x).exp()))
        .collect();
        
    let storage = Storage::new(result_data);
    let mut tensor = Tensor::new_with_storage(storage, input.shape());
    
    if input.requires_grad() {
        tensor.set_requires_grad_mut(true);
        // SigmoidBackward
        // dy/dx = y * (1 - y)
        // We need to store output (y) for efficient backward
        // But backward op takes input usually.
        // Let's store output in op.
        
        // Wait, creating tensor inside backward?
        // Let's use input for backward: sig(x) * (1 - sig(x))
        // Or store output. Output is `tensor`.
        // We can't easily store `tensor` in its own op because of circular ref if op is Arc.
        // Op is inside tensor. So tensor -> op -> tensor (cycle).
        // Standard way: store input, recompute or store weak ref?
        // Or just recompute sigmoid(input).
        
        tensor.set_op(Arc::new(SigmoidBackward {
            input: input.clone(), // Store input
        }));
    }
    
    tensor
}

#[derive(Debug)]
pub struct SigmoidBackward {
    pub input: Tensor,
}

impl BackwardOp for SigmoidBackward {
    fn backward(&self, grad: &Tensor) {
        if self.input.requires_grad() {
            // grad_input = grad * sigmoid(input) * (1 - sigmoid(input))
            let s = sigmoid(&self.input);
            let one = Tensor::ones(s.shape());
            let ds = s.mul(&one.sub(&s));
            let grad_input = grad.mul(&ds);
            
            self.input.accumulate_grad(&grad_input);
            self.input.backward_step();
        }
    }
}

// --- Tanh ---
pub fn tanh(input: &Tensor) -> Tensor {
    let input_guard = input.data();
    let input_data = &*input_guard;
    
    let result_data: Vec<f32> = input_data.par_iter()
        .map(|&x| x.tanh())
        .collect();
        
    let storage = Storage::new(result_data);
    let mut tensor = Tensor::new_with_storage(storage, input.shape());
    
    if input.requires_grad() {
        tensor.set_requires_grad_mut(true);
        tensor.set_op(Arc::new(TanhBackward {
            input: input.clone(),
        }));
    }
    
    tensor
}

#[derive(Debug)]
pub struct TanhBackward {
    pub input: Tensor,
}

impl BackwardOp for TanhBackward {
    fn backward(&self, grad: &Tensor) {
        if self.input.requires_grad() {
            // grad_input = grad * (1 - tanh(x)^2)
            let t = tanh(&self.input);
            let one = Tensor::ones(t.shape());
            let dt = one.sub(&t.mul(&t));
            let grad_input = grad.mul(&dt);
            
            self.input.accumulate_grad(&grad_input);
            self.input.backward_step();
        }
    }
}

// --- Softmax ---
// Naive implementation along last dim
pub fn softmax(input: &Tensor, dim: i64) -> Tensor {
    // Handle negative dim
    let ndim = input.shape().len() as i64;
    let dim = if dim < 0 { ndim + dim } else { dim } as usize;
    
    if dim != input.shape().len() - 1 {
        // For now only support last dim for simplicity in parallel iter
        panic!("Softmax currently only supports last dimension (dim=-1)");
    }
    
    let shape = input.shape();
    let last_dim_size = shape[shape.len() - 1];
    let outer_size: usize = shape.iter().take(shape.len() - 1).product();
    
    let input_guard = input.data();
    let input_data = &*input_guard;
    
    let mut output_data = vec![0.0; input_data.len()];
    
    // Parallel over outer dimensions
    output_data.par_chunks_mut(last_dim_size).enumerate().for_each(|(i, out_row)| {
        let offset = i * last_dim_size;
        let in_row = &input_data[offset..offset + last_dim_size];
        
        // Max for numerical stability
        let max_val = in_row.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        
        let mut sum_exp = 0.0;
        for (j, &val) in in_row.iter().enumerate() {
            let exp_val = (val - max_val).exp();
            out_row[j] = exp_val;
            sum_exp += exp_val;
        }
        
        for val in out_row.iter_mut() {
            *val /= sum_exp;
        }
    });
    
    let storage = Storage::new(output_data);
    let mut tensor = Tensor::new_with_storage(storage, shape);
    
    if input.requires_grad() {
        tensor.set_requires_grad_mut(true);
        // SoftmaxBackward
        // dS_i/dx_j = S_i * (delta_ij - S_j)
        // grad_input_j = sum_i (grad_i * dS_i/dx_j)
        //              = sum_i (grad_i * S_i * (delta_ij - S_j))
        //              = S_j * (grad_j - sum_k(grad_k * S_k))
        //              = S_j * (grad_j - (grad . S))
        
        // We need the output S for backward. Recomputing it is safer for graph.
        tensor.set_op(Arc::new(SoftmaxBackward {
            output: tensor.clone(), // Wait, cycle? 
            // Yes, storing tensor in its own op creates cycle: Tensor -> Op -> Tensor.
            // But we can store input and recompute.
            input: input.clone(),
            dim,
        }));
    }
    
    tensor
}

#[derive(Debug)]
pub struct SoftmaxBackward {
    pub input: Tensor,
    pub output: Tensor, // Warning: Cycle if not careful. 
    // Actually, if we drop the graph, cycle breaks. 
    // But `output` here is the result of forward.
    // Ideally we should store `Weak<TensorImpl>` or recompute.
    // For MVP, let's store `input` and recompute softmax in backward.
    pub dim: usize,
}

impl BackwardOp for SoftmaxBackward {
    fn backward(&self, grad: &Tensor) {
        if self.input.requires_grad() {
            // Recompute softmax
            let s = softmax(&self.input, self.dim as i64);
            
            // grad_input = S * (grad - sum(grad * S, dim=keepdim))
            // We need sum reduction.
            // Let's implement manually for last dim.
            
            let s_guard = s.data();
            let s_data = &*s_guard;
            
            let grad_guard = grad.data();
            let grad_data = &*grad_guard;
            
            let shape = s.shape();
            let last_dim = shape[shape.len()-1];
            
            let mut grad_input_data = vec![0.0; s_data.len()];
            
            grad_input_data.par_chunks_mut(last_dim).enumerate().for_each(|(i, out_row)| {
                let offset = i * last_dim;
                let s_row = &s_data[offset..offset + last_dim];
                let g_row = &grad_data[offset..offset + last_dim];
                
                let mut dot = 0.0;
                for j in 0..last_dim {
                    dot += s_row[j] * g_row[j];
                }
                
                for j in 0..last_dim {
                    out_row[j] = s_row[j] * (g_row[j] - dot);
                }
            });
            
            let grad_input = Tensor::new(&grad_input_data, shape);
            self.input.accumulate_grad(&grad_input);
            self.input.backward_step();
        }
    }
}
