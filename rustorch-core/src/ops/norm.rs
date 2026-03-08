use std::sync::{Arc, Mutex};
use rayon::prelude::*;
use crate::Tensor;
use crate::autograd::BackwardOp;
use crate::storage::Storage;

// --- BatchNorm2d ---

#[derive(Debug)]
pub struct BatchNorm2dBackwardFull {
    pub input: Tensor,
    pub gamma: Option<Tensor>,
    pub beta: Option<Tensor>,
    pub mean: Vec<f32>,
    pub inv_std: Vec<f32>,
    pub eps: f32,
}

impl BackwardOp for BatchNorm2dBackwardFull {
    fn backward(&self, grad: &Tensor) {
        if self.input.requires_grad() {
            // Simplified Backward for BatchNorm
            // Assuming training mode logic (complex)
            // Or just use a simplified diagonal approximation if strictly needed for MVP.
            // But let's try to be reasonably correct.
            
            let n = self.input.shape()[0];
            let c = self.input.shape()[1];
            let h = self.input.shape()[2];
            let w = self.input.shape()[3];
            let num_pixels = (h * w) as f32;
            let m = (n * h * w) as f32;
            
            let grad_guard = grad.data();
            let grad_data = &*grad_guard;
            
            let input_guard = self.input.data();
            let input_data = &*input_guard;
            
            let gamma_data = self.gamma.as_ref().map(|g| g.data());
            
            // Grads for gamma and beta
            let mut d_gamma = vec![0.0; c];
            let mut d_beta = vec![0.0; c];
            
            // First pass: compute d_gamma, d_beta
            // Parallelize over C?
            // Accumulate over N, H, W
            
            // This is reduction.
            // Let's do it serial over C for simplicity or parallel over C.
            
            let grads: Vec<(f32, f32)> = (0..c).into_par_iter().map(|ci| {
                let mut dg = 0.0;
                let mut db = 0.0;
                let mean = self.mean[ci];
                let inv_std = self.inv_std[ci];
                
                for b in 0..n {
                    for hi in 0..h {
                        for wi in 0..w {
                            let idx = ((b * c + ci) * h + hi) * w + wi;
                            let dy = grad_data[idx];
                            let x = input_data[idx];
                            let x_hat = (x - mean) * inv_std;
                            
                            dg += dy * x_hat;
                            db += dy;
                        }
                    }
                }
                (dg, db)
            }).collect();
            
            for (i, (dg, db)) in grads.iter().enumerate() {
                d_gamma[i] = *dg;
                d_beta[i] = *db;
            }
            
            // Update gamma/beta grads
            if let Some(gamma) = &self.gamma {
                if gamma.requires_grad() {
                    let g_grad = Tensor::new(&d_gamma, gamma.shape());
                    gamma.accumulate_grad(&g_grad);
                    gamma.backward_step();
                }
            }
            if let Some(beta) = &self.beta {
                if beta.requires_grad() {
                    let b_grad = Tensor::new(&d_beta, beta.shape());
                    beta.accumulate_grad(&b_grad);
                    beta.backward_step();
                }
            }
            
            // Compute d_input
            // dx_hat = dy * gamma
            // dx = (1/m) * inv_std * (m * dx_hat - sum(dx_hat) - x_hat * sum(dx_hat * x_hat))
            
            let mut d_input_data = vec![0.0; input_data.len()];
            
            d_input_data.par_chunks_mut(h * w * c).enumerate().for_each(|(b, batch_slice)| {
                for ci in 0..c {
                    let mean = self.mean[ci];
                    let inv_std = self.inv_std[ci];
                    let gamma_val = if let Some(g) = &gamma_data { g[ci] } else { 1.0 };
                    
                    // We need sum(dx_hat) and sum(dx_hat * x_hat) over the whole batch!
                    // But we are inside parallel chunk.
                    // We computed these sums in d_gamma (sum(dy*x_hat)) and d_beta (sum(dy)).
                    // sum(dx_hat) = sum(dy * gamma) = gamma * sum(dy) = gamma * d_beta
                    // sum(dx_hat * x_hat) = sum(dy * gamma * x_hat) = gamma * sum(dy * x_hat) = gamma * d_gamma
                    
                    let sum_dx_hat = gamma_val * d_beta[ci];
                    let sum_dx_hat_x_hat = gamma_val * d_gamma[ci];
                    
                    for hi in 0..h {
                        for wi in 0..w {
                            let local_idx = (ci * h + hi) * w + wi;
                            let global_idx = ((b * c + ci) * h + hi) * w + wi; // Wait, b is batch index?
                            // batch_slice is slice of size C*H*W for batch b.
                            
                            let dy = grad_data[global_idx];
                            let x = input_data[global_idx];
                            let x_hat = (x - mean) * inv_std;
                            let dx_hat = dy * gamma_val;
                            
                            let val = (1.0 / m) * inv_std * (m * dx_hat - sum_dx_hat - x_hat * sum_dx_hat_x_hat);
                            batch_slice[local_idx] = val;
                        }
                    }
                }
            });
            
            let d_input = Tensor::new(&d_input_data, self.input.shape());
            self.input.accumulate_grad(&d_input);
            self.input.backward_step();
        }
    }
}

pub fn batch_norm2d(
    input: &Tensor,
    gamma: Option<&Tensor>,
    beta: Option<&Tensor>,
    running_mean: &Tensor,
    running_var: &Tensor,
    training: bool,
    momentum: f32,
    eps: f32
) -> Tensor {
    let shape = input.shape();
    if shape.len() != 4 {
        panic!("BatchNorm2d requires 4D tensor");
    }
    let n = shape[0];
    let c = shape[1];
    let h = shape[2];
    let w = shape[3];
    
    let input_guard = input.data();
    let input_data = &*input_guard;
    
    // Compute mean and var per channel
    let (mean, var) = if training {
        let stats: Vec<(f32, f32)> = (0..c).into_par_iter().map(|ci| {
            let mut sum = 0.0;
            let mut sq_sum = 0.0;
            let count = (n * h * w) as f32;
            
            for b in 0..n {
                for hi in 0..h {
                    for wi in 0..w {
                        let val = input_data[((b * c + ci) * h + hi) * w + wi];
                        sum += val;
                        sq_sum += val * val;
                    }
                }
            }
            
            let m = sum / count;
            let v = sq_sum / count - m * m;
            (m, v)
        }).collect();
        
        let mut means = Vec::with_capacity(c);
        let mut vars = Vec::with_capacity(c);
        for (m, v) in stats {
            means.push(m);
            vars.push(v);
        }
        
        // Update running stats
        let mut r_mean = running_mean.data_mut();
        let mut r_var = running_var.data_mut();
        
        for i in 0..c {
            r_mean[i] = (1.0 - momentum) * r_mean[i] + momentum * means[i];
            r_var[i] = (1.0 - momentum) * r_var[i] + momentum * vars[i];
        }
        
        (means, vars)
    } else {
        // Inference: use running stats
        let r_mean = running_mean.data().clone();
        let r_var = running_var.data().clone();
        (r_mean, r_var)
    };
    
    // Normalize
    let mut output_data = vec![0.0; n * c * h * w];
    let gamma_data = gamma.map(|g| g.data());
    let beta_data = beta.map(|b| b.data());
    
    // Precompute inv_std
    let inv_std: Vec<f32> = var.iter().map(|v| 1.0 / (v + eps).sqrt()).collect();
    
    output_data.par_chunks_mut(h * w).enumerate().for_each(|(i, plane)| {
        let ci = (i / n) % c; // Wait, chunk is H*W? No.
        // Index mapping:
        // i goes from 0 to N*C-1.
        // idx = i * (H*W)
        // b = i / c;
        // ci = i % c;
        // Wait, memory layout is N, C, H, W.
        // So continuous chunks of H*W belong to (b, ci).
        
        let ci = i % c;
        // let b = i / c;
        
        let m = mean[ci];
        let inv_s = inv_std[ci];
        let g = if let Some(gd) = &gamma_data { gd[ci] } else { 1.0 };
        let b_val = if let Some(bd) = &beta_data { bd[ci] } else { 0.0 };
        
        for x in plane.iter_mut() { // plane is pre-filled with 0.0, we need to read input
             // Oops, we don't have input slice here easily unless we zip or calculate index.
             // Let's iterate index.
        }
    });
    
    // Rewrite parallel loop properly
    output_data.par_chunks_mut(h * w).enumerate().for_each(|(i, output_plane)| {
        let ci = i % c;
        let m = mean[ci];
        let inv_s = inv_std[ci];
        let g = if let Some(gd) = &gamma_data { gd[ci] } else { 1.0 };
        let b_val = if let Some(bd) = &beta_data { bd[ci] } else { 0.0 };
        
        let start_idx = i * h * w;
        for j in 0..h*w {
            let val = input_data[start_idx + j];
            output_plane[j] = (val - m) * inv_s * g + b_val;
        }
    });
    
    let storage = Storage::new(output_data);
    let mut tensor = Tensor::new_with_storage(storage, shape);
    
    if training && (input.requires_grad() || gamma.map_or(false, |g| g.requires_grad()) || beta.map_or(false, |b| b.requires_grad())) {
        tensor.set_requires_grad_mut(true);
        // Need to pass mean/inv_std to backward
        // But mean/inv_std are Vec<f32>.
        // Backward op expects them?
        // We defined BackwardOp with Tensor or Vec?
        // Let's use Vec for internal storage in op struct.
        
        tensor.set_op(Arc::new(BatchNorm2dBackwardFull {
            input: input.clone(),
            gamma: gamma.map(|g| g.clone()),
            beta: beta.map(|b| b.clone()),
            mean,
            inv_std,
            eps,
        }));
    }
    
    tensor
}

// --- LayerNorm ---
// Input: (N, *, Normalized_Shape)
// Output: Same
// Mean/Var computed over last D dims.

#[derive(Debug)]
pub struct LayerNormBackward {
    pub input: Tensor,
    pub weight: Option<Tensor>,
    pub bias: Option<Tensor>,
    pub mean: Tensor,
    pub inv_std: Tensor,
    pub normalized_shape: Vec<usize>,
}

impl BackwardOp for LayerNormBackward {
    fn backward(&self, grad: &Tensor) {
        if self.input.requires_grad() {
            // Simplified backward for LayerNorm
            // dL/dx = ... complex formula ...
            // Similar to BatchNorm but over different dimensions.
            // For now, implement a simplified version or placeholder.
            // Implementing full LayerNorm backward is verbose.
            // Let's assume standard implementation.
            
            // TODO: Full implementation.
            // Just passing gradient through for now (Incorrect but allows graph flow)
            // Or better: Implement correctly for 1D case (common in Transformer).
            
            // Assuming input (N, L, D) and norm over D.
            // Mean/Var is (N, L, 1).
            
            // Let's implement correct logic for last dim normalization.
            let shape = self.input.shape();
            let dim = *shape.last().unwrap();
            let n_elements: usize = shape.iter().take(shape.len()-1).product();
            
            // Grad input calculation...
            // It's quite involved to write out from scratch in one go.
            // Given constraints, I'll use a simplified approximation or placeholder 
            // that warns it's not fully implemented.
            // Or, if I want to be diligent:
            // dL/dx_i = 1/D * gamma * inv_std * ( D*dy_i - sum(dy) - x_hat_i * sum(dy * x_hat) )
            
            let grad_guard = grad.data();
            let grad_data = &*grad_guard;
            
            let input_guard = self.input.data();
            let input_data = &*input_guard;
            
            let mean_guard = self.mean.data(); // (N*L)
            let mean_data = &*mean_guard;
            
            let inv_std_guard = self.inv_std.data(); // (N*L)
            let inv_std_data = &*inv_std_guard;
            
            let weight_data = self.weight.as_ref().map(|w| w.data());
            
            let mut grad_input_data = vec![0.0; grad_data.len()];
            
            // Iterate over each instance (N*L)
            grad_input_data.par_chunks_mut(dim).enumerate().for_each(|(i, dx_row)| {
                let m = mean_data[i];
                let inv_s = inv_std_data[i];
                let offset = i * dim;
                
                let mut sum_dy = 0.0;
                let mut sum_dy_x_hat = 0.0;
                
                for j in 0..dim {
                    let dy = grad_data[offset + j];
                    let x = input_data[offset + j];
                    let x_hat = (x - m) * inv_s;
                    let g = if let Some(wd) = &weight_data { wd[j] } else { 1.0 };
                    
                    // dy is dL/dy. 
                    // Effective dy for normalization part is dy * gamma
                    let dy_eff = dy * g;
                    
                    sum_dy += dy_eff;
                    sum_dy_x_hat += dy_eff * x_hat;
                }
                
                let factor = inv_s / (dim as f32);
                
                for j in 0..dim {
                    let dy = grad_data[offset + j];
                    let x = input_data[offset + j];
                    let x_hat = (x - m) * inv_s;
                    let g = if let Some(wd) = &weight_data { wd[j] } else { 1.0 };
                    let dy_eff = dy * g;
                    
                    dx_row[j] = factor * ( (dim as f32) * dy_eff - sum_dy - x_hat * sum_dy_x_hat );
                }
            });
            
            let grad_input = Tensor::new(&grad_input_data, shape);
            self.input.accumulate_grad(&grad_input);
            self.input.backward_step();
            
            // Weight/Bias grads
            if let Some(weight) = &self.weight {
                if weight.requires_grad() {
                    // sum(dy * x_hat) over all batches
                    let mut dw_data = vec![0.0; dim];
                    // Need atomic add or reduce.
                    // Simple serial reduce for now.
                    for i in 0..n_elements {
                        let offset = i * dim;
                        let m = mean_data[i];
                        let inv_s = inv_std_data[i];
                        for j in 0..dim {
                            let dy = grad_data[offset + j];
                            let x = input_data[offset + j];
                            let x_hat = (x - m) * inv_s;
                            dw_data[j] += dy * x_hat;
                        }
                    }
                    let dw = Tensor::new(&dw_data, weight.shape());
                    weight.accumulate_grad(&dw);
                    weight.backward_step();
                }
            }
            
            if let Some(bias) = &self.bias {
                if bias.requires_grad() {
                    let mut db_data = vec![0.0; dim];
                    for i in 0..n_elements {
                        let offset = i * dim;
                        for j in 0..dim {
                            db_data[j] += grad_data[offset + j];
                        }
                    }
                    let db = Tensor::new(&db_data, bias.shape());
                    bias.accumulate_grad(&db);
                    bias.backward_step();
                }
            }
        }
    }
}

pub fn layer_norm(
    input: &Tensor,
    normalized_shape: &[usize],
    weight: Option<&Tensor>,
    bias: Option<&Tensor>,
    eps: f32
) -> Tensor {
    // Assume normalized_shape is last D dims.
    // Flatten input to (N, D_norm)
    let shape = input.shape();
    let norm_dims = normalized_shape.len();
    if shape.len() < norm_dims {
        panic!("LayerNorm: Input shape {:?} smaller than normalized shape {:?}", shape, normalized_shape);
    }
    
    // Check suffixes match
    let start_dim = shape.len() - norm_dims;
    if &shape[start_dim..] != normalized_shape {
        panic!("LayerNorm: Input shape {:?} does not match normalized shape {:?}", shape, normalized_shape);
    }
    
    let outer_dim: usize = shape[0..start_dim].iter().product();
    let inner_dim: usize = normalized_shape.iter().product(); // D
    
    let input_guard = input.data();
    let input_data = &*input_guard;
    
    let weight_data = weight.map(|w| w.data());
    let bias_data = bias.map(|b| b.data());
    
    let mut output_data = vec![0.0; input_data.len()];
    let mut means = vec![0.0; outer_dim];
    let mut inv_stds = vec![0.0; outer_dim];
    
    // Compute mean/var per outer instance
    // Parallelize over outer_dim
    // We need to return means/inv_stds, so maybe collect.
    
    let stats: Vec<(f32, f32)> = (0..outer_dim).into_par_iter().map(|i| {
        let offset = i * inner_dim;
        let mut sum = 0.0;
        let mut sq_sum = 0.0;
        for j in 0..inner_dim {
            let val = input_data[offset + j];
            sum += val;
            sq_sum += val * val;
        }
        let m = sum / inner_dim as f32;
        let v = sq_sum / inner_dim as f32 - m * m;
        let inv_s = 1.0 / (v + eps).sqrt();
        (m, inv_s)
    }).collect();
    
    for (i, (m, inv_s)) in stats.iter().enumerate() {
        means[i] = *m;
        inv_stds[i] = *inv_s;
        
        let offset = i * inner_dim;
        for j in 0..inner_dim {
            let val = input_data[offset + j];
            let x_hat = (val - m) * inv_s;
            let g = if let Some(wd) = &weight_data { wd[j] } else { 1.0 };
            let b = if let Some(bd) = &bias_data { bd[j] } else { 0.0 };
            output_data[offset + j] = x_hat * g + b;
        }
    }
    
    let storage = Storage::new(output_data);
    let mut tensor = Tensor::new_with_storage(storage, shape);
    
    if input.requires_grad() || weight.map_or(false, |w| w.requires_grad()) || bias.map_or(false, |b| b.requires_grad()) {
        tensor.set_requires_grad_mut(true);
        // Store mean/inv_std for backward
        // They are (OuterDim). We can store as (OuterDim) tensor.
        let mean_tensor = Tensor::new(&means, &[outer_dim]); // Flattened stats
        let inv_std_tensor = Tensor::new(&inv_stds, &[outer_dim]);
        
        tensor.set_op(Arc::new(LayerNormBackward {
            input: input.clone(),
            weight: weight.map(|w| w.clone()),
            bias: bias.map(|b| b.clone()),
            mean: mean_tensor,
            inv_std: inv_std_tensor,
            normalized_shape: normalized_shape.to_vec(),
        }));
    }
    
    tensor
}
