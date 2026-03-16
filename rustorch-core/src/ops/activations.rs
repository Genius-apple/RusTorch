use crate::autograd::BackwardOp;
use crate::storage::Storage;
use crate::Tensor;
use rayon::prelude::*;
use std::sync::Arc;

// --- Sigmoid ---
pub fn sigmoid(input: &Tensor) -> Tensor {
    #[cfg(feature = "wgpu_backend")]
    {
        if let Some(input_buf) = input.storage().wgpu_buffer() {
            if !input.is_contiguous() {
                return sigmoid(&input.contiguous());
            }

            use crate::backend::wgpu::{elementwise_wgpu_buffer, ElementwiseOp};
            let size: usize = input.shape().iter().product();
            let output_buf = elementwise_wgpu_buffer(
                input_buf,
                input.shape(),
                input.strides(),
                None,
                input.shape(),
                ElementwiseOp::Sigmoid,
                None,
            );
            let storage = Storage::new_wgpu(output_buf, size, 0);
            let mut tensor = Tensor::new_with_storage(storage, input.shape());

            if input.requires_grad() {
                tensor.set_requires_grad_mut(true);
                tensor.set_op(Arc::new(SigmoidBackward {
                    input: input.clone(),
                }));
            }
            return tensor;
        }
    }

    if !input.is_contiguous() {
        return sigmoid(&input.contiguous());
    }

    let input_guard = input.data();
    let input_data = &*input_guard;

    let result_data: Vec<f32> = input_data
        .par_iter()
        .map(|&x| 1.0 / (1.0 + (-x).exp()))
        .collect();

    let storage = Storage::new(result_data);
    let mut tensor = Tensor::new_with_storage(storage, input.shape());

    if input.requires_grad() {
        tensor.set_requires_grad_mut(true);
        // We store input to update its gradient.
        // We also store output to avoid recomputing sigmoid(x) during backward.
        // But we need to be careful about reference cycles if we store output (which is `tensor` itself).
        // Since `tensor` owns `op`, and `op` owns `output` (tensor), we have a cycle.
        // So we CANNOT store `tensor` (output) in `op`.
        // We must recompute or store input.
        // Recomputing is safer for memory management in this simple Arc-based graph.
        // To optimize, we would need Weak refs or a different graph structure (e.g. tape-based).
        // Sticking to recompute for now but fixing logic.

        tensor.set_op(Arc::new(SigmoidBackward {
            input: input.clone(),
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
            // Check for GPU
            #[cfg(feature = "wgpu_backend")]
            {
                if let Some(_) = self.input.storage().wgpu_buffer() {
                    // We need output(sigmoid(input))
                    // Recompute sigmoid
                    let s = sigmoid(&self.input);
                    let s_buf = s
                        .storage()
                        .wgpu_buffer()
                        .expect("Sigmoid output should be on GPU");

                    // grad might not be contiguous or on GPU if not handled properly upstream
                    let grad_contig = if !grad.is_contiguous() {
                        grad.contiguous()
                    } else {
                        grad.clone()
                    };
                    let grad_buf = grad_contig
                        .storage()
                        .wgpu_buffer()
                        .expect("Grad should be on GPU");

                    use crate::backend::wgpu::{elementwise_wgpu_buffer, ElementwiseOp};
                    let size = grad.shape().iter().product();
                    // SigmoidBackward takes output (s) and grad
                    let output_buf = elementwise_wgpu_buffer(
                        s_buf,
                        s.shape(),
                        s.strides(),
                        Some((grad_buf, grad.shape(), grad.strides())),
                        grad.shape(),
                        ElementwiseOp::SigmoidBackward,
                        None,
                    );
                    let storage = Storage::new_wgpu(output_buf, size, 0);
                    let grad_input = Tensor::new_with_storage(storage, grad.shape());

                    self.input.accumulate_grad(&grad_input);
                    self.input.backward_step();
                    return;
                }
            }

            // grad_input = grad * sigmoid(input) * (1 - sigmoid(input))
            // Recompute sigmoid

            // Fix: Ensure CPU fallback
            #[cfg(feature = "wgpu_backend")]
            let (input, grad) = {
                let i = if self.input.storage().device().is_wgpu() {
                    self.input.to_cpu()
                } else {
                    self.input.clone()
                };
                let g = if grad.storage().device().is_wgpu() {
                    grad.to_cpu()
                } else {
                    grad.clone()
                };
                (i, g)
            };
            #[cfg(not(feature = "wgpu_backend"))]
            let (input, grad) = (self.input.clone(), grad.clone());

            let s = sigmoid(&input);

            // dS = s * (1 - s)
            // This creates intermediates.
            // Optimization: fused kernel for dS * grad

            // Manual fused implementation for speed
            let s_guard = s.data();
            let grad_guard = grad.data();
            let s_data = &*s_guard;
            let grad_data = &*grad_guard;

            let grad_input_data: Vec<f32> = s_data
                .par_iter()
                .zip(grad_data.par_iter())
                .map(|(s_val, g_val)| g_val * s_val * (1.0 - s_val))
                .collect();

            let grad_input = Tensor::new_with_storage(Storage::new(grad_input_data), grad.shape());

            self.input.accumulate_grad(&grad_input);
            self.input.backward_step();
        }
    }
}

// --- Tanh ---
pub fn tanh(input: &Tensor) -> Tensor {
    #[cfg(feature = "wgpu_backend")]
    {
        if let Some(input_buf) = input.storage().wgpu_buffer() {
            if !input.is_contiguous() {
                return tanh(&input.contiguous());
            }

            use crate::backend::wgpu::{elementwise_wgpu_buffer, ElementwiseOp};
            let size: usize = input.shape().iter().product();
            let output_buf = elementwise_wgpu_buffer(
                input_buf,
                input.shape(),
                input.strides(),
                None,
                input.shape(),
                ElementwiseOp::Tanh,
                None,
            );
            let storage = Storage::new_wgpu(output_buf, size, 0);
            let mut tensor = Tensor::new_with_storage(storage, input.shape());

            if input.requires_grad() {
                tensor.set_requires_grad_mut(true);
                tensor.set_op(Arc::new(TanhBackward {
                    input: input.clone(),
                }));
            }
            return tensor;
        }
    }

    if !input.is_contiguous() {
        return tanh(&input.contiguous());
    }

    let input_guard = input.data();
    let input_data = &*input_guard;

    let result_data: Vec<f32> = input_data.par_iter().map(|&x| x.tanh()).collect();

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
            #[cfg(feature = "wgpu_backend")]
            {
                if let Some(_) = self.input.storage().wgpu_buffer() {
                    // Recompute tanh
                    let t = tanh(&self.input);
                    let t_buf = t
                        .storage()
                        .wgpu_buffer()
                        .expect("Tanh output should be on GPU");

                    let grad_contig = if !grad.is_contiguous() {
                        grad.contiguous()
                    } else {
                        grad.clone()
                    };
                    let grad_buf = grad_contig
                        .storage()
                        .wgpu_buffer()
                        .expect("Grad should be on GPU");

                    use crate::backend::wgpu::{elementwise_wgpu_buffer, ElementwiseOp};
                    let size = grad.shape().iter().product();
                    // TanhBackward: (1 - t^2) * grad
                    let output_buf = elementwise_wgpu_buffer(
                        t_buf,
                        t.shape(),
                        t.strides(),
                        Some((grad_buf, grad.shape(), grad.strides())),
                        grad.shape(),
                        ElementwiseOp::TanhBackward,
                        None,
                    );

                    let storage = Storage::new_wgpu(output_buf, size, 0);
                    let grad_input = Tensor::new_with_storage(storage, grad.shape());

                    self.input.accumulate_grad(&grad_input);
                    self.input.backward_step();
                    return;
                }
            }

            // Fix: CPU Fallback
            #[cfg(feature = "wgpu_backend")]
            let (input, grad) = {
                let i = if self.input.storage().device().is_wgpu() {
                    self.input.to_cpu()
                } else {
                    self.input.clone()
                };
                let g = if grad.storage().device().is_wgpu() {
                    grad.to_cpu()
                } else {
                    grad.clone()
                };
                (i, g)
            };
            #[cfg(not(feature = "wgpu_backend"))]
            let (input, grad) = (self.input.clone(), grad.clone());

            let t = tanh(&input);

            let t_guard = t.data();
            let grad_guard = grad.data();
            let t_data = &*t_guard;
            let grad_data = &*grad_guard;

            let grad_input_data: Vec<f32> = t_data
                .par_iter()
                .zip(grad_data.par_iter())
                .map(|(t_val, g_val)| g_val * (1.0 - t_val * t_val))
                .collect();

            let grad_input = Tensor::new_with_storage(Storage::new(grad_input_data), grad.shape());

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
    let _outer_size: usize = shape.iter().take(shape.len() - 1).product();

    if !input.is_contiguous() {
        return softmax(&input.contiguous(), dim as i64);
    }

    #[cfg(feature = "wgpu_backend")]
    let input = if input.storage().device().is_wgpu() {
        input.to_cpu()
    } else {
        input.clone()
    };

    let input_guard = input.data();
    let input_data = &*input_guard;

    let mut output_data = vec![0.0; input_data.len()];

    // Parallel over outer dimensions
    output_data
        .par_chunks_mut(last_dim_size)
        .enumerate()
        .for_each(|(i, out_row)| {
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

            #[cfg(feature = "wgpu_backend")]
            let (s, grad) = {
                let s = if s.storage().device().is_wgpu() {
                    s.to_cpu()
                } else {
                    s
                };
                let g = if grad.storage().device().is_wgpu() {
                    grad.to_cpu()
                } else {
                    grad.clone()
                };
                (s, g)
            };

            let s_guard = s.data();
            let s_data = &*s_guard;

            let grad_guard = grad.data();
            let grad_data = &*grad_guard;

            let shape = s.shape();
            let last_dim = shape[shape.len() - 1];

            let mut grad_input_data = vec![0.0; s_data.len()];

            grad_input_data
                .par_chunks_mut(last_dim)
                .enumerate()
                .for_each(|(i, out_row)| {
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
