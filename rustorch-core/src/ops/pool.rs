use std::sync::Arc;
use rayon::prelude::*;
use crate::Tensor;
use crate::autograd::BackwardOp;
use crate::storage::Storage;

// --- MaxPool2d ---
// Input: (N, C, H, W)
// Output: (N, C, H_out, W_out)
// H_out = (H + 2*pad - kernel_size) / stride + 1

#[derive(Debug)]
pub struct MaxPool2dBackward {
    pub input: Tensor,
    pub kernel_size: (usize, usize),
    pub stride: (usize, usize),
    pub padding: (usize, usize),
}

impl BackwardOp for MaxPool2dBackward {
    fn backward(&self, grad: &Tensor) {
        if self.input.requires_grad() {
            let (k_h, k_w) = self.kernel_size;
            let (stride_h, stride_w) = self.stride;
            let (pad_h, pad_w) = self.padding;
            
            let input_shape = self.input.shape();
            let grad_shape = grad.shape();
            
            let n = input_shape[0];
            let c = input_shape[1];
            let h_in = input_shape[2];
            let w_in = input_shape[3];
            
            let h_out = grad_shape[2];
            let w_out = grad_shape[3];
            
            let input_guard = self.input.data();
            let grad_guard = grad.data();
            let input_data = &*input_guard;
            let grad_data = &*grad_guard;
            
            // We need to scatter gradients back to the max indices.
            // Since multiple output pixels might map to same input pixel (with overlap), we accumulate.
            // However, maxpool usually takes the gradient from the max position.
            
            // Since we can't easily do scatter_add in parallel without atomics or locking, 
            // let's iterate over output and add to a local buffer, then reduce?
            // Or use sequential update for simplicity first, or parallel over N, C.
            
            let mut grad_input_data = vec![0.0; n * c * h_in * w_in];
            
            // For MaxPool backward, we need to find which index was the max.
            // We re-compute the forward pass window to find the index.
            
            // Parallelize over N, C
            // Note: Parallel writing to grad_input_data is unsafe if windows overlap.
            // But MaxPool windows usually stride >= kernel_size for non-overlapping.
            // If they overlap, we need atomic adds.
            // For now, let's assume standard non-overlapping or handle overlap sequentially within a thread?
            // Actually, if stride < kernel_size, multiple output pixels depend on same input.
            // So we can't parallelize purely by output pixel without atomic add to input.
            // But we CAN parallelize by N and C, as they are independent.
            
            // We can use chunks_mut to split grad_input_data by N*C
            let chunk_size = h_in * w_in;
            grad_input_data.par_chunks_mut(chunk_size).enumerate().for_each(|(i, grad_in_chunk)| {
                let b = i / c;
                let ci = i % c;
                
                // Corresponding section in input and grad
                let input_offset = (b * c + ci) * h_in * w_in;
                let grad_offset = (b * c + ci) * h_out * w_out;
                
                for ho in 0..h_out {
                    for wo in 0..w_out {
                        let h_start = (ho * stride_h).saturating_sub(pad_h);
                        let w_start = (wo * stride_w).saturating_sub(pad_w);
                        let h_end = (h_start + k_h).min(h_in);
                        let w_end = (w_start + k_w).min(w_in);
                        
                        // Find max in window
                        let mut max_val = -f32::INFINITY;
                        let mut max_idx = (h_start, w_start); // Default to start
                        
                        for h in h_start..h_end {
                            for w in w_start..w_end {
                                let val = input_data[input_offset + h * w_in + w];
                                if val > max_val {
                                    max_val = val;
                                    max_idx = (h, w);
                                }
                            }
                        }
                        
                        // Add gradient to max index
                        // Safety: max_idx is within h_in, w_in bounds
                        let g_val = grad_data[grad_offset + ho * w_out + wo];
                        grad_in_chunk[max_idx.0 * w_in + max_idx.1] += g_val;
                    }
                }
            });
            
            let grad_input_tensor = Tensor::new_with_storage(Storage::new(grad_input_data), self.input.shape());
            self.input.accumulate_grad(&grad_input_tensor);
            self.input.backward_step();
        }
    }
}

pub fn max_pool2d(input: &Tensor, kernel_size: (usize, usize), stride: (usize, usize), padding: (usize, usize)) -> Tensor {
    let shape = input.shape();
    if shape.len() != 4 {
        panic!("MaxPool2d requires 4D tensor (N, C, H, W)");
    }
    
    let n = shape[0];
    let c = shape[1];
    let h_in = shape[2];
    let w_in = shape[3];
    
    let (k_h, k_w) = kernel_size;
    let (stride_h, stride_w) = stride;
    let (pad_h, pad_w) = padding;
    
    let h_out = (h_in + 2 * pad_h - k_h) / stride_h + 1;
    let w_out = (w_in + 2 * pad_w - k_w) / stride_w + 1;
    
    let input_guard = input.data();
    let input_data = &*input_guard;
    
    let total_elements = n * c * h_out * w_out;
    let result_data: Vec<f32> = (0..total_elements).into_par_iter().map(|idx| {
        let wo = idx % w_out;
        let ho = (idx / w_out) % h_out;
        let ci = (idx / (w_out * h_out)) % c;
        let b = idx / (w_out * h_out * c);
        
        let h_start_raw = (ho * stride_h) as isize - pad_h as isize;
        let w_start_raw = (wo * stride_w) as isize - pad_w as isize;
        
        let mut max_val = -f32::INFINITY;
        
        for kh in 0..k_h {
            for kw in 0..k_w {
                let h_in_idx = h_start_raw + kh as isize;
                let w_in_idx = w_start_raw + kw as isize;
                
                if h_in_idx >= 0 && h_in_idx < h_in as isize && w_in_idx >= 0 && w_in_idx < w_in as isize {
                     let val = input_data[((b * c + ci) * h_in + h_in_idx as usize) * w_in + w_in_idx as usize];
                     if val > max_val {
                         max_val = val;
                     }
                }
            }
        }
        max_val
    }).collect();
    
    let storage = Storage::new(result_data);
    let mut tensor = Tensor::new_with_storage(storage, &[n, c, h_out, w_out]);
    
    if input.requires_grad() {
        tensor.set_requires_grad_mut(true);
        tensor.set_op(Arc::new(MaxPool2dBackward {
            input: input.clone(),
            kernel_size,
            stride,
            padding,
        }));
    }
    
    tensor
}
