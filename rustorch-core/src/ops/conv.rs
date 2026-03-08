use std::sync::Arc;
use rayon::prelude::*;
use crate::Tensor;
use crate::autograd::BackwardOp;
use crate::storage::Storage;

// --- Conv2d ---
// Input: (N, C_in, H, W)
// Weight: (C_out, C_in, kH, kW)
// Output: (N, C_out, H_out, W_out)

#[derive(Debug)]
pub struct Conv2dBackward {
    pub input: Tensor,
    pub weight: Tensor,
    pub stride: (usize, usize),
    pub padding: (usize, usize),
}

impl BackwardOp for Conv2dBackward {
    fn backward(&self, grad: &Tensor) {
        let (stride_h, stride_w) = self.stride;
        let (pad_h, pad_w) = self.padding;
        
        let input_shape = self.input.shape();
        let weight_shape = self.weight.shape();
        let grad_shape = grad.shape();
        
        let n = input_shape[0];
        let c_in = input_shape[1];
        let h_in = input_shape[2];
        let w_in = input_shape[3];
        
        let c_out = weight_shape[0];
        let k_h = weight_shape[2];
        let k_w = weight_shape[3];
        
        let h_out = grad_shape[2];
        let w_out = grad_shape[3];
        
        // Compute grad_input
        if self.input.requires_grad() {
            let grad_guard = grad.data();
            let weight_guard = self.weight.data();
            let grad_data = &*grad_guard;
            let weight_data = &*weight_guard;
            
            let total_elements = n * c_in * h_in * w_in;
            let grad_input_data: Vec<f32> = (0..total_elements).into_par_iter().map(|idx| {
                 let w = idx % w_in;
                 let h = (idx / w_in) % h_in;
                 let ci = (idx / (w_in * h_in)) % c_in;
                 let b = idx / (w_in * h_in * c_in);
                 
                 let mut sum = 0.0;
                 
                 // h_out range
                 let h_out_start = if h + pad_h >= k_h { (h + pad_h - k_h + 1 + stride_h - 1) / stride_h } else { 0 };
                 let h_out_end = std::cmp::min(h_out, (h + pad_h) / stride_h + 1);
                 
                 for ho in h_out_start..h_out_end {
                     let kh = h + pad_h - ho * stride_h;
                     
                     let w_out_start = if w + pad_w >= k_w { (w + pad_w - k_w + 1 + stride_w - 1) / stride_w } else { 0 };
                     let w_out_end = std::cmp::min(w_out, (w + pad_w) / stride_w + 1);

                     for wo in w_out_start..w_out_end {
                         let kw = w + pad_w - wo * stride_w;
                         
                         for co in 0..c_out {
                             let g_val = grad_data[((b * c_out + co) * h_out + ho) * w_out + wo];
                             let w_val = weight_data[((co * c_in + ci) * k_h + kh) * k_w + kw];
                             sum += g_val * w_val;
                         }
                     }
                 }
                 sum
            }).collect();
            
            let grad_input_tensor = Tensor::new_with_storage(Storage::new(grad_input_data), self.input.shape());
            self.input.accumulate_grad(&grad_input_tensor);
            self.input.backward_step();
        }
        
        // Compute grad_weight
        if self.weight.requires_grad() {
            let input_guard = self.input.data();
            let grad_guard = grad.data();
            let input_data = &*input_guard;
            let grad_data = &*grad_guard;
            
            let total_elements = c_out * c_in * k_h * k_w;
            let grad_weight_data: Vec<f32> = (0..total_elements).into_par_iter().map(|idx| {
                let kw = idx % k_w;
                let kh = (idx / k_w) % k_h;
                let ci = (idx / (k_w * k_h)) % c_in;
                let co = idx / (k_w * k_h * c_in);
                
                let mut sum = 0.0;
                for b in 0..n {
                    for ho in 0..h_out {
                        for wo in 0..w_out {
                            let h_in_idx = ho * stride_h + kh;
                            let w_in_idx = wo * stride_w + kw;
                            
                            if h_in_idx >= pad_h && w_in_idx >= pad_w {
                                let hi = h_in_idx - pad_h;
                                let wi = w_in_idx - pad_w;
                                
                                if hi < h_in && wi < w_in {
                                    let val_in = input_data[((b * c_in + ci) * h_in + hi) * w_in + wi];
                                    let val_g = grad_data[((b * c_out + co) * h_out + ho) * w_out + wo];
                                    sum += val_in * val_g;
                                }
                            }
                        }
                    }
                }
                sum
            }).collect();
            
            let grad_weight_tensor = Tensor::new_with_storage(Storage::new(grad_weight_data), self.weight.shape());
            self.weight.accumulate_grad(&grad_weight_tensor);
            self.weight.backward_step();
        }
    }
}

pub fn conv2d(input: &Tensor, weight: &Tensor, stride: (usize, usize), padding: (usize, usize)) -> Tensor {
    let input_shape = input.shape();
    let weight_shape = weight.shape();
    
    if input_shape.len() != 4 || weight_shape.len() != 4 {
        panic!("Conv2d requires 4D tensors");
    }
    
    let n = input_shape[0];
    let c_in = input_shape[1];
    let h_in = input_shape[2];
    let w_in = input_shape[3];
    
    let c_out = weight_shape[0];
    let k_h = weight_shape[2];
    let k_w = weight_shape[3];
    
    if weight_shape[1] != c_in {
        panic!("Weight input channels {} must match input channels {}", weight_shape[1], c_in);
    }
    
    let (stride_h, stride_w) = stride;
    let (pad_h, pad_w) = padding;
    
    let h_out = (h_in + 2 * pad_h - k_h) / stride_h + 1;
    let w_out = (w_in + 2 * pad_w - k_w) / stride_w + 1;
    
    let input_guard = input.data();
    let weight_guard = weight.data();
    let input_data = &*input_guard;
    let weight_data = &*weight_guard;
    
    let total_elements = n * c_out * h_out * w_out;
    let result_data: Vec<f32> = (0..total_elements).into_par_iter().map(|idx| {
        let wo = idx % w_out;
        let ho = (idx / w_out) % h_out;
        let co = (idx / (w_out * h_out)) % c_out;
        let b = idx / (w_out * h_out * c_out);
        
        let mut sum = 0.0;
        for ci in 0..c_in {
            for kh in 0..k_h {
                for kw in 0..k_w {
                    let h_in_idx = ho * stride_h + kh;
                    let w_in_idx = wo * stride_w + kw;
                    
                    if h_in_idx >= pad_h && w_in_idx >= pad_w {
                        let hi = h_in_idx - pad_h;
                        let wi = w_in_idx - pad_w;
                        
                        if hi < h_in && wi < w_in {
                            let val_in = input_data[((b * c_in + ci) * h_in + hi) * w_in + wi];
                            let val_w = weight_data[((co * c_in + ci) * k_h + kh) * k_w + kw];
                            sum += val_in * val_w;
                        }
                    }
                }
            }
        }
        sum
    }).collect();
    
    let storage = Storage::new(result_data);
    let mut tensor = Tensor::new_with_storage(storage, &[n, c_out, h_out, w_out]);
    
    if input.requires_grad() || weight.requires_grad() {
        tensor.set_requires_grad_mut(true);
        tensor.set_op(Arc::new(Conv2dBackward {
            input: input.clone(),
            weight: weight.clone(),
            stride,
            padding,
        }));
    }
    
    if crate::graph::is_tracing() {
        crate::graph::record_op(crate::graph::NodeOp::Conv2d { stride, padding }, &[input, weight], &tensor);
    }
    
    tensor
}
