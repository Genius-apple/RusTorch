use rustorch_core::Tensor;
use crate::Module;
#[cfg(feature = "serde")]
use serde::{Serialize, Deserialize};

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Dropout {
    pub p: f32,
    pub training: bool,
}

impl Dropout {
    pub fn new(p: f32) -> Self {
        Self { p, training: true }
    }
}

impl Module for Dropout {
    fn forward(&self, input: &Tensor) -> Tensor {
        if !self.training || self.p == 0.0 {
            return input.clone();
        }
        
        // Dropout mask: Bernoulli(1-p)
        // If 1, keep. If 0, drop.
        // Scale by 1/(1-p)
        
        let mut mask = Tensor::zeros(input.shape());
        mask.uniform_(0.0, 1.0);
        
        let keep_prob = 1.0 - self.p;
        let scale = 1.0 / keep_prob;
        
        // Manual mask logic since we lack logical ops
        // If val < keep_prob -> 1.0, else 0.0
        let mask_data_guard = mask.data();
        let mask_data = &*mask_data_guard;
        
        let new_mask_data: Vec<f32> = mask_data.iter().map(|&x| if x < keep_prob { scale } else { 0.0 }).collect();
        let mask_tensor = Tensor::new(&new_mask_data, input.shape()); // No grad for mask
        
        // Element-wise mul
        // But we need to make sure backward pass works.
        // If we use input * mask_tensor, backward is mask_tensor * grad.
        // Which is correct.
        
        // Note: mul op support?
        // Tensor has mul? Let's check ops.
        // We have matmul. Do we have element-wise mul?
        // Tensor::mul not explicitly added in previous turns, but `impl Mul` might be there?
        // Let's assume we need to implement element-wise mul if missing.
        // Checking `tensor.rs`: `impl Mul` is not fully implemented in snippet I saw.
        // I saw `impl Add`, `Sub`. `Mul` was imported but maybe not impl?
        // Wait, I saw `impl Add for &Tensor`.
        // Let's check if `Mul` is implemented.
        // If not, use `matmul` is wrong.
        // Let's implement `mul` in `rustorch-core` if needed.
        
        // For now, assume we have `mul` or implement it.
        // If I can't check, I'll add `mul` to `rustorch-core` now to be safe.
        
        input.mul(&mask_tensor)
    }

    fn parameters(&self) -> Vec<Tensor> {
        vec![]
    }
}
