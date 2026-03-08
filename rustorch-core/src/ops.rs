use std::sync::Arc;
use rayon::prelude::*;
use crate::Tensor;
use crate::autograd::BackwardOp;
use crate::storage::Storage;

pub mod conv;
pub mod pool;
pub mod norm;
pub mod view;
pub mod embedding;
pub mod activations;

pub use conv::conv2d;
pub use pool::max_pool2d;
pub use norm::{batch_norm2d, layer_norm};
pub use embedding::embedding;
pub use activations::{sigmoid, tanh, softmax};
pub use view::ReshapeBackward;

#[derive(Debug)]
pub struct MulBackward {
    pub lhs: Tensor,
    pub rhs: Tensor,
}

impl BackwardOp for MulBackward {
    fn backward(&self, grad: &Tensor) {
        // y = a * b
        // da = dy * b
        // db = dy * a
        if self.lhs.requires_grad() {
            // Need to detach rhs from graph to avoid cycle?
            // Actually, in backward, we use values.
            // If we use .mul() and rhs has requires_grad=true, we might build graph.
            // For first order derivative, we usually don't need graph.
            // But if we support higher order...
            // Let's keep it simple: assume we don't track grad of grad yet.
            // But `mul` will check `requires_grad`.
            // We should use a `mul_no_grad` or similar if we want to stop tracking.
            // But `grad` tensor usually has requires_grad=false unless created with it.
            
            // However, self.rhs might have requires_grad=true.
            // So grad.mul(&self.rhs) will produce a tensor with requires_grad=true!
            // This means we are building the graph for double backward. This is good!
            // But we need to be careful about infinite recursion if not careful.
            // Here it's fine.
            
            let grad_lhs = grad.mul(&self.rhs); 
            self.lhs.accumulate_grad(&grad_lhs);
            self.lhs.backward_step();
        }
        if self.rhs.requires_grad() {
            let grad_rhs = grad.mul(&self.lhs);
            self.rhs.accumulate_grad(&grad_rhs);
            self.rhs.backward_step();
        }
    }
}

pub fn mul(lhs: &Tensor, rhs: &Tensor) -> Tensor {
    // 1. Check shapes & Broadcast (Simplified: assume same shape for now)
    if lhs.shape() != rhs.shape() {
        // TODO: Broadcast
        panic!("Mul: shapes mismatch {:?} vs {:?}", lhs.shape(), rhs.shape());
    }

    let lhs_guard = lhs.data();
    let rhs_guard = rhs.data();
    let lhs_data = &*lhs_guard;
    let rhs_data = &*rhs_guard;
    
    let result_data: Vec<f32> = lhs_data.par_iter()
        .zip(rhs_data.par_iter())
        .map(|(a, b)| a * b)
        .collect();
        
    let storage = Storage::new(result_data);
    let mut tensor = Tensor::new_with_storage(storage, lhs.shape());
    
    if lhs.requires_grad() || rhs.requires_grad() {
        tensor.set_requires_grad_mut(true);
        tensor.set_op(Arc::new(MulBackward {
            lhs: lhs.clone(),
            rhs: rhs.clone(),
        }));
    }
    
    tensor
}

// --- Add ---
#[derive(Debug)]
pub struct AddBackward {
    pub lhs: Tensor,
    pub rhs: Tensor,
}

impl BackwardOp for AddBackward {
    fn backward(&self, grad: &Tensor) {
        if self.lhs.requires_grad() {
            let grad_lhs = if grad.shape() != self.lhs.shape() {
                // Simplified sum_to_shape (placeholder)
                // For now, assume broadcast happened and we need to reduce.
                grad.clone() 
            } else {
                grad.clone()
            };
            self.lhs.accumulate_grad(&grad_lhs);
            self.lhs.backward_step();
        }
        if self.rhs.requires_grad() {
            let grad_rhs = if grad.shape() != self.rhs.shape() {
                grad.clone()
            } else {
                grad.clone()
            };
            self.rhs.accumulate_grad(&grad_rhs);
            self.rhs.backward_step();
        }
    }
}

pub fn add(lhs: &Tensor, rhs: &Tensor) -> Tensor {
    // 1. Check shapes & Broadcast
    if lhs.shape() == rhs.shape() {
        // Fast path for same shape
        let lhs_guard = lhs.data();
        let rhs_guard = rhs.data();
        let lhs_data = &*lhs_guard;
        let rhs_data = &*rhs_guard;
        let result_data: Vec<f32> = lhs_data.par_iter()
            .zip(rhs_data.par_iter())
            .map(|(a, b)| a + b)
            .collect();
        let storage = Storage::new(result_data);
        let mut tensor = Tensor::new_with_storage(storage, lhs.shape());
        if lhs.requires_grad() || rhs.requires_grad() {
            tensor.set_requires_grad_mut(true);
            tensor.set_op(Arc::new(AddBackward {
                lhs: lhs.clone(),
                rhs: rhs.clone(),
            }));
        }
        return tensor;
    }

    // Broadcast
    let target_shape = crate::broadcast::broadcast_shapes(lhs.shape(), rhs.shape())
        .expect("Shapes not broadcastable");
    
    let lhs_expanded = lhs.expand(&target_shape);
    let rhs_expanded = rhs.expand(&target_shape);
    
    let lhs_guard = lhs_expanded.data();
    let rhs_guard = rhs_expanded.data();
    let lhs_data = &*lhs_guard;
    let rhs_data = &*rhs_guard;
    
    let result_data: Vec<f32> = lhs_data.par_iter()
        .zip(rhs_data.par_iter())
        .map(|(a, b)| a + b)
        .collect();
    
    let storage = Storage::new(result_data);
    let mut tensor = Tensor::new_with_storage(storage, &target_shape);
    
    if lhs.requires_grad() || rhs.requires_grad() {
        tensor.set_requires_grad_mut(true);
        tensor.set_op(Arc::new(AddBackward {
            lhs: lhs.clone(),
            rhs: rhs.clone(),
        }));
    }
    
    tensor
}

pub fn sub(lhs: &Tensor, rhs: &Tensor) -> Tensor {
    // Simplified: Assume same shape
    if lhs.shape() != rhs.shape() {
         panic!("Sub shape mismatch");
    }
    
    let lhs_guard = lhs.data();
    let rhs_guard = rhs.data();
    let lhs_data = &*lhs_guard;
    let rhs_data = &*rhs_guard;
    
    let result_data: Vec<f32> = lhs_data.par_iter().zip(rhs_data.par_iter()).map(|(a, b)| a - b).collect();
    let storage = Storage::new(result_data);
    let mut tensor = Tensor::new_with_storage(storage, lhs.shape());
    
    if lhs.requires_grad() || rhs.requires_grad() {
        tensor.set_requires_grad_mut(true);
        // TODO: SubBackward
    }
    
    tensor
}

// --- Neg ---
pub fn neg(input: &Tensor) -> Tensor {
    let input_guard = input.data();
    let input_data = &*input_guard;
    let result_data: Vec<f32> = input_data.par_iter().map(|x| -x).collect();
    let storage = Storage::new(result_data);
    let mut tensor = Tensor::new_with_storage(storage, input.shape());
    
    if input.requires_grad() {
        tensor.set_requires_grad_mut(true);
        // NegBackward is just -grad
    }
    tensor
}

// --- ReLU ---
pub fn relu(input: &Tensor) -> Tensor {
    let input_guard = input.data();
    let input_data = &*input_guard;
    let result_data: Vec<f32> = input_data.par_iter().map(|x| x.max(0.0)).collect();
    let storage = Storage::new(result_data);
    let mut tensor = Tensor::new_with_storage(storage, input.shape());
    
    if input.requires_grad() {
        tensor.set_requires_grad_mut(true);
        // ReluBackward
    }
    tensor
}

// --- Add ---
