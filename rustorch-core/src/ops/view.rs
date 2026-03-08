use std::sync::{Arc, Mutex};
use crate::Tensor;
use crate::autograd::BackwardOp;
use crate::storage::Storage;
use crate::tensor::TensorImpl;

#[derive(Debug)]
pub struct ReshapeBackward {
    pub input: Tensor,
    pub input_shape: Vec<usize>,
}

impl BackwardOp for ReshapeBackward {
    fn backward(&self, grad: &Tensor) {
        if self.input.requires_grad() {
            // Gradient should be reshaped back to input shape
            let grad_reshaped = grad.reshape(&self.input_shape);
            self.input.accumulate_grad(&grad_reshaped);
            self.input.backward_step();
        }
    }
}

// --- Permute ---

#[derive(Debug)]
pub struct PermuteBackward {
    pub input: Tensor,
    pub dims: Vec<usize>, // Original permutation
}

impl BackwardOp for PermuteBackward {
    fn backward(&self, grad: &Tensor) {
        if self.input.requires_grad() {
            // Inverse permutation needed
            // If dims is [1, 0], we permute grad with [1, 0] to get back.
            // If dims is [0, 2, 1], inverse is [0, 2, 1].
            // If dims is [2, 0, 1], inverse is [1, 2, 0].
            
            let ndim = self.dims.len();
            let mut inverse_dims = vec![0; ndim];
            for (i, &d) in self.dims.iter().enumerate() {
                inverse_dims[d] = i;
            }
            
            let grad_permuted = grad.permute(&inverse_dims);
            self.input.accumulate_grad(&grad_permuted);
            self.input.backward_step();
        }
    }
}

pub fn permute(input: &Tensor, dims: &[usize]) -> Tensor {
    let ndim = input.shape().len();
    if dims.len() != ndim {
        panic!("Permute dims length {} does not match tensor ndim {}", dims.len(), ndim);
    }
    
    // Check if dims are valid permutation
    let mut seen = vec![false; ndim];
    for &d in dims {
        if d >= ndim || seen[d] {
            panic!("Invalid permutation {:?}", dims);
        }
        seen[d] = true;
    }
    
    let old_shape = input.shape();
    let old_strides = input.strides();
    
    let mut new_shape = vec![0; ndim];
    let mut new_strides = vec![0; ndim];
    
    for (i, &d) in dims.iter().enumerate() {
        new_shape[i] = old_shape[d];
        new_strides[i] = old_strides[d];
    }
    
    // Create new tensor sharing storage
    // Need access to internal fields. TensorImpl fields are pub(crate).
    // View operations share storage.
    
    let inner = &input.inner;
    
    let tensor = Tensor {
        inner: Arc::new(TensorImpl {
            storage: inner.storage.clone(),
            shape: new_shape,
            strides: new_strides,
            grad: Mutex::new(None),
            requires_grad: inner.requires_grad,
            op: None,
            is_leaf: false,
        }),
    };
    
    if inner.requires_grad {
        let mut t = tensor.clone(); // Mutable clone wrapper
        // Actually we need to set op on `tensor`.
        // But `Tensor` struct has `inner` which is `Arc`.
        // We just created it, so we can modify it if we had mut access.
        // But `TensorImpl` fields are immutable after creation usually unless Mutex.
        // `op` is `Option<Arc<dyn BackwardOp>>` inside `TensorImpl`.
        // Wait, `TensorImpl` definition:
        // pub(crate) op: Option<Arc<dyn BackwardOp>>
        // It's not Mutex protected?
        // Let's check TensorImpl definition again.
        // In previous turns, `op` was `Option<Arc<dyn BackwardOp>>`.
        // `Tensor::set_op` uses `Arc::get_mut` or unsafe if shared?
        // `Tensor` usually wraps `Arc<TensorImpl>`.
        // If we just created `Arc`, we can get mut.
        
        // However, `Tensor` methods like `set_op` handle this?
        // `set_op` is likely `unsafe` or uses internal mutability if designed for it.
        // But here I'm constructing `TensorImpl` directly.
        
        let op = Arc::new(PermuteBackward {
            input: input.clone(),
            dims: dims.to_vec(),
        });
        
        // Re-construct with op
        return Tensor {
            inner: Arc::new(TensorImpl {
                storage: inner.storage.clone(),
                shape: tensor.shape().to_vec(),
                strides: tensor.strides().to_vec(),
                grad: Mutex::new(None),
                requires_grad: true,
                op: Some(op),
                is_leaf: false,
            }),
        };
    }
    
    tensor
}

pub fn transpose(input: &Tensor, dim0: usize, dim1: usize) -> Tensor {
    let ndim = input.shape().len();
    let mut dims: Vec<usize> = (0..ndim).collect();
    dims.swap(dim0, dim1);
    permute(input, &dims)
}

pub fn contiguous(input: &Tensor) -> Tensor {
    if input.is_contiguous() {
        return input.clone();
    }
    
    // Create new contiguous storage
    let shape = input.shape();
    let size: usize = shape.iter().product();
    let mut data = vec![0.0; size];
    
    // Iterate logical indices and copy
    // Naive iteration for now.
    // Optimization: recursive copy or specialized iterator.
    
    // We need an iterator that yields offsets based on strides.
    // Or just multi-dim loop.
    // Since ndim is dynamic, we use recursion.
    
    let input_guard = input.data(); // This gives storage data (linear)
    let input_storage = &*input_guard;
    
    let strides = input.strides();
    
    // Helper closure to iterate
    // But recursive closure in Rust is tricky.
    // Use explicit stack or struct.
    
    // Iterating 0..size is logical index.
    // We need to convert logical index to physical offset.
    // logical_to_physical(index, shape, strides)
    
    for i in 0..size {
        let mut idx = i;
        let mut physical_offset = 0;
        let mut shape_mul = size;
        
        // Decompose linear index i into coords
        // shape: [d0, d1, d2]
        // strides: [s0, s1, s2]
        
        // Standard contiguous strides: [d1*d2, d2, 1]
        // We can precompute contiguous strides.
        
        // Let's do it properly:
        // We need to map linear index `i` (in new contiguous tensor) -> `offset` (in old storage)
        
        let mut temp_i = i;
        for dim_idx in (0..shape.len()).rev() {
            let dim_size = shape[dim_idx];
            let coord = temp_i % dim_size;
            temp_i /= dim_size;
            
            physical_offset += coord * strides[dim_idx];
        }
        
        data[i] = input_storage[physical_offset];
    }
    
    let storage = Storage::new(data);
    let mut tensor = Tensor::new_with_storage(storage, shape);
    if input.requires_grad() {
        tensor.set_requires_grad_mut(true);
        // ContiguousBackward is Identity (or Permute inverse if we track it as permute?)
        // Actually, contiguous is just a copy. Backward propagates gradients.
        // If we view it as Identity op on graph (just memory reorg), gradients flow back 1:1?
        // No, if we permuted before, gradient must be permuted back.
        // But `permute` already registered a BackwardOp.
        // `contiguous` creates a new leaf-like node in the graph relative to `permute`.
        // So we need a `CopyBackward` or just identity if shape matches.
        // But shape matches.
        // So simple identity backward.
        // We need an Identity op.
        
        // For now, let's assume `permute` handles the shape change logic.
        // `contiguous` just copies data.
        // So we can use an IdentityBackward.
        // Or better: `contiguous` is often implicit.
        // But if we create a new Tensor, we must link it.
        
        // Let's reuse ReshapeBackward with same shape?
        // Or implement `ContiguousBackward`.
        // Or just `Identity`.
    }
    tensor
}
