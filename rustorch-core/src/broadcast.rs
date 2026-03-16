use crate::autograd::BackwardOp;
// use crate::storage::Storage;
use crate::Tensor;
use std::sync::Arc;

#[derive(Debug)]
pub struct ExpandBackward {
    pub input: Tensor,
    pub input_shape: Vec<usize>,
}

impl BackwardOp for ExpandBackward {
    fn backward(&self, grad: &Tensor) {
        if self.input.requires_grad() {
            // Gradient reduction: sum over broadcasted dimensions
            let grad_reduced = crate::ops::sum_to(grad, &self.input_shape);
            self.input.accumulate_grad(&grad_reduced);
            self.input.backward_step();
        }
    }
}

pub fn broadcast_shapes(shape1: &[usize], shape2: &[usize]) -> Option<Vec<usize>> {
    let len1 = shape1.len();
    let len2 = shape2.len();
    let max_len = std::cmp::max(len1, len2);

    let mut result_shape = vec![0; max_len];

    for i in 0..max_len {
        let dim1 = if i < len1 { shape1[len1 - 1 - i] } else { 1 };
        let dim2 = if i < len2 { shape2[len2 - 1 - i] } else { 1 };

        if dim1 == dim2 {
            result_shape[max_len - 1 - i] = dim1;
        } else if dim1 == 1 {
            result_shape[max_len - 1 - i] = dim2;
        } else if dim2 == 1 {
            result_shape[max_len - 1 - i] = dim1;
        } else {
            return None;
        }
    }

    Some(result_shape)
}

impl Tensor {
    // Lazy expansion: returns a view with modified strides (0 stride for broadcasted dims)
    pub fn expand(&self, new_shape: &[usize]) -> Tensor {
        if self.shape() == new_shape {
            return self.clone();
        }

        let current_shape = self.shape();
        let current_strides = self.strides();

        let ndim_new = new_shape.len();
        let ndim_old = current_shape.len();

        if ndim_new < ndim_old {
            panic!("expand: new shape must have >= dims than current shape");
        }

        let mut new_strides = vec![0; ndim_new];
        let offset = ndim_new - ndim_old;

        for i in 0..ndim_new {
            if i < offset {
                // New dimension added at front (broadcasting)
                // If new_shape[i] > 1, stride is 0.
                // If new_shape[i] == 1, stride is arbitrary (say 0).
                new_strides[i] = 0;
            } else {
                let old_idx = i - offset;
                let old_dim = current_shape[old_idx];
                let new_dim = new_shape[i];

                if old_dim == 1 && new_dim > 1 {
                    // Broadcast existing dim: stride 0
                    new_strides[i] = 0;
                } else if old_dim == new_dim {
                    // Inherit stride
                    new_strides[i] = current_strides[old_idx];
                } else {
                    panic!(
                        "expand: invalid shape {:?} -> {:?}",
                        current_shape, new_shape
                    );
                }
            }
        }

        // Return view
        // We construct a new Tensor sharing the same storage
        // This requires accessing private fields or using a constructor that takes strides.
        // Tensor::new_with_storage usually assumes contiguous.
        // We need `Tensor::new_with_storage_and_strides` or similar.
        // Or modify `new_with_storage`?
        // Let's check `tensor.rs`.

        // For now, I will assume I can create it.
        // I need to add `new_view` or similar to Tensor.
        // But wait, `Tensor::new_with_storage` computes strides from shape assuming contiguous.
        // I need to add a method to Tensor to create from storage + strides.

        // HACK: I cannot easily modify `Tensor` private fields from here if they are private to crate.
        // `broadcast.rs` is in `src/`, same crate. So I can access `pub(crate)` fields.
        // But `Tensor` struct definition is in `tensor.rs`.
        // `Tensor` wraps `Arc<TensorImpl>`. `TensorImpl` fields are `pub(crate)`.

        use crate::tensor::TensorImpl;
        use std::sync::Mutex;

        let inner = TensorImpl {
            storage: self.storage().clone(),
            shape: new_shape.to_vec(),
            strides: new_strides,
            grad: Mutex::new(None),
            requires_grad: self.requires_grad(),
            op: if self.requires_grad() {
                Some(Arc::new(ExpandBackward {
                    input: self.clone(),
                    input_shape: self.shape().to_vec(),
                }))
            } else {
                None
            },
            is_leaf: false, // Views are not leaf usually? Or if they share storage...
                            // If it has history, it's not leaf.
        };

        Tensor {
            inner: Arc::new(inner),
        }
    }
}
