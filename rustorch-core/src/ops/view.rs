use crate::autograd::BackwardOp;
use crate::storage::Storage;
use crate::tensor::TensorImpl;
use crate::Tensor;
use std::sync::{Arc, Mutex};

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
        panic!(
            "Permute dims length {} does not match tensor ndim {}",
            dims.len(),
            ndim
        );
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

    let mut tensor = Tensor {
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

    if input.requires_grad() {
        tensor.set_op(Arc::new(PermuteBackward {
            input: input.clone(),
            dims: dims.to_vec(),
        }));
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

    #[cfg(feature = "wgpu_backend")]
    {
        if input.storage().device().is_wgpu() {
            if let Some(input_buf) = input.storage().wgpu_buffer() {
                let output_buf = crate::backend::wgpu::contiguous_wgpu(
                    input_buf,
                    input.shape(),
                    input.strides(),
                );

                let size: usize = input.shape().iter().product();
                let storage = Storage::new_wgpu(output_buf, size, 0);
                let mut tensor = Tensor::new_with_storage(storage, input.shape());
                if input.requires_grad() {
                    tensor.set_requires_grad_mut(true);
                    tensor.set_op(Arc::new(ContiguousBackward {
                        input: input.clone(),
                    }));
                }
                return tensor;
            }
        }
    }

    let shape = input.shape();
    let size: usize = shape.iter().product();
    let mut data = vec![0.0; size];

    let input_guard = input.data();
    let input_storage = &*input_guard;
    let strides = input.strides();
    let storage_len = input_storage.len();

    for (i, val) in data.iter_mut().enumerate().take(size) {
        let mut physical_offset = 0;
        let mut temp_i = i;
        for dim_idx in (0..shape.len()).rev() {
            let dim_size = shape[dim_idx];
            let coord = temp_i % dim_size;
            temp_i /= dim_size;
            physical_offset += coord * strides[dim_idx];
        }
        if storage_len == 1 {
            *val = input_storage[0];
        } else if physical_offset < storage_len {
            *val = input_storage[physical_offset];
        } else {
            *val = 0.0;
        }
    }

    let storage = Storage::new(data);
    let mut tensor = Tensor::new_with_storage(storage, shape);
    if input.requires_grad() {
        tensor.set_requires_grad_mut(true);
        tensor.set_op(Arc::new(ContiguousBackward {
            input: input.clone(),
        }));
    }
    tensor
}

#[derive(Debug)]
pub struct ContiguousBackward {
    pub input: Tensor,
}

impl BackwardOp for ContiguousBackward {
    fn backward(&self, grad: &Tensor) {
        if self.input.requires_grad() {
            let grad_contig = if grad.is_contiguous() {
                grad.clone()
            } else {
                grad.contiguous()
            };
            let grad_view = grad_contig.reshape(self.input.shape());

            let mut grad_input = if self.input.is_contiguous() {
                grad_view
            } else {
                let mut data = vec![0.0; self.input.shape().iter().product()];
                let strides = self.input.strides();
                let shape = self.input.shape();

                let grad_guard = grad_view.data();
                let grad_data = &*grad_guard;

                for (i, &g) in grad_data.iter().enumerate() {
                    let mut physical_offset = 0;
                    let mut temp_i = i;
                    for dim_idx in (0..shape.len()).rev() {
                        let dim_size = shape[dim_idx];
                        let coord = temp_i % dim_size;
                        temp_i /= dim_size;
                        physical_offset += coord * strides[dim_idx];
                    }
                    data[physical_offset] = g;
                }

                Tensor::new_with_storage(Storage::new(data), shape)
            };

            grad_input.set_requires_grad_mut(true);
            self.input.accumulate_grad(&grad_input);
            self.input.backward_step();
        }
    }
}

pub fn sum_to(input: &Tensor, shape: &[usize]) -> Tensor {
    if input.shape() == shape {
        return input.clone();
    }

    #[cfg(feature = "wgpu_backend")]
    {
        if input.storage().device().is_wgpu() {
            let input_shape = input.shape();
            let input_ndim = input_shape.len();
            let output_ndim = shape.len();

            if output_ndim == 0 || (output_ndim == 1 && shape[0] == 1) {
                let input_contig = if input.is_contiguous() {
                    input.clone()
                } else {
                    input.contiguous()
                };
                if let Some(input_buf) = input_contig.storage().wgpu_buffer() {
                    let total_size: usize = input_contig.shape().iter().product();
                    let output_buf =
                        crate::backend::wgpu::reduce_sum_all_wgpu(input_buf, total_size);
                    let storage = Storage::new_wgpu(output_buf, 1, 0);
                    return Tensor::new_with_storage(storage, shape);
                }
            }

            if input_ndim == 2 && output_ndim == 1 && input_shape[1] == shape[0] {
                let input_contig = if input.is_contiguous() {
                    input.clone()
                } else {
                    input.contiguous()
                };
                if let Some(input_buf) = input_contig.storage().wgpu_buffer() {
                    let output_buf =
                        crate::backend::wgpu::reduce_sum_dim0_wgpu(input_buf, input_contig.shape());
                    let size: usize = shape.iter().product();
                    let storage = Storage::new_wgpu(output_buf, size, 0);
                    return Tensor::new_with_storage(storage, shape);
                }
            }

            if input_ndim == 2 && output_ndim == 1 && input_shape[0] == shape[0] {
                let input_contig = if input.is_contiguous() {
                    input.clone()
                } else {
                    input.contiguous()
                };
                if let Some(input_buf) = input_contig.storage().wgpu_buffer() {
                    let output_buf = crate::backend::wgpu::reduce_sum_dim_wgpu(
                        input_buf,
                        input_contig.shape(),
                        1,
                    );
                    let size: usize = shape.iter().product();
                    let storage = Storage::new_wgpu(output_buf, size, 0);
                    return Tensor::new_with_storage(storage, shape);
                }
            }
        }
    }

    let input_contig = if input.is_contiguous() {
        input.clone()
    } else {
        input.contiguous()
    };
    let input_shape = input_contig.shape();
    let output_shape = shape;

    if input_shape.len() == 2 && output_shape.len() == 1 {
        if input_shape[1] == output_shape[0] {
            let m = input_shape[0];
            let n = input_shape[1];
            let data = input_contig.data();
            let mut result = vec![0.0; n];

            for (j, out) in result.iter_mut().enumerate().take(n) {
                let mut col = vec![0.0f32; m];
                for i in 0..m {
                    col[i] = data[i * n + j];
                }
                *out = crate::ops::sum_auto(&col);
            }

            return Tensor::new_with_storage(Storage::new(result), output_shape);
        }
    }

    if input_shape.len() == 1 && output_shape.len() == 1 {
        if input_shape[0] == output_shape[0] {
            return input_contig.clone();
        }
    }

    if output_shape.iter().product::<usize>() == 1 {
        let data = input_contig.data();
        let sum = crate::ops::sum_auto(&data);
        return Tensor::new_with_storage(Storage::new(vec![sum]), output_shape);
    }

    panic!(
        "General sum_to not implemented for shape {:?} -> {:?}",
        input_shape, output_shape
    );
}
