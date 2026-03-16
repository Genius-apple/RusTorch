use crate::autograd::BackwardOp;
use crate::storage::Storage;
use crate::Tensor;
use parking_lot::Mutex;
use rayon::prelude::*;
use std::collections::HashMap;
use std::hint::black_box;
use std::sync::{Arc, OnceLock};
use std::time::Instant;
use wide::f32x8;

pub mod activations;
pub mod conv;
pub mod embedding;
pub mod norm;
pub mod pool;
pub mod view;

pub use activations::{sigmoid, softmax, tanh};
pub use conv::conv2d;
pub use embedding::embedding;
pub use norm::{batch_norm2d, layer_norm};
pub use pool::max_pool2d;
pub use view::ReshapeBackward;

#[derive(Debug)]
pub struct MulBackward {
    pub lhs: Tensor,
    pub rhs: Tensor,
}

impl BackwardOp for MulBackward {
    fn backward(&self, grad: &Tensor) {
        let same_input = Arc::ptr_eq(&self.lhs.inner, &self.rhs.inner);
        if same_input && self.lhs.requires_grad() {
            let mut grad_lhs = crate::ops::mul(grad, &self.rhs);
            let mut grad_rhs = crate::ops::mul(grad, &self.lhs);
            if grad_lhs.shape() != self.lhs.shape() {
                grad_lhs = sum_to(&grad_lhs, self.lhs.shape());
            }
            if grad_rhs.shape() != self.lhs.shape() {
                grad_rhs = sum_to(&grad_rhs, self.lhs.shape());
            }
            let grad_total = add(&grad_lhs, &grad_rhs);
            self.lhs.accumulate_grad(&grad_total);
            self.lhs.backward_step();
            return;
        }

        if self.lhs.requires_grad() {
            let mut grad_lhs = crate::ops::mul(grad, &self.rhs);
            if grad_lhs.shape() != self.lhs.shape() {
                grad_lhs = sum_to(&grad_lhs, self.lhs.shape());
            }
            self.lhs.accumulate_grad(&grad_lhs);
            self.lhs.backward_step();
        }
        if self.rhs.requires_grad() {
            let mut grad_rhs = crate::ops::mul(grad, &self.lhs);
            if grad_rhs.shape() != self.rhs.shape() {
                grad_rhs = sum_to(&grad_rhs, self.rhs.shape());
            }
            self.rhs.accumulate_grad(&grad_rhs);
            self.rhs.backward_step();
        }
    }
}

pub fn mul(lhs: &Tensor, rhs: &Tensor) -> Tensor {
    #[cfg(feature = "wgpu_backend")]
    {
        if let (Some(lhs_buf), Some(rhs_buf)) =
            (lhs.storage().wgpu_buffer(), rhs.storage().wgpu_buffer())
        {
            let target_shape = crate::broadcast::broadcast_shapes(lhs.shape(), rhs.shape())
                .expect("Shapes not broadcastable");

            use crate::backend::wgpu::{elementwise_wgpu_buffer, ElementwiseOp};
            let output_buf = elementwise_wgpu_buffer(
                lhs_buf,
                lhs.shape(),
                lhs.strides(),
                Some((rhs_buf, rhs.shape(), rhs.strides())),
                &target_shape,
                ElementwiseOp::Mul,
                None,
            );

            let size: usize = target_shape.iter().product();
            let storage = Storage::new_wgpu(output_buf, size, 0);
            let mut tensor = Tensor::new_with_storage(storage, &target_shape);

            if lhs.requires_grad() || rhs.requires_grad() {
                tensor.set_requires_grad_mut(true);
                tensor.set_op(Arc::new(MulBackward {
                    lhs: lhs.clone(),
                    rhs: rhs.clone(),
                }));
            }
            return tensor;
        }
    }

    if lhs.shape() != rhs.shape() {
        // CPU Broadcast (simplified)
        let target_shape = crate::broadcast::broadcast_shapes(lhs.shape(), rhs.shape())
            .expect("Shapes not broadcastable");
        let lhs_expanded = lhs.expand(&target_shape);
        let rhs_expanded = rhs.expand(&target_shape);
        return mul(&lhs_expanded, &rhs_expanded);
    }

    let lhs_contig = if lhs.is_contiguous() {
        lhs.clone()
    } else {
        lhs.contiguous()
    };
    let rhs_contig = if rhs.is_contiguous() {
        rhs.clone()
    } else {
        rhs.contiguous()
    };

    let lhs_guard = lhs_contig.data();
    let rhs_guard = rhs_contig.data();
    let lhs_data = &*lhs_guard;
    let rhs_data = &*rhs_guard;

    let result_data = elemwise_auto(lhs_data, rhs_data, ElemwiseKind::Mul);

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

pub fn div(lhs: &Tensor, rhs: &Tensor) -> Tensor {
    // Basic Div
    if lhs.shape() != rhs.shape() {
        let target_shape = crate::broadcast::broadcast_shapes(lhs.shape(), rhs.shape())
            .expect("Shapes not broadcastable");
        let lhs_expanded = lhs.expand(&target_shape);
        let rhs_expanded = rhs.expand(&target_shape);
        return div(&lhs_expanded, &rhs_expanded);
    }

    let lhs_contig = if lhs.is_contiguous() {
        lhs.clone()
    } else {
        lhs.contiguous()
    };
    let rhs_contig = if rhs.is_contiguous() {
        rhs.clone()
    } else {
        rhs.contiguous()
    };

    let lhs_guard = lhs_contig.data();
    let rhs_guard = rhs_contig.data();
    let result_data: Vec<f32> = lhs_guard
        .par_iter()
        .zip(rhs_guard.par_iter())
        .map(|(a, b)| a / b)
        .collect();
    let storage = Storage::new(result_data);
    let tensor = Tensor::new_with_storage(storage, lhs.shape());

    // DivBackward... (omitted for brevity unless needed)
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
        let same_input = Arc::ptr_eq(&self.lhs.inner, &self.rhs.inner);
        if same_input && self.lhs.requires_grad() {
            let grad_lhs = if grad.shape() != self.lhs.shape() {
                sum_to(grad, self.lhs.shape())
            } else {
                grad.clone()
            };
            let grad_total = add(&grad_lhs, &grad_lhs);
            self.lhs.accumulate_grad(&grad_total);
            self.lhs.backward_step();
            return;
        }

        if self.lhs.requires_grad() {
            let grad_lhs = if grad.shape() != self.lhs.shape() {
                sum_to(grad, self.lhs.shape())
            } else {
                grad.clone()
            };
            self.lhs.accumulate_grad(&grad_lhs);
            self.lhs.backward_step();
        }
        if self.rhs.requires_grad() {
            let grad_rhs = if grad.shape() != self.rhs.shape() {
                sum_to(grad, self.rhs.shape())
            } else {
                grad.clone()
            };
            self.rhs.accumulate_grad(&grad_rhs);
            self.rhs.backward_step();
        }
    }
}

pub fn add(lhs: &Tensor, rhs: &Tensor) -> Tensor {
    #[cfg(feature = "wgpu_backend")]
    {
        if let (Some(lhs_buf), Some(rhs_buf)) =
            (lhs.storage().wgpu_buffer(), rhs.storage().wgpu_buffer())
        {
            let target_shape = crate::broadcast::broadcast_shapes(lhs.shape(), rhs.shape())
                .expect("Shapes not broadcastable");

            use crate::backend::wgpu::{elementwise_wgpu_buffer, ElementwiseOp};
            let output_buf = elementwise_wgpu_buffer(
                lhs_buf,
                lhs.shape(),
                lhs.strides(),
                Some((rhs_buf, rhs.shape(), rhs.strides())),
                &target_shape,
                ElementwiseOp::Add,
                None,
            );

            let size: usize = target_shape.iter().product();
            let storage = Storage::new_wgpu(output_buf, size, 0);
            let mut tensor = Tensor::new_with_storage(storage, &target_shape);

            if lhs.requires_grad() || rhs.requires_grad() {
                tensor.set_requires_grad_mut(true);
                tensor.set_op(Arc::new(AddBackward {
                    lhs: lhs.clone(),
                    rhs: rhs.clone(),
                }));
            }
            return tensor;
        }
    }

    if lhs.shape() != rhs.shape() {
        let target_shape = crate::broadcast::broadcast_shapes(lhs.shape(), rhs.shape())
            .expect("Shapes not broadcastable");
        let lhs_expanded = lhs.expand(&target_shape);
        let rhs_expanded = rhs.expand(&target_shape);
        return add(&lhs_expanded, &rhs_expanded);
    }

    let lhs_contig = if lhs.is_contiguous() {
        lhs.clone()
    } else {
        lhs.contiguous()
    };
    let rhs_contig = if rhs.is_contiguous() {
        rhs.clone()
    } else {
        rhs.contiguous()
    };

    let lhs_guard = lhs_contig.data();
    let rhs_guard = rhs_contig.data();
    let result_data = elemwise_auto(&lhs_guard, &rhs_guard, ElemwiseKind::Add);
    let storage = Storage::new(result_data);
    let mut tensor = Tensor::new_with_storage(storage, lhs.shape());
    if lhs.requires_grad() || rhs.requires_grad() {
        tensor.set_requires_grad_mut(true);
        tensor.set_op(Arc::new(AddBackward {
            lhs: lhs.clone(),
            rhs: rhs.clone(),
        }));
    }
    tensor
}

#[derive(Debug)]
pub struct SubBackward {
    pub lhs: Tensor,
    pub rhs: Tensor,
}

impl BackwardOp for SubBackward {
    fn backward(&self, grad: &Tensor) {
        let same_input = Arc::ptr_eq(&self.lhs.inner, &self.rhs.inner);
        if same_input && self.lhs.requires_grad() {
            return;
        }

        if self.lhs.requires_grad() {
            let mut grad_lhs = grad.clone();
            if grad_lhs.shape() != self.lhs.shape() {
                grad_lhs = sum_to(&grad_lhs, self.lhs.shape());
            }
            self.lhs.accumulate_grad(&grad_lhs);
            self.lhs.backward_step();
        }
        if self.rhs.requires_grad() {
            let mut grad_rhs = neg(grad);
            if grad_rhs.shape() != self.rhs.shape() {
                grad_rhs = sum_to(&grad_rhs, self.rhs.shape());
            }
            self.rhs.accumulate_grad(&grad_rhs);
            self.rhs.backward_step();
        }
    }
}

pub fn sub(lhs: &Tensor, rhs: &Tensor) -> Tensor {
    #[cfg(feature = "wgpu_backend")]
    {
        if let (Some(lhs_buf), Some(rhs_buf)) =
            (lhs.storage().wgpu_buffer(), rhs.storage().wgpu_buffer())
        {
            let target_shape = crate::broadcast::broadcast_shapes(lhs.shape(), rhs.shape())
                .expect("Shapes not broadcastable");

            use crate::backend::wgpu::{elementwise_wgpu_buffer, ElementwiseOp};
            let output_buf = elementwise_wgpu_buffer(
                lhs_buf,
                lhs.shape(),
                lhs.strides(),
                Some((rhs_buf, rhs.shape(), rhs.strides())),
                &target_shape,
                ElementwiseOp::Sub,
                None,
            );

            let size: usize = target_shape.iter().product();
            let storage = Storage::new_wgpu(output_buf, size, 0);
            let mut tensor = Tensor::new_with_storage(storage, &target_shape);

            if lhs.requires_grad() || rhs.requires_grad() {
                tensor.set_requires_grad_mut(true);
                tensor.set_op(Arc::new(SubBackward {
                    lhs: lhs.clone(),
                    rhs: rhs.clone(),
                }));
            }
            return tensor;
        }
    }

    if lhs.shape() != rhs.shape() {
        let target_shape = crate::broadcast::broadcast_shapes(lhs.shape(), rhs.shape())
            .expect("Shapes not broadcastable");
        let lhs_expanded = lhs.expand(&target_shape);
        let rhs_expanded = rhs.expand(&target_shape);
        return sub(&lhs_expanded, &rhs_expanded);
    }

    let lhs_contig = if lhs.is_contiguous() {
        lhs.clone()
    } else {
        lhs.contiguous()
    };
    let rhs_contig = if rhs.is_contiguous() {
        rhs.clone()
    } else {
        rhs.contiguous()
    };
    let lhs_guard = lhs_contig.data();
    let rhs_guard = rhs_contig.data();
    let result_data = elemwise_auto(&lhs_guard, &rhs_guard, ElemwiseKind::Sub);
    let storage = Storage::new(result_data);
    let mut tensor = Tensor::new_with_storage(storage, lhs.shape());

    if lhs.requires_grad() || rhs.requires_grad() {
        tensor.set_requires_grad_mut(true);
        tensor.set_op(Arc::new(SubBackward {
            lhs: lhs.clone(),
            rhs: rhs.clone(),
        }));
    }
    tensor
}

pub fn neg(input: &Tensor) -> Tensor {
    let input_guard = input.data();
    let result_data: Vec<f32> = input_guard.par_iter().map(|x| -x).collect();
    let storage = Storage::new(result_data);
    Tensor::new_with_storage(storage, input.shape())
}

#[derive(Debug)]
pub struct ReluBackward {
    pub input: Tensor,
    pub output: Tensor,
}

impl BackwardOp for ReluBackward {
    fn backward(&self, grad: &Tensor) {
        if self.input.requires_grad() {
            #[cfg(feature = "wgpu_backend")]
            {
                if let (Some(out_buf), Some(grad_buf)) = (
                    self.output.storage().wgpu_buffer(),
                    grad.storage().wgpu_buffer(),
                ) {
                    use crate::backend::wgpu::{elementwise_wgpu_buffer, ElementwiseOp};
                    let grad_input_buf = elementwise_wgpu_buffer(
                        out_buf,
                        self.output.shape(),
                        self.output.strides(),
                        Some((grad_buf, grad.shape(), grad.strides())),
                        grad.shape(),
                        ElementwiseOp::ReLUBackward,
                        None,
                    );
                    let size: usize = grad.shape().iter().product();
                    let storage = Storage::new_wgpu(grad_input_buf, size, 0);
                    let grad_input = Tensor::new_with_storage(storage, grad.shape());
                    self.input.accumulate_grad(&grad_input);
                    self.input.backward_step();
                    return;
                }
            }

            let input_guard = self.input.data();
            let grad_guard = grad.data();
            let grad_input: Vec<f32> = input_guard
                .par_iter()
                .zip(grad_guard.par_iter())
                .map(|(x, g)| if *x > 0.0 { *g } else { 0.0 })
                .collect();
            let storage = Storage::new(grad_input);
            let grad_input_tensor = Tensor::new_with_storage(storage, grad.shape());
            self.input.accumulate_grad(&grad_input_tensor);
            self.input.backward_step();
        }
    }
}

pub fn relu(input: &Tensor) -> Tensor {
    #[cfg(feature = "wgpu_backend")]
    {
        if let Some(buf) = input.storage().wgpu_buffer() {
            use crate::backend::wgpu::{elementwise_wgpu_buffer, ElementwiseOp};
            let output_buf = elementwise_wgpu_buffer(
                buf,
                input.shape(),
                input.strides(),
                None,
                input.shape(),
                ElementwiseOp::ReLU,
                None,
            );
            let size: usize = input.shape().iter().product();
            let storage = Storage::new_wgpu(output_buf, size, 0);
            let mut tensor = Tensor::new_with_storage(storage, input.shape());

            if input.requires_grad() {
                tensor.set_requires_grad_mut(true);
                tensor.set_op(Arc::new(ReluBackward {
                    input: input.clone(),
                    output: tensor.detach(),
                }));
            }
            return tensor;
        }
    }

    let input_guard = input.data();
    let result_data: Vec<f32> = input_guard.par_iter().map(|x| x.max(0.0)).collect();
    let storage = Storage::new(result_data);
    let mut tensor = Tensor::new_with_storage(storage, input.shape());

    if input.requires_grad() {
        tensor.set_requires_grad_mut(true);
        tensor.set_op(Arc::new(ReluBackward {
            input: input.clone(),
            output: tensor.detach(),
        }));
    }
    tensor
}

pub fn sgd_step(param: &Tensor, grad: &Tensor, lr: f32) -> Tensor {
    // param - lr * grad
    #[cfg(feature = "wgpu_backend")]
    {
        if let (Some(p_buf), Some(g_buf)) =
            (param.storage().wgpu_buffer(), grad.storage().wgpu_buffer())
        {
            use crate::backend::wgpu::{elementwise_wgpu_buffer, ElementwiseOp};
            let output_buf = elementwise_wgpu_buffer(
                p_buf,
                param.shape(),
                param.strides(),
                Some((g_buf, grad.shape(), grad.strides())),
                param.shape(),
                ElementwiseOp::SGDStep,
                Some(lr),
            );
            let size: usize = param.shape().iter().product();
            let storage = Storage::new_wgpu(output_buf, size, 0);
            return Tensor::new_with_storage(storage, param.shape());
        }
    }

    // CPU
    let p_data = param.data();
    let g_data = grad.data();
    let res_data: Vec<f32> = p_data
        .par_iter()
        .zip(g_data.par_iter())
        .map(|(p, g)| p - lr * g)
        .collect();
    let storage = Storage::new(res_data);
    Tensor::new_with_storage(storage, param.shape())
}

#[derive(Debug)]
pub struct MatmulBackward {
    pub lhs: Tensor,
    pub rhs: Tensor,
}

impl BackwardOp for MatmulBackward {
    fn backward(&self, grad: &Tensor) {
        #[cfg(feature = "wgpu_backend")]
        {
            let grad_is_wgpu = grad.storage().wgpu_buffer().is_some();
            let lhs_is_wgpu = self.lhs.storage().wgpu_buffer().is_some();
            let rhs_is_wgpu = self.rhs.storage().wgpu_buffer().is_some();

            if grad_is_wgpu && lhs_is_wgpu && rhs_is_wgpu {
                if self.lhs.requires_grad() {
                    let rhs_t = self.rhs.t();
                    let grad_lhs = matmul(grad, &rhs_t).detach();
                    self.lhs.accumulate_grad(&grad_lhs);
                    self.lhs.backward_step();
                }
                if self.rhs.requires_grad() {
                    let grad_rhs = matmul(&self.lhs.t(), grad).detach();
                    self.rhs.accumulate_grad(&grad_rhs);
                    self.rhs.backward_step();
                }
                return;
            }
        }

        if self.lhs.requires_grad() {
            let rhs_t = self.rhs.t();
            let grad_lhs = matmul(grad, &rhs_t);
            self.lhs.accumulate_grad(&grad_lhs);
            self.lhs.backward_step();
        }
        if self.rhs.requires_grad() {
            let grad_rhs = matmul(&self.lhs.t(), grad);
            self.rhs.accumulate_grad(&grad_rhs);
            self.rhs.backward_step();
        }
    }
}

#[cfg(feature = "wgpu_backend")]
#[allow(dead_code)]
fn matmul_gpu_aware_no_grad(lhs: &Tensor, rhs: &Tensor) -> Tensor {
    let lhs_shape = lhs.shape();
    let rhs_shape = rhs.shape();

    if lhs_shape.len() != 2 || rhs_shape.len() != 2 {
        panic!("Matmul only supports 2D");
    }

    let m = lhs_shape[0];
    let k = lhs_shape[1];
    let k2 = rhs_shape[0];
    let n = rhs_shape[1];

    if k != k2 {
        panic!("Matmul dimension mismatch");
    }

    if let (Some(_), Some(_)) = (lhs.storage().wgpu_buffer(), rhs.storage().wgpu_buffer()) {
        let lhs_contig = if lhs.is_contiguous() {
            lhs.clone()
        } else {
            lhs.contiguous()
        };
        let rhs_contig = if rhs.is_contiguous() {
            rhs.clone()
        } else {
            rhs.contiguous()
        };

        if lhs_contig.storage().wgpu_buffer().is_none()
            || rhs_contig.storage().wgpu_buffer().is_none()
        {
            return matmul(&lhs_contig, &rhs_contig);
        }

        use crate::backend::wgpu::{matmul_wgpu_buffer, Activation};
        let output_buf = matmul_wgpu_buffer(
            lhs_contig.storage().wgpu_buffer().unwrap(),
            lhs_contig.shape(),
            rhs_contig.storage().wgpu_buffer().unwrap(),
            rhs_contig.shape(),
            Activation::None,
        );

        let storage = Storage::new_wgpu(output_buf, m * n, 0);
        return Tensor::new_with_storage(storage, &[m, n]);
    }

    matmul(lhs, rhs)
}

pub fn sum_to(tensor: &Tensor, shape: &[usize]) -> Tensor {
    if tensor.shape() == shape {
        return tensor.clone();
    }
    view::sum_to(tensor, shape)
}

#[derive(Debug)]
pub struct SumBackward {
    pub input: Tensor,
}

impl BackwardOp for SumBackward {
    fn backward(&self, grad: &Tensor) {
        if self.input.requires_grad() {
            #[cfg(feature = "wgpu_backend")]
            let grad_cpu = if grad.storage().device().is_wgpu() {
                grad.to_cpu()
            } else {
                grad.clone()
            };
            #[cfg(not(feature = "wgpu_backend"))]
            let grad_cpu = grad.clone();

            let grad_val = grad_cpu.data()[0];
            let mut grad_input = Tensor::full(self.input.shape(), grad_val);
            #[cfg(feature = "wgpu_backend")]
            if self.input.storage().device().is_wgpu() {
                grad_input = grad_input.to_wgpu();
            }
            self.input.accumulate_grad(&grad_input);
            self.input.backward_step();
        }
    }
}

pub fn sum(tensor: &Tensor) -> Tensor {
    let total_size: usize = tensor.shape().iter().product();

    let mut output = {
        #[cfg(feature = "wgpu_backend")]
        {
            if let Some(input_buf) = tensor.storage().wgpu_buffer() {
                let output_buf = crate::backend::wgpu::reduce_sum_all_wgpu(input_buf, total_size);
                let storage = Storage::new_wgpu(output_buf, 1, 0);
                Tensor::new_with_storage(storage, &[])
            } else {
                let data = tensor.data();
                let sum_val = sum_auto(&data);
                Tensor::new_with_storage(Storage::new(vec![sum_val]), &[])
            }
        }
        #[cfg(not(feature = "wgpu_backend"))]
        {
            let data = tensor.data();
            let sum_val = sum_auto(&data);
            Tensor::new_with_storage(Storage::new(vec![sum_val]), &[])
        }
    };

    if tensor.requires_grad() {
        output.set_requires_grad_mut(true);
        output.set_op(Arc::new(SumBackward {
            input: tensor.clone(),
        }));
    }

    output
}

#[derive(Debug)]
pub struct MeanBackward {
    pub input: Tensor,
}

impl BackwardOp for MeanBackward {
    fn backward(&self, grad: &Tensor) {
        if self.input.requires_grad() {
            #[cfg(feature = "wgpu_backend")]
            let grad_cpu = if grad.storage().device().is_wgpu() {
                grad.to_cpu()
            } else {
                grad.clone()
            };
            #[cfg(not(feature = "wgpu_backend"))]
            let grad_cpu = grad.clone();

            let grad_val = grad_cpu.data()[0];
            let numel = self.input.shape().iter().product::<usize>() as f32;
            let mut grad_input = Tensor::full(self.input.shape(), grad_val / numel);
            #[cfg(feature = "wgpu_backend")]
            if self.input.storage().device().is_wgpu() {
                grad_input = grad_input.to_wgpu();
            }
            self.input.accumulate_grad(&grad_input);
            self.input.backward_step();
        }
    }
}

pub fn mean(tensor: &Tensor) -> Tensor {
    let t_cpu = {
        #[cfg(feature = "wgpu_backend")]
        {
            if tensor.storage().device().is_wgpu() {
                tensor.to_cpu()
            } else {
                tensor.clone()
            }
        }
        #[cfg(not(feature = "wgpu_backend"))]
        {
            tensor.clone()
        }
    };
    let data = t_cpu.data();
    let numel = data.len() as f32;
    let mean_val = sum_auto(&data) / numel;
    let mut out = Tensor::new_with_storage(Storage::new(vec![mean_val]), &[]);
    if tensor.requires_grad() {
        out.set_requires_grad_mut(true);
        out.set_op(Arc::new(MeanBackward {
            input: tensor.clone(),
        }));
    }
    out
}

#[derive(Debug)]
pub struct VarBackward {
    pub input: Tensor,
    pub mean: f32,
}

impl BackwardOp for VarBackward {
    fn backward(&self, grad: &Tensor) {
        if self.input.requires_grad() {
            #[cfg(feature = "wgpu_backend")]
            let grad_cpu = if grad.storage().device().is_wgpu() {
                grad.to_cpu()
            } else {
                grad.clone()
            };
            #[cfg(not(feature = "wgpu_backend"))]
            let grad_cpu = grad.clone();
            let grad_val = grad_cpu.data()[0];
            let input_cpu = {
                #[cfg(feature = "wgpu_backend")]
                {
                    if self.input.storage().device().is_wgpu() {
                        self.input.to_cpu()
                    } else {
                        self.input.clone()
                    }
                }
                #[cfg(not(feature = "wgpu_backend"))]
                {
                    self.input.clone()
                }
            };
            let input_data = input_cpu.data();
            let numel = input_data.len() as f32;
            let scale = grad_val * 2.0 / numel;
            let grad_data: Vec<f32> = input_data.iter().map(|x| (x - self.mean) * scale).collect();
            let mut grad_input = Tensor::new(&grad_data, self.input.shape());
            #[cfg(feature = "wgpu_backend")]
            if self.input.storage().device().is_wgpu() {
                grad_input = grad_input.to_wgpu();
            }
            self.input.accumulate_grad(&grad_input);
            self.input.backward_step();
        }
    }
}

pub fn var(tensor: &Tensor) -> Tensor {
    let t_cpu = {
        #[cfg(feature = "wgpu_backend")]
        {
            if tensor.storage().device().is_wgpu() {
                tensor.to_cpu()
            } else {
                tensor.clone()
            }
        }
        #[cfg(not(feature = "wgpu_backend"))]
        {
            tensor.clone()
        }
    };
    let data = t_cpu.data();
    let numel = data.len() as f32;
    let m = sum_auto(&data) / numel;
    let sq: Vec<f32> = data.iter().map(|x| (x - m) * (x - m)).collect();
    let v = sum_auto(&sq) / numel;
    let mut out = Tensor::new_with_storage(Storage::new(vec![v]), &[]);
    if tensor.requires_grad() {
        out.set_requires_grad_mut(true);
        out.set_op(Arc::new(VarBackward {
            input: tensor.clone(),
            mean: m,
        }));
    }
    out
}

pub fn linear_mse_grads(input: &Tensor, output: &Tensor, target: &Tensor) -> (f32, Tensor, Tensor) {
    let x = if input.is_contiguous() {
        input.clone()
    } else {
        input.contiguous()
    };
    let y = if output.is_contiguous() {
        output.clone()
    } else {
        output.contiguous()
    };
    let t = if target.is_contiguous() {
        target.clone()
    } else {
        target.contiguous()
    };

    let x_shape = x.shape();
    let y_shape = y.shape();
    let t_shape = t.shape();
    if x_shape.len() != 2 || y_shape.len() != 2 || t_shape.len() != 2 {
        panic!("linear_mse_grads expects 2D tensors");
    }
    if y_shape != t_shape {
        panic!("linear_mse_grads output and target shape mismatch");
    }
    if x_shape[0] != y_shape[0] {
        panic!("linear_mse_grads batch mismatch");
    }

    let batch = x_shape[0];
    let in_dim = x_shape[1];
    let out_dim = y_shape[1];
    let numel = (batch * out_dim) as f32;
    let grad_scale = 2.0 / numel;

    let x_data = x.data();
    let y_data = y.data();
    let t_data = t.data();

    let mut grad_w = vec![0.0f32; out_dim * in_dim];
    let mut grad_b = vec![0.0f32; out_dim];
    let mut loss = 0.0f32;

    for b in 0..batch {
        for (o, gb) in grad_b.iter_mut().enumerate().take(out_dim) {
            let idx = b * out_dim + o;
            let d = y_data[idx] - t_data[idx];
            loss += d * d;
            let go = d * grad_scale;
            *gb += go;
            let w_row_offset = o * in_dim;
            let x_row_offset = b * in_dim;
            for i in 0..in_dim {
                grad_w[w_row_offset + i] += go * x_data[x_row_offset + i];
            }
        }
    }

    let loss = loss / numel;
    let grad_w_t = Tensor::new(&grad_w, &[out_dim, in_dim]);
    let grad_b_t = Tensor::new(&grad_b, &[out_dim]);
    (loss, grad_w_t, grad_b_t)
}

#[derive(Debug)]
pub struct FusedMatmulBackward {
    pub lhs: Tensor,
    pub rhs: Tensor,
    pub bias: Option<Tensor>,
    pub output: Tensor,
    pub activation: crate::backend::Activation,
}

impl BackwardOp for FusedMatmulBackward {
    fn backward(&self, grad_output: &Tensor) {
        let grad_pre_act = match self.activation {
            crate::backend::Activation::ReLU => {
                #[cfg(feature = "wgpu_backend")]
                {
                    if let (Some(out_buf), Some(grad_buf)) = (
                        self.output.storage().wgpu_buffer(),
                        grad_output.storage().wgpu_buffer(),
                    ) {
                        // ... (existing code)
                        use crate::backend::wgpu::{elementwise_wgpu_buffer, ElementwiseOp};
                        let target_shape = grad_output.shape();
                        let out_buf = elementwise_wgpu_buffer(
                            out_buf,
                            self.output.shape(),
                            self.output.strides(),
                            Some((grad_buf, grad_output.shape(), grad_output.strides())),
                            target_shape,
                            ElementwiseOp::ReLUBackward,
                            None,
                        );
                        let storage = Storage::new_wgpu(out_buf, target_shape.iter().product(), 0);
                        Tensor::new_with_storage(storage, target_shape)
                    } else {
                        grad_output.clone()
                    }
                }
                #[cfg(not(feature = "wgpu_backend"))]
                grad_output.clone()
            }
            crate::backend::Activation::Sigmoid => {
                #[cfg(feature = "wgpu_backend")]
                {
                    if let (Some(out_buf), Some(grad_buf)) = (
                        self.output.storage().wgpu_buffer(),
                        grad_output.storage().wgpu_buffer(),
                    ) {
                        use crate::backend::wgpu::{elementwise_wgpu_buffer, ElementwiseOp};
                        let target_shape = grad_output.shape();
                        let out_buf = elementwise_wgpu_buffer(
                            out_buf,
                            self.output.shape(),
                            self.output.strides(),
                            Some((grad_buf, grad_output.shape(), grad_output.strides())),
                            target_shape,
                            ElementwiseOp::SigmoidBackward,
                            None,
                        );
                        let storage = Storage::new_wgpu(out_buf, target_shape.iter().product(), 0);
                        Tensor::new_with_storage(storage, target_shape)
                    } else {
                        grad_output.clone()
                    }
                }
                #[cfg(not(feature = "wgpu_backend"))]
                grad_output.clone()
            }
            crate::backend::Activation::Tanh => {
                #[cfg(feature = "wgpu_backend")]
                {
                    if let (Some(out_buf), Some(grad_buf)) = (
                        self.output.storage().wgpu_buffer(),
                        grad_output.storage().wgpu_buffer(),
                    ) {
                        use crate::backend::wgpu::{elementwise_wgpu_buffer, ElementwiseOp};
                        let target_shape = grad_output.shape();
                        let out_buf = elementwise_wgpu_buffer(
                            out_buf,
                            self.output.shape(),
                            self.output.strides(),
                            Some((grad_buf, grad_output.shape(), grad_output.strides())),
                            target_shape,
                            ElementwiseOp::TanhBackward,
                            None,
                        );
                        let storage = Storage::new_wgpu(out_buf, target_shape.iter().product(), 0);
                        Tensor::new_with_storage(storage, target_shape)
                    } else {
                        grad_output.clone()
                    }
                }
                #[cfg(not(feature = "wgpu_backend"))]
                grad_output.clone()
            }
            crate::backend::Activation::None => grad_output.clone(),
        };

        if let Some(bias) = &self.bias {
            if bias.requires_grad() {
                let grad_bias = sum_to(&grad_pre_act, bias.shape());
                bias.accumulate_grad(&grad_bias);
                bias.backward_step();
            }
        }

        if self.lhs.requires_grad() {
            let rhs_t = self.rhs.t();
            let grad_lhs = matmul(&grad_pre_act, &rhs_t);
            self.lhs.accumulate_grad(&grad_lhs);
            self.lhs.backward_step();
        }

        if self.rhs.requires_grad() {
            let grad_rhs = matmul(&self.lhs.t(), &grad_pre_act);
            self.rhs.accumulate_grad(&grad_rhs);
            self.rhs.backward_step();
        }
    }
}

#[inline]
fn parse_usize_env(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(default)
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum CpuMatmulStrategy {
    Auto,
    Profile,
    Sgemm,
    Parallel,
}

#[derive(Clone, Copy)]
struct CpuMatmulConfig {
    strategy: CpuMatmulStrategy,
    min_m: usize,
    min_k: usize,
    max_n: usize,
    profile_iters: usize,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum CpuKernelChoice {
    Sgemm,
    Parallel,
}

type MatmulPerfKey = (usize, usize, usize, bool);
type MatmulPerfCache = HashMap<MatmulPerfKey, CpuKernelChoice>;

fn cpu_matmul_config() -> CpuMatmulConfig {
    static CFG: OnceLock<CpuMatmulConfig> = OnceLock::new();
    *CFG.get_or_init(|| {
        let strategy = match std::env::var("RUSTORCH_CPU_MATMUL_STRATEGY")
            .unwrap_or_else(|_| "auto".to_string())
            .to_ascii_lowercase()
            .as_str()
        {
            "parallel" => CpuMatmulStrategy::Parallel,
            "sgemm" => CpuMatmulStrategy::Sgemm,
            "profile" => CpuMatmulStrategy::Profile,
            _ => CpuMatmulStrategy::Auto,
        };

        CpuMatmulConfig {
            strategy,
            min_m: parse_usize_env("RUSTORCH_CPU_MATMUL_MIN_M", 128),
            min_k: parse_usize_env("RUSTORCH_CPU_MATMUL_MIN_K", 256),
            max_n: parse_usize_env("RUSTORCH_CPU_MATMUL_MAX_N", 128),
            profile_iters: parse_usize_env("RUSTORCH_CPU_MATMUL_PROFILE_ITERS", 2),
        }
    })
}

#[inline]
fn should_use_parallel_auto(m: usize, k: usize, n: usize) -> bool {
    let cfg = cpu_matmul_config();
    m >= cfg.min_m && k >= cfg.min_k && n <= cfg.max_n
}

fn matmul_profile_cache() -> &'static Mutex<MatmulPerfCache> {
    static CACHE: OnceLock<Mutex<MatmulPerfCache>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

fn matmul_cpu_sgemm_core(
    lhs_data: &[f32],
    rhs_data: &[f32],
    m: usize,
    k: usize,
    n: usize,
    lhs_stride0: isize,
    lhs_stride1: isize,
    rhs_stride0: isize,
    rhs_stride1: isize,
    bias: Option<&[f32]>,
) -> Vec<f32> {
    let mut out = vec![0.0f32; m * n];
    unsafe {
        matrixmultiply::sgemm(
            m,
            k,
            n,
            1.0,
            lhs_data.as_ptr(),
            lhs_stride0,
            lhs_stride1,
            rhs_data.as_ptr(),
            rhs_stride0,
            rhs_stride1,
            0.0,
            out.as_mut_ptr(),
            n as isize,
            1,
        );
    }
    if let Some(bias_data) = bias {
        out.par_chunks_mut(n).for_each(|row| {
            row.iter_mut()
                .zip(bias_data.iter())
                .for_each(|(v, b)| *v += *b);
        });
    }
    out
}

fn bench_kernel<F: Fn() -> Vec<f32>>(f: F, iters: usize) -> u128 {
    let mut total_ns = 0u128;
    let mut acc = 0.0f32;
    for _ in 0..iters {
        let t0 = Instant::now();
        let out = f();
        total_ns += t0.elapsed().as_nanos();
        if let Some(v) = out.first() {
            acc += *v;
        }
        black_box(acc);
    }
    total_ns
}

fn choose_cpu_kernel(
    m: usize,
    k: usize,
    n: usize,
    has_bias: bool,
    lhs_data: &[f32],
    rhs_data: &[f32],
    lhs_stride0: isize,
    lhs_stride1: isize,
    rhs_stride0: isize,
    rhs_stride1: isize,
    bias: Option<&[f32]>,
) -> CpuKernelChoice {
    let cfg = cpu_matmul_config();
    match cfg.strategy {
        CpuMatmulStrategy::Parallel => CpuKernelChoice::Parallel,
        CpuMatmulStrategy::Sgemm => CpuKernelChoice::Sgemm,
        CpuMatmulStrategy::Auto => {
            if should_use_parallel_auto(m, k, n) {
                CpuKernelChoice::Parallel
            } else {
                CpuKernelChoice::Sgemm
            }
        }
        CpuMatmulStrategy::Profile => {
            let key = (m, k, n, has_bias);
            if let Some(cached) = matmul_profile_cache().lock().get(&key).copied() {
                return cached;
            }

            let iters = cfg.profile_iters.max(1);
            let sgemm_ns = bench_kernel(
                || {
                    matmul_cpu_sgemm_core(
                        lhs_data,
                        rhs_data,
                        m,
                        k,
                        n,
                        lhs_stride0,
                        lhs_stride1,
                        rhs_stride0,
                        rhs_stride1,
                        bias,
                    )
                },
                iters,
            );
            let parallel_ns = bench_kernel(
                || matmul_cpu_parallel_core(lhs_data, rhs_data, m, k, n, bias),
                iters,
            );
            let choice = if parallel_ns < sgemm_ns {
                CpuKernelChoice::Parallel
            } else {
                CpuKernelChoice::Sgemm
            };
            matmul_profile_cache().lock().insert(key, choice);
            choice
        }
    }
}

fn matmul_cpu_parallel_core(
    lhs_data: &[f32],
    rhs_data: &[f32],
    m: usize,
    k: usize,
    n: usize,
    bias: Option<&[f32]>,
) -> Vec<f32> {
    let mut result = vec![0.0f32; m * n];
    result.par_chunks_mut(n).enumerate().for_each(|(i, row)| {
        let lhs_row = &lhs_data[i * k..(i + 1) * k];
        for j in 0..n {
            let mut sum = bias.map_or(0.0, |b| b[j]);
            let mut p = 0usize;
            while p + 8 <= k {
                sum += lhs_row[p] * rhs_data[p * n + j];
                sum += lhs_row[p + 1] * rhs_data[(p + 1) * n + j];
                sum += lhs_row[p + 2] * rhs_data[(p + 2) * n + j];
                sum += lhs_row[p + 3] * rhs_data[(p + 3) * n + j];
                sum += lhs_row[p + 4] * rhs_data[(p + 4) * n + j];
                sum += lhs_row[p + 5] * rhs_data[(p + 5) * n + j];
                sum += lhs_row[p + 6] * rhs_data[(p + 6) * n + j];
                sum += lhs_row[p + 7] * rhs_data[(p + 7) * n + j];
                p += 8;
            }
            while p < k {
                sum += lhs_row[p] * rhs_data[p * n + j];
                p += 1;
            }
            row[j] = sum;
        }
    });
    result
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum ElemwiseKind {
    Add,
    Sub,
    Mul,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum CpuElemwiseStrategy {
    Auto,
    Profile,
    Scalar,
    Simd,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum ElemwiseKernelChoice {
    Scalar,
    Simd,
}

#[derive(Clone, Copy)]
struct CpuElemwiseConfig {
    strategy: CpuElemwiseStrategy,
    min_len: usize,
    profile_iters: usize,
}

type ElemwisePerfKey = (usize, ElemwiseKind);
type ElemwisePerfCache = HashMap<ElemwisePerfKey, ElemwiseKernelChoice>;

fn cpu_elemwise_config() -> CpuElemwiseConfig {
    static CFG: OnceLock<CpuElemwiseConfig> = OnceLock::new();
    *CFG.get_or_init(|| {
        let strategy = match std::env::var("RUSTORCH_CPU_ELEMWISE_STRATEGY")
            .unwrap_or_else(|_| "auto".to_string())
            .to_ascii_lowercase()
            .as_str()
        {
            "simd" => CpuElemwiseStrategy::Simd,
            "scalar" => CpuElemwiseStrategy::Scalar,
            "profile" => CpuElemwiseStrategy::Profile,
            _ => CpuElemwiseStrategy::Auto,
        };
        CpuElemwiseConfig {
            strategy,
            min_len: parse_usize_env("RUSTORCH_CPU_ELEMWISE_MIN_LEN", 2048),
            profile_iters: parse_usize_env("RUSTORCH_CPU_ELEMWISE_PROFILE_ITERS", 2),
        }
    })
}

fn elemwise_profile_cache() -> &'static Mutex<ElemwisePerfCache> {
    static CACHE: OnceLock<Mutex<ElemwisePerfCache>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

#[inline]
fn apply_elemwise_scalar(a: f32, b: f32, kind: ElemwiseKind) -> f32 {
    match kind {
        ElemwiseKind::Add => a + b,
        ElemwiseKind::Sub => a - b,
        ElemwiseKind::Mul => a * b,
    }
}

#[inline]
fn apply_elemwise_simd(a: f32x8, b: f32x8, kind: ElemwiseKind) -> f32x8 {
    match kind {
        ElemwiseKind::Add => a + b,
        ElemwiseKind::Sub => a - b,
        ElemwiseKind::Mul => a * b,
    }
}

fn elemwise_scalar(lhs: &[f32], rhs: &[f32], kind: ElemwiseKind) -> Vec<f32> {
    lhs.par_iter()
        .zip(rhs.par_iter())
        .map(|(a, b)| apply_elemwise_scalar(*a, *b, kind))
        .collect()
}

fn elemwise_simd(lhs: &[f32], rhs: &[f32], kind: ElemwiseKind) -> Vec<f32> {
    let len = lhs.len();
    let mut out = vec![0.0f32; len];
    let vec_len = len / 8 * 8;
    let lanes = 8usize;

    out[..vec_len]
        .par_chunks_mut(1024)
        .enumerate()
        .for_each(|(chunk_idx, out_chunk)| {
            let chunk_start = chunk_idx * 1024;
            let lhs_chunk = &lhs[chunk_start..chunk_start + out_chunk.len()];
            let rhs_chunk = &rhs[chunk_start..chunk_start + out_chunk.len()];
            let mut i = 0usize;
            while i + lanes <= out_chunk.len() {
                let mut la = [0.0f32; 8];
                let mut lb = [0.0f32; 8];
                la.copy_from_slice(&lhs_chunk[i..i + lanes]);
                lb.copy_from_slice(&rhs_chunk[i..i + lanes]);
                let va = f32x8::from(la);
                let vb = f32x8::from(lb);
                let vc = apply_elemwise_simd(va, vb, kind);
                let oc: [f32; 8] = vc.into();
                out_chunk[i..i + lanes].copy_from_slice(&oc);
                i += lanes;
            }
            while i < out_chunk.len() {
                out_chunk[i] = apply_elemwise_scalar(lhs_chunk[i], rhs_chunk[i], kind);
                i += 1;
            }
        });

    for i in vec_len..len {
        out[i] = apply_elemwise_scalar(lhs[i], rhs[i], kind);
    }
    out
}

fn choose_elemwise_kernel(
    len: usize,
    kind: ElemwiseKind,
    lhs: &[f32],
    rhs: &[f32],
) -> ElemwiseKernelChoice {
    let cfg = cpu_elemwise_config();
    match cfg.strategy {
        CpuElemwiseStrategy::Simd => ElemwiseKernelChoice::Simd,
        CpuElemwiseStrategy::Scalar => ElemwiseKernelChoice::Scalar,
        CpuElemwiseStrategy::Auto => {
            if len >= cfg.min_len {
                ElemwiseKernelChoice::Simd
            } else {
                ElemwiseKernelChoice::Scalar
            }
        }
        CpuElemwiseStrategy::Profile => {
            let key = (len, kind);
            if let Some(cached) = elemwise_profile_cache().lock().get(&key).copied() {
                return cached;
            }
            let iters = cfg.profile_iters.max(1);
            let scalar_ns = {
                let mut total = 0u128;
                for _ in 0..iters {
                    let t0 = Instant::now();
                    let out = elemwise_scalar(lhs, rhs, kind);
                    black_box(out.len());
                    total += t0.elapsed().as_nanos();
                }
                total
            };
            let simd_ns = {
                let mut total = 0u128;
                for _ in 0..iters {
                    let t0 = Instant::now();
                    let out = elemwise_simd(lhs, rhs, kind);
                    black_box(out.len());
                    total += t0.elapsed().as_nanos();
                }
                total
            };
            let choice = if simd_ns < scalar_ns {
                ElemwiseKernelChoice::Simd
            } else {
                ElemwiseKernelChoice::Scalar
            };
            elemwise_profile_cache().lock().insert(key, choice);
            choice
        }
    }
}

fn elemwise_auto(lhs: &[f32], rhs: &[f32], kind: ElemwiseKind) -> Vec<f32> {
    match choose_elemwise_kernel(lhs.len(), kind, lhs, rhs) {
        ElemwiseKernelChoice::Simd => elemwise_simd(lhs, rhs, kind),
        ElemwiseKernelChoice::Scalar => elemwise_scalar(lhs, rhs, kind),
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum CpuReductionStrategy {
    Auto,
    Profile,
    Scalar,
    Simd,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum ReductionKernelChoice {
    Scalar,
    Simd,
}

#[derive(Clone, Copy)]
struct CpuReductionConfig {
    strategy: CpuReductionStrategy,
    min_len: usize,
    profile_iters: usize,
}

fn cpu_reduction_config() -> CpuReductionConfig {
    static CFG: OnceLock<CpuReductionConfig> = OnceLock::new();
    *CFG.get_or_init(|| {
        let strategy = match std::env::var("RUSTORCH_CPU_REDUCTION_STRATEGY")
            .unwrap_or_else(|_| "auto".to_string())
            .to_ascii_lowercase()
            .as_str()
        {
            "simd" => CpuReductionStrategy::Simd,
            "scalar" => CpuReductionStrategy::Scalar,
            "profile" => CpuReductionStrategy::Profile,
            _ => CpuReductionStrategy::Auto,
        };
        CpuReductionConfig {
            strategy,
            min_len: parse_usize_env("RUSTORCH_CPU_REDUCTION_MIN_LEN", 4096),
            profile_iters: parse_usize_env("RUSTORCH_CPU_REDUCTION_PROFILE_ITERS", 2),
        }
    })
}

fn reduction_profile_cache() -> &'static Mutex<HashMap<usize, ReductionKernelChoice>> {
    static CACHE: OnceLock<Mutex<HashMap<usize, ReductionKernelChoice>>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

fn sum_scalar(data: &[f32]) -> f32 {
    data.par_iter().copied().sum()
}

fn sum_simd_chunk(chunk: &[f32]) -> f32 {
    let lanes = 8usize;
    let vec_len = chunk.len() / lanes * lanes;
    let mut acc = f32x8::from([0.0; 8]);
    let mut i = 0usize;
    while i < vec_len {
        let mut v = [0.0f32; 8];
        v.copy_from_slice(&chunk[i..i + lanes]);
        acc += f32x8::from(v);
        i += lanes;
    }
    let a: [f32; 8] = acc.into();
    let mut s = a.iter().sum::<f32>();
    while i < chunk.len() {
        s += chunk[i];
        i += 1;
    }
    s
}

fn sum_simd(data: &[f32]) -> f32 {
    data.par_chunks(4096).map(sum_simd_chunk).sum()
}

fn choose_reduction_kernel(len: usize, data: &[f32]) -> ReductionKernelChoice {
    let cfg = cpu_reduction_config();
    match cfg.strategy {
        CpuReductionStrategy::Simd => ReductionKernelChoice::Simd,
        CpuReductionStrategy::Scalar => ReductionKernelChoice::Scalar,
        CpuReductionStrategy::Auto => {
            if len >= cfg.min_len {
                ReductionKernelChoice::Simd
            } else {
                ReductionKernelChoice::Scalar
            }
        }
        CpuReductionStrategy::Profile => {
            if let Some(cached) = reduction_profile_cache().lock().get(&len).copied() {
                return cached;
            }
            let iters = cfg.profile_iters.max(1);
            let mut scalar_ns = 0u128;
            let mut simd_ns = 0u128;
            for _ in 0..iters {
                let t0 = Instant::now();
                let s = sum_scalar(data);
                scalar_ns += t0.elapsed().as_nanos();
                black_box(s);

                let t1 = Instant::now();
                let v = sum_simd(data);
                simd_ns += t1.elapsed().as_nanos();
                black_box(v);
            }
            let choice = if simd_ns < scalar_ns {
                ReductionKernelChoice::Simd
            } else {
                ReductionKernelChoice::Scalar
            };
            reduction_profile_cache().lock().insert(len, choice);
            choice
        }
    }
}

pub(crate) fn sum_auto(data: &[f32]) -> f32 {
    match choose_reduction_kernel(data.len(), data) {
        ReductionKernelChoice::Simd => sum_simd(data),
        ReductionKernelChoice::Scalar => sum_scalar(data),
    }
}

pub fn matmul_fused(
    lhs: &Tensor,
    rhs: &Tensor,
    bias: Option<&Tensor>,
    activation: crate::backend::Activation,
) -> Tensor {
    #[cfg(feature = "wgpu_backend")]
    {
        if let (Some(lhs_buf), Some(rhs_buf)) =
            (lhs.storage().wgpu_buffer(), rhs.storage().wgpu_buffer())
        {
            let m = lhs.shape()[0];
            let _k = lhs.shape()[1];
            let n = rhs.shape()[1];

            let bias_data =
                bias.and_then(|b| b.storage().wgpu_buffer().map(|buf| (buf, b.shape())));

            use crate::backend::wgpu::matmul_fused_wgpu_buffer;
            let output_buf = matmul_fused_wgpu_buffer(
                lhs_buf,
                lhs.shape(),
                rhs_buf,
                rhs.shape(),
                bias_data,
                activation,
            );

            let storage = Storage::new_wgpu(output_buf, m * n, 0);
            let mut tensor = Tensor::new_with_storage(storage, &[m, n]);

            if lhs.requires_grad()
                || rhs.requires_grad()
                || bias.map_or(false, |b| b.requires_grad())
            {
                tensor.set_requires_grad_mut(true);
                tensor.set_op(Arc::new(FusedMatmulBackward {
                    lhs: lhs.clone(),
                    rhs: rhs.clone(),
                    bias: bias.cloned(),
                    output: tensor.detach(),
                    activation,
                }));
            }
            return tensor;
        }
    }

    if matches!(activation, crate::backend::Activation::None) {
        let lhs_shape = lhs.shape();
        let rhs_shape = rhs.shape();
        if lhs_shape.len() == 2 && rhs_shape.len() == 2 && lhs_shape[1] == rhs_shape[0] {
            let m = lhs_shape[0];
            let k = lhs_shape[1];
            let n = rhs_shape[1];

            let lhs_contig = if lhs.is_contiguous() {
                lhs.clone()
            } else {
                lhs.contiguous()
            };
            let rhs_contig = if rhs.is_contiguous() {
                rhs.clone()
            } else {
                rhs.contiguous()
            };

            #[cfg(feature = "wgpu_backend")]
            let (lhs_contig, rhs_contig) = {
                let l = if lhs_contig.storage().device().is_wgpu() {
                    lhs_contig.to_cpu()
                } else {
                    lhs_contig
                };
                let r = if rhs_contig.storage().device().is_wgpu() {
                    rhs_contig.to_cpu()
                } else {
                    rhs_contig
                };
                (l, r)
            };

            let lhs_guard = lhs_contig.data();
            let rhs_guard = rhs_contig.data();
            let lhs_data = &*lhs_guard;
            let rhs_data = &*rhs_guard;

            let bias_vec = bias.and_then(|b| {
                if b.shape().len() == 1 && b.shape()[0] == n {
                    let b_cpu = {
                        #[cfg(feature = "wgpu_backend")]
                        {
                            if b.storage().device().is_wgpu() {
                                b.to_cpu()
                            } else {
                                b.clone()
                            }
                        }
                        #[cfg(not(feature = "wgpu_backend"))]
                        {
                            b.clone()
                        }
                    };
                    let bg = b_cpu.data();
                    Some(bg.to_vec())
                } else {
                    None
                }
            });
            let bias_slice = bias_vec.as_deref();

            let lhs_s0 = lhs_contig.strides()[0] as isize;
            let lhs_s1 = lhs_contig.strides()[1] as isize;
            let rhs_s0 = rhs_contig.strides()[0] as isize;
            let rhs_s1 = rhs_contig.strides()[1] as isize;
            let kernel = choose_cpu_kernel(
                m,
                k,
                n,
                bias_slice.is_some(),
                lhs_data,
                rhs_data,
                lhs_s0,
                lhs_s1,
                rhs_s0,
                rhs_s1,
                bias_slice,
            );
            let result_data = match kernel {
                CpuKernelChoice::Parallel => {
                    matmul_cpu_parallel_core(lhs_data, rhs_data, m, k, n, bias_slice)
                }
                CpuKernelChoice::Sgemm => matmul_cpu_sgemm_core(
                    lhs_data, rhs_data, m, k, n, lhs_s0, lhs_s1, rhs_s0, rhs_s1, bias_slice,
                ),
            };

            let storage = Storage::new(result_data);
            let mut tensor = Tensor::new_with_storage(storage, &[m, n]);
            if lhs.requires_grad()
                || rhs.requires_grad()
                || bias.map_or(false, |b| b.requires_grad())
            {
                tensor.set_requires_grad_mut(true);
                tensor.set_op(Arc::new(FusedMatmulBackward {
                    lhs: lhs.clone(),
                    rhs: rhs.clone(),
                    bias: bias.cloned(),
                    output: tensor.detach(),
                    activation,
                }));
            }
            return tensor;
        }
    }

    bump_pipeline_stat("staged");
    let mut out = lhs.matmul(rhs);
    if let Some(b) = bias {
        out = out.add(b);
    }
    match activation {
        crate::backend::Activation::ReLU => out.relu(),
        crate::backend::Activation::Sigmoid => crate::ops::activations::sigmoid(&out),
        crate::backend::Activation::Tanh => crate::ops::activations::tanh(&out),
        crate::backend::Activation::None => out,
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum FusedPipelineStrategy {
    Auto,
    Profile,
    Staged,
    Fused,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum FusedPipelineChoice {
    Staged,
    Fused,
}

#[derive(Clone, Copy)]
struct FusedPipelineConfig {
    strategy: FusedPipelineStrategy,
    profile_iters: usize,
}

type FusedPipelineKey = (usize, usize, usize, bool, bool, i32);

fn fused_pipeline_config() -> FusedPipelineConfig {
    static CFG: OnceLock<FusedPipelineConfig> = OnceLock::new();
    *CFG.get_or_init(|| {
        let strategy = match std::env::var("RUSTORCH_FUSED_PIPELINE_STRATEGY")
            .unwrap_or_else(|_| "auto".to_string())
            .to_ascii_lowercase()
            .as_str()
        {
            "fused" => FusedPipelineStrategy::Fused,
            "staged" => FusedPipelineStrategy::Staged,
            "profile" => FusedPipelineStrategy::Profile,
            _ => FusedPipelineStrategy::Auto,
        };
        FusedPipelineConfig {
            strategy,
            profile_iters: parse_usize_env("RUSTORCH_FUSED_PIPELINE_PROFILE_ITERS", 1),
        }
    })
}

fn fused_pipeline_cache() -> &'static Mutex<HashMap<FusedPipelineKey, FusedPipelineChoice>> {
    static CACHE: OnceLock<Mutex<HashMap<FusedPipelineKey, FusedPipelineChoice>>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

fn fused_pipeline_stats() -> &'static Mutex<HashMap<String, u64>> {
    static STATS: OnceLock<Mutex<HashMap<String, u64>>> = OnceLock::new();
    STATS.get_or_init(|| Mutex::new(HashMap::new()))
}

fn bump_pipeline_stat(key: &str) {
    let mut s = fused_pipeline_stats().lock();
    *s.entry(key.to_string()).or_insert(0) += 1;
}

pub fn get_fused_pipeline_stats() -> HashMap<String, u64> {
    fused_pipeline_stats().lock().clone()
}

fn apply_activation(t: Tensor, activation: crate::backend::Activation) -> Tensor {
    match activation {
        crate::backend::Activation::ReLU => t.relu(),
        crate::backend::Activation::Sigmoid => crate::ops::activations::sigmoid(&t),
        crate::backend::Activation::Tanh => crate::ops::activations::tanh(&t),
        crate::backend::Activation::None => t,
    }
}

fn pipeline_staged(
    lhs: &Tensor,
    rhs: &Tensor,
    bias: Option<&Tensor>,
    norm_weight: Option<&Tensor>,
    norm_bias: Option<&Tensor>,
    eps: f32,
    activation: crate::backend::Activation,
) -> Tensor {
    let mut out = matmul(lhs, rhs);
    if let Some(b) = bias {
        out = add(&out, b);
    }
    let norm_shape = [rhs.shape()[1]];
    out = layer_norm(&out, &norm_shape, norm_weight, norm_bias, eps);
    apply_activation(out, activation)
}

fn pipeline_fused(
    lhs: &Tensor,
    rhs: &Tensor,
    bias: Option<&Tensor>,
    norm_weight: Option<&Tensor>,
    norm_bias: Option<&Tensor>,
    eps: f32,
    activation: crate::backend::Activation,
) -> Tensor {
    let out = matmul_fused(lhs, rhs, bias, crate::backend::Activation::None);
    let norm_shape = [rhs.shape()[1]];
    let out = layer_norm(&out, &norm_shape, norm_weight, norm_bias, eps);
    apply_activation(out, activation)
}

pub fn matmul_bias_norm_activation(
    lhs: &Tensor,
    rhs: &Tensor,
    bias: Option<&Tensor>,
    norm_weight: Option<&Tensor>,
    norm_bias: Option<&Tensor>,
    eps: f32,
    activation: crate::backend::Activation,
) -> Tensor {
    let m = lhs.shape()[0];
    let k = lhs.shape()[1];
    let n = rhs.shape()[1];
    let key: FusedPipelineKey = (
        m,
        k,
        n,
        norm_weight.is_some(),
        norm_bias.is_some(),
        activation as i32,
    );
    let cfg = fused_pipeline_config();
    let choice = match cfg.strategy {
        FusedPipelineStrategy::Fused => FusedPipelineChoice::Fused,
        FusedPipelineStrategy::Staged => FusedPipelineChoice::Staged,
        FusedPipelineStrategy::Auto => {
            if m >= 128 && k >= 128 && n >= 32 {
                FusedPipelineChoice::Fused
            } else {
                FusedPipelineChoice::Staged
            }
        }
        FusedPipelineStrategy::Profile => {
            if let Some(cached) = fused_pipeline_cache().lock().get(&key).copied() {
                cached
            } else {
                let iters = cfg.profile_iters.max(1);
                let mut staged_ns = 0u128;
                let mut fused_ns = 0u128;
                for _ in 0..iters {
                    let t0 = Instant::now();
                    let s =
                        pipeline_staged(lhs, rhs, bias, norm_weight, norm_bias, eps, activation);
                    staged_ns += t0.elapsed().as_nanos();
                    black_box(s.shape()[0]);

                    let t1 = Instant::now();
                    let f = pipeline_fused(lhs, rhs, bias, norm_weight, norm_bias, eps, activation);
                    fused_ns += t1.elapsed().as_nanos();
                    black_box(f.shape()[0]);
                }
                let c = if fused_ns < staged_ns {
                    FusedPipelineChoice::Fused
                } else {
                    FusedPipelineChoice::Staged
                };
                fused_pipeline_cache().lock().insert(key, c);
                c
            }
        }
    };

    match choice {
        FusedPipelineChoice::Fused => {
            bump_pipeline_stat("fused");
            pipeline_fused(lhs, rhs, bias, norm_weight, norm_bias, eps, activation)
        }
        FusedPipelineChoice::Staged => {
            bump_pipeline_stat("staged");
            pipeline_staged(lhs, rhs, bias, norm_weight, norm_bias, eps, activation)
        }
    }
}

pub fn matmul(lhs: &Tensor, rhs: &Tensor) -> Tensor {
    let lhs_shape = lhs.shape();
    let rhs_shape = rhs.shape();

    if lhs_shape.len() != 2 || rhs_shape.len() != 2 {
        panic!("Matmul only supports 2D");
    }

    let m = lhs_shape[0];
    let k = lhs_shape[1];
    let k2 = rhs_shape[0];
    let n = rhs_shape[1];

    if k != k2 {
        panic!("Matmul dimension mismatch");
    }

    #[cfg(feature = "wgpu_backend")]
    {
        // Ensure both tensors are on the same device
        let lhs_is_wgpu = lhs.storage().device().is_wgpu();
        let rhs_is_wgpu = rhs.storage().device().is_wgpu();

        let (lhs, rhs) = if lhs_is_wgpu && !rhs_is_wgpu {
            // LHS is GPU, RHS is CPU - move RHS to GPU
            (lhs.clone(), rhs.to_wgpu())
        } else if !lhs_is_wgpu && rhs_is_wgpu {
            // LHS is CPU, RHS is GPU - move LHS to GPU
            (lhs.to_wgpu(), rhs.clone())
        } else {
            (lhs.clone(), rhs.clone())
        };

        if let (Some(lhs_buf), Some(rhs_buf)) =
            (lhs.storage().wgpu_buffer(), rhs.storage().wgpu_buffer())
        {
            let lhs_strides = lhs.strides();
            let rhs_strides = rhs.strides();

            let lhs_is_contig = lhs.is_contiguous();
            let rhs_is_contig = rhs.is_contiguous();

            // Check if tensor is transposed: strides = [1, rows] means transposed from [cols, rows]
            // Original: shape [rows, cols], strides [cols, 1]
            // Transposed: shape [cols, rows], strides [1, cols]
            let lhs_is_transposed = !lhs_is_contig && lhs_strides[0] == 1;
            let rhs_is_transposed = !rhs_is_contig && rhs_strides[0] == 1;

            if lhs_is_contig && rhs_is_contig {
                use crate::backend::wgpu::{matmul_wgpu_buffer, Activation};
                let output_buf =
                    matmul_wgpu_buffer(lhs_buf, lhs_shape, rhs_buf, rhs_shape, Activation::None);

                let storage = Storage::new_wgpu(output_buf, m * n, 0);
                let mut tensor = Tensor::new_with_storage(storage, &[m, n]);

                if lhs.requires_grad() || rhs.requires_grad() {
                    tensor.set_requires_grad_mut(true);
                    tensor.set_op(Arc::new(MatmulBackward {
                        lhs: lhs.clone(),
                        rhs: rhs.clone(),
                    }));
                }
                return tensor;
            }

            if lhs_is_transposed || rhs_is_transposed {
                // Ensure previous GPU commands are executed before reading transposed data
                crate::backend::wgpu::flush_queue();

                // For transposed tensor, the data is stored in transposed layout
                // We need to make it contiguous OR adjust the matmul logic
                // Let's make it contiguous for now
                let lhs_contig = if lhs_is_transposed {
                    lhs.contiguous()
                } else {
                    lhs.clone()
                };
                let rhs_contig = if rhs_is_transposed {
                    rhs.contiguous()
                } else {
                    rhs.clone()
                };

                // Ensure contiguous commands are executed
                crate::backend::wgpu::flush_queue();

                if lhs_contig.storage().wgpu_buffer().is_some()
                    && rhs_contig.storage().wgpu_buffer().is_some()
                {
                    let lhs_buf = lhs_contig.storage().wgpu_buffer().unwrap();
                    let rhs_buf = rhs_contig.storage().wgpu_buffer().unwrap();

                    use crate::backend::wgpu::{matmul_wgpu_buffer, Activation};
                    let output_buf = matmul_wgpu_buffer(
                        lhs_buf,
                        lhs_contig.shape(),
                        rhs_buf,
                        rhs_contig.shape(),
                        Activation::None,
                    );

                    let storage = Storage::new_wgpu(output_buf, m * n, 0);
                    let mut tensor = Tensor::new_with_storage(storage, &[m, n]);

                    if lhs.requires_grad() || rhs.requires_grad() {
                        tensor.set_requires_grad_mut(true);
                        tensor.set_op(Arc::new(MatmulBackward {
                            lhs: lhs.clone(),
                            rhs: rhs.clone(),
                        }));
                    }
                    return tensor;
                }
            }

            return matmul(&lhs.contiguous(), &rhs.contiguous());
        }
    }

    // CPU MatrixMultiply
    let lhs_contig = if lhs.is_contiguous() {
        lhs.clone()
    } else {
        lhs.contiguous()
    };
    let rhs_contig = if rhs.is_contiguous() {
        rhs.clone()
    } else {
        rhs.contiguous()
    };

    #[cfg(feature = "wgpu_backend")]
    let (lhs_contig, rhs_contig) = {
        let l = if lhs_contig.storage().device().is_wgpu() {
            lhs_contig.to_cpu()
        } else {
            lhs_contig
        };
        let r = if rhs_contig.storage().device().is_wgpu() {
            rhs_contig.to_cpu()
        } else {
            rhs_contig
        };
        (l, r)
    };

    let lhs_guard = lhs_contig.data();
    let rhs_guard = rhs_contig.data();
    let lhs_data = &*lhs_guard;
    let rhs_data = &*rhs_guard;

    let lhs_s0 = lhs_contig.strides()[0] as isize;
    let lhs_s1 = lhs_contig.strides()[1] as isize;
    let rhs_s0 = rhs_contig.strides()[0] as isize;
    let rhs_s1 = rhs_contig.strides()[1] as isize;
    let kernel = choose_cpu_kernel(
        m, k, n, false, lhs_data, rhs_data, lhs_s0, lhs_s1, rhs_s0, rhs_s1, None,
    );
    let result_data = match kernel {
        CpuKernelChoice::Parallel => matmul_cpu_parallel_core(lhs_data, rhs_data, m, k, n, None),
        CpuKernelChoice::Sgemm => matmul_cpu_sgemm_core(
            lhs_data, rhs_data, m, k, n, lhs_s0, lhs_s1, rhs_s0, rhs_s1, None,
        ),
    };

    let storage = Storage::new(result_data);
    let mut tensor = Tensor::new_with_storage(storage, &[m, n]);

    if lhs.requires_grad() || rhs.requires_grad() {
        tensor.set_requires_grad_mut(true);
        tensor.set_op(Arc::new(MatmulBackward {
            lhs: lhs.clone(),
            rhs: rhs.clone(),
        }));
    }

    tensor
}
