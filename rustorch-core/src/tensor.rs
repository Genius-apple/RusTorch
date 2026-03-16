use parking_lot::{RwLockReadGuard, RwLockWriteGuard};
use std::fmt;
use std::ops::{Add, Div, Mul, Sub};
use std::sync::{Arc, Mutex};
// use rand::Rng;
use rand_distr::{Distribution, Normal, Uniform};
// use rayon::prelude::*;
// use rayon::iter::{IntoParallelRefIterator, ParallelIterator, IndexedParallelIterator};
// use rayon::slice::ParallelSliceMut;
use crate::autograd::BackwardOp;
use crate::storage::Storage;
#[cfg(feature = "serde")]
use serde::{Deserialize, Deserializer, Serialize, Serializer};

#[derive(Clone, Debug)]
pub struct Tensor {
    pub(crate) inner: Arc<TensorImpl>,
}

impl PartialEq for Tensor {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.inner, &other.inner)
    }
}

#[derive(Debug)]
pub(crate) struct TensorImpl {
    pub(crate) storage: Storage,
    pub(crate) shape: Vec<usize>,
    pub(crate) strides: Vec<usize>,
    pub(crate) grad: Mutex<Option<Tensor>>, // Gradient
    pub(crate) requires_grad: bool,
    pub(crate) op: Option<Arc<dyn BackwardOp>>, // Operation that created this tensor
    pub(crate) is_leaf: bool,
}

#[cfg(feature = "wgpu_backend")]
#[derive(Debug)]
struct ToCpuBackward {
    input: Tensor,
}

#[cfg(feature = "wgpu_backend")]
impl BackwardOp for ToCpuBackward {
    fn backward(&self, grad: &Tensor) {
        if self.input.requires_grad() {
            let grad_wgpu = grad.to_wgpu();
            self.input.accumulate_grad(&grad_wgpu);
            self.input.backward_step();
        }
    }
}

impl Tensor {
    pub fn new(data: &[f32], shape: &[usize]) -> Self {
        let size: usize = shape.iter().product();
        if data.len() != size {
            panic!(
                "Data size {} does not match shape {:?} (expected {})",
                data.len(),
                shape,
                size
            );
        }

        let strides = Self::compute_strides(shape);
        let storage = Storage::from_slice(data);

        Self {
            inner: Arc::new(TensorImpl {
                storage,
                shape: shape.to_vec(),
                strides,
                grad: Mutex::new(None),
                requires_grad: false,
                op: None,
                is_leaf: true,
            }),
        }
    }

    pub fn new_with_storage(storage: Storage, shape: &[usize]) -> Self {
        let strides = Self::compute_strides(shape);
        Self {
            inner: Arc::new(TensorImpl {
                storage,
                shape: shape.to_vec(),
                strides,
                grad: Mutex::new(None),
                requires_grad: false,
                op: None,
                is_leaf: true,
            }),
        }
    }

    pub fn zeros(shape: &[usize]) -> Self {
        let size: usize = shape.iter().product();
        Self::new(&vec![0.0; size], shape)
    }

    pub fn full(shape: &[usize], value: f32) -> Self {
        let size: usize = shape.iter().product();
        let data = vec![value; size];
        let storage = Storage::new(data);
        Self::new_with_storage(storage, shape)
    }

    pub fn ones(shape: &[usize]) -> Self {
        let size: usize = shape.iter().product();
        Self::new(&vec![1.0; size], shape)
    }

    pub fn storage(&self) -> &Storage {
        &self.inner.storage
    }

    #[cfg(feature = "wgpu_backend")]
    pub fn to_wgpu(&self) -> Self {
        if let Some(_) = self.storage().wgpu_buffer() {
            return self.clone();
        }

        let contig = if self.is_contiguous() {
            self.clone()
        } else {
            self.contiguous()
        };

        let data = contig.data();
        let ctx = crate::backend::wgpu::get_context().expect("WGPU context not initialized");

        use wgpu::util::DeviceExt;
        let buffer = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Tensor Buffer"),
                contents: bytemuck::cast_slice(&data),
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
            });

        let storage = Storage::new_wgpu(buffer, data.len(), 0);

        let inner = TensorImpl {
            storage,
            shape: contig.shape().to_vec(),
            strides: contig.strides().to_vec(),
            grad: Mutex::new(None),
            requires_grad: self.requires_grad(),
            op: None,
            is_leaf: self.inner.is_leaf,
        };

        Tensor {
            inner: Arc::new(inner),
        }
    }

    #[cfg(feature = "wgpu_backend")]
    pub fn to_cpu(&self) -> Self {
        if let Some(buffer) = self.storage().wgpu_buffer() {
            // Flush any pending commands to ensure buffer is ready
            crate::backend::wgpu::flush_queue();

            let ctx = crate::backend::wgpu::get_context().expect("WGPU context not initialized");

            let buf_size = buffer.size();

            let staging_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Staging Buffer"),
                size: buf_size,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            let mut encoder = ctx
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Download Encoder"),
                });
            encoder.copy_buffer_to_buffer(buffer, 0, &staging_buffer, 0, buf_size);
            ctx.queue.submit(Some(encoder.finish()));

            let buffer_slice = staging_buffer.slice(..);
            let (sender, receiver) = std::sync::mpsc::channel();
            buffer_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());

            ctx.device.poll(wgpu::Maintain::Wait);
            receiver.recv().unwrap().unwrap();

            let data = buffer_slice.get_mapped_range();
            let result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
            drop(data);
            staging_buffer.unmap();

            // Create CPU tensor with SAME shape and strides, but using the downloaded storage
            let storage = Storage::new(result);

            // We need to construct Tensor manually to preserve strides/offset
            let inner = TensorImpl {
                storage,
                shape: self.shape().to_vec(),
                strides: self.strides().to_vec(),
                grad: Mutex::new(None),
                requires_grad: self.requires_grad(),
                op: if self.requires_grad() {
                    Some(Arc::new(ToCpuBackward {
                        input: self.clone(),
                    }))
                } else {
                    None
                },
                is_leaf: self.inner.is_leaf,
            };
            return Self {
                inner: Arc::new(inner),
            };
        }
        // println!("DEBUG: to_cpu falling back to clone (no WGPU buffer)");
        self.clone()
    }

    pub fn shape(&self) -> &[usize] {
        &self.inner.shape
    }

    pub fn strides(&self) -> &[usize] {
        &self.inner.strides
    }

    pub fn set_requires_grad(self, requires_grad: bool) -> Self {
        let inner = &self.inner;
        let new_impl = TensorImpl {
            storage: inner.storage.clone(),
            shape: inner.shape.clone(),
            strides: inner.strides.clone(),
            grad: Mutex::new(None),
            requires_grad,
            op: inner.op.clone(),
            is_leaf: inner.is_leaf,
        };
        Self {
            inner: Arc::new(new_impl),
        }
    }

    pub fn set_requires_grad_mut(&mut self, requires_grad: bool) {
        if let Some(inner) = Arc::get_mut(&mut self.inner) {
            inner.requires_grad = requires_grad;
        } else {
            // Clone if shared
            *self = self.clone().set_requires_grad(requires_grad);
        }
    }

    pub fn requires_grad(&self) -> bool {
        self.inner.requires_grad
    }

    pub fn data(&self) -> RwLockReadGuard<'_, Vec<f32>> {
        self.inner.storage.data()
    }

    pub fn data_mut(&self) -> RwLockWriteGuard<'_, Vec<f32>> {
        self.inner.storage.data_mut()
    }

    pub fn grad(&self) -> Option<Tensor> {
        self.inner.grad.lock().unwrap().clone()
    }

    pub fn zero_grad(&self) {
        *self.inner.grad.lock().unwrap() = None;
    }

    pub fn accumulate_grad(&self, grad: &Tensor) {
        let mut g = self.inner.grad.lock().unwrap();
        if let Some(existing) = &*g {
            #[cfg(feature = "wgpu_backend")]
            {
                let existing_is_wgpu = existing.storage().wgpu_buffer().is_some();
                let grad_is_wgpu = grad.storage().wgpu_buffer().is_some();

                if existing_is_wgpu && grad_is_wgpu {
                    *g = Some(existing.add(grad));
                } else if existing_is_wgpu {
                    *g = Some(existing.add(&grad.to_wgpu()));
                } else if grad_is_wgpu {
                    *g = Some(existing.add(&grad.to_cpu()));
                } else {
                    *g = Some(existing.add(grad));
                }
            }
            #[cfg(not(feature = "wgpu_backend"))]
            {
                *g = Some(existing.add(grad));
            }
        } else {
            *g = Some(grad.clone());
        }
    }

    pub fn backward(&self) {
        // Gradient of scalar output is 1.0
        if self.shape().len() != 1 || self.shape()[0] != 1 {
            // Usually backward() is called on scalar loss.
            // If not scalar, PyTorch requires gradient argument.
            // RusTorch: implicitly assume 1.0 if scalar?
            // If tensor is not scalar, we should probably fill ones.
            // But for simplicity, let's assume scalar 1.0 or Tensor::ones.
        }

        let grad = Tensor::ones(self.shape());
        self.accumulate_grad(&grad);
        self.backward_step();
    }

    pub fn backward_step(&self) {
        if let Some(op) = &self.inner.op {
            if let Some(grad) = self.grad() {
                op.backward(&grad);
            }
        }
    }

    /// Returns a new Tensor, detached from the current graph.
    /// The result will never require gradient.
    pub fn detach(&self) -> Tensor {
        Tensor {
            inner: Arc::new(TensorImpl {
                storage: self.inner.storage.clone(),
                shape: self.inner.shape.clone(),
                strides: self.inner.strides.clone(),
                grad: Mutex::new(None),
                requires_grad: false,
                op: None,
                is_leaf: true,
            }),
        }
    }

    pub fn set_op(&mut self, op: Arc<dyn BackwardOp>) {
        if let Some(inner) = Arc::get_mut(&mut self.inner) {
            inner.op = Some(op);
        } else {
            // Panic or clone?
            // Usually set_op is called during construction where we have unique ownership.
            // If not, it means something is wrong.
            // But `permute` cloned `inner`...
            // In `permute`, I created a new Tensor with `inner: Arc::new(...)`.
            // So `self.inner` is unique there.
            panic!("Cannot set op on shared tensor storage wrapper");
        }
    }

    pub fn matmul(&self, rhs: &Tensor) -> Tensor {
        crate::ops::matmul(self, rhs)
    }

    pub fn t(&self) -> Tensor {
        crate::ops::view::transpose(self, 0, 1) // Default to 2D transpose
    }

    pub fn sub(&self, rhs: &Tensor) -> Tensor {
        crate::ops::sub(self, rhs)
    }

    pub fn add(&self, rhs: &Tensor) -> Tensor {
        crate::ops::add(self, rhs)
    }

    pub fn neg(&self) -> Tensor {
        crate::ops::neg(self)
    }

    pub fn relu(&self) -> Tensor {
        crate::ops::relu(self)
    }

    pub fn sigmoid(&self) -> Tensor {
        crate::ops::sigmoid(self)
    }

    pub fn tanh(&self) -> Tensor {
        crate::ops::tanh(self)
    }

    pub fn softmax(&self, dim: i64) -> Tensor {
        crate::ops::softmax(self, dim)
    }

    pub fn conv2d(
        &self,
        weight: &Tensor,
        stride: (usize, usize),
        padding: (usize, usize),
    ) -> Tensor {
        crate::ops::conv2d(self, weight, stride, padding)
    }

    pub fn max_pool2d(
        &self,
        kernel_size: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
    ) -> Tensor {
        crate::ops::max_pool2d(self, kernel_size, stride, padding)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn batch_norm2d(
        &self,
        gamma: Option<&Tensor>,
        beta: Option<&Tensor>,
        running_mean: &Tensor,
        running_var: &Tensor,
        training: bool,
        momentum: f32,
        eps: f32,
    ) -> Tensor {
        crate::ops::batch_norm2d(
            self,
            gamma,
            beta,
            running_mean,
            running_var,
            training,
            momentum,
            eps,
        )
    }

    pub fn layer_norm(
        &self,
        normalized_shape: &[usize],
        weight: Option<&Tensor>,
        bias: Option<&Tensor>,
        eps: f32,
    ) -> Tensor {
        crate::ops::layer_norm(self, normalized_shape, weight, bias, eps)
    }

    pub fn permute(&self, dims: &[usize]) -> Tensor {
        crate::ops::view::permute(self, dims)
    }

    pub fn transpose(&self, dim0: usize, dim1: usize) -> Tensor {
        crate::ops::view::transpose(self, dim0, dim1)
    }

    pub fn contiguous(&self) -> Tensor {
        if self.is_contiguous() {
            return self.clone();
        }

        #[cfg(feature = "wgpu_backend")]
        if let Some(input_buf) = self.storage().wgpu_buffer() {
            use crate::backend::wgpu::contiguous_wgpu;
            let output_buf = contiguous_wgpu(input_buf, self.shape(), self.strides());
            let size: usize = self.shape().iter().product();
            let storage = Storage::new_wgpu(output_buf, size, 0);
            let mut tensor = Tensor::new_with_storage(storage, self.shape());
            tensor.set_requires_grad_mut(self.requires_grad());
            return tensor;
        }

        crate::ops::view::contiguous(self)
    }

    pub fn is_contiguous(&self) -> bool {
        let default_strides = Self::compute_strides(self.shape());
        if self.strides() != default_strides {
            return false;
        }
        let expected_size: usize = self.shape().iter().product();
        let actual_size = self.storage().len();
        expected_size == actual_size
    }

    pub fn normal_(&self, mean: f32, std: f32) {
        let mut guard = self.data_mut();
        let mut rng = rand::thread_rng();
        let normal = Normal::new(mean, std).unwrap();
        for x in guard.iter_mut() {
            *x = normal.sample(&mut rng);
        }
    }

    pub fn uniform_(&self, low: f32, high: f32) {
        let mut guard = self.data_mut();
        let mut rng = rand::thread_rng();
        let uniform = Uniform::new(low, high);
        for x in guard.iter_mut() {
            *x = uniform.sample(&mut rng);
        }
    }

    pub fn fill_(&self, value: f32) {
        let mut guard = self.data_mut();
        for x in guard.iter_mut() {
            *x = value;
        }
    }

    pub fn reshape(&self, new_shape: &[usize]) -> Tensor {
        let size: usize = self.shape().iter().product();
        let new_size: usize = new_shape.iter().product();
        if size != new_size {
            panic!(
                "Reshape: element count mismatch: {:?} vs {:?}",
                self.shape(),
                new_shape
            );
        }

        let inner = &self.inner;
        let strides = Self::compute_strides(new_shape);

        // Share storage, create new TensorImpl
        let mut tensor = Self {
            inner: Arc::new(TensorImpl {
                storage: inner.storage.clone(),
                shape: new_shape.to_vec(),
                strides,
                grad: Mutex::new(None),
                requires_grad: inner.requires_grad,
                op: None,
                is_leaf: false,
            }),
        };

        if inner.requires_grad {
            tensor.set_op(Arc::new(crate::ops::ReshapeBackward {
                input_shape: inner.shape.clone(),
                input: self.clone(),
            }));
        }

        tensor
    }

    pub fn mul(&self, rhs: &Tensor) -> Tensor {
        crate::ops::mul(self, rhs)
    }

    pub fn div(&self, rhs: &Tensor) -> Tensor {
        crate::ops::div(self, rhs)
    }

    #[cfg(feature = "wgpu_backend")]
    pub fn matmul_relu(&self, rhs: &Tensor) -> Tensor {
        crate::ops::matmul_fused(self, rhs, None, crate::backend::wgpu::Activation::ReLU)
    }

    #[cfg(not(feature = "wgpu_backend"))]
    pub fn matmul_relu(&self, rhs: &Tensor) -> Tensor {
        self.matmul(rhs).relu()
    }

    pub fn sgd_step(&self, grad: &Tensor, lr: f32) -> Tensor {
        crate::ops::sgd_step(self, grad, lr)
    }

    pub fn copy_(&self, src: &Tensor) {
        #[cfg(feature = "wgpu_backend")]
        if self.storage().device().is_wgpu() {
            if let Some(dest_buf) = self.storage().wgpu_buffer() {
                let ctx = crate::backend::wgpu::get_context().expect("WGPU context missing");

                // Case 1: src is WGPU -> GPU Copy
                if let Some(src_buf) = src.storage().wgpu_buffer() {
                    let mut encoder =
                        ctx.device
                            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                                label: Some("Copy Encoder"),
                            });
                    encoder.copy_buffer_to_buffer(src_buf, 0, dest_buf, 0, dest_buf.size());
                    ctx.queue.submit(Some(encoder.finish()));
                    return;
                }

                // Case 2: src is CPU -> Upload
                let src_cpu = if src.storage().device().is_wgpu() {
                    src.to_cpu()
                } else {
                    src.clone()
                };
                let src_guard = src_cpu.data();
                ctx.queue
                    .write_buffer(dest_buf, 0, bytemuck::cast_slice(&src_guard));
                return;
            }
        }

        // Case 3: self is CPU -> Memcpy
        let src_cpu = if src.storage().device().is_wgpu() {
            src.to_cpu()
        } else {
            src.clone()
        };
        let mut dest_guard = self.data_mut();
        let src_guard = src_cpu.data();
        if dest_guard.len() != src_guard.len() {
            // panic!("copy_: element count mismatch");
            // Allow broadcasting? No, strict copy_ usually.
        }
        let len = std::cmp::min(dest_guard.len(), src_guard.len());
        dest_guard[..len].copy_from_slice(&src_guard[..len]);
    }

    fn compute_strides(shape: &[usize]) -> Vec<usize> {
        let mut strides = vec![0; shape.len()];
        let mut stride = 1;
        for i in (0..shape.len()).rev() {
            strides[i] = stride;
            stride *= shape[i];
        }
        strides
    }

    // pub fn expand(&self, target_shape: &[usize]) -> Tensor {
    //    crate::broadcast::expand(self, target_shape)
    // }

    pub fn copy_from_slice(&self, src: &[f32]) {
        let mut guard = self.data_mut();
        let len = std::cmp::min(guard.len(), src.len());
        guard[..len].copy_from_slice(&src[..len]);
    }
}

// Implement arithmetic traits for &Tensor
impl Add for &Tensor {
    type Output = Tensor;
    fn add(self, rhs: Self) -> Tensor {
        self.add(rhs)
    }
}

impl Add<Tensor> for Tensor {
    type Output = Tensor;
    fn add(self, rhs: Tensor) -> Tensor {
        Tensor::add(&self, &rhs)
    }
}

impl Sub<Tensor> for Tensor {
    type Output = Tensor;
    fn sub(self, rhs: Tensor) -> Tensor {
        Tensor::sub(&self, &rhs)
    }
}

impl Mul<Tensor> for Tensor {
    type Output = Tensor;
    fn mul(self, rhs: Tensor) -> Tensor {
        Tensor::mul(&self, &rhs)
    }
}

impl Div<Tensor> for Tensor {
    type Output = Tensor;
    fn div(self, rhs: Tensor) -> Tensor {
        Tensor::div(&self, &rhs)
    }
}

impl Sub for &Tensor {
    type Output = Tensor;
    fn sub(self, rhs: Self) -> Tensor {
        self.sub(rhs)
    }
}

impl Mul for &Tensor {
    type Output = Tensor;
    fn mul(self, rhs: Self) -> Tensor {
        self.mul(rhs)
    }
}

impl Div for &Tensor {
    type Output = Tensor;
    fn div(self, rhs: Self) -> Tensor {
        self.div(rhs)
    }
}

#[cfg(feature = "serde")]
#[derive(Serialize, Deserialize)]
struct TensorData {
    shape: Vec<usize>,
    data: Vec<f32>,
    requires_grad: bool,
}

#[cfg(feature = "serde")]
impl Serialize for Tensor {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let data = self.data().clone();
        let tensor_data = TensorData {
            shape: self.shape().to_vec(),
            data,
            requires_grad: self.requires_grad(),
        };
        tensor_data.serialize(serializer)
    }
}

#[cfg(feature = "serde")]
impl<'de> Deserialize<'de> for Tensor {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let tensor_data = TensorData::deserialize(deserializer)?;
        let tensor = Tensor::new(&tensor_data.data, &tensor_data.shape)
            .set_requires_grad(tensor_data.requires_grad);
        Ok(tensor)
    }
}

impl fmt::Display for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let data = self.data();
        let len = std::cmp::min(data.len(), 10);
        write!(
            f,
            "Tensor(shape={:?}, data={:?})",
            self.shape(),
            &data[..len]
        )
    }
}
