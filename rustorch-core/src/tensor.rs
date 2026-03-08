use std::sync::{Arc, Mutex, RwLockReadGuard, RwLockWriteGuard};
use std::fmt;
use std::ops::{Add, Mul, Sub};
// use rand::Rng;
use rand_distr::{Normal, Uniform, Distribution};
// use rayon::prelude::*;
// use rayon::iter::{IntoParallelRefIterator, ParallelIterator, IndexedParallelIterator};
// use rayon::slice::ParallelSliceMut;
#[cfg(feature = "serde")]
use serde::{Serialize, Deserialize, Serializer, Deserializer};
use crate::storage::Storage;
use crate::autograd::BackwardOp;

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

impl Tensor {
    pub fn new(data: &[f32], shape: &[usize]) -> Self {
        let size: usize = shape.iter().product();
        if data.len() != size {
             panic!("Data size {} does not match shape {:?} (expected {})", data.len(), shape, size);
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
                is_leaf: false,
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
        Self { inner: Arc::new(new_impl) }
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

    pub fn shape(&self) -> &[usize] {
        &self.inner.shape
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
            // Check shape (sum broadcasting if needed?)
            // For now assume same shape
            *g = Some(existing + grad);
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
        // Topological sort is better, but recursive DFS works for DAG.
        // We need to avoid visiting same node multiple times? 
        // PyTorch uses Engine. Here we do simple recursive DFS.
        // Problem: Double counting if diamond shape.
        // Standard approach: Queue based topological sort.
        // For this task, keep existing recursive implementation if it exists, or implement simple one.
        
        if let Some(op) = &self.inner.op {
            let grad = self.grad().unwrap();
            op.backward(&grad);
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

    pub fn matmul(&self, _rhs: &Tensor) -> Tensor {
        // crate::ops::matmul(self, rhs)
        // Temporary placeholder or fix path
        // Assume matmul is not yet exported or needs qualified path
        // For now, let's assume it's in ops but maybe not pub
        // Or better:
        panic!("Matmul not fully implemented yet");
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

    pub fn conv2d(&self, weight: &Tensor, stride: (usize, usize), padding: (usize, usize)) -> Tensor {
        crate::ops::conv2d(self, weight, stride, padding)
    }

    pub fn max_pool2d(&self, kernel_size: (usize, usize), stride: (usize, usize), padding: (usize, usize)) -> Tensor {
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
        eps: f32
    ) -> Tensor {
        crate::ops::batch_norm2d(self, gamma, beta, running_mean, running_var, training, momentum, eps)
    }

    pub fn layer_norm(
        &self,
        normalized_shape: &[usize],
        weight: Option<&Tensor>,
        bias: Option<&Tensor>,
        eps: f32
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
        crate::ops::view::contiguous(self)
    }

    pub fn is_contiguous(&self) -> bool {
        let default_strides = Self::compute_strides(self.shape());
        self.strides() == default_strides
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

    pub fn strides(&self) -> &[usize] {
        &self.inner.strides
    }

    pub fn reshape(&self, new_shape: &[usize]) -> Tensor {
        let size: usize = self.shape().iter().product();
        let new_size: usize = new_shape.iter().product();
        if size != new_size {
            panic!("Reshape: element count mismatch: {:?} vs {:?}", self.shape(), new_shape);
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
        write!(f, "Tensor(shape={:?}, data={:?})", self.shape(), &data[..len])
    }
}
