use rustorch_core::Tensor;
use crate::Module;
#[cfg(feature = "serde")]
use serde::{Serialize, Deserialize};

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct MaxPool2d {
    pub kernel_size: (usize, usize),
    pub stride: (usize, usize),
    pub padding: (usize, usize),
}

impl MaxPool2d {
    pub fn new(kernel_size: (usize, usize), stride: Option<(usize, usize)>, padding: (usize, usize)) -> Self {
        let stride = stride.unwrap_or(kernel_size);
        Self {
            kernel_size,
            stride,
            padding,
        }
    }
}

impl Module for MaxPool2d {
    fn forward(&self, input: &Tensor) -> Tensor {
        input.max_pool2d(self.kernel_size, self.stride, self.padding)
    }

    fn parameters(&self) -> Vec<Tensor> {
        vec![]
    }
}
