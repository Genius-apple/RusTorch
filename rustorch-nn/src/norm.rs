use rustorch_core::Tensor;
use crate::Module;
#[cfg(feature = "serde")]
use serde::{Serialize, Deserialize};

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct LayerNorm {
    pub normalized_shape: Vec<usize>,
    pub eps: f32,
    pub elementwise_affine: bool,
    pub weight: Option<Tensor>,
    pub bias: Option<Tensor>,
}

impl LayerNorm {
    pub fn new(normalized_shape: Vec<usize>) -> Self {
        // Default affine=true
        let weight = Tensor::ones(&normalized_shape).set_requires_grad(true);
        let bias = Tensor::zeros(&normalized_shape).set_requires_grad(true);
        
        Self {
            normalized_shape,
            eps: 1e-5,
            elementwise_affine: true,
            weight: Some(weight),
            bias: Some(bias),
        }
    }
}

impl Module for LayerNorm {
    fn forward(&self, input: &Tensor) -> Tensor {
        input.layer_norm(
            &self.normalized_shape,
            self.weight.as_ref(),
            self.bias.as_ref(),
            self.eps
        )
    }

    fn parameters(&self) -> Vec<Tensor> {
        let mut params = vec![];
        if let Some(w) = &self.weight {
            params.push(w.clone());
        }
        if let Some(b) = &self.bias {
            params.push(b.clone());
        }
        params
    }
}

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct BatchNorm2d {
    pub num_features: usize,
    pub eps: f32,
    pub momentum: f32,
    pub affine: bool,
    pub track_running_stats: bool,
    
    pub weight: Option<Tensor>, // gamma
    pub bias: Option<Tensor>,   // beta
    
    pub running_mean: Tensor,
    pub running_var: Tensor,
}

impl BatchNorm2d {
    pub fn new(num_features: usize) -> Self {
        // Default init
        let weight = Tensor::ones(&[num_features]).set_requires_grad(true);
        let bias = Tensor::zeros(&[num_features]).set_requires_grad(true);
        
        // Running stats: no grad
        let running_mean = Tensor::zeros(&[num_features]);
        let running_var = Tensor::ones(&[num_features]);
        
        Self {
            num_features,
            eps: 1e-5,
            momentum: 0.1,
            affine: true,
            track_running_stats: true,
            weight: Some(weight),
            bias: Some(bias),
            running_mean,
            running_var,
        }
    }
}

impl Module for BatchNorm2d {
    fn forward(&self, input: &Tensor) -> Tensor {
        // Determine training mode (TODO: Add train/eval mode to Module trait or Context)
        let training = true; 
        
        input.batch_norm2d(
            self.weight.as_ref(),
            self.bias.as_ref(),
            &self.running_mean,
            &self.running_var,
            training,
            self.momentum,
            self.eps
        )
    }

    fn parameters(&self) -> Vec<Tensor> {
        let mut params = vec![];
        if let Some(w) = &self.weight {
            params.push(w.clone());
        }
        if let Some(b) = &self.bias {
            params.push(b.clone());
        }
        params
    }
}
