use crate::Module;
use rand::Rng;
use rustorch_core::Tensor;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use std::sync::OnceLock;

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Linear {
    pub weight: Tensor,
    pub bias: Option<Tensor>,
}

impl Linear {
    pub fn new(in_features: usize, out_features: usize) -> Self {
        let mut rng = rand::thread_rng();

        // PyTorch default initialization for Linear:
        // Weights: Uniform(-k, k), where k = sqrt(1 / in_features)
        // Bias: Uniform(-k, k)

        let k = (1.0 / in_features as f32).sqrt();

        // Weight: [out_features, in_features]
        let w_size = out_features * in_features;
        let w_data: Vec<f32> = (0..w_size).map(|_| rng.gen_range(-k..k)).collect();

        let mut weight = Tensor::new(&w_data, &[out_features, in_features]);
        weight.set_requires_grad_mut(true);

        // Bias: [out_features]
        let b_data: Vec<f32> = (0..out_features).map(|_| rng.gen_range(-k..k)).collect();

        let mut bias = Tensor::new(&b_data, &[out_features]);
        bias.set_requires_grad_mut(true);

        Self {
            weight,
            bias: Some(bias),
        }
    }

    pub fn forward_fused(
        &self,
        input: &Tensor,
        activation: rustorch_core::backend::Activation,
    ) -> Tensor {
        // y = Fused(x @ W.t() + b, activation)
        let w_t = self.weight.t();
        rustorch_core::ops::matmul_fused(input, &w_t, self.bias.as_ref(), activation)
    }

    pub fn forward_fused_norm_activation(
        &self,
        input: &Tensor,
        norm_weight: Option<&Tensor>,
        norm_bias: Option<&Tensor>,
        eps: f32,
        activation: rustorch_core::backend::Activation,
    ) -> Tensor {
        let w_t = self.weight.t();
        rustorch_core::ops::matmul_bias_norm_activation(
            input,
            &w_t,
            self.bias.as_ref(),
            norm_weight,
            norm_bias,
            eps,
            activation,
        )
    }
}

impl Module for Linear {
    fn forward(&self, input: &Tensor) -> Tensor {
        static USE_FUSED: OnceLock<bool> = OnceLock::new();
        let use_fused = *USE_FUSED.get_or_init(|| {
            std::env::var("RUSTORCH_LINEAR_FUSED")
                .map(|v| matches!(v.as_str(), "1" | "true" | "TRUE" | "on" | "ON"))
                .unwrap_or(false)
        });
        if use_fused {
            return self.forward_fused(input, rustorch_core::backend::Activation::None);
        }

        let w_t = self.weight.t();
        let output = input.matmul(&w_t);

        if let Some(bias) = &self.bias {
            return output + bias.clone();
        }

        output
    }

    fn parameters(&self) -> Vec<Tensor> {
        let mut params = vec![self.weight.clone()];
        if let Some(bias) = &self.bias {
            params.push(bias.clone());
        }
        params
    }
}
