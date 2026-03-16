use crate::Module;
use rustorch_core::Tensor;
// use rand::Rng; // Unused
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Conv2d {
    pub weight: Tensor,
    pub bias: Option<Tensor>,
    pub stride: (usize, usize),
    pub padding: (usize, usize),
}

impl Conv2d {
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
    ) -> Self {
        // Weight: (Out, In, kH, kW)
        let k_h = kernel_size.0;
        let k_w = kernel_size.1;

        // Init weights (simple random for now, ideally Kaiming)
        let size = out_channels * in_channels * k_h * k_w;
        let w_data = vec![0.01; size];

        let weight =
            Tensor::new(&w_data, &[out_channels, in_channels, k_h, k_w]).set_requires_grad(true);

        // Init bias
        // Bias: (Out) -> usually broadcasted to (N, Out, H, W)
        // We need broadcast support for add.
        // Bias shape: [out_channels, 1, 1] for broadcasting? Or [1, out_channels, 1, 1]
        // Let's use [1, out_channels, 1, 1]
        let b_data = vec![0.0; out_channels];
        let bias = Tensor::new(&b_data, &[1, out_channels, 1, 1]).set_requires_grad(true);

        Self {
            weight,
            bias: Some(bias),
            stride,
            padding,
        }
    }
}

impl Module for Conv2d {
    fn forward(&self, input: &Tensor) -> Tensor {
        let output = input.conv2d(&self.weight, self.stride, self.padding);

        if let Some(bias) = &self.bias {
            // output is (N, C_out, H, W)
            // bias is (1, C_out, 1, 1)
            // Broadcasting should handle this.
            return output.add(bias);
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
