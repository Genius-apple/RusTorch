use rustorch_core::Tensor;
use rustorch_nn::optim::{Adam, Optimizer};
use rustorch_nn::{Linear, Module};
use serde_json::json;
use std::time::Instant;

#[allow(clippy::upper_case_acronyms)]
struct MLP {
    fc: Linear,
}

fn deterministic_unit(idx: u64, seed: u64) -> f32 {
    let mut x = idx.wrapping_add(seed.wrapping_mul(0x9E3779B97F4A7C15));
    x ^= x >> 30;
    x = x.wrapping_mul(0xBF58476D1CE4E5B9);
    x ^= x >> 27;
    x = x.wrapping_mul(0x94D049BB133111EB);
    x ^= x >> 31;
    (x as u32) as f32 / (u32::MAX as f32)
}

impl MLP {
    fn new() -> Self {
        Self {
            fc: Linear::new(784, 10),
        }
    }

    fn forward(&self, x: &Tensor) -> Tensor {
        self.fc.forward(x)
    }

    fn parameters(&self) -> Vec<Tensor> {
        self.fc.parameters()
    }
}

fn main() {
    let batch_size = 1024;
    let input_size = 784;
    let output_size = 10;
    let epochs = 50;
    let steps_per_epoch = 20;

    let mut teacher_w = vec![0.0f32; input_size * output_size];
    let mut teacher_b = vec![0.0f32; output_size];
    for (i, v) in teacher_w.iter_mut().enumerate() {
        *v = deterministic_unit(i as u64, 20260315) * 0.2 - 0.1;
    }
    for (i, v) in teacher_b.iter_mut().enumerate() {
        *v = deterministic_unit(i as u64, 20260401) * 0.2 - 0.1;
    }

    let model = MLP::new();
    model.fc.weight.fill_(0.0);
    if let Some(bias) = &model.fc.bias {
        bias.fill_(0.0);
    }
    let mut optimizer = Adam::new(model.parameters(), 0.001);

    let (x_train, y_train) = {
        let mut data_vec = vec![0.0; batch_size * input_size];
        let mut target_vec = vec![0.0; batch_size * output_size];
        for i in 0..batch_size {
            for j in 0..input_size {
                let idx = (i * input_size + j) as u64;
                data_vec[i * input_size + j] = deterministic_unit(idx, 42) - 0.5;
            }
            for j in 0..output_size {
                let mut z = teacher_b[j];
                for k in 0..input_size {
                    z += data_vec[i * input_size + k] * teacher_w[k * output_size + j];
                }
                target_vec[i * output_size + j] = z;
            }
        }
        let x = Tensor::new(&data_vec, &[batch_size, input_size]);
        let y = Tensor::new(&target_vec, &[batch_size, output_size]);
        (x, y)
    };

    let total_start = Instant::now();
    let mut last_speed = 0.0f32;
    let grad_path = std::env::var("RUSTORCH_GRAD_PATH").unwrap_or_else(|_| "tensor".to_string());

    for _ in 0..epochs {
        let start = Instant::now();
        for _ in 0..steps_per_epoch {
            optimizer.zero_grad();
            let output = model.forward(&x_train);
            let diff = output.sub(&y_train);
            let numel = diff.shape().iter().product::<usize>() as f32;
            if grad_path.eq_ignore_ascii_case("fused") {
                let (_loss_val, grad_w, grad_b) =
                    rustorch_core::ops::linear_mse_grads(&x_train, &output, &y_train);
                model.fc.weight.accumulate_grad(&grad_w);
                if let Some(bias) = &model.fc.bias {
                    bias.accumulate_grad(&grad_b);
                }
            } else {
                let grad_scale = 2.0 / numel;
                let diff_data = diff.data();
                let grad_out_data: Vec<f32> = diff_data.iter().map(|v| v * grad_scale).collect();
                let grad_out = Tensor::new(&grad_out_data, diff.shape());
                let grad_w = grad_out.t().matmul(&x_train);
                model.fc.weight.accumulate_grad(&grad_w);
                if let Some(bias) = &model.fc.bias {
                    let grad_b = rustorch_core::ops::sum_to(&grad_out, &[output_size]);
                    bias.accumulate_grad(&grad_b);
                }
            }
            optimizer.step();
        }
        let duration = start.elapsed();
        let samples = (batch_size * steps_per_epoch) as f32;
        last_speed = samples / duration.as_secs_f32();
    }

    let final_output = model.forward(&x_train);
    let final_diff = final_output.sub(&y_train);
    let final_sq = final_diff.clone() * final_diff.clone();
    let final_loss = {
        #[cfg(feature = "wgpu_backend")]
        {
            if final_sq.storage().wgpu_buffer().is_some() {
                let numel = final_sq.shape().iter().product::<usize>() as f32;
                rustorch_core::ops::sum(&final_sq).to_cpu().data()[0] / numel
            } else {
                final_sq.data().iter().sum::<f32>()
                    / final_sq.shape().iter().product::<usize>() as f32
            }
        }
        #[cfg(not(feature = "wgpu_backend"))]
        {
            final_sq.data().iter().sum::<f32>() / final_sq.shape().iter().product::<usize>() as f32
        }
    };

    let final_accuracy = {
        #[cfg(feature = "wgpu_backend")]
        let diff_host = if final_diff.storage().wgpu_buffer().is_some() {
            final_diff.to_cpu()
        } else {
            final_diff
        };
        #[cfg(not(feature = "wgpu_backend"))]
        let diff_host = final_diff;
        let d = diff_host.data();
        let c = d.iter().filter(|&&x| x.abs() < 0.1).count();
        c as f32 / d.len() as f32
    };

    let total_time = total_start.elapsed().as_secs_f32();
    let avg_speed = (batch_size * steps_per_epoch * epochs) as f32 / total_time;

    println!(
        "{}",
        json!({
            "type": "finish",
            "framework": "RusTorch",
            "final_loss": final_loss,
            "final_accuracy": final_accuracy,
            "avg_speed": avg_speed,
            "last_epoch_speed": last_speed
        })
    );
}
