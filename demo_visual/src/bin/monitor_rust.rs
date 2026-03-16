use rustorch_core::Tensor;
use rustorch_nn::{Linear, Module};
use serde::Deserialize;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, Write};

#[allow(dead_code)]
#[derive(Deserialize, Debug)]
struct Config {
    input_size: usize,
    hidden_size: usize,
    output_size: usize,
    batch_size: usize,
    learning_rate: f32,
    epochs: usize,
}

#[derive(Deserialize, Debug)]
struct InitData {
    config: Config,
    #[serde(rename = "X")]
    x: Vec<Vec<f32>>,
    #[serde(rename = "Y")]
    y: Vec<Vec<f32>>,
    fc1_weight: Vec<Vec<f32>>,
    fc1_bias: Vec<f32>,
    fc2_weight: Vec<Vec<f32>>,
    fc2_bias: Vec<f32>,
}

fn flatten_2d(data: &[Vec<f32>]) -> (Vec<f32>, Vec<usize>) {
    let rows = data.len();
    let cols = data[0].len();
    let flat: Vec<f32> = data.iter().flat_map(|r| r.clone()).collect();
    (flat, vec![rows, cols])
}

// Simple SGD
#[allow(clippy::upper_case_acronyms)]
struct SGD {
    params: Vec<Tensor>,
    lr: f32,
}

impl SGD {
    fn new(params: Vec<Tensor>, lr: f32) -> Self {
        Self { params, lr }
    }

    fn zero_grad(&self) {
        for param in &self.params {
            if param.requires_grad() {
                param.zero_grad();
            }
        }
    }

    fn step(&self) {
        for param in &self.params {
            if param.requires_grad() {
                if let Some(grad) = param.grad() {
                    let mut param_data = param.data_mut();
                    let grad_data = grad.data();
                    for (p, g) in param_data.iter_mut().zip(grad_data.iter()) {
                        *p -= self.lr * *g;
                    }
                }
            }
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Load Init Data
    let file = File::open("demo_visual/monitor/init_data.json")?;
    let reader = BufReader::new(file);
    let init_data: InitData = serde_json::from_reader(reader)?;
    let config = &init_data.config;

    // 2. Setup Tensors
    let (x_flat, x_shape) = flatten_2d(&init_data.x);
    let x = Tensor::new(&x_flat, &x_shape);

    let (y_flat, y_shape) = flatten_2d(&init_data.y);
    let y = Tensor::new(&y_flat, &y_shape);

    // 3. Setup Model
    let mut fc1 = Linear::new(config.input_size, config.hidden_size);
    let (w1_flat, w1_shape) = flatten_2d(&init_data.fc1_weight);
    fc1.weight = Tensor::new(&w1_flat, &w1_shape).set_requires_grad(true);

    let b1_flat = init_data.fc1_bias.clone();
    let b1_shape = vec![config.hidden_size];
    let b1 = Tensor::new(&b1_flat, &b1_shape).set_requires_grad(true);
    fc1.bias = Some(b1);

    let mut fc2 = Linear::new(config.hidden_size, config.output_size);
    let (w2_flat, w2_shape) = flatten_2d(&init_data.fc2_weight);
    fc2.weight = Tensor::new(&w2_flat, &w2_shape).set_requires_grad(true);

    let b2_flat = init_data.fc2_bias.clone();
    let b2_shape = vec![config.output_size];
    let b2 = Tensor::new(&b2_flat, &b2_shape).set_requires_grad(true);
    fc2.bias = Some(b2);

    let mut params = fc1.parameters();
    params.extend(fc2.parameters());

    // 4. Optimizer
    let optimizer = SGD::new(params, config.learning_rate);

    // 5. Monitor Log
    let log_file = File::create("demo_visual/monitor/rust_log.jsonl")?;
    let mut writer = std::io::BufWriter::new(log_file);

    println!("Starting Rust training...");

    for epoch in 0..config.epochs {
        optimizer.zero_grad();

        // Forward
        let fc1_out = fc1.forward(&x);
        let relu_out = fc1_out.relu();
        let output = fc2.forward(&relu_out);

        // Loss (MSE)
        let diff = output.sub(&y);
        let sq_diff = diff.mul(&diff);

        // Mean reduction
        let numel = sq_diff.shape().iter().product::<usize>() as f32;
        let scale = Tensor::full(sq_diff.shape(), 1.0 / numel);
        let loss_tensor = sq_diff.mul(&scale);

        loss_tensor.backward();

        // Calculate scalar loss for logging
        let loss_val: f32 = loss_tensor.data().iter().sum();

        // Capture Gradients
        let mut gradients = HashMap::new();
        let get_grad = |t: &Tensor| -> Option<Vec<f32>> { t.grad().map(|g| g.data().to_vec()) };

        if let Some(g) = get_grad(&fc1.weight) {
            gradients.insert("fc1_w_grad", g);
        }
        if let Some(ref b) = fc1.bias {
            if let Some(g) = get_grad(b) {
                gradients.insert("fc1_b_grad", g);
            }
        }
        if let Some(g) = get_grad(&fc2.weight) {
            gradients.insert("fc2_w_grad", g);
        }
        if let Some(ref b) = fc2.bias {
            if let Some(g) = get_grad(b) {
                gradients.insert("fc2_b_grad", g);
            }
        }

        // Capture Weights (Before Update)
        let mut weights = HashMap::new();
        weights.insert("fc1_w", fc1.weight.data().to_vec());
        if let Some(ref b) = fc1.bias {
            weights.insert("fc1_b", b.data().to_vec());
        }
        weights.insert("fc2_w", fc2.weight.data().to_vec());
        if let Some(ref b) = fc2.bias {
            weights.insert("fc2_b", b.data().to_vec());
        }

        // Capture Activations
        let mut activations = HashMap::new();
        activations.insert("fc1_out", fc1_out.data().to_vec());
        activations.insert("relu_out", relu_out.data().to_vec());
        activations.insert("fc2_out", output.data().to_vec());

        // Step
        optimizer.step();

        // Log
        let log_entry = serde_json::json!({
            "epoch": epoch,
            "loss": loss_val,
            "activations": activations,
            "gradients": gradients,
            "weights": weights
        });

        writeln!(writer, "{}", log_entry)?;
        writer.flush()?;

        if epoch % 10 == 0 {
            println!("Epoch {}, Loss: {:.6}", epoch, loss_val);
        }
    }

    println!("Rust training complete. Log saved to rust_log.jsonl");
    Ok(())
}
