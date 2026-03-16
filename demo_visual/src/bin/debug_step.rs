use rustorch_core::ops::activations::sigmoid;
use rustorch_core::Tensor;
use rustorch_nn::{
    optim::{Adam, Optimizer},
    Linear, Module,
};
use serde_json::Value;
use std::fs::File;
use std::io::BufReader;

fn main() {
    println!("Loading golden data...");
    let file = File::open("debug_golden.json").expect("file not found");
    let reader = BufReader::new(file);
    let json: Value = serde_json::from_reader(reader).expect("json error");

    // Load Data
    let input_data: Vec<f32> = json["input"]
        .as_array()
        .unwrap()
        .iter()
        .flat_map(|r| {
            r.as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_f64().unwrap() as f32)
        })
        .collect();
    let target_data: Vec<f32> = json["target"]
        .as_array()
        .unwrap()
        .iter()
        .flat_map(|r| {
            r.as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_f64().unwrap() as f32)
        })
        .collect();

    let input = Tensor::new(&input_data, &[4, 4]);
    let target = Tensor::new(&target_data, &[4, 2]);

    // Init Model
    // FC1: 4 -> 4
    let fc1 = Linear::new(4, 4);
    // FC2: 4 -> 2
    let fc2 = Linear::new(4, 2);

    // Load Init Weights
    let fc1_w: Vec<f32> = json["init"]["fc1_w"]
        .as_array()
        .unwrap()
        .iter()
        .flat_map(|r| {
            r.as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_f64().unwrap() as f32)
        })
        .collect();
    let fc1_b: Vec<f32> = json["init"]["fc1_b"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap() as f32)
        .collect();

    let fc2_w: Vec<f32> = json["init"]["fc2_w"]
        .as_array()
        .unwrap()
        .iter()
        .flat_map(|r| {
            r.as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_f64().unwrap() as f32)
        })
        .collect();
    let fc2_b: Vec<f32> = json["init"]["fc2_b"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap() as f32)
        .collect();

    fc1.weight.copy_from_slice(&fc1_w);
    if let Some(b) = &fc1.bias {
        b.copy_from_slice(&fc1_b);
    }
    fc2.weight.copy_from_slice(&fc2_w);
    if let Some(b) = &fc2.bias {
        b.copy_from_slice(&fc2_b);
    }

    // Optimizer
    let mut params = fc1.parameters();
    params.extend(fc2.parameters());
    let mut optimizer = Adam::new(params, 0.1);

    // Forward
    optimizer.zero_grad();
    let x = fc1.forward(&input);
    let x = sigmoid(&x);
    let output = fc2.forward(&x);

    // Loss
    let diff = output.sub(&target);
    let sq_diff = diff.clone() * diff.clone();
    let numel = sq_diff.shape().iter().product::<usize>() as f32;
    let loss_val = sq_diff.data().iter().sum::<f32>() / numel;

    println!("RusTorch Loss: {:.8}", loss_val);
    println!(
        "PyTorch Loss: {:.8}",
        json["step0"]["loss"].as_f64().unwrap()
    );

    // Backward
    let grad = Tensor::full(sq_diff.shape(), 1.0 / numel);
    sq_diff.accumulate_grad(&grad);
    sq_diff.backward_step();

    // Check Gradients
    println!("\n--- Checking Gradients ---");
    // Check FC2 Bias Grad
    if let Some(grad) = fc2.bias.as_ref().unwrap().grad() {
        let g = grad.data();
        let py_g: Vec<f32> = json["step0"]["grad"]["fc2_b"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap() as f32)
            .collect();
        println!("FC2 Bias Grad Diff: {:.8}", (g[0] - py_g[0]).abs());
    }

    // Step
    optimizer.step();

    // Check Updated Weights
    println!("\n--- Checking Updated Weights ---");
    let w = fc2.weight.data();
    let py_w: Vec<f32> = json["step0"]["updated"]["fc2_w"]
        .as_array()
        .unwrap()
        .iter()
        .flat_map(|r| {
            r.as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_f64().unwrap() as f32)
        })
        .collect();
    println!("FC2 Weight[0] Diff: {:.8}", (w[0] - py_w[0]).abs());
}
