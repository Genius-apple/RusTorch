use rustorch_core::Tensor;
use rustorch_nn::{Linear, Conv2d, MaxPool2d, ReLU, Module, CrossEntropyLoss, Dataset, DataLoader};
use rustorch_nn::optim::{Optimizer, SGD};
use std::time::Instant;

// --- Simple Model ---
// Conv -> ReLU -> Pool -> Flatten -> Linear
struct Net {
    conv1: Conv2d,
    pool: MaxPool2d,
    relu: ReLU,
    fc1: Linear,
}

impl Module for Net {
    fn forward(&self, input: &Tensor) -> Tensor {
        // Input: (N, 1, 8, 8)
        let x = self.conv1.forward(input); // (N, 4, 8, 8) (padding=1)
        let x = self.relu.forward(&x);
        let x = self.pool.forward(&x); // (N, 4, 4, 4)
        
        // Flatten
        // We don't have flatten op, reshape manually
        // Shape: (N, 4*4*4) = (N, 64)
        let n = x.shape()[0];
        let x_flat = x.reshape(&[n, 64]);
        
        self.fc1.forward(&x_flat) // (N, 10)
    }
    
    fn parameters(&self) -> Vec<Tensor> {
        let mut params = vec![];
        params.extend(self.conv1.parameters());
        params.extend(self.fc1.parameters());
        params
    }
}

// --- Dummy Dataset ---
#[derive(Clone)]
struct DummyDataset {
    size: usize,
    data: Vec<(Tensor, Tensor)>,
}

impl DummyDataset {
    fn new(size: usize) -> Self {
        // Create random data
        let mut data = Vec::new();
        for i in 0..size {
            // Random input: 1x8x8
            // Let's make it simple: if sum > threshold, class 1, else 0.
            // But we need 10 classes.
            // Just random for benchmark/flow test.
            
            // Note: Since we don't have `randn` yet, we use `ones` or manual filling.
            // Using a deterministic pattern to check if loss decreases.
            
            // Pattern: Class i has input with value i/10.0 at (0,0) and noise elsewhere?
            // Let's just do: target = (i % 10). Input = target / 10.0 filled.
            let target_val = (i % 10) as f32;
            let mut input_data = vec![0.0; 64];
            input_data[0] = target_val / 10.0 + 0.5; // Feature at [0,0]
            
            let input = Tensor::new(&input_data, &[1, 8, 8]);
            let target = Tensor::new(&[target_val], &[1]); // shape [1]
            data.push((input, target));
        }
        
        Self { size, data }
    }
}

impl Dataset for DummyDataset {
    fn len(&self) -> usize {
        self.size
    }
    
    fn get(&self, index: usize) -> (Tensor, Tensor) {
        let (input, target) = &self.data[index];
        (input.clone(), target.clone())
    }
}

fn main() {
    println!("--- RusTorch Full Training Demo ---");
    
    // 1. Setup Data
    let dataset = DummyDataset::new(100);
    
    // 2. Setup Model
    let model = Net {
        conv1: Conv2d::new(1, 4, (3, 3), (1, 1), (1, 1)),
        pool: MaxPool2d::new((2, 2), Some((2, 2)), (0, 0)),
        relu: ReLU::new(),
        fc1: Linear::new(64, 10),
    };
    
    // 3. Setup Loss and Optimizer
    let criterion = CrossEntropyLoss::new();
    let mut optimizer = SGD::new(model.parameters(), 0.01, 0.9);
    
    // 4. Training Loop
    let start_time = Instant::now();
    let epochs = 5;
    
    for epoch in 0..epochs {
        let mut total_loss = 0.0;
        let mut correct = 0;
        let mut total = 0;
        let mut batch_count = 0;
        
        // Iterate over batches
        // Re-create iterator for each epoch
        let mut dataloader = DataLoader::new(dataset.clone(), 10, true);
        
        while let Some((mut inputs, targets)) = dataloader.next() {
            batch_count += 1;
            
            // targets: (B, 1) -> Flatten to (B) for CrossEntropy?
            // CrossEntropy expects targets as (B) indices.
            // Our target is (B, 1). We need to flatten.
            // Reshape target to (B)
            let batch_size = targets.shape()[0];
            let targets_flat = targets.reshape(&[batch_size]); // (B)
            
            // Forward
            inputs.set_requires_grad_mut(true);
            let outputs = model.forward(&inputs); // (B, 10)
            let loss = criterion.forward(&outputs, &targets_flat);
            
            // Backward
            optimizer.zero_grad();
            loss.backward();
            optimizer.step();
            
            // Stats
            {
                let loss_guard = loss.data();
                if !loss_guard.is_empty() {
                    total_loss += loss_guard[0];
                }
            }
            
            // Accuracy
            {
                let output_guard = outputs.data();
                let target_guard = targets_flat.data();
                let output_data = &*output_guard;
                let target_data = &*target_guard;
                
                for i in 0..batch_size {
                    // Argmax
                    let mut max_val = -f32::INFINITY;
                    let mut max_idx = 0;
                    for j in 0..10 {
                        let val = output_data[i * 10 + j];
                        if val > max_val {
                            max_val = val;
                            max_idx = j;
                        }
                    }
                    if (max_idx as f32 - target_data[i]).abs() < 0.1 {
                        correct += 1;
                    }
                    total += 1;
                }
            }
        }
        
        let avg_loss = if batch_count > 0 { total_loss / batch_count as f32 } else { 0.0 };
        let accuracy = if total > 0 { 100.0 * correct as f32 / total as f32 } else { 0.0 };
        
        println!("Epoch [{}/{}], Loss: {:.4}, Accuracy: {:.2}%", 
            epoch+1, epochs, avg_loss, accuracy);
    }
    
    println!("Training finished in {:.2?}", start_time.elapsed());
}
