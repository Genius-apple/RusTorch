use rustorch_core::Tensor;
use rustorch_nn::{Conv2d, Linear, Module};
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
#[cfg(feature = "serde")]
use std::fs::File;
#[cfg(feature = "serde")]
use std::io::{Read, Write};

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
struct MyModel {
    fc: Linear,
    conv: Conv2d,
}

impl Module for MyModel {
    fn forward(&self, input: &Tensor) -> Tensor {
        // Just for structure test
        self.fc.forward(input)
    }
    fn parameters(&self) -> Vec<Tensor> {
        let mut p = self.fc.parameters();
        p.extend(self.conv.parameters());
        p
    }
}

fn main() {
    println!("--- Testing Serialization ---");

    let model = MyModel {
        fc: Linear::new(10, 5),
        conv: Conv2d::new(1, 1, (3, 3), (1, 1), (0, 0)),
    };

    // Modify weight to test it's not just default init
    model.fc.weight.fill_(0.5);

    println!("Original Weight[0]: {}", model.fc.weight.data()[0]);

    // Save
    #[cfg(feature = "serde")]
    {
        let serialized = serde_json::to_string(&model).unwrap();
        println!("Serialized length: {}", serialized.len());

        let mut file = File::create("model.json").unwrap();
        file.write_all(serialized.as_bytes()).unwrap();

        // Load
        let mut file = File::open("model.json").unwrap();
        let mut contents = String::new();
        file.read_to_string(&mut contents).unwrap();

        let loaded_model: MyModel = serde_json::from_str(&contents).unwrap();

        println!("Loaded Weight[0]: {}", loaded_model.fc.weight.data()[0]);

        assert!((model.fc.weight.data()[0] - loaded_model.fc.weight.data()[0]).abs() < 1e-6);
        println!("Serialization test passed!");

        // Cleanup
        std::fs::remove_file("model.json").unwrap();
    }
}
