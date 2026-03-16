// use rustorch_core::Tensor;
// use rustorch_nn::{Conv2d, Module, ReLU};
// use rustorch_core::graph::{start_tracing, stop_tracing, register_input, NodeOp};

fn main() {
    println!("--- Testing Graph Tracing (Disabled for now) ---");
    /*
    // Define a simple model
    struct MyModel {
        conv1: Conv2d,
        relu: ReLU,
        conv2: Conv2d,
    }

    impl Module for MyModel {
        fn forward(&self, input: &Tensor) -> Tensor {
            let x = self.conv1.forward(input);
            let x = self.relu.forward(&x);
            self.conv2.forward(&x)
        }

        fn parameters(&self) -> Vec<Tensor> {
            vec![]
        }
    }

    let model = MyModel {
        conv1: Conv2d::new(1, 4, (3, 3), (1, 1), (1, 1)),
        relu: ReLU::new(),
        conv2: Conv2d::new(4, 2, (3, 3), (1, 1), (1, 1)),
    };

    // Input
    let input = Tensor::new(&vec![1.0; 1*1*8*8], &[1, 1, 8, 8]);

    println!("Starting tracing...");
    start_tracing();

    // Register inputs
    register_input(&input, "input".to_string());

    // Run forward
    let output = model.forward(&input);

    println!("Forward done. Output shape: {:?}", output.shape());

    // Stop tracing
    let graph = stop_tracing().expect("Tracing failed");

    // Print graph
    graph.print();

    // Simple optimization pass: Find Conv2d followed by ReLU
    println!("\n--- Optimization Pass: Fuse Conv+ReLU ---");
    for node in &graph.nodes {
        if let NodeOp::ReLU = node.op {
            // Check input
            if let Some(input_id) = node.inputs.first() {
                let input_node = &graph.nodes[*input_id];
                if let NodeOp::Conv2d { .. } = input_node.op {
                    println!("Found candidate for fusion: Conv2d (Node {}) -> ReLU (Node {})", input_node.id, node.id);
                }
            }
        }
    }
    */
}
