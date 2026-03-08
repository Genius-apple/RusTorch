use rustorch_core::{Tensor, jit::{Graph, NodeType, Optimizer, Executor}};

fn main() {
    println!("--- JIT Optimization & Permute Demo ---");
    
    // 1. Permute Demo
    let t = Tensor::new(&vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    println!("Original (2x3): \n{}", t);
    
    let p = t.permute(&[1, 0]); // Transpose -> (3, 2)
    println!("Permuted (3x2): \n{}", p);
    println!("Permuted shape: {:?}", p.shape());
    println!("Permuted strides: {:?}", p.strides());
    
    let c = p.contiguous();
    println!("Contiguous (3x2): \n{}", c);
    println!("Contiguous strides: {:?}", c.strides());
    
    assert_eq!(c.data()[0], 1.0); // (0,0) -> 1
    assert_eq!(c.data()[1], 4.0); // (0,1) -> (1,0) in orig -> 4
    
    println!("Permute OK!");
    
    // 2. JIT Graph Demo
    // Scenario: Conv2d -> ReLU fusion
    
    let mut graph = Graph::new();
    
    // Define inputs
    let input_shape = vec![1, 1, 4, 4];
    let weight_shape = vec![1, 1, 3, 3];
    
    let input_id = graph.add_input(input_shape.clone());
    let weight_id = graph.add_input(weight_shape.clone()); // Treat weight as input for simplicity or add_weight
    
    // Add Conv2d Node
    let conv_out_shape = vec![1, 1, 2, 2];
    let conv_id = graph.add_node(NodeType::Conv2d(input_id, weight_id, (1, 1), (0, 0)), conv_out_shape.clone());
    
    // Add ReLU Node
    let relu_id = graph.add_node(NodeType::Relu(conv_id), conv_out_shape.clone());
    
    graph.outputs.push(relu_id);
    
    println!("Original Graph Nodes: {:?}", graph.nodes.len());
    for node in &graph.nodes {
        println!(" - {:?}", node.op);
    }
    
    // Optimize
    Optimizer::optimize(&mut graph);
    
    println!("Optimized Graph Nodes: {:?}", graph.nodes.len());
    for node in &graph.nodes {
        println!(" - {:?}", node.op);
    }
    
    // Verify Fusion
    let fused = graph.nodes.iter().any(|n| matches!(n.op, NodeType::Conv2dRelu(..)));
    if fused {
        println!("Fusion Successful: Conv2dRelu found!");
    } else {
        println!("Fusion Failed!");
    }
    
    // Execute
    let input = Tensor::ones(&[1, 1, 4, 4]);
    let weight = Tensor::ones(&[1, 1, 3, 3]);
    
    let results = Executor::run(&graph, &[input, weight]);
    let output = &results[0];
    
    println!("Output shape: {:?}", output.shape());
    println!("Output data[0]: {}", output.data()[0]);
    
    // Expected: Conv of ones with 3x3 kernel of ones -> 9.0. ReLU(9.0) -> 9.0.
    // Wait, 3x3 ones sum is 9.
    
    assert!((output.data()[0] - 9.0).abs() < 1e-5);
    println!("Execution OK!");
}
