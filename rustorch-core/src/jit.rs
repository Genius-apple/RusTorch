use std::collections::HashMap;
use crate::Tensor;
use crate::ops;

// --- IR Definition ---

#[derive(Clone, Debug, PartialEq)]
pub enum NodeType {
    Input(usize), // Input index
    Weight(Tensor), // Captured weight (constant)
    
    // Ops
    Add(usize, usize), // LHS, RHS node indices
    Mul(usize, usize),
    MatMul(usize, usize),
    Relu(usize),
    Conv2d(usize, usize, (usize, usize), (usize, usize)), // Input, Weight, Stride, Padding
    
    // Fused Ops
    Conv2dRelu(usize, usize, (usize, usize), (usize, usize)),
    LinearRelu(usize, usize, usize), // Input, Weight, Bias (Optional?)
}

#[derive(Debug)]
pub struct Node {
    pub op: NodeType,
    pub shape: Vec<usize>,
    pub id: usize,
    // dependencies, users, etc.
}

#[derive(Debug)]
pub struct Graph {
    pub nodes: Vec<Node>,
    pub inputs: Vec<usize>,
    pub outputs: Vec<usize>,
}

impl Graph {
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            inputs: Vec::new(),
            outputs: Vec::new(),
        }
    }
    
    pub fn add_node(&mut self, op: NodeType, shape: Vec<usize>) -> usize {
        let id = self.nodes.len();
        self.nodes.push(Node { op, shape, id });
        id
    }
    
    pub fn add_input(&mut self, shape: Vec<usize>) -> usize {
        let id = self.add_node(NodeType::Input(self.inputs.len()), shape);
        self.inputs.push(id);
        id
    }
    
    pub fn add_weight(&mut self, tensor: Tensor) -> usize {
        self.add_node(NodeType::Weight(tensor.clone()), tensor.shape().to_vec())
    }
}

// --- Tracer ---
// A simple tracer that records operations.
// In a real framework, we would use a thread-local graph context or proxy tensors.
// Here we simulate tracing by manually building graph or using a "TracedTensor" wrapper.

// Let's implement a simple "Optimizer" pass.

pub struct Optimizer;

impl Optimizer {
    pub fn optimize(graph: &mut Graph) {
        Self::fuse_conv_relu(graph);
        // Self::eliminate_dead_code(graph);
    }
    
    fn fuse_conv_relu(graph: &mut Graph) {
        // Look for Conv2d -> Relu pattern
        // This requires analyzing graph topology.
        // For simplicity: Iterate nodes, if Relu(Conv2d(idx)), replace op.
        
        // We can't easily modify Vec while iterating.
        // And we need to redirect edges.
        // Simplified approach: Build new graph.
        
        let mut new_nodes = Vec::new();
        let mut mapping = HashMap::new(); // Old ID -> New ID
        
        // We iterate old nodes.
        // If we see Conv2d, we look ahead? No, usually we look at Relu and check input.
        
        // But to rebuild, we visit in topological order (which is index order here).
        
        let n = graph.nodes.len();
        let mut consumed = vec![false; n];
        
        for i in 0..n {
            if consumed[i] { continue; }
            
            let node = &graph.nodes[i];
            
            match &node.op {
                NodeType::Conv2d(input_id, weight_id, stride, padding) => {
                    // Check if this node is used ONLY by a Relu
                    // If so, we can fuse.
                    // We need use-def chains.
                    // For this demo, let's peek ahead. 
                    // If next node is Relu and takes this Conv2d as input, fuse.
                    // (This assumes linear ordering which is not guaranteed but common in sequential models)
                    
                    let mut fused = false;
                    // Find if any future node is Relu(i)
                    // Optimization: just check if next one is Relu(i)
                    if i + 1 < n {
                        if let NodeType::Relu(inp) = graph.nodes[i+1].op {
                            if inp == i {
                                // Found Fusion!
                                let new_id = new_nodes.len();
                                mapping.insert(i+1, new_id); // Relu maps to Fused
                                // Conv2d node maps to Fused? 
                                // Actually the output of Relu is the output of Fused.
                                // The output of Conv2d is consumed.
                                
                                // Remap inputs
                                let new_input = *mapping.get(input_id).unwrap_or(input_id);
                                let new_weight = *mapping.get(weight_id).unwrap_or(weight_id);
                                
                                new_nodes.push(Node {
                                    op: NodeType::Conv2dRelu(new_input, new_weight, *stride, *padding),
                                    shape: graph.nodes[i+1].shape.clone(),
                                    id: new_id,
                                });
                                
                                consumed[i+1] = true; // Skip Relu
                                fused = true;
                            }
                        }
                    }
                    
                    if !fused {
                        // Copy Conv2d
                        let new_id = new_nodes.len();
                        mapping.insert(i, new_id);
                        let new_input = *mapping.get(input_id).unwrap_or(input_id);
                        let new_weight = *mapping.get(weight_id).unwrap_or(weight_id);
                        
                        new_nodes.push(Node {
                            op: NodeType::Conv2d(new_input, new_weight, *stride, *padding),
                            shape: node.shape.clone(),
                            id: new_id,
                        });
                    }
                },
                
                // Generic copy for others
                op => {
                    let new_id = new_nodes.len();
                    mapping.insert(i, new_id);
                    
                    // Remap inputs
                    let new_op = match op {
                        NodeType::Add(a, b) => NodeType::Add(*mapping.get(a).unwrap_or(a), *mapping.get(b).unwrap_or(b)),
                        NodeType::Mul(a, b) => NodeType::Mul(*mapping.get(a).unwrap_or(a), *mapping.get(b).unwrap_or(b)),
                        NodeType::Relu(a) => NodeType::Relu(*mapping.get(a).unwrap_or(a)),
                        // ... copy others
                        _ => op.clone(),
                    };
                    
                    new_nodes.push(Node {
                        op: new_op,
                        shape: node.shape.clone(),
                        id: new_id,
                    });
                }
            }
        }
        
        graph.nodes = new_nodes;
        // Remap outputs
        for out in &mut graph.outputs {
            if let Some(&new_id) = mapping.get(out) {
                *out = new_id;
            }
        }
        // Remap inputs (Node IDs)
        for inp in &mut graph.inputs {
            if let Some(&new_id) = mapping.get(inp) {
                *inp = new_id;
            }
        }
    }
}

// --- Executor ---
pub struct Executor;

impl Executor {
    pub fn run(graph: &Graph, inputs: &[Tensor]) -> Vec<Tensor> {
        let mut values: HashMap<usize, Tensor> = HashMap::new();
        
        // Load inputs
        for (i, &id) in graph.inputs.iter().enumerate() {
            values.insert(id, inputs[i].clone());
        }
        
        for node in &graph.nodes {
            if values.contains_key(&node.id) { continue; } // Already computed (Input/Weight)
            
            let val = match &node.op {
                NodeType::Input(_) => panic!("Input should be loaded"),
                NodeType::Weight(t) => t.clone(),
                
                NodeType::Add(a, b) => {
                    let va = values.get(a).unwrap();
                    let vb = values.get(b).unwrap();
                    va.add(vb)
                },
                NodeType::Mul(a, b) => {
                    let va = values.get(a).unwrap();
                    let vb = values.get(b).unwrap();
                    va.mul(vb)
                },
                NodeType::Relu(a) => {
                    let va = values.get(a).unwrap();
                    va.relu()
                },
                NodeType::Conv2d(inp, w, stride, padding) => {
                    let va = values.get(inp).unwrap();
                    let vw = values.get(w).unwrap();
                    va.conv2d(vw, *stride, *padding)
                },
                
                // Fused Ops
                NodeType::Conv2dRelu(inp, w, stride, padding) => {
                    let va = values.get(inp).unwrap();
                    let vw = values.get(w).unwrap();
                    // In real XLA, this calls a fused kernel.
                    // Here we emulate by calling conv then relu.
                    // But we could dispatch to a specialized kernel if we had one.
                    let conv = va.conv2d(vw, *stride, *padding);
                    conv.relu()
                },
                
                _ => panic!("Op not implemented in executor"),
            };
            
            values.insert(node.id, val);
        }
        
        graph.outputs.iter().map(|id| values.get(id).unwrap().clone()).collect()
    }
}
