use crate::Tensor;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

// --- Graph Tracing ---

#[derive(Debug, Clone)]
pub enum NodeOp {
    Input,
    Constant,
    Add,
    Sub,
    MatMul,
    Conv2d {
        stride: (usize, usize),
        padding: (usize, usize),
    },
    ReLU,
    MaxPool2d {
        kernel_size: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
    },
    BatchNorm2d,
    // ... other ops
}

#[derive(Debug, Clone)]
pub struct Node {
    pub id: usize,
    pub op: NodeOp,
    pub inputs: Vec<usize>, // Input Node IDs
    pub shape: Vec<usize>,
    pub name: Option<String>,
}

#[derive(Debug, Clone)]
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

    pub fn add_node(
        &mut self,
        op: NodeOp,
        inputs: Vec<usize>,
        shape: Vec<usize>,
        name: Option<String>,
    ) -> usize {
        let id = self.nodes.len();
        self.nodes.push(Node {
            id,
            op,
            inputs,
            shape,
            name,
        });
        id
    }

    pub fn print(&self) {
        println!("Graph:");
        for node in &self.nodes {
            println!(
                "  Node {}: {:?} shape={:?} inputs={:?}",
                node.id, node.op, node.shape, node.inputs
            );
        }
    }
}

// Global Tracer Context (Thread Local)
pub struct TracerContext {
    pub graph: Graph,
    pub tensor_map: HashMap<usize, usize>, // Tensor Inner Ptr -> Node ID
}

thread_local! {
    static TRACER_CTX: Mutex<Option<TracerContext>> = Mutex::new(None);
}

pub fn start_tracing() {
    TRACER_CTX.with(|ctx| {
        *ctx.lock().unwrap() = Some(TracerContext {
            graph: Graph::new(),
            tensor_map: HashMap::new(),
        });
    });
}

pub fn stop_tracing() -> Option<Graph> {
    TRACER_CTX.with(|ctx| {
        let mut guard = ctx.lock().unwrap();
        guard.take().map(|c| c.graph)
    })
}

pub fn is_tracing() -> bool {
    TRACER_CTX.with(|ctx| ctx.lock().unwrap().is_some())
}

fn get_node_id(tensor: &Tensor) -> Option<usize> {
    TRACER_CTX.with(|ctx| {
        if let Some(c) = ctx.lock().unwrap().as_ref() {
            let ptr = Arc::as_ptr(&tensor.inner) as usize;
            c.tensor_map.get(&ptr).cloned()
        } else {
            None
        }
    })
}

pub fn register_input(tensor: &Tensor, name: String) {
    TRACER_CTX.with(|ctx| {
        if let Some(c) = ctx.lock().unwrap().as_mut() {
            let node_id =
                c.graph
                    .add_node(NodeOp::Input, vec![], tensor.shape().to_vec(), Some(name));
            let ptr = Arc::as_ptr(&tensor.inner) as usize;
            c.tensor_map.insert(ptr, node_id);
            c.graph.inputs.push(node_id);
        }
    });
}

pub fn record_op(op: NodeOp, inputs: &[&Tensor], output: &Tensor) {
    // Collect input IDs
    let mut input_ids = Vec::new();
    for t in inputs {
        if let Some(id) = get_node_id(t) {
            input_ids.push(id);
        } else {
            // If input is not tracked, register as constant?
            // For now, let's assume it's a constant/param
            TRACER_CTX.with(|ctx| {
                if let Some(c) = ctx.lock().unwrap().as_mut() {
                    let id = c
                        .graph
                        .add_node(NodeOp::Constant, vec![], t.shape().to_vec(), None);
                    let ptr = Arc::as_ptr(&t.inner) as usize;
                    c.tensor_map.insert(ptr, id);
                    input_ids.push(id);
                }
            });
        }
    }

    TRACER_CTX.with(|ctx| {
        if let Some(c) = ctx.lock().unwrap().as_mut() {
            let node_id = c
                .graph
                .add_node(op, input_ids, output.shape().to_vec(), None);
            let ptr = Arc::as_ptr(&output.inner) as usize;
            c.tensor_map.insert(ptr, node_id);
        }
    });
}
