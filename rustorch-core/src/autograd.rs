use std::sync::{Arc, Mutex};
use crate::Tensor;

pub trait BackwardOp: std::fmt::Debug + Send + Sync {
    fn backward(&self, grad: &Tensor);
}

// Simple engine for now
pub fn backward(tensor: &Tensor, grad: &Tensor) {
    // DFS with visited set to avoid infinite loops if cycles exist (though usually DAG)
    // But for correct gradient accumulation in DAG, we need topological sort.
    // Here we just do recursive call.
    // To avoid re-computation, we should use a proper engine.
    
    // For now, let's just delegate to the Op.
    if let Some(op) = tensor.op() {
        op.backward(grad);
    }
}
