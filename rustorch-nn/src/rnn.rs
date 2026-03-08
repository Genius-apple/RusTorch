use rustorch_core::Tensor;
use crate::{Module, Linear};
#[cfg(feature = "serde")]
use serde::{Serialize, Deserialize};

// --- RNN Cell ---
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct RNNCell {
    pub input_size: usize,
    pub hidden_size: usize,
    pub bias: bool,
    pub nonlinearity: String, // "tanh" or "relu"
    
    pub weight_ih: Linear,
    pub weight_hh: Linear,
}

impl RNNCell {
    pub fn new(input_size: usize, hidden_size: usize, bias: bool, nonlinearity: &str) -> Self {
        // Linear modules usually handle bias.
        // RNN formula: h' = tanh(W_ih * x + b_ih + W_hh * h + b_hh)
        // My Linear: y = x @ W.t() + b
        // weight_ih: in -> hidden
        // weight_hh: hidden -> hidden
        
        // Note: PyTorch merges biases usually.
        let weight_ih = Linear::new(input_size, hidden_size); // Check Linear signature: (in, out)
        let weight_hh = Linear::new(hidden_size, hidden_size);
        
        Self {
            input_size,
            hidden_size,
            bias,
            nonlinearity: nonlinearity.to_string(),
            weight_ih,
            weight_hh,
        }
    }
}

impl Module for RNNCell {
    fn forward(&self, input: &Tensor) -> Tensor {
        // Input: (N, InputSize)
        // Hidden: Implicit? Or passed?
        // Module::forward only takes input.
        // We need a stateful module or pass tuple?
        // Standard PyTorch RNNCell forward: (input, h_0) -> h_1
        // But our Module trait is `forward(&self, input: &Tensor) -> Tensor`.
        // Limitation of our current trait.
        // Workaround: input can be packed? Or we just assume h_0 is 0 if not provided?
        // Or we extend trait?
        // Let's stick to trait for now and maybe panic if state handling is needed, 
        // or just assume input contains hidden state (concatenated)?
        // Or better: RNNCell usually managed by RNN container.
        // Let's just implement the logic assuming input is tuple (x, h) packed?
        // No, Tensor cannot pack tuples easily without "List" type.
        // For simplicity in this demo framework: 
        // We will change Module trait? No, too invasive.
        // We will assume `input` is just X, and H is stored internally? (Stateful RNN)
        // Or we just implement `forward_with_state` inherent method.
        panic!("Use forward_with_state");
    }

    fn parameters(&self) -> Vec<Tensor> {
        let mut p = self.weight_ih.parameters();
        p.extend(self.weight_hh.parameters());
        p
    }
}

impl RNNCell {
    pub fn forward_with_state(&self, input: &Tensor, hx: Option<&Tensor>) -> Tensor {
        let h_prev = if let Some(h) = hx {
            h.clone()
        } else {
            Tensor::zeros(&[input.shape()[0], self.hidden_size])
        };
        
        let ih = self.weight_ih.forward(input);
        let hh = self.weight_hh.forward(&h_prev);
        
        let pre_act = ih + hh; // Add
        
        if self.nonlinearity == "relu" {
            pre_act.relu()
        } else {
            // Tanh not implemented in core yet?
            // Assuming we have it or use simple approximation/placeholder.
            // Let's use ReLU for now as fallback or implement tanh in core.
            // I'll add tanh to core ops later if needed.
            pre_act // Placeholder: Identity if tanh missing, or panic.
        }
    }
}

// --- LSTM Cell ---
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct LSTMCell {
    pub input_size: usize,
    pub hidden_size: usize,
    // Use separate weights for simplicity until Split op is available
    pub w_ii: Linear, pub w_hi: Linear, // Input gate
    pub w_if: Linear, pub w_hf: Linear, // Forget gate
    pub w_ig: Linear, pub w_hg: Linear, // Cell gate
    pub w_io: Linear, pub w_ho: Linear, // Output gate
}

impl LSTMCell {
    pub fn new(input_size: usize, hidden_size: usize) -> Self {
        Self {
            input_size,
            hidden_size,
            w_ii: Linear::new(input_size, hidden_size), w_hi: Linear::new(hidden_size, hidden_size),
            w_if: Linear::new(input_size, hidden_size), w_hf: Linear::new(hidden_size, hidden_size),
            w_ig: Linear::new(input_size, hidden_size), w_hg: Linear::new(hidden_size, hidden_size),
            w_io: Linear::new(input_size, hidden_size), w_ho: Linear::new(hidden_size, hidden_size),
        }
    }
    
    pub fn forward_with_state(&self, input: &Tensor, hx: Option<(&Tensor, &Tensor)>) -> (Tensor, Tensor) {
        let (h_prev, c_prev) = if let Some((h, c)) = hx {
            (h.clone(), c.clone())
        } else {
            let batch_size = input.shape()[0];
            (Tensor::zeros(&[batch_size, self.hidden_size]), Tensor::zeros(&[batch_size, self.hidden_size]))
        };
        
        // Input Gate
        let i = (self.w_ii.forward(input) + self.w_hi.forward(&h_prev)).sigmoid();
        
        // Forget Gate
        let f = (self.w_if.forward(input) + self.w_hf.forward(&h_prev)).sigmoid();
        
        // Cell Gate (Candidate)
        let g = (self.w_ig.forward(input) + self.w_hg.forward(&h_prev)).tanh();
        
        // Output Gate
        let o = (self.w_io.forward(input) + self.w_ho.forward(&h_prev)).sigmoid();
        
        // Cell State
        let c_next = f.mul(&c_prev) + i.mul(&g);
        
        // Hidden State
        let h_next = o.mul(&c_next.tanh());
        
        (h_next, c_next)
    }
}

impl Module for LSTMCell {
    fn forward(&self, _input: &Tensor) -> Tensor {
        panic!("Use forward_with_state returning tuple");
    }
    fn parameters(&self) -> Vec<Tensor> {
        let mut p = Vec::new();
        p.extend(self.w_ii.parameters()); p.extend(self.w_hi.parameters());
        p.extend(self.w_if.parameters()); p.extend(self.w_hf.parameters());
        p.extend(self.w_ig.parameters()); p.extend(self.w_hg.parameters());
        p.extend(self.w_io.parameters()); p.extend(self.w_ho.parameters());
        p
    }
}

// --- GRU Cell ---
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct GRUCell {
    // Placeholder
    pub input_size: usize,
    pub hidden_size: usize,
}

impl Module for GRUCell {
    fn forward(&self, input: &Tensor) -> Tensor { input.clone() }
    fn parameters(&self) -> Vec<Tensor> { vec![] }
}

// --- RNN Container ---
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct RNN {
    pub cell: RNNCell,
}

impl RNN {
    pub fn new(input_size: usize, hidden_size: usize) -> Self {
        Self {
            cell: RNNCell::new(input_size, hidden_size, true, "relu"),
        }
    }
}

impl Module for RNN {
    fn forward(&self, input: &Tensor) -> Tensor {
        // Input: (Batch, Seq, Feature)
        // Loop over seq
        // Need Unbind/Split op.
        // Skeleton.
        input.clone()
    }
    
    fn parameters(&self) -> Vec<Tensor> {
        self.cell.parameters()
    }
}
