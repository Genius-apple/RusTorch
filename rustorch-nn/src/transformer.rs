use rustorch_core::Tensor;
use crate::{Module, Linear, LayerNorm, Dropout};
#[cfg(feature = "serde")]
use serde::{Serialize, Deserialize};

// --- Multihead Attention ---
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct MultiheadAttention {
    pub embed_dim: usize,
    pub num_heads: usize,
    pub head_dim: usize,
    
    pub q_proj: Linear,
    pub k_proj: Linear,
    pub v_proj: Linear,
    pub out_proj: Linear,
}

impl MultiheadAttention {
    pub fn new(embed_dim: usize, num_heads: usize) -> Self {
        assert_eq!(embed_dim % num_heads, 0, "embed_dim must be divisible by num_heads");
        let head_dim = embed_dim / num_heads;
        
        Self {
            embed_dim,
            num_heads,
            head_dim,
            q_proj: Linear::new(embed_dim, embed_dim),
            k_proj: Linear::new(embed_dim, embed_dim),
            v_proj: Linear::new(embed_dim, embed_dim),
            out_proj: Linear::new(embed_dim, embed_dim),
        }
    }
    
    pub fn forward_multi(&self, query: &Tensor, key: &Tensor, value: &Tensor) -> Tensor {
        // Simple implementation: Single head equivalent if reshape not supported nicely.
        // Q: (N, L, E) -> (N, L, E)
        let q = self.q_proj.forward(query);
        let k = self.k_proj.forward(key);
        let v = self.v_proj.forward(value);
        
        // Scaled Dot Product
        // scores = Q @ K.T / sqrt(d)
        // K is (N, S, E). K.T should be (N, E, S).
        // RusTorch .t() is only 2D.
        // If N=1 (Batch 1), we can squeeze?
        
        // For MVP: Assume 2D input (L, E) (Batch size = 1 implicit or L=Batch*Seq)
        // But Transformer needs Seq and Batch.
        
        // Let's implement a simplified attention:
        // Assume input is (Batch, Seq, E) but we treat it as (Batch*Seq, E) for Linear?
        // No, Attention needs to mix Seq.
        
        // If we don't have Batched MatMul, we can loop over batch?
        // Or if Batch=1, it's just (Seq, E) @ (E, Seq).
        
        // Let's assume Batch=1 for this demo implementation if shape is 3D.
        // Or if shape is 2D (Seq, E), it works directly.
        
        // Check rank
        let q_shape = q.shape();
        let scores = if q_shape.len() == 2 {
            // (Seq, E)
            let kt = k.t(); // (E, Seq)
            let raw_scores = q.matmul(&kt); // (Seq, Seq)
            let d_k = self.head_dim as f32;
            // scale
            // We need scalar div or mul.
            // Implement scalar mul/div.
            // For now, use tensor filled with 1/sqrt(d_k)
            let scale = 1.0 / d_k.sqrt();
            // We don't have scalar mul op exposed on Tensor yet?
            // Let's multiply by constant tensor.
            let scale_t = Tensor::full(raw_scores.shape(), scale); 
            raw_scores.mul(&scale_t)
        } else {
             // 3D case: (Batch, Seq, E)
             // Fallback: Panic or simplify.
             // Let's panic for now or treat as 2D if B=1
             panic!("MultiheadAttention currently supports only 2D inputs (Batch=1 squeezed)");
        };
        
        // Softmax
        let attn_weights = scores.softmax(-1); // (Seq, Seq)
        
        // Output = Weights @ V
        let output = attn_weights.matmul(&v); // (Seq, Seq) @ (Seq, E) -> (Seq, E)
        
        let out = self.out_proj.forward(&output);
        out
    }
}

impl Module for MultiheadAttention {
    fn forward(&self, input: &Tensor) -> Tensor {
        // Self-attention
        self.forward_multi(input, input, input)
    }

    fn parameters(&self) -> Vec<Tensor> {
        let mut p = self.q_proj.parameters();
        p.extend(self.k_proj.parameters());
        p.extend(self.v_proj.parameters());
        p.extend(self.out_proj.parameters());
        p
    }
}

// --- Transformer Encoder Layer ---
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct TransformerEncoderLayer {
    pub self_attn: MultiheadAttention,
    pub linear1: Linear,
    pub dropout: Dropout,
    pub linear2: Linear,
    pub norm1: LayerNorm,
    pub norm2: LayerNorm,
    pub dropout1: Dropout,
    pub dropout2: Dropout,
}

impl TransformerEncoderLayer {
    pub fn new(d_model: usize, nhead: usize, dim_feedforward: usize, dropout: f32) -> Self {
        Self {
            self_attn: MultiheadAttention::new(d_model, nhead),
            linear1: Linear::new(d_model, dim_feedforward),
            dropout: Dropout::new(dropout),
            linear2: Linear::new(dim_feedforward, d_model),
            norm1: LayerNorm::new(vec![d_model]),
            norm2: LayerNorm::new(vec![d_model]),
            dropout1: Dropout::new(dropout),
            dropout2: Dropout::new(dropout),
        }
    }
}

impl Module for TransformerEncoderLayer {
    fn forward(&self, src: &Tensor) -> Tensor {
        // src: (S, N, E)
        let src2 = self.self_attn.forward(src);
        let src = src + &self.dropout1.forward(&src2);
        let src = self.norm1.forward(&src);
        
        let src2 = self.linear2.forward(&self.dropout.forward(&self.linear1.forward(&src).relu()));
        let src = src + &self.dropout2.forward(&src2);
        let src = self.norm2.forward(&src);
        src
    }

    fn parameters(&self) -> Vec<Tensor> {
        let mut p = self.self_attn.parameters();
        p.extend(self.linear1.parameters());
        p.extend(self.linear2.parameters());
        p.extend(self.norm1.parameters());
        p.extend(self.norm2.parameters());
        p
    }
}

// --- Transformer Encoder ---
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct TransformerEncoder {
    pub layers: Vec<TransformerEncoderLayer>,
}

impl Module for TransformerEncoder {
    fn forward(&self, input: &Tensor) -> Tensor {
        let mut out = input.clone();
        for layer in &self.layers {
            out = layer.forward(&out);
        }
        out
    }
    
    fn parameters(&self) -> Vec<Tensor> {
        self.layers.iter().flat_map(|l| l.parameters()).collect()
    }
}

// --- Transformer ---
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Transformer {
    pub encoder: TransformerEncoder,
    // Decoder...
}
