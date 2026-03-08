use rustorch_core::Tensor;
use crate::Module;
#[cfg(feature = "serde")]
use serde::{Serialize, Deserialize};

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Embedding {
    pub num_embeddings: usize,
    pub embedding_dim: usize,
    pub padding_idx: Option<usize>,
    pub max_norm: Option<f32>,
    pub norm_type: f32,
    pub scale_grad_by_freq: bool,
    pub sparse: bool,
    pub weight: Tensor,
}

impl Embedding {
    pub fn new(num_embeddings: usize, embedding_dim: usize) -> Self {
        // Init weight: Standard Normal (0, 1) usually.
        let weight = Tensor::new(&vec![0.0; num_embeddings * embedding_dim], &[num_embeddings, embedding_dim])
            .set_requires_grad(true);
        weight.normal_(0.0, 1.0);
        
        Self {
            num_embeddings,
            embedding_dim,
            padding_idx: None,
            max_norm: None,
            norm_type: 2.0,
            scale_grad_by_freq: false,
            sparse: false,
            weight,
        }
    }
}

impl Module for Embedding {
    fn forward(&self, input: &Tensor) -> Tensor {
        rustorch_core::ops::embedding(
            input,
            &self.weight,
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse
        )
    }

    fn parameters(&self) -> Vec<Tensor> {
        vec![self.weight.clone()]
    }
}
