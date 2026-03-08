use std::sync::Arc;
use crate::Tensor;
use crate::autograd::BackwardOp;
use crate::storage::Storage;
use rayon::prelude::*;

#[derive(Debug)]
pub struct EmbeddingBackward {
    pub input: Tensor, // indices
    pub weight: Tensor,
    pub num_embeddings: usize,
    pub embedding_dim: usize,
    pub padding_idx: Option<usize>,
}

impl BackwardOp for EmbeddingBackward {
    fn backward(&self, grad: &Tensor) {
        if self.weight.requires_grad() {
            // Grad is (N, *, EmbeddingDim)
            // Input is (N, *) indices
            // We need to scatter add grad to weight.grad
            // weight.grad shape: (NumEmbeddings, EmbeddingDim)
            
            // This is sparse update.
            // For simplicity, we can iterate over indices and accumulate.
            // But we need to lock weight.grad.
            
            // Let's create a dense grad tensor for weight first (inefficient but simple)
            // Or better: accumulate directly if possible.
            // Tensor::accumulate_grad expects a Tensor.
            
            // We need to implement a "SparseAccumulate" or just create a dense Zero tensor and fill it.
            let mut weight_grad_data = vec![0.0; self.num_embeddings * self.embedding_dim];
            
            let grad_guard = grad.data();
            let grad_data = &*grad_guard;
            
            let input_guard = self.input.data(); // These are f32, need to cast to usize
            let input_data = &*input_guard;
            
            // Check shapes
            // Input: (B...)
            // Grad: (B..., Dim)
            // Input len * Dim == Grad len
            let num_indices = input_data.len();
            let dim = self.embedding_dim;
            
            if grad_data.len() != num_indices * dim {
                panic!("Embedding backward shape mismatch");
            }
            
            // Iterate and accumulate
            // This part is hard to parallelize without atomic adds on weight_grad_data.
            // So run serial or use localized buffers. Serial for now.
            for (i, &idx_f) in input_data.iter().enumerate() {
                let idx = idx_f as usize;
                if let Some(pad) = self.padding_idx {
                    if idx == pad { continue; }
                }
                if idx >= self.num_embeddings {
                     // Index out of bounds, ignore or panic? PyTorch panics or errors.
                     continue; 
                }
                
                let grad_offset = i * dim;
                let weight_offset = idx * dim;
                
                for j in 0..dim {
                    weight_grad_data[weight_offset + j] += grad_data[grad_offset + j];
                }
            }
            
            let weight_grad = Tensor::new(&weight_grad_data, self.weight.shape());
            self.weight.accumulate_grad(&weight_grad);
            self.weight.backward_step();
        }
    }
}

pub fn embedding(
    input: &Tensor,
    weight: &Tensor,
    padding_idx: Option<usize>,
    _max_norm: Option<f32>,
    _norm_type: f32,
    _scale_grad_by_freq: bool,
    _sparse: bool
) -> Tensor {
    // Input: Indices (Arbitrary Shape) -> but stored as f32 in Tensor
    // Weight: (NumEmbeddings, EmbeddingDim)
    // Output: (InputShape..., EmbeddingDim)
    
    let weight_shape = weight.shape();
    if weight_shape.len() != 2 {
        panic!("Embedding weight must be 2D");
    }
    let num_embeddings = weight_shape[0];
    let embedding_dim = weight_shape[1];
    
    let input_guard = input.data();
    let input_data = &*input_guard;
    
    let weight_guard = weight.data();
    let weight_data = &*weight_guard;
    
    let num_indices = input_data.len();
    let mut output_data = vec![0.0; num_indices * embedding_dim];
    
    // Parallel lookup
    output_data.par_chunks_mut(embedding_dim).enumerate().for_each(|(i, out_row)| {
        let idx_f = input_data[i];
        let idx = idx_f as usize;
        
        if idx >= num_embeddings {
            // Panic in real scenario
            // panic!("Index {} out of bounds for embedding size {}", idx, num_embeddings);
            // But inside parallel iterator panic is messy.
            // Let's just fill 0 or clamp?
            // PyTorch: runtime error.
            return; 
        }
        
        if let Some(pad) = padding_idx {
            if idx == pad {
                // Zero vector
                out_row.fill(0.0);
                return;
            }
        }
        
        let weight_offset = idx * embedding_dim;
        let w_row = &weight_data[weight_offset..weight_offset + embedding_dim];
        out_row.copy_from_slice(w_row);
    });
    
    let mut output_shape = input.shape().to_vec();
    output_shape.push(embedding_dim);
    
    let storage = Storage::new(output_data);
    let mut tensor = Tensor::new_with_storage(storage, &output_shape);
    
    if weight.requires_grad() {
        tensor.set_requires_grad_mut(true);
        tensor.set_op(Arc::new(EmbeddingBackward {
            input: input.clone(),
            weight: weight.clone(),
            num_embeddings,
            embedding_dim,
            padding_idx,
        }));
    }
    
    tensor
}
