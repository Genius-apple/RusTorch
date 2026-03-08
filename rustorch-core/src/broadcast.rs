use crate::Tensor;
use crate::storage::Storage;

pub fn broadcast_shapes(shape1: &[usize], shape2: &[usize]) -> Option<Vec<usize>> {
    let len1 = shape1.len();
    let len2 = shape2.len();
    let max_len = std::cmp::max(len1, len2);
    
    let mut result_shape = vec![0; max_len];
    
    for i in 0..max_len {
        let dim1 = if i < len1 { shape1[len1 - 1 - i] } else { 1 };
        let dim2 = if i < len2 { shape2[len2 - 1 - i] } else { 1 };
        
        if dim1 == dim2 {
            result_shape[max_len - 1 - i] = dim1;
        } else if dim1 == 1 {
            result_shape[max_len - 1 - i] = dim2;
        } else if dim2 == 1 {
            result_shape[max_len - 1 - i] = dim1;
        } else {
            return None;
        }
    }
    
    Some(result_shape)
}

impl Tensor {
    // Eager expansion: copies data
    pub fn expand(&self, new_shape: &[usize]) -> Tensor {
        let current_shape = self.shape();
        
        // Basic compatibility check and compute strides for expansion
        // Here we simulate expansion by calculating indices
        
        let num_elements: usize = new_shape.iter().product();
        let mut new_data = Vec::with_capacity(num_elements);
        
        // We need to map index in new_shape to index in current_shape
        // This is slow but correct.
        
        // Precompute strides for index calculation
        let mut new_strides = vec![0; new_shape.len()];
        let mut stride = 1;
        for i in (0..new_shape.len()).rev() {
            new_strides[i] = stride;
            stride *= new_shape[i];
        }
        
        let mut old_strides = vec![0; current_shape.len()];
        let mut stride = 1;
        for i in (0..current_shape.len()).rev() {
            old_strides[i] = stride;
            stride *= current_shape[i];
        }
        
        let offset = new_shape.len() - current_shape.len();
        
        let old_data = self.data();
        
        for i in 0..num_elements {
            // Convert flat index i to new_shape coords
            let mut temp_i = i;
            let mut old_idx = 0;
            
            for dim in 0..new_shape.len() {
                let coord = temp_i / new_strides[dim];
                temp_i %= new_strides[dim];
                
                // Map coord to old coord
                if dim >= offset {
                    let old_dim = dim - offset;
                    let old_coord = if current_shape[old_dim] == 1 {
                        0
                    } else {
                        coord // Must match
                    };
                    old_idx += old_coord * old_strides[old_dim];
                }
            }
            
            new_data.push(old_data[old_idx]);
        }
        
        let storage = Storage::new(new_data);
        // Expand usually doesn't require grad on the expansion itself, 
        // but propagates grad.
        // If self requires grad, new tensor should too?
        // Yes, if input requires grad, output does.
        
        let mut tensor = Tensor::new_with_storage(storage, new_shape);
        if self.requires_grad() {
            tensor.set_requires_grad_mut(true);
            // Register ExpandBackward? 
            // For eager implementation, we don't strictly need ExpandBackward if we implement Sum correctly
            // But we need to link the graph.
            // Let's implement ExpandBackward in ops.rs or here.
        }
        tensor
    }
}
