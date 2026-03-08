use rustorch_core::Tensor;

pub trait Dataset {
    fn len(&self) -> usize;
    fn get(&self, index: usize) -> (Tensor, Tensor); // (Input, Target)
}

pub struct DataLoader<D: Dataset> {
    dataset: D,
    batch_size: usize,
    shuffle: bool,
    // Iterator state
    indices: Vec<usize>,
    current_idx: usize,
}

impl<D: Dataset> DataLoader<D> {
    pub fn new(dataset: D, batch_size: usize, shuffle: bool) -> Self {
        let len = dataset.len();
        let indices: Vec<usize> = (0..len).collect();
        // TODO: Implement shuffle if true
        Self {
            dataset,
            batch_size,
            shuffle,
            indices,
            current_idx: 0,
        }
    }
}

impl<D: Dataset> Iterator for DataLoader<D> {
    type Item = (Tensor, Tensor); // (Batch Input, Batch Target)

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_idx >= self.dataset.len() {
            // Reset for next epoch? Usually Iterator consumes.
            // But DataLoader is often re-used.
            // For standard Iterator, return None.
            return None;
        }

        let end_idx = std::cmp::min(self.current_idx + self.batch_size, self.dataset.len());
        let batch_indices = &self.indices[self.current_idx..end_idx];
        self.current_idx = end_idx;

        let mut batch_inputs = Vec::new();
        let mut batch_targets = Vec::new();

        // Collect samples
        for &idx in batch_indices {
            let (input, target) = self.dataset.get(idx);
            // Assuming input and target are Tensors with shape (C, H, W) or (Features)
            // We need to stack them.
            // Since we don't have a `stack` op yet, let's manually concat data.
            batch_inputs.push(input);
            batch_targets.push(target);
        }

        // Stack logic
        // Verify shapes match
        if batch_inputs.is_empty() {
            return None;
        }
        let input_shape = batch_inputs[0].shape();
        let target_shape = batch_targets[0].shape();
        
        // Flatten data
        let mut batched_input_data = Vec::new();
        let mut batched_target_data = Vec::new();
        
        for t in &batch_inputs {
             let guard = t.data();
             batched_input_data.extend_from_slice(&*guard);
        }
        for t in &batch_targets {
             let guard = t.data();
             batched_target_data.extend_from_slice(&*guard);
        }
        
        // New shape: (Batch, ...)
        let mut new_input_shape = vec![batch_inputs.len()];
        new_input_shape.extend_from_slice(input_shape);
        
        // Target shape logic: if target is scalar (size 1, shape [1] or []), batch shape should be [Batch] or [Batch, 1]
        // If target is already Tensor(shape=[1]), new shape is [Batch, 1].
        // If target is scalar, shape might be [].
        // Let's assume target is always at least [1] or [C].
        
        let mut new_target_shape = vec![batch_targets.len()];
        new_target_shape.extend_from_slice(target_shape);
        
        let input_tensor = Tensor::new(&batched_input_data, &new_input_shape);
        let target_tensor = Tensor::new(&batched_target_data, &new_target_shape);
        
        Some((input_tensor, target_tensor))
    }
}
