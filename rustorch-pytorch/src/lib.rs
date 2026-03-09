use rustorch_core::Tensor;
use tch::{self, Kind, Device as TchDevice};
use anyhow::Result;
use std::path::Path;

pub struct PyTorchAdapter;

impl PyTorchAdapter {
    /// Convert a RusTorch Tensor to a PyTorch Tensor
    pub fn to_torch(tensor: &Tensor) -> tch::Tensor {
        let storage = tensor.storage();
        let data = storage.data(); // Read lock
        let shape: Vec<i64> = tensor.shape().iter().map(|&x| x as i64).collect();
        
        // Create a PyTorch tensor from the data
        // Currently assumes f32 and CPU
        let t = tch::Tensor::from_slice(&data);
        t.reshape(&shape)
    }

    /// Convert a PyTorch Tensor to a RusTorch Tensor
    pub fn from_torch(tensor: &tch::Tensor) -> Result<Tensor> {
        let size: Vec<usize> = tensor.size().iter().map(|&x| x as usize).collect();
        
        // Ensure the tensor is on CPU and is contiguous
        let cpu_tensor = tensor.to_device(TchDevice::Cpu).contiguous();
        
        // Check if the tensor is Float (f32)
        if cpu_tensor.kind() != Kind::Float {
            // Cast to float if not
            // For now, we only support f32 in RusTorch
            let float_tensor = cpu_tensor.to_kind(Kind::Float);
            let data: Vec<f32> = float_tensor.into();
            Ok(Tensor::new(&data, &size))
        } else {
            let data: Vec<f32> = cpu_tensor.into();
            Ok(Tensor::new(&data, &size))
        }
    }

    /// Load a PyTorch model (.pth) and return a dictionary of tensors (state_dict)
    pub fn load_state_dict<P: AsRef<Path>>(path: P) -> Result<std::collections::HashMap<String, Tensor>> {
        let tensors = tch::Tensor::load_multi(path)?;
        let mut result = std::collections::HashMap::new();
        
        for (name, tensor) in tensors {
            let rt_tensor = Self::from_torch(&tensor)?;
            result.insert(name, rt_tensor);
        }
        
        Ok(result)
    }

    /// Save a dictionary of RusTorch tensors to a .pth file
    pub fn save_state_dict<P: AsRef<Path>>(tensors: &std::collections::HashMap<String, Tensor>, path: P) -> Result<()> {
        let mut named_tensors = Vec::new();
        for (name, tensor) in tensors {
            let t = Self::to_torch(tensor);
            named_tensors.push((name.clone(), t));
        }
        
        // Convert to slice of (&str, Tensor)
        let named_tensors_refs: Vec<(&str, tch::Tensor)> = named_tensors.iter()
            .map(|(n, t)| (n.as_str(), t.shallow_clone())) // shallow_clone is cheap
            .collect();

        tch::Tensor::save_multi(&named_tensors_refs, path)?;
        Ok(())
    }
}

/// Operator mapping layer: Run operations using PyTorch backend
pub mod ops {
    use super::*;
    use rustorch_core::Tensor;

    pub fn add(a: &Tensor, b: &Tensor) -> Result<Tensor> {
        let ta = PyTorchAdapter::to_torch(a);
        let tb = PyTorchAdapter::to_torch(b);
        let res = ta + tb;
        PyTorchAdapter::from_torch(&res)
    }

    pub fn sub(a: &Tensor, b: &Tensor) -> Result<Tensor> {
        let ta = PyTorchAdapter::to_torch(a);
        let tb = PyTorchAdapter::to_torch(b);
        let res = ta - tb;
        PyTorchAdapter::from_torch(&res)
    }

    pub fn mul(a: &Tensor, b: &Tensor) -> Result<Tensor> {
        let ta = PyTorchAdapter::to_torch(a);
        let tb = PyTorchAdapter::to_torch(b);
        let res = ta * tb;
        PyTorchAdapter::from_torch(&res)
    }

    pub fn matmul(a: &Tensor, b: &Tensor) -> Result<Tensor> {
        let ta = PyTorchAdapter::to_torch(a);
        let tb = PyTorchAdapter::to_torch(b);
        let res = ta.matmul(&tb);
        PyTorchAdapter::from_torch(&res)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustorch_core::Tensor;

    #[test]
    fn test_conversion() {
        // Create RusTorch tensor
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let shape = vec![2, 2];
        let rt_tensor = Tensor::new(&data, &shape);

        // Convert to PyTorch
        let torch_tensor = PyTorchAdapter::to_torch(&rt_tensor);
        assert_eq!(torch_tensor.size(), vec![2, 2]);
        assert_eq!(Vec::<f32>::from(&torch_tensor), data);

        // Convert back
        let rt_tensor_back = PyTorchAdapter::from_torch(&torch_tensor).unwrap();
        assert_eq!(rt_tensor_back.shape(), shape.as_slice());
        assert_eq!(*rt_tensor_back.storage().data(), data);
    }
    
    #[test]
    fn test_ops() {
        let t1 = Tensor::new(&vec![1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let t2 = Tensor::new(&vec![1.0, 1.0, 1.0, 1.0], &[2, 2]);
        
        let res = ops::add(&t1, &t2).unwrap();
        assert_eq!(*res.storage().data(), vec![2.0, 3.0, 4.0, 5.0]);
    }
}
