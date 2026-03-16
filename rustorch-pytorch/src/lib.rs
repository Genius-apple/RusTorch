use anyhow::{anyhow, Result};
use rustorch_core::Tensor;
use std::path::Path;

pub struct PyTorchAdapter;

#[cfg(feature = "pytorch_backend")]
impl PyTorchAdapter {
    pub fn to_torch(tensor: &Tensor) -> tch::Tensor {
        let storage = tensor.storage();
        let data = storage.data();
        let shape: Vec<i64> = tensor.shape().iter().map(|&x| x as i64).collect();
        let t = tch::Tensor::from_slice(&data);
        t.reshape(&shape)
    }

    pub fn from_torch(tensor: &tch::Tensor) -> Result<Tensor> {
        let size: Vec<usize> = tensor.size().iter().map(|&x| x as usize).collect();
        let cpu_tensor = tensor.to_device(tch::Device::Cpu).contiguous();
        if cpu_tensor.kind() != tch::Kind::Float {
            let float_tensor = cpu_tensor.to_kind(tch::Kind::Float);
            let numel = float_tensor.numel();
            let mut data = vec![0.0f32; numel];
            float_tensor.copy_data(&mut data, numel);
            Ok(Tensor::new(&data, &size))
        } else {
            let numel = cpu_tensor.numel();
            let mut data = vec![0.0f32; numel];
            cpu_tensor.copy_data(&mut data, numel);
            Ok(Tensor::new(&data, &size))
        }
    }

    pub fn load_state_dict<P: AsRef<Path>>(
        path: P,
    ) -> Result<std::collections::HashMap<String, Tensor>> {
        let tensors = tch::Tensor::load_multi(path)?;
        let mut result = std::collections::HashMap::new();
        for (name, tensor) in tensors {
            let rt_tensor = Self::from_torch(&tensor)?;
            result.insert(name, rt_tensor);
        }
        Ok(result)
    }

    pub fn save_state_dict<P: AsRef<Path>>(
        tensors: &std::collections::HashMap<String, Tensor>,
        path: P,
    ) -> Result<()> {
        let mut named_tensors = Vec::new();
        for (name, tensor) in tensors {
            let t = Self::to_torch(tensor);
            named_tensors.push((name.clone(), t));
        }
        let named_tensors_refs: Vec<(&str, tch::Tensor)> = named_tensors
            .iter()
            .map(|(n, t)| (n.as_str(), t.shallow_clone()))
            .collect();
        tch::Tensor::save_multi(&named_tensors_refs, path)?;
        Ok(())
    }
}

#[cfg(not(feature = "pytorch_backend"))]
impl PyTorchAdapter {
    pub fn load_state_dict<P: AsRef<Path>>(
        _path: P,
    ) -> Result<std::collections::HashMap<String, Tensor>> {
        Err(anyhow!("pytorch_backend feature is disabled"))
    }

    pub fn save_state_dict<P: AsRef<Path>>(
        _tensors: &std::collections::HashMap<String, Tensor>,
        _path: P,
    ) -> Result<()> {
        Err(anyhow!("pytorch_backend feature is disabled"))
    }
}

#[cfg(feature = "pytorch_backend")]
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

#[cfg(not(feature = "pytorch_backend"))]
pub mod ops {
    use super::*;
    use rustorch_core::Tensor;

    pub fn add(_a: &Tensor, _b: &Tensor) -> Result<Tensor> {
        Err(anyhow!("pytorch_backend feature is disabled"))
    }

    pub fn sub(_a: &Tensor, _b: &Tensor) -> Result<Tensor> {
        Err(anyhow!("pytorch_backend feature is disabled"))
    }

    pub fn mul(_a: &Tensor, _b: &Tensor) -> Result<Tensor> {
        Err(anyhow!("pytorch_backend feature is disabled"))
    }

    pub fn matmul(_a: &Tensor, _b: &Tensor) -> Result<Tensor> {
        Err(anyhow!("pytorch_backend feature is disabled"))
    }
}

#[cfg(all(test, feature = "pytorch_backend"))]
mod tests {
    use super::*;
    use rustorch_core::Tensor;

    #[test]
    fn test_conversion() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let shape = vec![2, 2];
        let rt_tensor = Tensor::new(&data, &shape);
        let torch_tensor = PyTorchAdapter::to_torch(&rt_tensor);
        assert_eq!(torch_tensor.size(), vec![2, 2]);
        let numel = torch_tensor.numel();
        let mut data_vec = vec![0.0f32; numel as usize];
        torch_tensor.copy_data(&mut data_vec, numel as usize);
        assert_eq!(data_vec, data);
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
