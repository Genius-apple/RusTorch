use std::fmt;
use std::sync::{Arc, RwLock, RwLockReadGuard, RwLockWriteGuard};

#[cfg(feature = "cuda")]
use cudarc::driver::CudaSlice;
#[cfg(feature = "wgpu_backend")]
use wgpu;
#[cfg(feature = "vulkan_backend")]
use vulkano::buffer::Subbuffer;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Device {
    Cpu,
    Cuda(usize),
    Metal(usize),
    Wgpu(usize),
    Vulkan(usize),
}

#[derive(Clone)]
enum StorageImpl {
    Cpu(Arc<RwLock<Vec<f32>>>),
    #[cfg(feature = "cuda")]
    Cuda(Arc<CudaSlice<f32>>),
    #[cfg(not(feature = "cuda"))]
    #[allow(dead_code)]
    CudaStub,
    #[cfg(feature = "wgpu_backend")]
    Wgpu(Arc<wgpu::Buffer>, usize), // Buffer and element count
    #[cfg(feature = "vulkan_backend")]
    Vulkan(Arc<Subbuffer<[f32]>>),
}

#[derive(Clone)]
pub struct Storage {
    inner: StorageImpl,
    device: Device,
}

impl Storage {
    pub fn new(data: Vec<f32>) -> Self {
        Self {
            inner: StorageImpl::Cpu(Arc::new(RwLock::new(data))),
            device: Device::Cpu,
        }
    }

    #[cfg(feature = "cuda")]
    pub fn new_cuda(data: CudaSlice<f32>, device_id: usize) -> Self {
        Self {
            inner: StorageImpl::Cuda(Arc::new(data)),
            device: Device::Cuda(device_id),
        }
    }

    #[cfg(feature = "wgpu_backend")]
    pub fn new_wgpu(buffer: Arc<wgpu::Buffer>, size: usize, device_id: usize) -> Self {
        Self {
            inner: StorageImpl::Wgpu(buffer, size),
            device: Device::Wgpu(device_id),
        }
    }

    #[cfg(feature = "vulkan_backend")]
    pub fn new_vulkan(buffer: Arc<Subbuffer<[f32]>>, device_id: usize) -> Self {
        Self {
            inner: StorageImpl::Vulkan(buffer),
            device: Device::Vulkan(device_id),
        }
    }

    #[cfg(feature = "wgpu_backend")]
    pub fn wgpu_buffer(&self) -> Option<&wgpu::Buffer> {
        match &self.inner {
            StorageImpl::Wgpu(buffer, _) => Some(buffer),
            _ => None,
        }
    }

    #[cfg(feature = "vulkan_backend")]
    pub fn vulkan_buffer(&self) -> Option<&Subbuffer<[f32]>> {
        match &self.inner {
            StorageImpl::Vulkan(buffer) => Some(buffer),
            _ => None,
        }
    }

    pub fn from_slice(data: &[f32]) -> Self {
        Self::new(data.to_vec())
    }

    pub fn zeros(size: usize) -> Self {
        Self::new(vec![0.0; size])
    }

    pub fn data(&self) -> RwLockReadGuard<'_, Vec<f32>> {
        match &self.inner {
            StorageImpl::Cpu(data) => data.read().expect("Lock poisoned"),
            _ => panic!("data() accessor only supported on CPU tensors. Use to_device() to move to CPU first."),
        }
    }

    pub fn data_mut(&self) -> RwLockWriteGuard<'_, Vec<f32>> {
        match &self.inner {
            StorageImpl::Cpu(data) => data.write().expect("Lock poisoned"),
            _ => panic!("data_mut() accessor only supported on CPU tensors."),
        }
    }

    pub fn as_slice(&self) -> RwLockReadGuard<'_, Vec<f32>> {
        self.data()
    }

    pub fn len(&self) -> usize {
        match &self.inner {
            StorageImpl::Cpu(data) => data.read().expect("Lock poisoned").len(),
            #[cfg(feature = "cuda")]
            StorageImpl::Cuda(data) => data.len(),
            #[cfg(not(feature = "cuda"))]
            #[allow(unused_variables)]
            StorageImpl::CudaStub => 0,
            #[cfg(feature = "wgpu_backend")]
            StorageImpl::Wgpu(_, size) => *size,
            #[cfg(feature = "vulkan_backend")]
            StorageImpl::Vulkan(buf) => buf.len() as usize,
        }
    }#[cfg(feature = "wgpu_backend")]
            StorageImpl::Wgpu(_, size) => *size,
            #[cfg(feature = "vulkan_backend")]
            StorageImpl::Vulkan(buf) => buf.len() as usize,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn device(&self) -> Device {
        self.device
    }

    pub fn to_device(&self, device: Device) -> Self {
        if self.device == device {
            return self.clone();
        }

        match (self.device, device) {
            (Device::Cpu, Device::Cuda(_id)) => {
                // Implement CPU -> CUDA transfer
                #[cfg(feature = "cuda")]
                {
                    // Need a way to get CudaDevice instance.
                    // Usually managed by a global context manager.
                    // For now, panic or todo.
                    todo!("Implement CPU -> CUDA transfer")
                }
                #[cfg(not(feature = "cuda"))]
                panic!("CUDA feature not enabled")
            }
            (Device::Cuda(_), Device::Cpu) => {
                // Implement CUDA -> CPU transfer
                #[cfg(feature = "cuda")]
                {
                    // Read from GPU
                    todo!("Implement CUDA -> CPU transfer")
                }
                #[cfg(not(feature = "cuda"))]
                panic!("CUDA feature not enabled")
            }
            _ => todo!(
                "Transfer between {:?} and {:?} not implemented",
                self.device,
                device
            ),
        }
    }
}

impl fmt::Debug for Storage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.inner {
            StorageImpl::Cpu(data) => {
                let guard = data.read().unwrap();
                write!(f, "Storage({:?}, size={})", self.device, guard.len())
            }
            #[cfg(feature = "cuda")]
            StorageImpl::Cuda(data) => {
                write!(f, "CudaStorage({:?}, size={})", self.device, data.len())
            }
            #[cfg(not(feature = "cuda"))]
            StorageImpl::CudaStub => {
                write!(f, "CudaStorageStub({:?})", self.device)
            }
            #[cfg(feature = "wgpu_backend")]
            StorageImpl::Wgpu(_, size) => {
                write!(f, "WgpuStorage({:?}, size={})", self.device, size)
            }
            #[cfg(feature = "vulkan_backend")]
            StorageImpl::Vulkan(buf) => {
                write!(f, "VulkanStorage({:?}, size={})", self.device, buf.len())
            }
        }
    }
}
