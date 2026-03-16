use parking_lot::{RwLock, RwLockReadGuard, RwLockWriteGuard};
use std::fmt;
use std::sync::Arc;

#[cfg(feature = "cuda")]
use cudarc::driver::CudaSlice;
#[cfg(feature = "vulkan_backend")]
use vulkano::buffer::Subbuffer;
#[cfg(feature = "wgpu_backend")]
use wgpu;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Device {
    Cpu,
    Cuda(usize),
    Metal(usize),
    Wgpu(usize),
    Vulkan(usize),
}

impl Device {
    pub fn is_wgpu(&self) -> bool {
        match self {
            Device::Wgpu(_) => true,
            _ => false,
        }
    }
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
    Wgpu(Arc<PooledBuffer>, usize),

    #[cfg(feature = "vulkan_backend")]
    Vulkan(Arc<Subbuffer<[f32]>>),
}

#[cfg(feature = "wgpu_backend")]
pub struct PooledBuffer {
    buffer: Option<wgpu::Buffer>,
    size: u64,
}

#[cfg(feature = "wgpu_backend")]
impl Drop for PooledBuffer {
    fn drop(&mut self) {
        if let Some(buf) = self.buffer.take() {
            crate::backend::wgpu::get_memory_pool().return_buffer(buf, self.size);
        }
    }
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
    pub fn new_wgpu(buffer: wgpu::Buffer, size: usize, _device_id: usize) -> Self {
        let size_bytes = (size * std::mem::size_of::<f32>()) as u64;
        let pooled = PooledBuffer {
            buffer: Some(buffer),
            size: size_bytes,
        };
        Self {
            inner: StorageImpl::Wgpu(Arc::new(pooled), size),
            device: Device::Wgpu(_device_id),
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
            StorageImpl::Wgpu(pooled, _) => pooled.buffer.as_ref(),
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
            StorageImpl::Cpu(data) => data.read(),
            #[cfg(feature = "wgpu_backend")]
            StorageImpl::Wgpu(_, _) => {
                // Temporary workaround: panic with clear message
                // Ideally, we should not access data() on WGPU tensor without to_cpu()
                // But some code paths might do it implicitly.
                // We CANNOT return a RwLockReadGuard here because we don't have the data locally locked.
                // We must panic or refactor `data()` to return `Cow<[f32]>` or similar, but that breaks API.

                println!(
                    "CRITICAL ERROR: data() called on non-CPU storage. Device: {:?}",
                    self.device
                );
                panic!("data() accessor only supported on CPU tensors. Use to_device() to move to CPU first.");
            }
            _ => {
                println!(
                    "CRITICAL ERROR: data() called on non-CPU storage. Device: {:?}",
                    self.device
                );
                panic!("data() accessor only supported on CPU tensors. Use to_device() to move to CPU first.");
            }
        }
    }

    pub fn data_mut(&self) -> RwLockWriteGuard<'_, Vec<f32>> {
        match &self.inner {
            StorageImpl::Cpu(data) => data.write(),
            _ => panic!("data_mut() accessor only supported on CPU tensors."),
        }
    }

    pub fn as_slice(&self) -> RwLockReadGuard<'_, Vec<f32>> {
        self.data()
    }

    pub fn len(&self) -> usize {
        match &self.inner {
            StorageImpl::Cpu(data) => data.read().len(),
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
                let guard = data.read();
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
