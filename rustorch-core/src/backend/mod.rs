#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Activation {
    None = 0,
    ReLU = 1,
    Sigmoid = 2,
    Tanh = 3,
}

#[cfg(feature = "wgpu_backend")]
pub mod wgpu;
