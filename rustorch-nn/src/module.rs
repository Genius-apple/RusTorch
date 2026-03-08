use rustorch_core::Tensor;

pub trait Module {
    fn forward(&self, input: &Tensor) -> Tensor;
    fn parameters(&self) -> Vec<Tensor>;
}
