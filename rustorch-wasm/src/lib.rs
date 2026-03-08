use wasm_bindgen::prelude::*;
use rustorch_core::Tensor;

#[wasm_bindgen]
pub struct JsTensor {
    inner: Tensor
}

#[wasm_bindgen]
impl JsTensor {
    #[wasm_bindgen(constructor)]
    pub fn new(val: f32) -> Self {
        JsTensor {
            inner: Tensor::new(&[val], &[1])
        }
    }
    
    pub fn add(&self, other: &JsTensor) -> JsTensor {
        JsTensor {
            inner: &self.inner + &other.inner
        }
    }
}
