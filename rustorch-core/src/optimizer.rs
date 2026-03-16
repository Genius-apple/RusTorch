use crate::Tensor;
use rayon::prelude::*;
use std::collections::HashMap;
use std::sync::Arc;
use wide::f32x8;

pub struct Adam {
    params: Vec<Tensor>,
    lr: f32,
    beta1: f32,
    beta2: f32,
    epsilon: f32,
    step: u32,
    state: HashMap<usize, (Tensor, Tensor)>, // (m, v) keyed by param unique ID
}

impl Adam {
    pub fn new(params: Vec<Tensor>, lr: f32) -> Self {
        Self {
            params,
            lr,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            step: 0,
            state: HashMap::new(),
        }
    }

    pub fn step(&mut self) {
        self.step += 1;

        for param in self.params.iter() {
            let grad = if let Some(g) = param.grad() {
                g
            } else {
                continue;
            };

            let is_wgpu = param.storage().wgpu_buffer().is_some();

            let param_id = Arc::as_ptr(&param.inner) as usize;

            self.state.entry(param_id).or_insert_with(|| {
                let m = if is_wgpu {
                    Tensor::zeros(param.shape()).to_wgpu()
                } else {
                    Tensor::zeros(param.shape())
                };
                let v = if is_wgpu {
                    Tensor::zeros(param.shape()).to_wgpu()
                } else {
                    Tensor::zeros(param.shape())
                };
                (m, v)
            });

            let (m, v) = self.state.get(&param_id).unwrap();

            let beta1 = self.beta1;
            let beta2 = self.beta2;
            let eps = self.epsilon;
            let lr = self.lr;
            let step = self.step as f32;
            let bias_correction1 = 1.0 - beta1.powf(step);
            let bias_correction2 = 1.0 - beta2.powf(step);

            if is_wgpu {
                #[cfg(feature = "wgpu_backend")]
                {
                    let grad_wgpu = if grad.storage().wgpu_buffer().is_none() {
                        grad.to_wgpu()
                    } else {
                        grad
                    };

                    let param_buf = param.storage().wgpu_buffer().unwrap();
                    let grad_buf = grad_wgpu.storage().wgpu_buffer().unwrap();
                    let m_buf = m.storage().wgpu_buffer().unwrap();
                    let v_buf = v.storage().wgpu_buffer().unwrap();

                    crate::backend::wgpu::adam_step_wgpu(
                        param_buf,
                        grad_buf,
                        m_buf,
                        v_buf,
                        param.shape().iter().product(),
                        self.lr,
                        self.beta1,
                        self.beta2,
                        self.epsilon,
                        self.step,
                    );
                    crate::backend::wgpu::flush_queue();
                }
            } else {
                let mut p_data = param.data_mut();
                let g_data = grad.data();
                let mut m_data = m.data_mut();
                let mut v_data = v.data_mut();
                let lanes = 8usize;
                let head = (p_data.len() / lanes) * lanes;

                p_data[..head]
                    .par_chunks_mut(lanes)
                    .zip(m_data[..head].par_chunks_mut(lanes))
                    .zip(v_data[..head].par_chunks_mut(lanes))
                    .zip(g_data[..head].par_chunks(lanes))
                    .for_each(|(((p_chunk, m_chunk), v_chunk), g_chunk)| {
                        let p_vec = f32x8::from([
                            p_chunk[0], p_chunk[1], p_chunk[2], p_chunk[3], p_chunk[4], p_chunk[5],
                            p_chunk[6], p_chunk[7],
                        ]);
                        let m_vec = f32x8::from([
                            m_chunk[0], m_chunk[1], m_chunk[2], m_chunk[3], m_chunk[4], m_chunk[5],
                            m_chunk[6], m_chunk[7],
                        ]);
                        let v_vec = f32x8::from([
                            v_chunk[0], v_chunk[1], v_chunk[2], v_chunk[3], v_chunk[4], v_chunk[5],
                            v_chunk[6], v_chunk[7],
                        ]);
                        let g_vec = f32x8::from([
                            g_chunk[0], g_chunk[1], g_chunk[2], g_chunk[3], g_chunk[4], g_chunk[5],
                            g_chunk[6], g_chunk[7],
                        ]);

                        let m_t = f32x8::splat(beta1) * m_vec + f32x8::splat(1.0 - beta1) * g_vec;
                        let v_t =
                            f32x8::splat(beta2) * v_vec + f32x8::splat(1.0 - beta2) * g_vec * g_vec;
                        let m_hat = m_t / f32x8::splat(bias_correction1);
                        let v_hat = v_t / f32x8::splat(bias_correction2);
                        let p_new =
                            p_vec - f32x8::splat(lr) * m_hat / (v_hat.sqrt() + f32x8::splat(eps));

                        let m_arr = m_t.to_array();
                        let v_arr = v_t.to_array();
                        let p_arr = p_new.to_array();

                        m_chunk.copy_from_slice(&m_arr);
                        v_chunk.copy_from_slice(&v_arr);
                        p_chunk.copy_from_slice(&p_arr);
                    });

                for i in head..p_data.len() {
                    let g = g_data[i];
                    let m_t = beta1 * m_data[i] + (1.0 - beta1) * g;
                    let v_t = beta2 * v_data[i] + (1.0 - beta2) * g * g;
                    m_data[i] = m_t;
                    v_data[i] = v_t;
                    let m_hat = m_t / bias_correction1;
                    let v_hat = v_t / bias_correction2;
                    p_data[i] -= lr * m_hat / (v_hat.sqrt() + eps);
                }
            }
        }
    }

    pub fn zero_grad(&self) {
        for param in &self.params {
            param.zero_grad();
        }
    }
}
