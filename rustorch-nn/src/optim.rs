use rustorch_core::Tensor;

pub trait Optimizer {
    fn step(&mut self);
    fn zero_grad(&mut self);
}

pub struct SGD {
    params: Vec<Tensor>,
    lr: f32,
    momentum: f32,
    velocities: Vec<Option<Tensor>>,
}

impl SGD {
    pub fn new(params: Vec<Tensor>, lr: f32, momentum: f32) -> Self {
        let len = params.len();
        Self {
            params,
            lr,
            momentum,
            velocities: vec![None; len],
        }
    }
}

impl Optimizer for SGD {
    fn zero_grad(&mut self) {
        for param in &self.params {
            param.zero_grad();
        }
    }

    fn step(&mut self) {
        for (i, param) in self.params.iter().enumerate() {
            if !param.requires_grad() {
                continue;
            }

            let grad_opt = param.grad();
            if let Some(grad) = grad_opt {
                let mut param_data = param.data_mut();
                let grad_data = grad.data();

                if self.momentum != 0.0 {
                    // Initialize velocity if needed
                    if self.velocities[i].is_none() {
                        self.velocities[i] = Some(Tensor::zeros(param.shape()));
                    }

                    let velocity_tensor = self.velocities[i].as_ref().unwrap();
                    let mut velocity_data = velocity_tensor.data_mut();

                    for ((p, g), v) in param_data
                        .iter_mut()
                        .zip(grad_data.iter())
                        .zip(velocity_data.iter_mut())
                    {
                        *v = self.momentum * *v + *g;
                        *p -= self.lr * *v;
                    }
                } else {
                    // Simple SGD: p = p - lr * g
                    for (p, g) in param_data.iter_mut().zip(grad_data.iter()) {
                        *p -= self.lr * *g;
                    }
                }
            }
        }
    }
}

pub struct Adam {
    params: Vec<Tensor>,
    lr: f32,
    beta1: f32,
    beta2: f32,
    epsilon: f32,
    step_t: usize,
    exp_avg: Vec<Option<Vec<f64>>>,
    exp_avg_sq: Vec<Option<Vec<f64>>>,
    #[cfg(feature = "wgpu_backend")]
    exp_avg_gpu: Vec<Option<Tensor>>,
    #[cfg(feature = "wgpu_backend")]
    exp_avg_sq_gpu: Vec<Option<Tensor>>,
}

impl Adam {
    pub fn new(params: Vec<Tensor>, lr: f32) -> Self {
        let len = params.len();
        Self {
            params,
            lr,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            step_t: 0,
            exp_avg: vec![None; len],
            exp_avg_sq: vec![None; len],
            #[cfg(feature = "wgpu_backend")]
            exp_avg_gpu: vec![None; len],
            #[cfg(feature = "wgpu_backend")]
            exp_avg_sq_gpu: vec![None; len],
        }
    }
}

impl Optimizer for Adam {
    fn step(&mut self) {
        self.step_t += 1;

        #[cfg(feature = "wgpu_backend")]
        {
            self.step_wgpu();
        }

        #[cfg(not(feature = "wgpu_backend"))]
        {
            self.step_cpu();
        }
    }

    fn zero_grad(&mut self) {
        for param in &self.params {
            param.zero_grad();
        }
    }
}

impl Adam {
    #[cfg(not(feature = "wgpu_backend"))]
    fn step_cpu(&mut self) {
        let t = self.step_t as f32;
        let beta1 = self.beta1;
        let beta2 = self.beta2;
        let epsilon = self.epsilon;
        let lr = self.lr;

        let beta1_f64 = beta1 as f64;
        let beta2_f64 = beta2 as f64;
        let epsilon_f64 = epsilon as f64;
        let lr_f64 = lr as f64;
        let t_f64 = t as f64;
        let bc1_f64 = 1.0 - beta1_f64.powf(t_f64);
        let bc2_f64 = 1.0 - beta2_f64.powf(t_f64);

        for (i, param) in self.params.iter().enumerate() {
            if !param.requires_grad() {
                continue;
            }

            let grad_opt = param.grad();
            if let Some(grad) = grad_opt {
                let grad_c = if grad.is_contiguous() {
                    grad
                } else {
                    grad.contiguous()
                };

                let mut param_data = param.data_mut();
                let grad_data = grad_c.data();
                let numel = param_data.len();

                if self.exp_avg[i].is_none() {
                    self.exp_avg[i] = Some(vec![0.0; numel]);
                    self.exp_avg_sq[i] = Some(vec![0.0; numel]);
                }

                let m_vec = self.exp_avg[i].as_mut().unwrap();
                let v_vec = self.exp_avg_sq[i].as_mut().unwrap();

                let chunk_size = 4096;

                if numel > chunk_size {
                    use rayon::prelude::*;
                    param_data
                        .par_iter_mut()
                        .zip(grad_data.par_iter())
                        .zip(m_vec.par_iter_mut())
                        .zip(v_vec.par_iter_mut())
                        .for_each(|(((p, g), m), v)| {
                            let g_f64 = *g as f64;
                            *m = beta1_f64 * *m + (1.0 - beta1_f64) * g_f64;
                            *v = beta2_f64 * *v + (1.0 - beta2_f64) * g_f64 * g_f64;
                            let m_hat = *m / bc1_f64;
                            let v_hat = *v / bc2_f64;
                            let step = lr_f64 * m_hat / (v_hat.sqrt() + epsilon_f64);
                            *p -= step as f32;
                        });
                } else {
                    for ((p, g), (m, v)) in param_data
                        .iter_mut()
                        .zip(grad_data.iter())
                        .zip(m_vec.iter_mut().zip(v_vec.iter_mut()))
                    {
                        let g_f64 = *g as f64;
                        *m = beta1_f64 * *m + (1.0 - beta1_f64) * g_f64;
                        *v = beta2_f64 * *v + (1.0 - beta2_f64) * g_f64 * g_f64;
                        let m_hat = *m / bc1_f64;
                        let v_hat = *v / bc2_f64;
                        let step = lr_f64 * m_hat / (v_hat.sqrt() + epsilon_f64);
                        *p -= step as f32;
                    }
                }
            }
        }
    }

    #[cfg(feature = "wgpu_backend")]
    fn step_wgpu(&mut self) {
        use rustorch_core::backend::wgpu::adam_step_wgpu;
        use rustorch_core::backend::wgpu::flush_queue;

        for (i, param) in self.params.iter().enumerate() {
            if !param.requires_grad() {
                continue;
            }

            let grad_opt = param.grad();
            if let Some(grad) = grad_opt {
                let param_buf = param.storage().wgpu_buffer();
                let grad_buf = grad.storage().wgpu_buffer();

                let use_gpu = param_buf.is_some() && grad_buf.is_some();

                if use_gpu {
                    flush_queue();
                    let grad_cpu = grad.to_cpu();
                    let grad_data = grad_cpu.data();
                    let _grad_sum: f32 = grad_data.iter().sum();

                    let numel = param.shape().iter().product();

                    if self.exp_avg_gpu[i].is_none() {
                        let m = Tensor::zeros(param.shape()).to_wgpu();
                        let v = Tensor::zeros(param.shape()).to_wgpu();
                        self.exp_avg_gpu[i] = Some(m);
                        self.exp_avg_sq_gpu[i] = Some(v);
                    }

                    let m_tensor = self.exp_avg_gpu[i].as_ref().unwrap();
                    let v_tensor = self.exp_avg_sq_gpu[i].as_ref().unwrap();

                    let m_buf = m_tensor.storage().wgpu_buffer().unwrap();
                    let v_buf = v_tensor.storage().wgpu_buffer().unwrap();

                    adam_step_wgpu(
                        param_buf.unwrap(),
                        grad_buf.unwrap(),
                        m_buf,
                        v_buf,
                        numel,
                        self.lr,
                        self.beta1,
                        self.beta2,
                        self.epsilon,
                        self.step_t as u32,
                    );
                } else {
                    let grad_c = if grad.is_contiguous() {
                        grad
                    } else {
                        grad.contiguous()
                    };

                    let mut param_data = param.data_mut();
                    let grad_data = grad_c.data();
                    let numel = param_data.len();

                    if self.exp_avg[i].is_none() {
                        self.exp_avg[i] = Some(vec![0.0; numel]);
                        self.exp_avg_sq[i] = Some(vec![0.0; numel]);
                    }

                    let m_vec = self.exp_avg[i].as_mut().unwrap();
                    let v_vec = self.exp_avg_sq[i].as_mut().unwrap();

                    let beta1_f64 = self.beta1 as f64;
                    let beta2_f64 = self.beta2 as f64;
                    let epsilon_f64 = self.epsilon as f64;
                    let lr_f64 = self.lr as f64;
                    let t_f64 = self.step_t as f64;
                    let bc1_f64 = 1.0 - beta1_f64.powf(t_f64);
                    let bc2_f64 = 1.0 - beta2_f64.powf(t_f64);

                    for ((p, g), (m, v)) in param_data
                        .iter_mut()
                        .zip(grad_data.iter())
                        .zip(m_vec.iter_mut().zip(v_vec.iter_mut()))
                    {
                        let g_f64 = *g as f64;
                        *m = beta1_f64 * *m + (1.0 - beta1_f64) * g_f64;
                        *v = beta2_f64 * *v + (1.0 - beta2_f64) * g_f64 * g_f64;
                        let m_hat = *m / bc1_f64;
                        let v_hat = *v / bc2_f64;
                        let step = lr_f64 * m_hat / (v_hat.sqrt() + epsilon_f64);
                        *p -= step as f32;
                    }
                }
            }
        }
    }
}
