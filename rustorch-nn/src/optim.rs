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
                    
                    for ((p, g), v) in param_data.iter_mut().zip(grad_data.iter()).zip(velocity_data.iter_mut()) {
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
    exp_avg: Vec<Option<Tensor>>,
    exp_avg_sq: Vec<Option<Tensor>>,
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
        }
    }
}

impl Optimizer for Adam {
    fn zero_grad(&mut self) {
        for param in &self.params {
            param.zero_grad();
        }
    }

    fn step(&mut self) {
        self.step_t += 1;
        let t = self.step_t as f32;
        let beta1_pow = self.beta1.powf(t);
        let beta2_pow = self.beta2.powf(t);

        for (i, param) in self.params.iter().enumerate() {
            if !param.requires_grad() {
                continue;
            }

            let grad_opt = param.grad();
            if let Some(grad) = grad_opt {
                let mut param_data = param.data_mut();
                let grad_data = grad.data();

                // Init state
                if self.exp_avg[i].is_none() {
                    self.exp_avg[i] = Some(Tensor::zeros(param.shape()));
                    self.exp_avg_sq[i] = Some(Tensor::zeros(param.shape()));
                }

                let m_tensor = self.exp_avg[i].as_ref().unwrap();
                let v_tensor = self.exp_avg_sq[i].as_ref().unwrap();
                
                let mut m_data = m_tensor.data_mut();
                let mut v_data = v_tensor.data_mut();

                let one_minus_beta1 = 1.0 - self.beta1;
                let one_minus_beta2 = 1.0 - self.beta2;
                let bias_correction1 = 1.0 - beta1_pow;
                let bias_correction2 = 1.0 - beta2_pow;

                for ((p, g), (m, v)) in param_data.iter_mut().zip(grad_data.iter())
                    .zip(m_data.iter_mut().zip(v_data.iter_mut())) 
                {
                    *m = self.beta1 * *m + one_minus_beta1 * *g;
                    *v = self.beta2 * *v + one_minus_beta2 * *g * *g;

                    let m_hat = *m / bias_correction1;
                    let v_hat = *v / bias_correction2;
                    
                    *p -= self.lr * m_hat / (v_hat.sqrt() + self.epsilon);
                }
            }
        }
    }
}
