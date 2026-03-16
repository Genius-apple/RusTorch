struct AdamParams {
    lr: f32,
    beta1: f32,
    beta2: f32,
    epsilon: f32,
    step: u32,
    numel: u32,
    pad1: u32,
    pad2: u32,
}

@group(0) @binding(0) var<storage, read_write> param: array<f32>;
@group(0) @binding(1) var<storage, read> grad: array<f32>;
@group(0) @binding(2) var<storage, read_write> m: array<f32>;
@group(0) @binding(3) var<storage, read_write> v: array<f32>;
@group(0) @binding(4) var<uniform> config: AdamParams;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    if (i >= config.numel) {
        return;
    }

    let g = grad[i];
    let m_prev = m[i];
    let v_prev = v[i];

    let m_t = config.beta1 * m_prev + (1.0 - config.beta1) * g;
    let v_t = config.beta2 * v_prev + (1.0 - config.beta2) * g * g;

    m[i] = m_t;
    v[i] = v_t;

    // Bias correction
    let beta1_pow = pow(config.beta1, f32(config.step));
    let beta2_pow = pow(config.beta2, f32(config.step));
    
    let m_hat = m_t / (1.0 - beta1_pow);
    let v_hat = v_t / (1.0 - beta2_pow);

    let p = param[i];
    param[i] = p - config.lr * m_hat / (sqrt(v_hat) + config.epsilon);
}
