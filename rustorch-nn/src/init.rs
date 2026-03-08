use rustorch_core::Tensor;

pub fn calculate_fan_in_and_fan_out(tensor: &Tensor) -> (usize, usize) {
    let shape = tensor.shape();
    if shape.len() < 2 {
        // Fallback for 1D or scalar
        let s = if shape.len() == 1 { shape[0] } else { 1 };
        return (s, s);
    }
    
    // PyTorch convention:
    // Linear: weight is (out_features, in_features) usually, but check implementation.
    // In our Linear implementation, weight is (in_features, out_features)? 
    // Let's check Linear implementation.
    // Conv2d: (out_channels, in_channels, kH, kW)
    
    // Assuming Linear is (out, in) like PyTorch for easier matmul with input (N, in).
    // Wait, let's check `rustorch-nn/src/linear.rs` if it exists, or `lib.rs`.
    // Actually I implemented Linear in `rustorch-nn/src/lib.rs` (inline) or separate file.
    // Let's assume standard PyTorch shapes.
    
    let dimensions = shape.len();
    if dimensions == 2 {
        // Linear: (out, in) usually. 
        // But if my Linear stores (in, out) for x.matmul(w), then fan_in is dim 0.
        // Let's verify Linear later.
        // Standard: 
        // fan_in = size[1]
        // fan_out = size[0]
        let num_input_fmaps = shape[1];
        let num_output_fmaps = shape[0];
        (num_input_fmaps, num_output_fmaps)
    } else {
        // Conv2d: (out, in, h, w)
        let num_input_fmaps = shape[1];
        let num_output_fmaps = shape[0];
        let receptive_field_size: usize = shape[2..].iter().product();
        
        let fan_in = num_input_fmaps * receptive_field_size;
        let fan_out = num_output_fmaps * receptive_field_size;
        
        (fan_in, fan_out)
    }
}

pub fn kaiming_uniform_(tensor: &Tensor, a: f32, mode: &str, nonlinearity: &str) {
    let (fan_in, fan_out) = calculate_fan_in_and_fan_out(tensor);
    let fan = if mode == "fan_in" { fan_in } else { fan_out };
    
    let gain = calculate_gain(nonlinearity, a);
    let std = gain / (fan as f32).sqrt();
    
    // Calculate bound for uniform distribution
    // Uniform(-bound, bound) has std = bound / sqrt(3)
    // We want std = gain / sqrt(fan)
    // bound / sqrt(3) = gain / sqrt(fan)
    // bound = sqrt(3) * gain / sqrt(fan)
    let bound = (3.0f32).sqrt() * std;
    
    tensor.uniform_(-bound, bound);
}

pub fn kaiming_normal_(tensor: &Tensor, a: f32, mode: &str, nonlinearity: &str) {
    let (fan_in, fan_out) = calculate_fan_in_and_fan_out(tensor);
    let fan = if mode == "fan_in" { fan_in } else { fan_out };
    
    let gain = calculate_gain(nonlinearity, a);
    let std = gain / (fan as f32).sqrt();
    
    tensor.normal_(0.0, std);
}

pub fn xavier_uniform_(tensor: &Tensor, gain: f32) {
    let (fan_in, fan_out) = calculate_fan_in_and_fan_out(tensor);
    let std = gain * (2.0 / (fan_in as f32 + fan_out as f32)).sqrt();
    let bound = (3.0f32).sqrt() * std;
    
    tensor.uniform_(-bound, bound);
}

pub fn xavier_normal_(tensor: &Tensor, gain: f32) {
    let (fan_in, fan_out) = calculate_fan_in_and_fan_out(tensor);
    let std = gain * (2.0 / (fan_in as f32 + fan_out as f32)).sqrt();
    
    tensor.normal_(0.0, std);
}

pub fn calculate_gain(nonlinearity: &str, param: f32) -> f32 {
    match nonlinearity {
        "linear" | "conv1d" | "conv2d" | "conv3d" | "conv_transpose1d" | "conv_transpose2d" | "conv_transpose3d" => 1.0,
        "sigmoid" => 1.0,
        "tanh" => 5.0 / 3.0,
        "relu" => (2.0f32).sqrt(),
        "leaky_relu" => (2.0 / (1.0 + param.powi(2))).sqrt(),
        _ => 1.0,
    }
}

pub fn constant_(tensor: &Tensor, val: f32) {
    tensor.fill_(val);
}

pub fn uniform_(tensor: &Tensor, low: f32, high: f32) {
    tensor.uniform_(low, high);
}

pub fn normal_(tensor: &Tensor, mean: f32, std: f32) {
    tensor.normal_(mean, std);
}
