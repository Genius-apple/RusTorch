struct Params {
    numel_vec4: u32,
    op: u32,
    alpha: f32,
    stride_mode_1: u32, // 0: Contiguous, 1: Scalar Broadcast
    stride_mode_2: u32, // 0: Contiguous, 1: Scalar Broadcast
}

@group(0) @binding(0) var<storage, read> input1: array<f32>;
@group(0) @binding(1) var<storage, read> input2: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    if (i >= params.numel_vec4) {
        return;
    }

    // Load Input 1
    var a: vec4<f32>;
    if (params.stride_mode_1 == 0u) {
        let base = i * 4u;
        a = vec4<f32>(input1[base], input1[base + 1u], input1[base + 2u], input1[base + 3u]);
    } else {
        let val = input1[0];
        a = vec4<f32>(val, val, val, val);
    }

    // Load Input 2 (if binary op)
    var b: vec4<f32>;
    if (params.op <= 3u || params.op >= 10u || params.op == 20u) {
        if (params.stride_mode_2 == 0u) {
             let base = i * 4u;
             b = vec4<f32>(input2[base], input2[base + 1u], input2[base + 2u], input2[base + 3u]);
        } else {
             let val = input2[0];
             b = vec4<f32>(val, val, val, val);
        }
    }

    var res: vec4<f32>;
    let zero = vec4<f32>(0.0);
    let one = vec4<f32>(1.0);

    if (params.op == 0u) { // Add
        res = a + b;
    } else if (params.op == 1u) { // Sub
        res = a - b;
    } else if (params.op == 2u) { // Mul
        res = a * b;
    } else if (params.op == 3u) { // Div
        res = a / b;
    } else if (params.op == 4u) { // ReLU
        res = max(a, zero);
    } else if (params.op == 5u) { // Sigmoid
        res = one / (one + exp(-a));
    } else if (params.op == 6u) { // Tanh
        res = tanh(a);
    } else if (params.op == 10u) { // ReLU Backward
        res = select(zero, b, a > zero);
    } else if (params.op == 11u) { // Sigmoid Backward
        // sigmoid_backward(out, grad) -> out * (1 - out) * grad
        // Here `a` is output of sigmoid, `b` is grad
        res = a * (one - a) * b;
    } else if (params.op == 12u) { // Tanh Backward
        // tanh_backward(out, grad) -> (1 - out^2) * grad
        res = (one - a * a) * b;
    } else if (params.op == 20u) { // SGD Step
        res = a - params.alpha * b;
    } else if (params.op == 30u) { // Copy
        res = a;
    }

    // Write Output
    let base = i * 4u;
    output[base] = res.x;
    output[base+1u] = res.y;
    output[base+2u] = res.z;
    output[base+3u] = res.w;
}
