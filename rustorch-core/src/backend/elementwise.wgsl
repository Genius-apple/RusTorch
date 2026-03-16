
const MAX_RANK: u32 = 4u;

struct Params {
    numel: u32,
    op: u32, 
    alpha: f32,
    ndim: u32,
    shape: vec4<u32>,
    stride_out: vec4<u32>,
    stride_in1: vec4<u32>,
    stride_in2: vec4<u32>,
}

@group(0) @binding(0) var<storage, read> input1: array<f32>;
@group(0) @binding(1) var<storage, read> input2: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= params.numel) {
        return;
    }

    var offset1: u32 = 0u;
    var offset2: u32 = 0u;
    var rem = index;
    
    // Iterate from last dim (ndim-1) to 0
    for (var k = 0u; k < params.ndim; k = k + 1u) {
        let i = params.ndim - 1u - k;
        let dim_val = params.shape[i];
        
        let coord = rem % dim_val;
        rem = rem / dim_val;
        
        offset1 = offset1 + coord * params.stride_in1[i];
        offset2 = offset2 + coord * params.stride_in2[i];
    }

    let a = input1[offset1];
    var b = 0.0;
    
    // Binary/Backward ops
    if (params.op <= 3u || params.op >= 10u || params.op == 20u) {
        b = input2[offset2];
    }

    var result = 0.0;

    if (params.op == 0u) { // Add
        result = a + b;
    } else if (params.op == 1u) { // Sub
        result = a - b;
    } else if (params.op == 2u) { // Mul
        result = a * b;
    } else if (params.op == 3u) { // Div
        result = a / b;
    } else if (params.op == 4u) { // ReLU
        result = max(a, 0.0);
    } else if (params.op == 5u) { // Sigmoid
        result = 1.0 / (1.0 + exp(-a));
    } else if (params.op == 6u) { // Tanh
        result = tanh(a);
    } else if (params.op == 10u) { // ReLU Backward
        if (a > 0.0) { result = b; } else { result = 0.0; }
    } else if (params.op == 11u) { // Sigmoid Backward
        result = a * (1.0 - a) * b;
    } else if (params.op == 12u) { // Tanh Backward
        result = (1.0 - a * a) * b;
    } else if (params.op == 20u) { // SGD Step
        result = a - params.alpha * b;
    } else if (params.op == 30u) { // Copy (Expand/Contiguous)
        result = a;
    }

    // Output is written linearly (assuming contiguous output buffer we are generating)
    output[index] = result;
}
