struct Uniforms {
    input_size: u32,
    reduce_dim_size: u32,
    reduce_dim_stride: u32,
    outer_size: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: Uniforms;

@compute @workgroup_size(256)
fn reduce_sum_dim0(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= params.input_size) {
        return;
    }

    var sum: f32 = 0.0;
    let batch = params.reduce_dim_size;
    let stride = params.reduce_dim_stride;

    for (var i: u32 = 0u; i < batch; i = i + 1u) {
        let offset = i * stride + idx;
        sum = sum + input[offset];
    }

    output[idx] = sum;
}

@compute @workgroup_size(256)
fn reduce_sum_general(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let output_idx = global_id.x;
    let total_outputs = params.input_size / params.reduce_dim_size;
    
    if (output_idx >= total_outputs) {
        return;
    }
    
    let outer_idx = output_idx / params.outer_size;
    let inner_idx = output_idx % params.outer_size;
    
    var sum: f32 = 0.0;
    
    for (var i: u32 = 0u; i < params.reduce_dim_size; i = i + 1u) {
        let input_idx = outer_idx * params.reduce_dim_size * params.outer_size 
                      + i * params.outer_size 
                      + inner_idx;
        sum = sum + input[input_idx];
    }
    
    output[output_idx] = sum;
}

@compute @workgroup_size(256)
fn reduce_sum_all(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    
    if (idx != 0u) {
        return;
    }
    
    var sum: f32 = 0.0;
    let size = params.input_size;
    
    for (var i: u32 = 0u; i < size; i = i + 1u) {
        sum = sum + input[i];
    }
    
    output[0] = sum;
}
