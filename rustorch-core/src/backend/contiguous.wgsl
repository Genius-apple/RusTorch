struct Params {
    size: u32,
    ndim: u32,
    pad0: u32,
    pad1: u32,
}

struct ShapeStride {
    shape0: u32,
    shape1: u32,
    shape2: u32,
    shape3: u32,
    shape4: u32,
    shape5: u32,
    shape6: u32,
    shape7: u32,
    stride0: u32,
    stride1: u32,
    stride2: u32,
    stride3: u32,
    stride4: u32,
    stride5: u32,
    stride6: u32,
    stride7: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;
@group(0) @binding(3) var<storage, read> shape_strides: ShapeStride;

fn get_shape(idx: u32) -> u32 {
    switch(idx) {
        case 0u: { return shape_strides.shape0; }
        case 1u: { return shape_strides.shape1; }
        case 2u: { return shape_strides.shape2; }
        case 3u: { return shape_strides.shape3; }
        case 4u: { return shape_strides.shape4; }
        case 5u: { return shape_strides.shape5; }
        case 6u: { return shape_strides.shape6; }
        case 7u: { return shape_strides.shape7; }
        default: { return 1u; }
    }
}

fn get_stride(idx: u32) -> u32 {
    switch(idx) {
        case 0u: { return shape_strides.stride0; }
        case 1u: { return shape_strides.stride1; }
        case 2u: { return shape_strides.stride2; }
        case 3u: { return shape_strides.stride3; }
        case 4u: { return shape_strides.stride4; }
        case 5u: { return shape_strides.stride5; }
        case 6u: { return shape_strides.stride6; }
        case 7u: { return shape_strides.stride7; }
        default: { return 0u; }
    }
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= params.size) {
        return;
    }
    
    let ndim = params.ndim;
    
    var physical_offset: u32 = 0u;
    var temp_i: u32 = idx;
    
    for (var dim_idx: u32 = ndim; dim_idx > 0u; dim_idx = dim_idx - 1u) {
        let dim = dim_idx - 1u;
        let dim_size = get_shape(dim);
        let stride = get_stride(dim);
        let coord = temp_i % dim_size;
        temp_i = temp_i / dim_size;
        physical_offset = physical_offset + coord * stride;
    }
    
    if (physical_offset < arrayLength(&input)) {
        output[idx] = input[physical_offset];
    } else {
        output[idx] = 0.0;
    }
}
