extern "C" {
    pub fn vector_add(
        n: i32,
        a: *const f32,
        b: *const f32,
        out: *mut f32
    );
}

#[cfg(feature = "cuda")]
pub mod cuda_kernels {
    use cudarc::driver::{CudaDevice, LaunchAsync, LaunchConfig};
    use cudarc::nvrtc::Ptx;
    use std::sync::Arc;

    // PTX source code
    pub const PTX_SRC: &str = r#"
    extern "C" __global__ void vector_add_kernel(int n, const float* a, const float* b, float* out) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n) {
            out[i] = a[i] + b[i];
        }
    }
    
    extern "C" __global__ void relu_kernel(int n, const float* inp, float* out) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n) {
            float val = inp[i];
            out[i] = val > 0.0f ? val : 0.0f;
        }
    }
    
    extern "C" __global__ void matmul_kernel(
        int M, int K, int N,
        const float* A, const float* B, float* C
    ) {
        int row = blockIdx.y * blockDim.y + threadIdx.y;
        int col = blockIdx.x * blockDim.x + threadIdx.x;
        
        if (row < M && col < N) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += A[row * K + k] * B[k * N + col];
            }
            C[row * N + col] = sum;
        }
    }
    
    extern "C" __global__ void max_pool2d_kernel(
        int n, int c, int h_in, int w_in,
        int k_h, int k_w,
        int stride_h, int stride_w,
        int pad_h, int pad_w,
        int h_out, int w_out,
        const float* input,
        float* output
    ) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int total = n * c * h_out * w_out;
        
        if (idx < total) {
            int wo = idx % w_out;
            int ho = (idx / w_out) % h_out;
            int ci = (idx / (w_out * h_out)) % c;
            int b = idx / (w_out * h_out * c);
            
            float max_val = -1e30f; // -inf
            
            int h_start = ho * stride_h - pad_h;
            int w_start = wo * stride_w - pad_w;
            
            for (int kh = 0; kh < k_h; ++kh) {
                for (int kw = 0; kw < k_w; ++kw) {
                    int h_in_idx = h_start + kh;
                    int w_in_idx = w_start + kw;
                    
                    if (h_in_idx >= 0 && h_in_idx < h_in && w_in_idx >= 0 && w_in_idx < w_in) {
                        float val = input[((b * c + ci) * h_in + h_in_idx) * w_in + w_in_idx];
                        if (val > max_val) max_val = val;
                    }
                }
            }
            output[idx] = max_val;
        }
    }

    extern "C" __global__ void conv2d_kernel(
        int n, int c_in, int h_in, int w_in,
        int c_out, int k_h, int k_w,
        int stride_h, int stride_w,
        int pad_h, int pad_w,
        int h_out, int w_out,
        const float* input,
        const float* weight,
        float* output
    ) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int total = n * c_out * h_out * w_out;
        
        if (idx < total) {
            int wo = idx % w_out;
            int ho = (idx / w_out) % h_out;
            int co = (idx / (w_out * h_out)) % c_out;
            int b = idx / (w_out * h_out * c_out);
            
            float sum = 0.0;
            
            for (int ci = 0; ci < c_in; ci++) {
                for (int kh = 0; kh < k_h; kh++) {
                    for (int kw = 0; kw < k_w; kw++) {
                        int h_in_idx = ho * stride_h + kh - pad_h;
                        int w_in_idx = wo * stride_w + kw - pad_w;
                        
                        if (h_in_idx >= 0 && h_in_idx < h_in && w_in_idx >= 0 && w_in_idx < w_in) {
                            int in_idx = ((b * c_in + ci) * h_in + h_in_idx) * w_in + w_in_idx;
                            int w_idx = ((co * c_in + ci) * k_h + kh) * k_w + kw;
                            sum += input[in_idx] * weight[w_idx];
                        }
                    }
                }
            }
            
            output[idx] = sum;
        }
    }
    
    extern "C" __global__ void batch_norm2d_inference_kernel(
        int n, int c, int h, int w,
        const float* input,
        const float* gamma,
        const float* beta,
        const float* mean,
        const float* var,
        float eps,
        float* output
    ) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int total = n * c * h * w;
        
        if (idx < total) {
            int hw = h * w;
            int ci = (idx / hw) % c;
            
            float m = mean[ci];
            float v = var[ci];
            float g = (gamma != 0) ? gamma[ci] : 1.0f;
            float b = (beta != 0) ? beta[ci] : 0.0f;
            
            float val = input[idx];
            float inv_std = rsqrtf(v + eps);
            
            output[idx] = (val - m) * inv_std * g + b;
        }
    }
    "#;

    pub fn load_kernels(dev: &Arc<CudaDevice>) -> Result<(), Box<dyn std::error::Error>> {
        let ptx = Ptx::from_src(PTX_SRC);
        dev.load_ptx(ptx, "kernels", &[
            "vector_add_kernel", 
            "relu_kernel",
            "matmul_kernel",
            "max_pool2d_kernel",
            "conv2d_kernel",
            "batch_norm2d_inference_kernel"
        ])?;
        Ok(())
    }
}
