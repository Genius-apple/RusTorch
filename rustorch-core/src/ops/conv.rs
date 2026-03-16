use crate::autograd::BackwardOp;
use crate::storage::Storage;
use crate::Tensor;
use parking_lot::Mutex;
use rayon::prelude::*;
use std::collections::HashMap;
use std::hint::black_box;
use std::sync::{Arc, OnceLock};
use std::time::Instant;

// --- Conv2d ---
// Input: (N, C_in, H, W)
// Weight: (C_out, C_in, kH, kW)
// Output: (N, C_out, H_out, W_out)

#[derive(Debug)]
pub struct Conv2dBackward {
    pub input: Tensor,
    pub weight: Tensor,
    pub stride: (usize, usize),
    pub padding: (usize, usize),
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum CpuConv2dStrategy {
    Auto,
    Profile,
    Direct,
    Im2col,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Conv2dKernelChoice {
    Direct,
    Im2col,
}

#[derive(Clone, Copy)]
struct CpuConv2dConfig {
    strategy: CpuConv2dStrategy,
    min_work: usize,
    profile_iters: usize,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum CpuConv2dBwdStrategy {
    Auto,
    Profile,
    Direct,
    Im2col,
}

#[derive(Clone, Copy)]
struct CpuConv2dBwdConfig {
    strategy: CpuConv2dBwdStrategy,
    min_work_grad_input: usize,
    min_work_grad_weight: usize,
    profile_iters_grad_input: usize,
    profile_iters_grad_weight: usize,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum Conv2dBwdTarget {
    GradInput,
    GradWeight,
}

type Conv2dPerfKey = (
    usize,
    usize,
    usize,
    usize,
    usize,
    usize,
    usize,
    usize,
    usize,
    usize,
    usize,
);
type Conv2dBwdPerfKey = (Conv2dBwdTarget, Conv2dPerfKey);

fn parse_usize_env(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(default)
}

fn cpu_conv2d_config() -> CpuConv2dConfig {
    static CFG: OnceLock<CpuConv2dConfig> = OnceLock::new();
    *CFG.get_or_init(|| {
        let strategy = match std::env::var("RUSTORCH_CPU_CONV2D_STRATEGY")
            .unwrap_or_else(|_| "auto".to_string())
            .to_ascii_lowercase()
            .as_str()
        {
            "im2col" => CpuConv2dStrategy::Im2col,
            "direct" => CpuConv2dStrategy::Direct,
            "profile" => CpuConv2dStrategy::Profile,
            _ => CpuConv2dStrategy::Auto,
        };
        CpuConv2dConfig {
            strategy,
            min_work: parse_usize_env("RUSTORCH_CPU_CONV2D_MIN_WORK", 65536),
            profile_iters: parse_usize_env("RUSTORCH_CPU_CONV2D_PROFILE_ITERS", 1),
        }
    })
}

fn conv2d_profile_cache() -> &'static Mutex<HashMap<Conv2dPerfKey, Conv2dKernelChoice>> {
    static CACHE: OnceLock<Mutex<HashMap<Conv2dPerfKey, Conv2dKernelChoice>>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

fn cpu_conv2d_bwd_config() -> CpuConv2dBwdConfig {
    static CFG: OnceLock<CpuConv2dBwdConfig> = OnceLock::new();
    *CFG.get_or_init(|| {
        let strategy = match std::env::var("RUSTORCH_CPU_CONV2D_BWD_STRATEGY")
            .unwrap_or_else(|_| "auto".to_string())
            .to_ascii_lowercase()
            .as_str()
        {
            "im2col" => CpuConv2dBwdStrategy::Im2col,
            "direct" => CpuConv2dBwdStrategy::Direct,
            "profile" => CpuConv2dBwdStrategy::Profile,
            _ => CpuConv2dBwdStrategy::Auto,
        };
        CpuConv2dBwdConfig {
            strategy,
            min_work_grad_input: parse_usize_env(
                "RUSTORCH_CPU_CONV2D_BWD_MIN_WORK_GRAD_INPUT",
                parse_usize_env("RUSTORCH_CPU_CONV2D_BWD_MIN_WORK", 65536),
            ),
            min_work_grad_weight: parse_usize_env(
                "RUSTORCH_CPU_CONV2D_BWD_MIN_WORK_GRAD_WEIGHT",
                parse_usize_env("RUSTORCH_CPU_CONV2D_BWD_MIN_WORK", 65536),
            ),
            profile_iters_grad_input: parse_usize_env(
                "RUSTORCH_CPU_CONV2D_BWD_PROFILE_ITERS_GRAD_INPUT",
                parse_usize_env("RUSTORCH_CPU_CONV2D_BWD_PROFILE_ITERS", 1),
            ),
            profile_iters_grad_weight: parse_usize_env(
                "RUSTORCH_CPU_CONV2D_BWD_PROFILE_ITERS_GRAD_WEIGHT",
                parse_usize_env("RUSTORCH_CPU_CONV2D_BWD_PROFILE_ITERS", 1),
            ),
        }
    })
}

fn conv2d_bwd_profile_cache() -> &'static Mutex<HashMap<Conv2dBwdPerfKey, Conv2dKernelChoice>> {
    static CACHE: OnceLock<Mutex<HashMap<Conv2dBwdPerfKey, Conv2dKernelChoice>>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

fn conv2d_direct_core(
    input_data: &[f32],
    weight_data: &[f32],
    n: usize,
    c_in: usize,
    h_in: usize,
    w_in: usize,
    c_out: usize,
    k_h: usize,
    k_w: usize,
    h_out: usize,
    w_out: usize,
    stride_h: usize,
    stride_w: usize,
    pad_h: usize,
    pad_w: usize,
) -> Vec<f32> {
    let total_elements = n * c_out * h_out * w_out;
    (0..total_elements)
        .into_par_iter()
        .map(|idx| {
            let wo = idx % w_out;
            let ho = (idx / w_out) % h_out;
            let co = (idx / (w_out * h_out)) % c_out;
            let b = idx / (w_out * h_out * c_out);

            let mut sum: f64 = 0.0;
            for ci in 0..c_in {
                for kh in 0..k_h {
                    for kw in 0..k_w {
                        let h_in_idx = ho * stride_h + kh;
                        let w_in_idx = wo * stride_w + kw;
                        if h_in_idx >= pad_h && w_in_idx >= pad_w {
                            let hi = h_in_idx - pad_h;
                            let wi = w_in_idx - pad_w;
                            if hi < h_in && wi < w_in {
                                let val_in =
                                    input_data[((b * c_in + ci) * h_in + hi) * w_in + wi] as f64;
                                let val_w =
                                    weight_data[((co * c_in + ci) * k_h + kh) * k_w + kw] as f64;
                                sum += val_in * val_w;
                            }
                        }
                    }
                }
            }
            sum as f32
        })
        .collect()
}

fn conv2d_im2col_core(
    input_data: &[f32],
    weight_data: &[f32],
    n: usize,
    c_in: usize,
    h_in: usize,
    w_in: usize,
    c_out: usize,
    k_h: usize,
    k_w: usize,
    h_out: usize,
    w_out: usize,
    stride_h: usize,
    stride_w: usize,
    pad_h: usize,
    pad_w: usize,
) -> Vec<f32> {
    let k_size = c_in * k_h * k_w;
    let patches = n * h_out * w_out;
    let mut col = vec![0.0f32; patches * k_size];
    for b in 0..n {
        for ho in 0..h_out {
            for wo in 0..w_out {
                let row = (b * h_out + ho) * w_out + wo;
                let mut col_idx = 0usize;
                for ci in 0..c_in {
                    for kh in 0..k_h {
                        for kw in 0..k_w {
                            let h_in_idx = ho * stride_h + kh;
                            let w_in_idx = wo * stride_w + kw;
                            let val = if h_in_idx >= pad_h && w_in_idx >= pad_w {
                                let hi = h_in_idx - pad_h;
                                let wi = w_in_idx - pad_w;
                                if hi < h_in && wi < w_in {
                                    input_data[((b * c_in + ci) * h_in + hi) * w_in + wi]
                                } else {
                                    0.0
                                }
                            } else {
                                0.0
                            };
                            col[row * k_size + col_idx] = val;
                            col_idx += 1;
                        }
                    }
                }
            }
        }
    }

    let col_t = Tensor::new(&col, &[patches, k_size]);
    let weight_t = Tensor::new(weight_data, &[c_out, k_size]).t();
    let out2d = crate::ops::matmul(&col_t, &weight_t);
    let out_data = out2d.data();
    let mut result = vec![0.0f32; n * c_out * h_out * w_out];
    for b in 0..n {
        for ho in 0..h_out {
            for wo in 0..w_out {
                let row = (b * h_out + ho) * w_out + wo;
                for co in 0..c_out {
                    result[((b * c_out + co) * h_out + ho) * w_out + wo] =
                        out_data[row * c_out + co];
                }
            }
        }
    }
    result
}

fn choose_conv2d_bwd_kernel<Fd, Fi>(
    target: Conv2dBwdTarget,
    key: Conv2dPerfKey,
    work: usize,
    direct_fn: Fd,
    im2col_fn: Fi,
) -> Conv2dKernelChoice
where
    Fd: Fn() -> Vec<f32>,
    Fi: Fn() -> Vec<f32>,
{
    let cfg = cpu_conv2d_bwd_config();
    let min_work = match target {
        Conv2dBwdTarget::GradInput => cfg.min_work_grad_input,
        Conv2dBwdTarget::GradWeight => cfg.min_work_grad_weight,
    };
    let profile_iters = match target {
        Conv2dBwdTarget::GradInput => cfg.profile_iters_grad_input,
        Conv2dBwdTarget::GradWeight => cfg.profile_iters_grad_weight,
    };
    match cfg.strategy {
        CpuConv2dBwdStrategy::Direct => Conv2dKernelChoice::Direct,
        CpuConv2dBwdStrategy::Im2col => Conv2dKernelChoice::Im2col,
        CpuConv2dBwdStrategy::Auto => {
            if work >= min_work {
                Conv2dKernelChoice::Im2col
            } else {
                Conv2dKernelChoice::Direct
            }
        }
        CpuConv2dBwdStrategy::Profile => {
            let cache_key = (target, key);
            if let Some(cached) = conv2d_bwd_profile_cache().lock().get(&cache_key).copied() {
                return cached;
            }
            let iters = profile_iters.max(1);
            let mut direct_ns = 0u128;
            let mut im2col_ns = 0u128;
            for _ in 0..iters {
                let t0 = Instant::now();
                let d = direct_fn();
                direct_ns += t0.elapsed().as_nanos();
                black_box(d.len());

                let t1 = Instant::now();
                let c = im2col_fn();
                im2col_ns += t1.elapsed().as_nanos();
                black_box(c.len());
            }
            let choice = if im2col_ns < direct_ns {
                Conv2dKernelChoice::Im2col
            } else {
                Conv2dKernelChoice::Direct
            };
            conv2d_bwd_profile_cache().lock().insert(cache_key, choice);
            choice
        }
    }
}

fn conv2d_grad_input_direct_core(
    grad_data: &[f32],
    weight_data: &[f32],
    n: usize,
    c_in: usize,
    h_in: usize,
    w_in: usize,
    c_out: usize,
    k_h: usize,
    k_w: usize,
    h_out: usize,
    w_out: usize,
    stride_h: usize,
    stride_w: usize,
    pad_h: usize,
    pad_w: usize,
) -> Vec<f32> {
    let total_elements = n * c_in * h_in * w_in;
    (0..total_elements)
        .into_par_iter()
        .map(|idx| {
            let w = idx % w_in;
            let h = (idx / w_in) % h_in;
            let ci = (idx / (w_in * h_in)) % c_in;
            let b = idx / (w_in * h_in * c_in);
            let mut sum: f64 = 0.0;

            let h_out_start = if h + pad_h >= k_h {
                (h + pad_h - k_h + 1).div_ceil(stride_h)
            } else {
                0
            };
            let h_out_end = std::cmp::min(h_out, (h + pad_h) / stride_h + 1);

            for ho in h_out_start..h_out_end {
                let kh = h + pad_h - ho * stride_h;
                let w_out_start = if w + pad_w >= k_w {
                    (w + pad_w - k_w + 1).div_ceil(stride_w)
                } else {
                    0
                };
                let w_out_end = std::cmp::min(w_out, (w + pad_w) / stride_w + 1);

                for wo in w_out_start..w_out_end {
                    let kw = w + pad_w - wo * stride_w;
                    for co in 0..c_out {
                        let g_val = grad_data[((b * c_out + co) * h_out + ho) * w_out + wo] as f64;
                        let w_val = weight_data[((co * c_in + ci) * k_h + kh) * k_w + kw] as f64;
                        sum += g_val * w_val;
                    }
                }
            }
            sum as f32
        })
        .collect()
}

fn conv2d_grad_input_im2col_core(
    grad_data: &[f32],
    weight_data: &[f32],
    n: usize,
    c_in: usize,
    h_in: usize,
    w_in: usize,
    c_out: usize,
    k_h: usize,
    k_w: usize,
    h_out: usize,
    w_out: usize,
    stride_h: usize,
    stride_w: usize,
    pad_h: usize,
    pad_w: usize,
) -> Vec<f32> {
    let patches = h_out * w_out;
    let k_size = c_in * k_h * k_w;
    let weight_flat = Tensor::new(weight_data, &[c_out, k_size]);
    let mut grad_input_data = vec![0.0f32; n * c_in * h_in * w_in];

    for b in 0..n {
        let mut gy = vec![0.0f32; patches * c_out];
        for ho in 0..h_out {
            for wo in 0..w_out {
                let row = ho * w_out + wo;
                for co in 0..c_out {
                    gy[row * c_out + co] = grad_data[((b * c_out + co) * h_out + ho) * w_out + wo];
                }
            }
        }
        let gy_t = Tensor::new(&gy, &[patches, c_out]);
        let dcol_t = crate::ops::matmul(&gy_t, &weight_flat);
        let dcol = dcol_t.data();

        for ho in 0..h_out {
            for wo in 0..w_out {
                let row = ho * w_out + wo;
                let mut col_idx = 0usize;
                for ci in 0..c_in {
                    for kh in 0..k_h {
                        for kw in 0..k_w {
                            let h_in_idx = ho * stride_h + kh;
                            let w_in_idx = wo * stride_w + kw;
                            if h_in_idx >= pad_h && w_in_idx >= pad_w {
                                let hi = h_in_idx - pad_h;
                                let wi = w_in_idx - pad_w;
                                if hi < h_in && wi < w_in {
                                    grad_input_data[((b * c_in + ci) * h_in + hi) * w_in + wi] +=
                                        dcol[row * k_size + col_idx];
                                }
                            }
                            col_idx += 1;
                        }
                    }
                }
            }
        }
    }

    grad_input_data
}

fn conv2d_grad_weight_direct_core(
    input_data: &[f32],
    grad_data: &[f32],
    n: usize,
    c_in: usize,
    h_in: usize,
    w_in: usize,
    c_out: usize,
    k_h: usize,
    k_w: usize,
    h_out: usize,
    w_out: usize,
    stride_h: usize,
    stride_w: usize,
    pad_h: usize,
    pad_w: usize,
) -> Vec<f32> {
    let total_elements = c_out * c_in * k_h * k_w;
    (0..total_elements)
        .into_par_iter()
        .map(|idx| {
            let kw = idx % k_w;
            let kh = (idx / k_w) % k_h;
            let ci = (idx / (k_w * k_h)) % c_in;
            let co = idx / (k_w * k_h * c_in);
            let mut sum: f64 = 0.0;
            for b in 0..n {
                for ho in 0..h_out {
                    for wo in 0..w_out {
                        let h_in_idx = ho * stride_h + kh;
                        let w_in_idx = wo * stride_w + kw;
                        if h_in_idx >= pad_h && w_in_idx >= pad_w {
                            let hi = h_in_idx - pad_h;
                            let wi = w_in_idx - pad_w;
                            if hi < h_in && wi < w_in {
                                let val_in =
                                    input_data[((b * c_in + ci) * h_in + hi) * w_in + wi] as f64;
                                let val_g =
                                    grad_data[((b * c_out + co) * h_out + ho) * w_out + wo] as f64;
                                sum += val_in * val_g;
                            }
                        }
                    }
                }
            }
            sum as f32
        })
        .collect()
}

fn conv2d_grad_weight_im2col_core(
    input_data: &[f32],
    grad_data: &[f32],
    n: usize,
    c_in: usize,
    h_in: usize,
    w_in: usize,
    c_out: usize,
    k_h: usize,
    k_w: usize,
    h_out: usize,
    w_out: usize,
    stride_h: usize,
    stride_w: usize,
    pad_h: usize,
    pad_w: usize,
) -> Vec<f32> {
    let patches = h_out * w_out;
    let k_size = c_in * k_h * k_w;
    let mut grad_weight = vec![0.0f32; c_out * k_size];

    for b in 0..n {
        let mut col = vec![0.0f32; patches * k_size];
        let mut gy = vec![0.0f32; patches * c_out];

        for ho in 0..h_out {
            for wo in 0..w_out {
                let row = ho * w_out + wo;
                let mut col_idx = 0usize;
                for ci in 0..c_in {
                    for kh in 0..k_h {
                        for kw in 0..k_w {
                            let h_in_idx = ho * stride_h + kh;
                            let w_in_idx = wo * stride_w + kw;
                            let val = if h_in_idx >= pad_h && w_in_idx >= pad_w {
                                let hi = h_in_idx - pad_h;
                                let wi = w_in_idx - pad_w;
                                if hi < h_in && wi < w_in {
                                    input_data[((b * c_in + ci) * h_in + hi) * w_in + wi]
                                } else {
                                    0.0
                                }
                            } else {
                                0.0
                            };
                            col[row * k_size + col_idx] = val;
                            col_idx += 1;
                        }
                    }
                }
                for co in 0..c_out {
                    gy[row * c_out + co] = grad_data[((b * c_out + co) * h_out + ho) * w_out + wo];
                }
            }
        }

        let gy_t = Tensor::new(&gy, &[patches, c_out]).t();
        let col_t = Tensor::new(&col, &[patches, k_size]);
        let gw_batch = crate::ops::matmul(&gy_t, &col_t);
        let gw_data = gw_batch.data();
        for i in 0..grad_weight.len() {
            grad_weight[i] += gw_data[i];
        }
    }

    grad_weight
}

impl BackwardOp for Conv2dBackward {
    fn backward(&self, grad: &Tensor) {
        let (stride_h, stride_w) = self.stride;
        let (pad_h, pad_w) = self.padding;

        let input_shape = self.input.shape();
        let weight_shape = self.weight.shape();
        let grad_shape = grad.shape();

        let n = input_shape[0];
        let c_in = input_shape[1];
        let h_in = input_shape[2];
        let w_in = input_shape[3];

        let c_out = weight_shape[0];
        let k_h = weight_shape[2];
        let k_w = weight_shape[3];

        let h_out = grad_shape[2];
        let w_out = grad_shape[3];

        // Compute grad_input
        if self.input.requires_grad() {
            let grad_guard = grad.data();
            let weight_guard = self.weight.data();
            let grad_data = &*grad_guard;
            let weight_data = &*weight_guard;
            let work = n * c_in * h_in * w_in * c_out * k_h * k_w;
            let key: Conv2dPerfKey = (
                n, c_in, h_in, w_in, c_out, k_h, k_w, stride_h, stride_w, pad_h, pad_w,
            );
            let kernel = choose_conv2d_bwd_kernel(
                Conv2dBwdTarget::GradInput,
                key,
                work,
                || {
                    conv2d_grad_input_direct_core(
                        grad_data,
                        weight_data,
                        n,
                        c_in,
                        h_in,
                        w_in,
                        c_out,
                        k_h,
                        k_w,
                        h_out,
                        w_out,
                        stride_h,
                        stride_w,
                        pad_h,
                        pad_w,
                    )
                },
                || {
                    conv2d_grad_input_im2col_core(
                        grad_data,
                        weight_data,
                        n,
                        c_in,
                        h_in,
                        w_in,
                        c_out,
                        k_h,
                        k_w,
                        h_out,
                        w_out,
                        stride_h,
                        stride_w,
                        pad_h,
                        pad_w,
                    )
                },
            );
            let grad_input_data = match kernel {
                Conv2dKernelChoice::Direct => conv2d_grad_input_direct_core(
                    grad_data,
                    weight_data,
                    n,
                    c_in,
                    h_in,
                    w_in,
                    c_out,
                    k_h,
                    k_w,
                    h_out,
                    w_out,
                    stride_h,
                    stride_w,
                    pad_h,
                    pad_w,
                ),
                Conv2dKernelChoice::Im2col => conv2d_grad_input_im2col_core(
                    grad_data,
                    weight_data,
                    n,
                    c_in,
                    h_in,
                    w_in,
                    c_out,
                    k_h,
                    k_w,
                    h_out,
                    w_out,
                    stride_h,
                    stride_w,
                    pad_h,
                    pad_w,
                ),
            };

            let grad_input_tensor =
                Tensor::new_with_storage(Storage::new(grad_input_data), self.input.shape());
            self.input.accumulate_grad(&grad_input_tensor);
            self.input.backward_step();
        }

        // Compute grad_weight
        if self.weight.requires_grad() {
            let input_guard = self.input.data();
            let grad_guard = grad.data();
            let input_data = &*input_guard;
            let grad_data = &*grad_guard;
            let work = n * c_in * h_in * w_in * c_out * k_h * k_w;
            let key: Conv2dPerfKey = (
                n, c_in, h_in, w_in, c_out, k_h, k_w, stride_h, stride_w, pad_h, pad_w,
            );
            let kernel = choose_conv2d_bwd_kernel(
                Conv2dBwdTarget::GradWeight,
                key,
                work,
                || {
                    conv2d_grad_weight_direct_core(
                        input_data, grad_data, n, c_in, h_in, w_in, c_out, k_h, k_w, h_out, w_out,
                        stride_h, stride_w, pad_h, pad_w,
                    )
                },
                || {
                    conv2d_grad_weight_im2col_core(
                        input_data, grad_data, n, c_in, h_in, w_in, c_out, k_h, k_w, h_out, w_out,
                        stride_h, stride_w, pad_h, pad_w,
                    )
                },
            );
            let grad_weight_data = match kernel {
                Conv2dKernelChoice::Direct => conv2d_grad_weight_direct_core(
                    input_data, grad_data, n, c_in, h_in, w_in, c_out, k_h, k_w, h_out, w_out,
                    stride_h, stride_w, pad_h, pad_w,
                ),
                Conv2dKernelChoice::Im2col => conv2d_grad_weight_im2col_core(
                    input_data, grad_data, n, c_in, h_in, w_in, c_out, k_h, k_w, h_out, w_out,
                    stride_h, stride_w, pad_h, pad_w,
                ),
            };

            let grad_weight_tensor =
                Tensor::new_with_storage(Storage::new(grad_weight_data), self.weight.shape());
            self.weight.accumulate_grad(&grad_weight_tensor);
            self.weight.backward_step();
        }
    }
}

pub fn conv2d(
    input: &Tensor,
    weight: &Tensor,
    stride: (usize, usize),
    padding: (usize, usize),
) -> Tensor {
    let input_shape = input.shape();
    let weight_shape = weight.shape();

    if input_shape.len() != 4 || weight_shape.len() != 4 {
        panic!("Conv2d requires 4D tensors");
    }

    let n = input_shape[0];
    let c_in = input_shape[1];
    let h_in = input_shape[2];
    let w_in = input_shape[3];

    let c_out = weight_shape[0];
    let k_h = weight_shape[2];
    let k_w = weight_shape[3];

    if weight_shape[1] != c_in {
        panic!(
            "Weight input channels {} must match input channels {}",
            weight_shape[1], c_in
        );
    }

    let (stride_h, stride_w) = stride;
    let (pad_h, pad_w) = padding;

    let h_out = (h_in + 2 * pad_h - k_h) / stride_h + 1;
    let w_out = (w_in + 2 * pad_w - k_w) / stride_w + 1;

    let input_contig = if input.is_contiguous() {
        input.clone()
    } else {
        input.contiguous()
    };
    let weight_contig = if weight.is_contiguous() {
        weight.clone()
    } else {
        weight.contiguous()
    };

    let input_guard = input_contig.data();
    let weight_guard = weight_contig.data();
    let input_data = &*input_guard;
    let weight_data = &*weight_guard;

    let cfg = cpu_conv2d_config();
    let work = n * c_out * h_out * w_out * c_in * k_h * k_w;
    let key: Conv2dPerfKey = (
        n, c_in, h_in, w_in, c_out, k_h, k_w, stride_h, stride_w, pad_h, pad_w,
    );
    let kernel = match cfg.strategy {
        CpuConv2dStrategy::Direct => Conv2dKernelChoice::Direct,
        CpuConv2dStrategy::Im2col => Conv2dKernelChoice::Im2col,
        CpuConv2dStrategy::Auto => {
            if work >= cfg.min_work {
                Conv2dKernelChoice::Im2col
            } else {
                Conv2dKernelChoice::Direct
            }
        }
        CpuConv2dStrategy::Profile => {
            if let Some(cached) = conv2d_profile_cache().lock().get(&key).copied() {
                cached
            } else {
                let iters = cfg.profile_iters.max(1);
                let mut direct_ns = 0u128;
                let mut im2col_ns = 0u128;
                for _ in 0..iters {
                    let t0 = Instant::now();
                    let d = conv2d_direct_core(
                        input_data,
                        weight_data,
                        n,
                        c_in,
                        h_in,
                        w_in,
                        c_out,
                        k_h,
                        k_w,
                        h_out,
                        w_out,
                        stride_h,
                        stride_w,
                        pad_h,
                        pad_w,
                    );
                    direct_ns += t0.elapsed().as_nanos();
                    black_box(d.len());
                    let t1 = Instant::now();
                    let c = conv2d_im2col_core(
                        input_data,
                        weight_data,
                        n,
                        c_in,
                        h_in,
                        w_in,
                        c_out,
                        k_h,
                        k_w,
                        h_out,
                        w_out,
                        stride_h,
                        stride_w,
                        pad_h,
                        pad_w,
                    );
                    im2col_ns += t1.elapsed().as_nanos();
                    black_box(c.len());
                }
                let choice = if im2col_ns < direct_ns {
                    Conv2dKernelChoice::Im2col
                } else {
                    Conv2dKernelChoice::Direct
                };
                conv2d_profile_cache().lock().insert(key, choice);
                choice
            }
        }
    };

    let result_data = match kernel {
        Conv2dKernelChoice::Direct => conv2d_direct_core(
            input_data,
            weight_data,
            n,
            c_in,
            h_in,
            w_in,
            c_out,
            k_h,
            k_w,
            h_out,
            w_out,
            stride_h,
            stride_w,
            pad_h,
            pad_w,
        ),
        Conv2dKernelChoice::Im2col => conv2d_im2col_core(
            input_data,
            weight_data,
            n,
            c_in,
            h_in,
            w_in,
            c_out,
            k_h,
            k_w,
            h_out,
            w_out,
            stride_h,
            stride_w,
            pad_h,
            pad_w,
        ),
    };

    let storage = Storage::new(result_data);
    let mut tensor = Tensor::new_with_storage(storage, &[n, c_out, h_out, w_out]);

    if input.requires_grad() || weight.requires_grad() {
        tensor.set_requires_grad_mut(true);
        tensor.set_op(Arc::new(Conv2dBackward {
            input: input.clone(),
            weight: weight.clone(),
            stride,
            padding,
        }));
    }

    // if crate::graph::is_tracing() {
    //     crate::graph::record_op(crate::graph::NodeOp::Conv2d { stride, padding }, &[input, weight], &tensor);
    // }

    tensor
}
