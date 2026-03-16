use crate::autograd::BackwardOp;
use crate::storage::Storage;
use crate::Tensor;
use parking_lot::Mutex;
use rayon::prelude::*;
use std::collections::HashMap;
use std::hint::black_box;
use std::sync::{Arc, OnceLock};
use std::time::Instant;
use wide::f32x8;

// --- BatchNorm2d ---

#[derive(Debug)]
pub struct BatchNorm2dBackwardFull {
    pub input: Tensor,
    pub gamma: Option<Tensor>,
    pub beta: Option<Tensor>,
    pub mean: Vec<f32>,
    pub inv_std: Vec<f32>,
    pub eps: f32,
}

impl BackwardOp for BatchNorm2dBackwardFull {
    fn backward(&self, grad: &Tensor) {
        if self.input.requires_grad() {
            // Simplified Backward for BatchNorm
            // Assuming training mode logic (complex)
            // Or just use a simplified diagonal approximation if strictly needed for MVP.
            // But let's try to be reasonably correct.

            let n = self.input.shape()[0];
            let c = self.input.shape()[1];
            let h = self.input.shape()[2];
            let w = self.input.shape()[3];
            let _num_pixels = (h * w) as f32;
            let m = (n * h * w) as f32;

            let grad_guard = grad.data();
            let grad_data = &*grad_guard;

            let input_guard = self.input.data();
            let input_data = &*input_guard;

            let gamma_data = self.gamma.as_ref().map(|g| g.data());

            // Grads for gamma and beta
            let mut d_gamma = vec![0.0; c];
            let mut d_beta = vec![0.0; c];

            // First pass: compute d_gamma, d_beta
            // Parallelize over C?
            // Accumulate over N, H, W

            // This is reduction.
            // Let's do it serial over C for simplicity or parallel over C.

            let grads: Vec<(f32, f32)> = (0..c)
                .into_par_iter()
                .map(|ci| {
                    let mut dg = 0.0;
                    let mut db = 0.0;
                    let mean = self.mean[ci];
                    let inv_std = self.inv_std[ci];

                    for b in 0..n {
                        for hi in 0..h {
                            for wi in 0..w {
                                let idx = ((b * c + ci) * h + hi) * w + wi;
                                let dy = grad_data[idx];
                                let x = input_data[idx];
                                let x_hat = (x - mean) * inv_std;

                                dg += dy * x_hat;
                                db += dy;
                            }
                        }
                    }
                    (dg, db)
                })
                .collect();

            for (i, (dg, db)) in grads.iter().enumerate() {
                d_gamma[i] = *dg;
                d_beta[i] = *db;
            }

            // Update gamma/beta grads
            if let Some(gamma) = &self.gamma {
                if gamma.requires_grad() {
                    let g_grad = Tensor::new(&d_gamma, gamma.shape());
                    gamma.accumulate_grad(&g_grad);
                    gamma.backward_step();
                }
            }
            if let Some(beta) = &self.beta {
                if beta.requires_grad() {
                    let b_grad = Tensor::new(&d_beta, beta.shape());
                    beta.accumulate_grad(&b_grad);
                    beta.backward_step();
                }
            }

            // Compute d_input
            // dx_hat = dy * gamma
            // dx = (1/m) * inv_std * (m * dx_hat - sum(dx_hat) - x_hat * sum(dx_hat * x_hat))

            let mut d_input_data = vec![0.0; input_data.len()];

            d_input_data
                .par_chunks_mut(h * w * c)
                .enumerate()
                .for_each(|(b, batch_slice)| {
                    for ci in 0..c {
                        let mean = self.mean[ci];
                        let inv_std = self.inv_std[ci];
                        let gamma_val = if let Some(g) = &gamma_data {
                            g[ci]
                        } else {
                            1.0
                        };

                        // We need sum(dx_hat) and sum(dx_hat * x_hat) over the whole batch!
                        // But we are inside parallel chunk.
                        // We computed these sums in d_gamma (sum(dy*x_hat)) and d_beta (sum(dy)).
                        // sum(dx_hat) = sum(dy * gamma) = gamma * sum(dy) = gamma * d_beta
                        // sum(dx_hat * x_hat) = sum(dy * gamma * x_hat) = gamma * sum(dy * x_hat) = gamma * d_gamma

                        let sum_dx_hat = gamma_val * d_beta[ci];
                        let sum_dx_hat_x_hat = gamma_val * d_gamma[ci];

                        for hi in 0..h {
                            for wi in 0..w {
                                let local_idx = (ci * h + hi) * w + wi;
                                let global_idx = ((b * c + ci) * h + hi) * w + wi; // Wait, b is batch index?
                                                                                   // batch_slice is slice of size C*H*W for batch b.

                                let dy = grad_data[global_idx];
                                let x = input_data[global_idx];
                                let x_hat = (x - mean) * inv_std;
                                let dx_hat = dy * gamma_val;

                                let val = (1.0 / m)
                                    * inv_std
                                    * (m * dx_hat - sum_dx_hat - x_hat * sum_dx_hat_x_hat);
                                batch_slice[local_idx] = val;
                            }
                        }
                    }
                });

            let d_input = Tensor::new(&d_input_data, self.input.shape());
            self.input.accumulate_grad(&d_input);
            self.input.backward_step();
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub fn batch_norm2d(
    input: &Tensor,
    gamma: Option<&Tensor>,
    beta: Option<&Tensor>,
    running_mean: &Tensor,
    running_var: &Tensor,
    training: bool,
    momentum: f32,
    eps: f32,
) -> Tensor {
    let shape = input.shape();
    if shape.len() != 4 {
        panic!("BatchNorm2d requires 4D tensor");
    }
    let n = shape[0];
    let c = shape[1];
    let h = shape[2];
    let w = shape[3];

    let input_guard = input.data();
    let input_data = &*input_guard;

    // Compute mean and var per channel
    let (mean, var) = if training {
        let stats: Vec<(f32, f32)> = (0..c)
            .into_par_iter()
            .map(|ci| {
                let mut sum = 0.0;
                let mut sq_sum = 0.0;
                let count = (n * h * w) as f32;

                for b in 0..n {
                    for hi in 0..h {
                        for wi in 0..w {
                            let val = input_data[((b * c + ci) * h + hi) * w + wi];
                            sum += val;
                            sq_sum += val * val;
                        }
                    }
                }

                let m = sum / count;
                let v = sq_sum / count - m * m;
                (m, v)
            })
            .collect();

        let mut means = Vec::with_capacity(c);
        let mut vars = Vec::with_capacity(c);
        for (m, v) in stats {
            means.push(m);
            vars.push(v);
        }

        // Update running stats
        let mut r_mean = running_mean.data_mut();
        let mut r_var = running_var.data_mut();

        for i in 0..c {
            r_mean[i] = (1.0 - momentum) * r_mean[i] + momentum * means[i];
            r_var[i] = (1.0 - momentum) * r_var[i] + momentum * vars[i];
        }

        (means, vars)
    } else {
        // Inference: use running stats
        let r_mean = running_mean.data().clone();
        let r_var = running_var.data().clone();
        (r_mean, r_var)
    };

    // Normalize
    let mut output_data = vec![0.0; n * c * h * w];
    let gamma_data = gamma.map(|g| g.data());
    let beta_data = beta.map(|b| b.data());

    // Precompute inv_std
    let inv_std: Vec<f32> = var.iter().map(|v| 1.0 / (v + eps).sqrt()).collect();

    output_data
        .par_chunks_mut(h * w)
        .enumerate()
        .for_each(|(i, plane)| {
            let _ci = (i / n) % c; // Wait, chunk is H*W? No.
                                   // Index mapping:
                                   // i goes from 0 to N*C-1.
                                   // idx = i * (H*W)
                                   // b = i / c;
                                   // ci = i % c;
                                   // Wait, memory layout is N, C, H, W.
                                   // So continuous chunks of H*W belong to (b, ci).

            let ci = i % c;
            // let b = i / c;

            let _m = mean[ci];
            let _inv_s = inv_std[ci];
            let _g = if let Some(gd) = &gamma_data {
                gd[ci]
            } else {
                1.0
            };
            let _b_val = if let Some(bd) = &beta_data {
                bd[ci]
            } else {
                0.0
            };

            for _x in plane.iter_mut() { // plane is pre-filled with 0.0, we need to read input
                 // Oops, we don't have input slice here easily unless we zip or calculate index.
                 // Let's iterate index.
            }
        });

    // Rewrite parallel loop properly
    output_data
        .par_chunks_mut(h * w)
        .enumerate()
        .for_each(|(i, output_plane)| {
            let ci = i % c;
            let m = mean[ci];
            let inv_s = inv_std[ci];
            let g = if let Some(gd) = &gamma_data {
                gd[ci]
            } else {
                1.0
            };
            let b_val = if let Some(bd) = &beta_data {
                bd[ci]
            } else {
                0.0
            };

            let start_idx = i * h * w;
            for j in 0..h * w {
                let val = input_data[start_idx + j];
                output_plane[j] = (val - m) * inv_s * g + b_val;
            }
        });

    let storage = Storage::new(output_data);
    let mut tensor = Tensor::new_with_storage(storage, shape);

    if training
        && (input.requires_grad()
            || gamma.is_some_and(|g| g.requires_grad())
            || beta.is_some_and(|b| b.requires_grad()))
    {
        tensor.set_requires_grad_mut(true);
        // Need to pass mean/inv_std to backward
        // But mean/inv_std are Vec<f32>.
        // Backward op expects them?
        // We defined BackwardOp with Tensor or Vec?
        // Let's use Vec for internal storage in op struct.

        tensor.set_op(Arc::new(BatchNorm2dBackwardFull {
            input: input.clone(),
            gamma: gamma.cloned(),
            beta: beta.cloned(),
            mean,
            inv_std,
            eps,
        }));
    }

    tensor
}

// --- LayerNorm ---
// Input: (N, *, Normalized_Shape)
// Output: Same
// Mean/Var computed over last D dims.

#[derive(Debug)]
pub struct LayerNormBackward {
    pub input: Tensor,
    pub weight: Option<Tensor>,
    pub bias: Option<Tensor>,
    pub mean: Tensor,
    pub inv_std: Tensor,
    pub normalized_shape: Vec<usize>,
}

impl BackwardOp for LayerNormBackward {
    fn backward(&self, grad: &Tensor) {
        if self.input.requires_grad() {
            // Simplified backward for LayerNorm
            // dL/dx = ... complex formula ...
            // Similar to BatchNorm but over different dimensions.
            // For now, implement a simplified version or placeholder.
            // Implementing full LayerNorm backward is verbose.
            // Let's assume standard implementation.

            // TODO: Full implementation.
            // Just passing gradient through for now (Incorrect but allows graph flow)
            // Or better: Implement correctly for 1D case (common in Transformer).

            // Assuming input (N, L, D) and norm over D.
            // Mean/Var is (N, L, 1).

            // Let's implement correct logic for last dim normalization.
            let shape = self.input.shape();
            let dim = *shape.last().unwrap();
            let n_elements: usize = shape.iter().take(shape.len() - 1).product();

            // Grad input calculation...
            // It's quite involved to write out from scratch in one go.
            // Given constraints, I'll use a simplified approximation or placeholder
            // that warns it's not fully implemented.
            // Or, if I want to be diligent:
            // dL/dx_i = 1/D * gamma * inv_std * ( D*dy_i - sum(dy) - x_hat_i * sum(dy * x_hat) )

            let grad_guard = grad.data();
            let grad_data = &*grad_guard;

            let input_guard = self.input.data();
            let input_data = &*input_guard;

            let mean_guard = self.mean.data(); // (N*L)
            let mean_data = &*mean_guard;

            let inv_std_guard = self.inv_std.data(); // (N*L)
            let inv_std_data = &*inv_std_guard;

            let weight_data = self.weight.as_ref().map(|w| w.data());

            let mut grad_input_data = vec![0.0; grad_data.len()];

            // Iterate over each instance (N*L)
            grad_input_data
                .par_chunks_mut(dim)
                .enumerate()
                .for_each(|(i, dx_row)| {
                    let m = mean_data[i];
                    let inv_s = inv_std_data[i];
                    let offset = i * dim;

                    let mut sum_dy = 0.0;
                    let mut sum_dy_x_hat = 0.0;

                    for j in 0..dim {
                        let dy = grad_data[offset + j];
                        let x = input_data[offset + j];
                        let x_hat = (x - m) * inv_s;
                        let g = if let Some(wd) = &weight_data {
                            wd[j]
                        } else {
                            1.0
                        };

                        // dy is dL/dy.
                        // Effective dy for normalization part is dy * gamma
                        let dy_eff = dy * g;

                        sum_dy += dy_eff;
                        sum_dy_x_hat += dy_eff * x_hat;
                    }

                    let factor = inv_s / (dim as f32);

                    for j in 0..dim {
                        let dy = grad_data[offset + j];
                        let x = input_data[offset + j];
                        let x_hat = (x - m) * inv_s;
                        let g = if let Some(wd) = &weight_data {
                            wd[j]
                        } else {
                            1.0
                        };
                        let dy_eff = dy * g;

                        dx_row[j] =
                            factor * ((dim as f32) * dy_eff - sum_dy - x_hat * sum_dy_x_hat);
                    }
                });

            let grad_input = Tensor::new(&grad_input_data, shape);
            self.input.accumulate_grad(&grad_input);
            self.input.backward_step();

            // Weight/Bias grads
            if let Some(weight) = &self.weight {
                if weight.requires_grad() {
                    // sum(dy * x_hat) over all batches
                    let mut dw_data = vec![0.0; dim];
                    // Need atomic add or reduce.
                    // Simple serial reduce for now.
                    for i in 0..n_elements {
                        let offset = i * dim;
                        let m = mean_data[i];
                        let inv_s = inv_std_data[i];
                        for j in 0..dim {
                            let dy = grad_data[offset + j];
                            let x = input_data[offset + j];
                            let x_hat = (x - m) * inv_s;
                            dw_data[j] += dy * x_hat;
                        }
                    }
                    let dw = Tensor::new(&dw_data, weight.shape());
                    weight.accumulate_grad(&dw);
                    weight.backward_step();
                }
            }

            if let Some(bias) = &self.bias {
                if bias.requires_grad() {
                    let mut db_data = vec![0.0; dim];
                    for i in 0..n_elements {
                        let offset = i * dim;
                        for j in 0..dim {
                            db_data[j] += grad_data[offset + j];
                        }
                    }
                    let db = Tensor::new(&db_data, bias.shape());
                    bias.accumulate_grad(&db);
                    bias.backward_step();
                }
            }
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum CpuLayerNormStrategy {
    Auto,
    Profile,
    Scalar,
    Simd,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum LayerNormKernelChoice {
    Scalar,
    Simd,
}

#[derive(Clone, Copy)]
struct CpuLayerNormConfig {
    strategy: CpuLayerNormStrategy,
    min_dim: usize,
    profile_iters: usize,
}

fn parse_usize_env(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(default)
}

fn cpu_layernorm_config() -> CpuLayerNormConfig {
    static CFG: OnceLock<CpuLayerNormConfig> = OnceLock::new();
    *CFG.get_or_init(|| {
        let strategy = match std::env::var("RUSTORCH_CPU_LAYERNORM_STRATEGY")
            .unwrap_or_else(|_| "auto".to_string())
            .to_ascii_lowercase()
            .as_str()
        {
            "simd" => CpuLayerNormStrategy::Simd,
            "scalar" => CpuLayerNormStrategy::Scalar,
            "profile" => CpuLayerNormStrategy::Profile,
            _ => CpuLayerNormStrategy::Auto,
        };
        CpuLayerNormConfig {
            strategy,
            min_dim: parse_usize_env("RUSTORCH_CPU_LAYERNORM_MIN_DIM", 256),
            profile_iters: parse_usize_env("RUSTORCH_CPU_LAYERNORM_PROFILE_ITERS", 2),
        }
    })
}

fn layernorm_profile_cache() -> &'static Mutex<HashMap<usize, LayerNormKernelChoice>> {
    static CACHE: OnceLock<Mutex<HashMap<usize, LayerNormKernelChoice>>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

fn row_sum_sq_scalar(row: &[f32]) -> (f32, f32) {
    let mut s = 0.0f32;
    let mut q = 0.0f32;
    for &v in row {
        s += v;
        q += v * v;
    }
    (s, q)
}

fn row_sum_sq_simd(row: &[f32]) -> (f32, f32) {
    let lanes = 8usize;
    let vec_len = row.len() / lanes * lanes;
    let mut s = f32x8::from([0.0; 8]);
    let mut q = f32x8::from([0.0; 8]);
    let mut i = 0usize;
    while i < vec_len {
        let mut a = [0.0f32; 8];
        a.copy_from_slice(&row[i..i + lanes]);
        let v = f32x8::from(a);
        s += v;
        q += v * v;
        i += lanes;
    }
    let sa: [f32; 8] = s.into();
    let qa: [f32; 8] = q.into();
    let mut sum = sa.iter().sum::<f32>();
    let mut sq = qa.iter().sum::<f32>();
    while i < row.len() {
        let v = row[i];
        sum += v;
        sq += v * v;
        i += 1;
    }
    (sum, sq)
}

fn choose_layernorm_kernel(inner_dim: usize, sample_row: &[f32]) -> LayerNormKernelChoice {
    let cfg = cpu_layernorm_config();
    match cfg.strategy {
        CpuLayerNormStrategy::Simd => LayerNormKernelChoice::Simd,
        CpuLayerNormStrategy::Scalar => LayerNormKernelChoice::Scalar,
        CpuLayerNormStrategy::Auto => {
            if inner_dim >= cfg.min_dim {
                LayerNormKernelChoice::Simd
            } else {
                LayerNormKernelChoice::Scalar
            }
        }
        CpuLayerNormStrategy::Profile => {
            if let Some(cached) = layernorm_profile_cache().lock().get(&inner_dim).copied() {
                return cached;
            }
            let iters = cfg.profile_iters.max(1);
            let mut scalar_ns = 0u128;
            let mut simd_ns = 0u128;
            for _ in 0..iters {
                let t0 = Instant::now();
                let s = row_sum_sq_scalar(sample_row);
                scalar_ns += t0.elapsed().as_nanos();
                black_box(s);
                let t1 = Instant::now();
                let v = row_sum_sq_simd(sample_row);
                simd_ns += t1.elapsed().as_nanos();
                black_box(v);
            }
            let choice = if simd_ns < scalar_ns {
                LayerNormKernelChoice::Simd
            } else {
                LayerNormKernelChoice::Scalar
            };
            layernorm_profile_cache().lock().insert(inner_dim, choice);
            choice
        }
    }
}

pub fn layer_norm(
    input: &Tensor,
    normalized_shape: &[usize],
    weight: Option<&Tensor>,
    bias: Option<&Tensor>,
    eps: f32,
) -> Tensor {
    // Assume normalized_shape is last D dims.
    // Flatten input to (N, D_norm)
    let shape = input.shape();
    let norm_dims = normalized_shape.len();
    if shape.len() < norm_dims {
        panic!(
            "LayerNorm: Input shape {:?} smaller than normalized shape {:?}",
            shape, normalized_shape
        );
    }

    // Check suffixes match
    let start_dim = shape.len() - norm_dims;
    if &shape[start_dim..] != normalized_shape {
        panic!(
            "LayerNorm: Input shape {:?} does not match normalized shape {:?}",
            shape, normalized_shape
        );
    }

    let outer_dim: usize = shape[0..start_dim].iter().product();
    let inner_dim: usize = normalized_shape.iter().product(); // D

    let input_guard = input.data();
    let input_data = &*input_guard;

    let weight_vec = weight.map(|w| w.data().to_vec());
    let bias_vec = bias.map(|b| b.data().to_vec());
    let weight_data = weight_vec.as_deref();
    let bias_data = bias_vec.as_deref();

    let mut output_data = vec![0.0; input_data.len()];
    let mut means = vec![0.0; outer_dim];
    let mut inv_stds = vec![0.0; outer_dim];

    // Compute mean/var per outer instance
    // Parallelize over outer_dim
    // We need to return means/inv_stds, so maybe collect.

    let sample_row = if outer_dim > 0 {
        &input_data[0..inner_dim]
    } else {
        &[][..]
    };
    let kernel = choose_layernorm_kernel(inner_dim, sample_row);

    let stats: Vec<(f32, f32)> = (0..outer_dim)
        .into_par_iter()
        .map(|i| {
            let offset = i * inner_dim;
            let row = &input_data[offset..offset + inner_dim];
            let (sum, sq_sum) = match kernel {
                LayerNormKernelChoice::Scalar => row_sum_sq_scalar(row),
                LayerNormKernelChoice::Simd => row_sum_sq_simd(row),
            };
            let m = sum / inner_dim as f32;
            let v = sq_sum / inner_dim as f32 - m * m;
            let inv_s = 1.0 / (v + eps).sqrt();
            (m, inv_s)
        })
        .collect();

    for (i, (m, inv_s)) in stats.iter().enumerate() {
        means[i] = *m;
        inv_stds[i] = *inv_s;
    }

    output_data
        .par_chunks_mut(inner_dim)
        .enumerate()
        .for_each(|(i, row_out)| {
            let m = means[i];
            let inv_s = inv_stds[i];
            let offset = i * inner_dim;
            for j in 0..inner_dim {
                let val = input_data[offset + j];
                let x_hat = (val - m) * inv_s;
                let g = if let Some(wd) = weight_data {
                    wd[j]
                } else {
                    1.0
                };
                let b = if let Some(bd) = bias_data { bd[j] } else { 0.0 };
                row_out[j] = x_hat * g + b;
            }
        });

    let storage = Storage::new(output_data);
    let mut tensor = Tensor::new_with_storage(storage, shape);

    if input.requires_grad()
        || weight.is_some_and(|w| w.requires_grad())
        || bias.is_some_and(|b| b.requires_grad())
    {
        tensor.set_requires_grad_mut(true);
        // Store mean/inv_std for backward
        // They are (OuterDim). We can store as (OuterDim) tensor.
        let mean_tensor = Tensor::new(&means, &[outer_dim]); // Flattened stats
        let inv_std_tensor = Tensor::new(&inv_stds, &[outer_dim]);

        tensor.set_op(Arc::new(LayerNormBackward {
            input: input.clone(),
            weight: weight.cloned(),
            bias: bias.cloned(),
            mean: mean_tensor,
            inv_std: inv_std_tensor,
            normalized_shape: normalized_shape.to_vec(),
        }));
    }

    tensor
}
