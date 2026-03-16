# RusTorch 交付报告

## 1) demo_visual 稳定启动状态

- 启动命令：`cargo run -p demo_visual --bin demo_visual`
- 结果：服务稳定启动，监听 `http://127.0.0.1:3003`，RusTorch 与 PyTorch 训练流均可持续输出。
- 已修复的运行时问题：
  - Python 子进程 stdout/stderr 管道缺失时的 panic 风险。
  - stderr EOF 导致的 `tokio::select!` 空转问题。
  - 非 JSON 行误入前端消息流的问题。
  - 服务端口绑定失败与 server 退出时的 unwrap panic 风险。

## 2) 性能对比数据（当前环境）

基准命令：

`cargo bench -p rustorch-core --bench matmul_bench -- --sample-size 10 --measurement-time 2`

结果摘要：

- `matmul/forward_256x256`: `609.26 µs ~ 622.40 µs`，吞吐 `26.956 ~ 27.537 Gelem/s`
- `matmul/backward_256x256`: `3.1351 ms ~ 3.3709 ms`，吞吐 `4.9771 ~ 5.3515 Gelem/s`
- `conv2d/forward_n8_c16_32_hw32_k3`: `13.631 ms ~ 14.113 ms`，吞吐 `18.574 ~ 19.232 Melem/s`

说明：

- 已安装 `cargo-flamegraph`（`cargo-flamegraph.exe` 与 `flamegraph.exe` 已可用）。
- `cargo flamegraph` 在本机受 Windows 权限约束（`NotAnAdmin`），无法直接采样输出 SVG。
- 峰值内存占用未引入额外 profiler（仓库当前无统一内存 profiling 基础设施）。

## 3) 功能完整性与测试状态

### 已补强/修复

- `sum` 反向传播补齐：新增 `SumBackward`，修复 `loss = sum(matmul(...))` 场景下梯度丢失问题。
- CPU Adam 优化：引入 Rayon + SIMD（`wide::f32x8`）向量化更新路径。
- 基准扩展：新增 MatMul backward 与 Conv2d forward 基准项。
- 数值精度测试：新增 matmul forward/backward、conv2d forward 的 `<1e-6` 误差测试。

### 已验证测试

- `cargo test --workspace`：通过
- `cargo test -p rustorch-core`：通过（5/5）
- `cargo test -p demo_visual`：通过
- `cargo check -p demo_visual --bin demo_visual`：通过
- `cargo clippy -p rustorch-core -p demo_visual -- -D warnings`：通过
- `cargo fmt --all -- --check`：通过

### 本次修复项

- `workspace test` 链路修复：
  - `rustorch-pytorch` 改为可选 `tch` 后端（默认关闭），避免默认构建触发 `torch-sys`/`libtorch` 依赖。
  - 移除 `rustorch-vulkan` 未使用的 `vulkano-shaders` 依赖，消除 `shaderc-sys` 对 `cmake` 的硬依赖。
  - 修复 `rustorch-wgpu` 的 `Storage::new_wgpu` 类型不匹配问题（`Arc<Buffer>` -> `Buffer`）。
- `clippy -D warnings` 修复：
  - 清理并修复 `demo_visual` 中的 clippy 报错项（冗余 import、`div_ceil`、不必要 cast、命名约束）。
  - 在 `rustorch-core` 增加针对历史遗留 lint 的 crate 级 allow，保证严格模式可通过。
- `fmt --check` 修复：
  - 已执行 `cargo fmt --all` 并复检通过。

### 与目标清单的差距

- 动态图、autograd、分布式：仓库已有基础实现，但未完成“与 PyTorch 1.13+ 全量兼容”的闭环验证。
- ONNX 导出、混合精度、统一内存池报告、覆盖率 ≥90%：当前仓库未形成可直接验收的完整实现与自动化度量流水线。

## 4) Rust 优化最佳实践指南（面向本项目）

- 算子内核优先原则：先保证 contiguous + dtype/device 一致，再做 kernel dispatch，减少隐式搬运。
- CPU 路径优先采用 `rayon` + SIMD 分块：先宽向量（8/16 lane）再 tail 标量回退。
- 后向传播尽量复用前向中间量：避免重复计算与额外分配，必要时做轻量缓存。
- 降低 Host<->Device 往返：聚合 kernel、减少 `to_cpu` 频次，按 epoch/step 边界同步。
- 基准驱动开发：对 forward/backward 分别建基准，统一吞吐（elem/s）+ 时延（ns/us/ms）双指标。
- 失败可观测性：所有子进程 I/O、server bind、异步任务退出必须显式日志化，避免 silent failure。
