
struct Params {
    M: u32,
    K: u32,
    N: u32,
    activation: u32, // 0: None, 1: ReLU, 2: Sigmoid, 3: Tanh
}

@group(0) @binding(0) var<storage, read> lhs: array<f32>;
@group(0) @binding(1) var<storage, read> rhs: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;
@group(0) @binding(4) var<storage, read> bias: array<f32>; // Optional bias

// TILE_K: How many K values we process per inner loop
// Each thread processes a 4x4 block of output C
// Workgroup size is 16x16 threads
// Workgroup output tile: (16*4) x (16*4) = 64x64 elements of C
// We need to fit two tiles (LHS: 64xTILE_K, RHS: TILE_Kx64) in shared memory.
// Let's keep TILE_K small to fit in shared memory limits (16KB/32KB typical).
// If TILE_K = 8:
// LHS: 64 * 8 * 4 bytes = 2048 bytes
// RHS: 8 * 64 * 4 bytes = 2048 bytes
// Total: 4KB << 16KB limit. Safe.

const TILE_K: u32 = 8u;
const WORKGROUP_SIZE: u32 = 16u;
const MICRO_TILE: u32 = 4u; // Each thread computes 4x4 elements

var<workgroup> tile_lhs: array<array<f32, TILE_K>, 64>; // [64][8]
var<workgroup> tile_rhs: array<array<f32, 64>, TILE_K>; // [8][64]

@compute @workgroup_size(16, 16, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) group_id: vec3<u32>
) {
    let m = params.M;
    let k = params.K;
    let n = params.N;

    let local_x = local_id.x; // 0..15
    let local_y = local_id.y; // 0..15
    let group_x = group_id.x; // row block index
    let group_y = group_id.y; // col block index

    // Global output indices for the top-left corner of this thread's 4x4 block
    // Each workgroup computes 64 rows.
    let global_row_start = group_x * 64u + local_x * 4u;
    // Each workgroup computes 64 cols.
    let global_col_start = group_y * 64u + local_y * 4u;

    // Accumulators for C[4][4] in registers
    var sum: array<array<f32, 4>, 4>;
    for (var i = 0u; i < 4u; i = i + 1u) {
        for (var j = 0u; j < 4u; j = j + 1u) {
            sum[i][j] = 0.0;
        }
    }

    // Number of K tiles
    let num_tiles = (k + TILE_K - 1u) / TILE_K;

    for (var t = 0u; t < num_tiles; t = t + 1u) {
        // --- Cooperative Loading into Shared Memory ---
        
        let flat_id = local_x * 16u + local_y; // 0..255

        // Load LHS: [64][8]
        // 256 threads loading 64*8 = 512 elements => 2 elements per thread
        
        // Element 0
        let lhs_row_0 = flat_id / TILE_K; // 0..31
        let lhs_col_0 = flat_id % TILE_K; // 0..7
        let global_lhs_row_0 = group_x * 64u + lhs_row_0;
        let global_lhs_col_0 = t * TILE_K + lhs_col_0;
        
        if (global_lhs_row_0 < m && global_lhs_col_0 < k) {
            tile_lhs[lhs_row_0][lhs_col_0] = lhs[global_lhs_row_0 * k + global_lhs_col_0];
        } else {
            tile_lhs[lhs_row_0][lhs_col_0] = 0.0;
        }

        // Element 1 (offset by 32)
        let lhs_row_1 = lhs_row_0 + 32u;
        let global_lhs_row_1 = group_x * 64u + lhs_row_1;
        // Same col
        if (global_lhs_row_1 < m && global_lhs_col_0 < k) {
            tile_lhs[lhs_row_1][lhs_col_0] = lhs[global_lhs_row_1 * k + global_lhs_col_0];
        } else {
            tile_lhs[lhs_row_1][lhs_col_0] = 0.0;
        }

        // Load RHS: [8][64]
        // 256 threads loading 8*64 = 512 elements => 2 elements per thread
        
        // Element 0
        let rhs_row_0 = flat_id / 64u; // 0..3
        let rhs_col_0 = flat_id % 64u; // 0..63
        let global_rhs_row_0 = t * TILE_K + rhs_row_0;
        let global_rhs_col_0 = group_y * 64u + rhs_col_0;

        if (global_rhs_row_0 < k && global_rhs_col_0 < n) {
            tile_rhs[rhs_row_0][rhs_col_0] = rhs[global_rhs_row_0 * n + global_rhs_col_0];
        } else {
            tile_rhs[rhs_row_0][rhs_col_0] = 0.0;
        }

        // Element 1 (offset by 4 rows)
        let rhs_row_1 = rhs_row_0 + 4u;
        let global_rhs_row_1 = t * TILE_K + rhs_row_1;
        
        if (global_rhs_row_1 < k && global_rhs_col_0 < n) {
            tile_rhs[rhs_row_1][rhs_col_0] = rhs[global_rhs_row_1 * n + global_rhs_col_0];
        } else {
            tile_rhs[rhs_row_1][rhs_col_0] = 0.0;
        }

        workgroupBarrier();

        // --- Compute Tile Product ---
        for (var k_i = 0u; k_i < TILE_K; k_i = k_i + 1u) {
            for (var i = 0u; i < 4u; i = i + 1u) {
                for (var j = 0u; j < 4u; j = j + 1u) {
                    let val_lhs = tile_lhs[local_x * 4u + i][k_i];
                    let val_rhs = tile_rhs[k_i][local_y * 4u + j];
                    sum[i][j] = sum[i][j] + val_lhs * val_rhs;
                }
            }
        }

        workgroupBarrier();
    }

    // Write back results
    for (var i = 0u; i < 4u; i = i + 1u) {
        for (var j = 0u; j < 4u; j = j + 1u) {
            let global_row = global_row_start + i;
            let global_col = global_col_start + j;

            if (global_row < m && global_col < n) {
                var val = sum[i][j];
                
                // Add Bias (if binding is valid - handled by logic or dummy buffer)
                // We can't check buffer size here. We assume bias is [N] if present.
                // Or [1, N] broadcasted.
                // WGSL binding 4 is array<f32>.
                // We use a param flag or assume if activation > 0 or special flag?
                // Wait, we need a flag for "has_bias".
                // Let's use `activation` high bits? Or a new field?
                // Params is uniform, 16-byte aligned.
                // M, K, N, activation.
                // Let's pack has_bias into activation: (activation & 0xFF) | (has_bias << 8)
                
                let act_type = params.activation & 0xFFu;
                let has_bias = (params.activation >> 8u) & 0xFFu;
                
                if (has_bias == 1u) {
                    val = val + bias[global_col]; // Broadcast bias [N]
                }

                // Apply Activation
                if (act_type == 1u) { // ReLU
                    val = max(val, 0.0);
                } else if (act_type == 2u) { // Sigmoid
                    val = 1.0 / (1.0 + exp(-val));
                } else if (act_type == 3u) { // Tanh
                    val = tanh(val);
                }
                
                output[global_row * n + global_col] = val;
            }
        }
    }
}
