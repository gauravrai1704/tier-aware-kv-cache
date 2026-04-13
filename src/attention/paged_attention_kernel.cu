/**
 * paged_attention_kernel.cu
 * Tier-Aware PagedAttention CUDA Kernel
 *
 * Extends standard PagedAttention to work with:
 *  1. A virtually-contiguous KV cache (vAttention-style VMM mapping).
 *  2. Block-level metadata indicating which tier each physical block resides in.
 *
 * The kernel itself only sees the virtual address range — tier transitions
 * happen transparently via CUDA VMM physical remapping before kernel launch.
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>
#include <assert.h>

// ---------------------------------------------------------------------------
// Compile-time constants
// ---------------------------------------------------------------------------
#define WARP_SIZE       32
#define MAX_SEQ_LEN     8192
#define BLOCK_SIZE      16      // tokens per KV block (must match allocator)
#define HEAD_DIM        128
#define UNROLL          4

// ---------------------------------------------------------------------------
// Data types
// ---------------------------------------------------------------------------
using scalar_t = __half;   // fp16 KV cache

// ---------------------------------------------------------------------------
// Block metadata passed to kernel (host→device, per request)
// ---------------------------------------------------------------------------
struct BlockMetadata {
    int64_t  virtual_offset;   ///< byte offset into the VMM virtual range
    uint8_t  tier;             ///< 0=TIER1_FAST, 1=TIER2_SLOW (informational)
    uint8_t  valid;
    uint16_t ref_count;
};

// ---------------------------------------------------------------------------
// Softmax helpers
// ---------------------------------------------------------------------------
__device__ __forceinline__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1)
        val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, mask));
    return val;
}

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1)
        val += __shfl_xor_sync(0xffffffff, val, mask);
    return val;
}

// ---------------------------------------------------------------------------
// Single-query Paged Attention kernel
// ---------------------------------------------------------------------------
/**
 * tier_aware_paged_attention_kernel
 *
 * @param output        [num_heads, head_dim]  — query output
 * @param query         [num_heads, head_dim]  — current query vector
 * @param kv_cache_base Virtual base pointer of the KV cache (VMM range)
 * @param block_table   [max_blocks]           — virtual offsets per block
 * @param context_lens  scalar                 — number of valid KV tokens
 * @param scale         1/sqrt(head_dim)
 * @param num_heads     KV heads
 * @param head_dim      must be HEAD_DIM
 * @param num_blocks    number of logical blocks in context
 */
extern "C" __global__ void tier_aware_paged_attention_kernel(
    scalar_t* __restrict__        output,
    const scalar_t* __restrict__  query,
    const scalar_t* __restrict__  kv_cache_base,  // virtual range start
    const BlockMetadata* __restrict__ block_meta,
    const int32_t* __restrict__   block_table,    // block_id → index in block_meta
    int                           context_len,
    float                         scale,
    int                           num_heads,
    int                           num_kv_heads,
    int                           head_dim)
{
    // Grid: (num_heads, ceil(context_len / BLOCK_SIZE))
    // Block: (WARP_SIZE, BLOCK_SIZE/WARP_SIZE)
    const int head_idx  = blockIdx.x;
    const int blk_idx   = blockIdx.y;
    const int lane      = threadIdx.x;

    if (head_idx >= num_heads) return;

    // Shared memory: query tile + partial softmax accumulators
    extern __shared__ float smem[];  // layout: [head_dim] query + [BLOCK_SIZE] attn
    float* sq    = smem;
    float* sattn = smem + head_dim;

    // Load query for this head
    const scalar_t* q_ptr = query + head_idx * head_dim;
    for (int i = lane; i < head_dim; i += WARP_SIZE) {
        sq[i] = __half2float(q_ptr[i]);
    }
    __syncwarp();

    // -----------------------------------------------------------------------
    // Compute attention scores for tokens in this block
    // -----------------------------------------------------------------------
    const int block_id     = block_table[blk_idx];
    const BlockMetadata& m = block_meta[block_id];
    if (!m.valid) return;

    // KV head (GQA support: num_kv_heads <= num_heads)
    int kv_head = head_idx % num_kv_heads;

    // Pointer into virtual KV cache for this block and KV head
    // Layout: [block_size, 2 (K/V), num_kv_heads, head_dim]
    const scalar_t* k_base = kv_cache_base
        + m.virtual_offset / sizeof(scalar_t)
        + kv_head * head_dim;
    const scalar_t* v_base = k_base + num_kv_heads * head_dim * BLOCK_SIZE;

    int tok_start = blk_idx * BLOCK_SIZE;
    int tok_end   = min(tok_start + BLOCK_SIZE, context_len);

    // Each thread handles a subset of tokens within the block
    for (int tok = tok_start + lane; tok < tok_end; tok += WARP_SIZE) {
        int local_tok = tok - tok_start;
        const scalar_t* k_tok = k_base + local_tok * num_kv_heads * head_dim;

        // Q·K dot product
        float qk = 0.0f;
        #pragma unroll 4
        for (int d = 0; d < head_dim; ++d) {
            qk += sq[d] * __half2float(k_tok[d]);
        }
        sattn[local_tok] = qk * scale;
    }
    __syncwarp();

    // -----------------------------------------------------------------------
    // Online softmax (max-first)
    // -----------------------------------------------------------------------
    float local_max = -1e9f;
    for (int t = lane; t < (tok_end - tok_start); t += WARP_SIZE)
        local_max = fmaxf(local_max, sattn[t]);
    local_max = warp_reduce_max(local_max);

    float local_sum = 0.0f;
    for (int t = lane; t < (tok_end - tok_start); t += WARP_SIZE) {
        sattn[t] = expf(sattn[t] - local_max);
        local_sum += sattn[t];
    }
    local_sum = warp_reduce_sum(local_sum);

    // Normalise
    float inv_sum = (local_sum > 1e-9f) ? 1.0f / local_sum : 0.0f;
    for (int t = lane; t < (tok_end - tok_start); t += WARP_SIZE)
        sattn[t] *= inv_sum;
    __syncwarp();

    // -----------------------------------------------------------------------
    // Weighted sum over V
    // -----------------------------------------------------------------------
    scalar_t* out_ptr = output + head_idx * head_dim;
    for (int d = lane; d < head_dim; d += WARP_SIZE) {
        float acc = 0.0f;
        for (int t = 0; t < (tok_end - tok_start); ++t) {
            const scalar_t* v_tok = v_base + t * num_kv_heads * head_dim;
            acc += sattn[t] * __half2float(v_tok[d]);
        }
        // Atomic add — multiple blocks in grid contribute to same head output
        atomicAdd(reinterpret_cast<float*>(out_ptr + d), acc);
    }
}

// ---------------------------------------------------------------------------
// Host-side launcher
// ---------------------------------------------------------------------------

/**
 * launch_tier_aware_paged_attention
 *
 * Ensures the KV blocks for the request are mapped into the contiguous
 * virtual range (via CUDA VMM), then launches the attention kernel.
 */
extern "C" cudaError_t launch_tier_aware_paged_attention(
    scalar_t*           output,
    const scalar_t*     query,
    const scalar_t*     kv_cache_base,
    const BlockMetadata* block_meta_d,
    const int32_t*      block_table_d,
    int                 context_len,
    int                 num_heads,
    int                 num_kv_heads,
    int                 head_dim,
    cudaStream_t        stream)
{
    float scale = 1.0f / sqrtf((float)head_dim);
    int num_blocks = (context_len + BLOCK_SIZE - 1) / BLOCK_SIZE;

    dim3 grid(num_heads, num_blocks, 1);
    dim3 block(WARP_SIZE, 1, 1);
    size_t smem = (head_dim + BLOCK_SIZE) * sizeof(float);

    tier_aware_paged_attention_kernel<<<grid, block, smem, stream>>>(
        output, query, kv_cache_base,
        block_meta_d, block_table_d,
        context_len, scale, num_heads, num_kv_heads, head_dim);

    return cudaGetLastError();
}

// ---------------------------------------------------------------------------
// KV cache append kernel (prefill / single token)
// ---------------------------------------------------------------------------
extern "C" __global__ void append_kv_cache_kernel(
    scalar_t* __restrict__       kv_cache_base,
    const scalar_t* __restrict__ new_keys,    // [num_heads, head_dim]
    const scalar_t* __restrict__ new_values,
    const int64_t*               block_offsets, // byte offsets per new token
    int                          num_heads,
    int                          head_dim,
    int                          token_idx)
{
    int head = blockIdx.x;
    int dim  = threadIdx.x;
    if (head >= num_heads || dim >= head_dim) return;

    int64_t off = block_offsets[token_idx] / sizeof(scalar_t);
    // Key
    kv_cache_base[off + head * head_dim + dim] = new_keys[head * head_dim + dim];
    // Value (stored right after all keys in the block)
    kv_cache_base[off + num_heads * head_dim + head * head_dim + dim] =
        new_values[head * head_dim + dim];
}
