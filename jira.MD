# TurboQuant: Metal (Apple Silicon) KV Cache Quantization Implementation Plan

## Overview

Implement TurboQuant KV cache quantization types (`TBQ3_0`, `TBQ4_0`) with **Apple Silicon / Metal GPU backend** support in llama.cpp. Based on:

- **Paper:** [TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate](https://arxiv.org/abs/2504.19874) (Google Research)
- **Existing CPU PR:** [#21089](https://github.com/ggml-org/llama.cpp/pull/21089) — adds `GGML_TYPE_TBQ3_0` / `GGML_TYPE_TBQ4_0` with CPU-only quantize/dequantize
- **MLX reference:** [PR #3328](https://github.com/ml-explore/mlx/pull/3328) — native Metal SDPA kernel with on-the-fly dequant (~4.9x speedup)

### What TurboQuant does

- KV cache quantization (not model weights)
- Applies **random orthogonal rotation** to unit-normalized vectors
- Per-coordinate **Lloyd-Max scalar quantization** using precomputed codebooks
- Two modes: MSE-optimal (`TBQ`) and inner-product (unbiased, `TBQP`)
- Achieves ~3.06 bpw (`TBQ3_0`) and ~4.06 bpw (`TBQ4_0`)

---

## Research Completed

### Algorithm Details

1. **Random rotation:** A global d×d random orthogonal matrix Q (generated once via SplitMix64 PRNG → Gaussian matrix → QR decomposition → Haar sign correction)
2. **Codebooks:** Precomputed Lloyd-Max-optimal scalar quantizers for Beta/normal distribution (2/3/4 bit). Stored in lookup tables.
3. **Quantize:** `v → normalize → Q·v → scalar quantize per coord → pack indices`
4. **Dequantize:** `indices → codebook lookup → Q^T · approx → rescale by norm`
5. **Block layout:** `block_tbq3_0` (98 bytes/256 elems), `block_tbq4_0` (130 bytes/256 elems). Each block stores an fp16 norm `d` + packed indices.

### CPU PR (#21089) Approach

- Full CPU quantize/dequantize in `ggml-turboq.c`
- On CPU, dequantizes K/V to f16 **before** attention (no on-the-fly dequant)
- Uses `ggml_cast` → standard flash attention on f16
- Rotation matrix: 128×128 per block-half, thread-locally cached, deterministic seed

### MLX PR (#3328) Approach — Key Reference for Metal

- **Pre-rotates queries** before attention (Q_rot = Q · Q_orthogonal)
- K stored as packed uint32 indices + separate float32 norm array
- V pre-dequantized to fp16 in separate buffer
- Custom Metal SDPA kernel reads packed 3-bit indices directly, does codebook lookup in inner loop
- ~5.3× less memory bandwidth vs fp16 → ~2-5× speedup on M4 Pro

---

## Implementation Strategy: Two-Phase

### Phase 1: Option A — Minimal Metal Support (Dequantize-then-Attend)

Follow the existing patterns in `ggml-metal.metal` for existing quant types (q4_0, q8_0, etc.). Dequantize to f32 on GPU before flash attention.

**Files to modify:**

#### 1. ggml/src/ggml-metal/ggml-metal.metal

Add Metal dequantize/quantize functions (analogous to `dequantize_q4_0`/`quantize_q4_0`):

```
// Dequantize TBQ block → float4x4
void dequantize_tbq3_0(device const block_tbq3_0 *xb, short il, thread type4x4 &reg)
void dequantize_tbq4_0(device const block_tbq4_0 *xb, short il, thread type4x4 &reg)

// Quantize float → TBQ block
void quantize_tbq3_0(device const float *src, device block_tbq3_0 &dst)
void quantize_tbq4_0(device const float *src, device block_tbq4_0 &dst)
```

The dequantize function must:
1. Read packed indices, codebook lookup → get rotated values
2. Apply inverse rotation (multiply by Q^T)
3. Rescale by block norm `d`
4. The rotation matrix Q must be generated on GPU (same SplitMix64 + QR algorithm) or read from a precomputed constant buffer

Add `[[host_name(...)]]` template instantiations for:

| Template | Instantiation Pattern |
|----------|----------------------|
| `kernel_set_rows_q32` | `kernel_set_rows_tbq3_0_i64`, `kernel_set_rows_tbq4_0_i64` |
| `kernel_cpy_f32_q` | `kernel_cpy_f32_tbq3_0`, `kernel_cpy_f32_tbq4_0` |
| `kernel_cpy_q_f32` | `kernel_cpy_tbq3_0_f32`, `kernel_cpy_tbq4_0_f32` |
| `kernel_cpy_q_f16` | `kernel_cpy_tbq3_0_f16`, `kernel_cpy_tbq4_0_f16` |
| `kernel_get_rows_q` | `kernel_get_rows_tbq3_0`, `kernel_get_rows_tbq4_0` |

**No changes** to `kernel_flash_attn_ext` — the TBQ data is dequantized to f32 before reaching attention.

#### 2. ggml/src/ggml-metal/ggml-metal-ops.cpp

The dispatch uses `ggml_type_name()` to construct pipeline names at runtime. If Metal shader has matching `[[host_name]]`, it's found automatically. No dispatch code changes needed.

#### 3. ggml/src/ggml-metal/ggml-metal-device.cpp

Pipeline name construction in `ggml_metal_library_get_pipeline_cpy` uses `ggml_type_name("tbq3_0")` → looks for `kernel_cpy_f32_tbq3_0`. This works automatically if the Metal shader provides it.

#### 4. ggml/src/ggml-turboq.c (from CPU PR)

The rotation matrix Q generation function (`turboq_generate_Q`) must be made available for both CPU and Metal. For Metal, the Q matrix will need to be uploaded to GPU as a buffer.

**Key detail:** The Q matrix (128×128 f32) is small (~64KB) and constant. It can be:
- Generated on CPU once, uploaded to Metal as a constant buffer
- Or generated on GPU using the same SplitMix64 + QR algorithm

#### 5. Validation

- `test-quantize-fns`: CPU quantize tests (already in CPU PR)
- `test-backend-ops -b METAL`: verify GET_ROWS, SET_ROWS, CPY ops for TBQ types on Metal
- `llama-bench`: benchmark KV cache throughput on Apple Silicon
- `llama-perplexity`: verify quality matches CPU results

---

### Phase 2: Option B — Optimized Metal SDPA (On-the-fly Dequant in Attention)

Implement MLX-style direct packed-index attention kernel for maximum performance.

#### Key Differences from Option A

| Aspect | Option A | Option B |
|--------|----------|----------|
| K storage | `block_tbq*_0` structs | Packed uint32 indices + separate float32 norms |
| Query rotation | In dequantize kernel | Pre-rotated before attention |
| V storage | `block_tbq*_0` structs | Pre-dequantized fp16 in separate buffer |
| Attention kernel | Standard `flash_attn_ext` (reads f32) | Custom SDPA kernel (reads packed indices) |
| Memory bandwidth | 130 bytes/256 elems decompressed to f32 first | 3-4 bits per element, read in inner loop |
| Speedup vs f16 | ~1.5-2× | ~2-5× |

#### Implementation Steps

##### 1. Custom Metal SDPA Kernel

Write `kernel_flash_attn_ext_tbq` in `ggml-metal.metal`:

```
template <int VPW, int D, int BITS>
kernel void kernel_flash_attn_ext_tbq(
    device const  float * q_rot,    // pre-rotated queries (f32)
    device const uint32 * k_packed, // packed K indices
    device const   float * k_norms, // per-vector K norms
    device const    half * v_val,   // pre-dequantized V (fp16)
    ...
)
```

Inner loop per query-vector pair:
```
float score = 0;
for each packed word in k_packed[token]:
    unpack indices
    codebook lookup → float values
    score += dot(q_rot[thread], codebook_values) * k_norms[token]
score *= inv_sqrt_dim
```

Template parameters:
- `VPW`: values per uint32 word (10 for 3-bit, 8 for 4-bit)
- `D`: head dimension (64, 128, etc.)
- `BITS`: quantization bits (3, 4)

##### 2. Pre-rotate Queries

Before attention, rotate queries by the orthogonal matrix Q:
```
Q_rot = Q · Q_query  // same Q used for K quantization
```
This can be a simple matmul kernel or use the existing `ggml_mul_mat`.

##### 3. KV Cache Storage Format Change for Metal

- For Option B, the Metal backend stores K differently than CPU:
  - K: `[n_tokens * n_kv_heads]` of packed `uint32` arrays + `float32` norms
  - V: pre-dequantized to fp16 on write, stored as `[n_embd_v, n_tokens * n_kv_heads]`
- This means CPU ↔ Metal tensor sharing for K/V would require format conversion
- **Architectural decision:** Use separate K/V storage per backend, or convert at transfer boundaries

##### 4. Alternative: Keep block_tbq*_0 format, but dequant inside attention

Instead of pre-rotated queries, keep the existing block format and implement inverse rotation inside the attention kernel:

```
// Inside attention inner loop:
read block_tbq3_0 from K cache
codebook lookup → rotated values
apply inverse rotation Q^T
dequant = rotated * norm
score = dot(q, dequant)  // standard query, no pre-rotation
```

This is simpler (no separate K/V storage formats, no query pre-rotation) but more compute inside the inner loop (inverse rotation + matrix multiply).

---

## Dependency Graph

```
CPU PR (#21089) — needed first
        │
        ▼
Option A (Dequantize-then-Attend)
   ┌────┴────┐
   │         │
Metal       CPU  (from CPU PR)
   │
   ▼
Option B (On-the-fly SDPA)
```

The CPU PR must be merged first since it provides:
- Block struct definitions (`block_tbq3_0`, `block_tbq4_0`)
- Type traits registration in `ggml.c`
- Codebook tables
- Rotation matrix generation (`turboq_generate_Q`)
- Quantize/dequantize reference implementations
- llama.cpp integration (CLI args, KV cache wiring)

---

## Testing Plan

1. **Unit:** `test-quantize-fns` — verify quantize/dequantize roundtrip matches CPU reference
2. **Backend ops:** `test-backend-ops -b METAL -o GET_ROWS,SET_ROWS,CPY -p tbq` — verify Metal kernel correctness
3. **End-to-end:** Run `llama-cli` with `--cache-type-k tbq4_0 --cache-type-v tbq4_0` — verify output matches f16 baseline
4. **Quality:** `llama-perplexity` — verify PPL matches CPU TBQ results
5. **Performance:** `llama-bench` — measure prompt/gen throughput vs f16, q8_0, q4_0 on Apple Silicon