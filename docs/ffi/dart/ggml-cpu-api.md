# ggml-cpu.h - CPU Backend API

> **Header**: `ggml/include/ggml-cpu.h` (151 lines)
> **Purpose**: CPU-specific operations, NUMA, threading, SIMD detection
> **FFI Priority**: MEDIUM - Needed for CPU configuration and SIMD detection

## Table of Contents

- [Overview](#overview)
- [Compute Plan](#compute-plan)
- [NUMA Support](#numa-support)
- [Tensor Value Access](#tensor-value-access)
- [Thread Pool](#thread-pool)
- [Graph Execution](#graph-execution)
- [CPU Feature Detection](#cpu-feature-detection)
- [CPU Backend Initialization](#cpu-backend-initialization)
- [Type Traits](#type-traits)
- [Dart FFI Notes](#dart-ffi-notes)

---

## Overview

The CPU backend provides the default implementation for all tensor operations, with optimizations for various SIMD instruction sets.

---

## Compute Plan

```c
struct ggml_cplan {
    size_t work_size;                        // Work buffer size
    uint8_t * work_data;                     // Work buffer (caller allocates)
    int n_threads;                           // Number of threads
    struct ggml_threadpool * threadpool;     // Thread pool
    ggml_abort_callback abort_callback;      // Abort callback
    void * abort_callback_data;              // Abort callback data
    bool use_ref;                            // Use reference implementations only
};
```

---

## NUMA Support

### NUMA Strategies

```c
enum ggml_numa_strategy {
    GGML_NUMA_STRATEGY_DISABLED = 0,    // No NUMA optimization
    GGML_NUMA_STRATEGY_DISTRIBUTE = 1,  // Distribute across NUMA nodes
    GGML_NUMA_STRATEGY_ISOLATE = 2,     // Isolate to specific nodes
    GGML_NUMA_STRATEGY_NUMACTL = 3,     // Use numactl
    GGML_NUMA_STRATEGY_MIRROR = 4,      // Mirror across nodes
    GGML_NUMA_STRATEGY_COUNT            // Number of strategies
};
```

### NUMA Functions

```c
GGML_BACKEND_API void ggml_numa_init(enum ggml_numa_strategy numa);
GGML_BACKEND_API bool ggml_is_numa(void);
```

Call `ggml_numa_init()` once at startup for better performance on NUMA systems.

---

## Tensor Value Access

### 1D Access

```c
GGML_BACKEND_API int32_t ggml_get_i32_1d(const struct ggml_tensor * tensor, int i);
GGML_BACKEND_API void ggml_set_i32_1d(const struct ggml_tensor * tensor, int i, int32_t value);
GGML_BACKEND_API float ggml_get_f32_1d(const struct ggml_tensor * tensor, int i);
GGML_BACKEND_API void ggml_set_f32_1d(const struct ggml_tensor * tensor, int i, float value);
```

### ND Access

```c
GGML_BACKEND_API int32_t ggml_get_i32_nd(const struct ggml_tensor * tensor, int i0, int i1, int i2, int i3);
GGML_BACKEND_API void ggml_set_i32_nd(const struct ggml_tensor * tensor, int i0, int i1, int i2, int i3, int32_t value);
GGML_BACKEND_API float ggml_get_f32_nd(const struct ggml_tensor * tensor, int i0, int i1, int i2, int i3);
GGML_BACKEND_API void ggml_set_f32_nd(const struct ggml_tensor * tensor, int i0, int i1, int i2, int i3, float value);
```

### Scalar Tensors

```c
GGML_BACKEND_API struct ggml_tensor * ggml_new_i32(struct ggml_context * ctx, int32_t value);
GGML_BACKEND_API struct ggml_tensor * ggml_new_f32(struct ggml_context * ctx, float value);
GGML_BACKEND_API struct ggml_tensor * ggml_set_i32(struct ggml_tensor * tensor, int32_t value);
GGML_BACKEND_API struct ggml_tensor * ggml_set_f32(struct ggml_tensor * tensor, float value);
```

---

## Thread Pool

```c
GGML_BACKEND_API struct ggml_threadpool * ggml_threadpool_new(struct ggml_threadpool_params * params);
GGML_BACKEND_API void ggml_threadpool_free(struct ggml_threadpool * threadpool);
GGML_BACKEND_API int ggml_threadpool_get_n_threads(struct ggml_threadpool * threadpool);
GGML_BACKEND_API void ggml_threadpool_pause(struct ggml_threadpool * threadpool);
GGML_BACKEND_API void ggml_threadpool_resume(struct ggml_threadpool * threadpool);
```

---

## Graph Execution

```c
GGML_BACKEND_API struct ggml_cplan ggml_graph_plan(
    const struct ggml_cgraph * cgraph,
    int n_threads,
    struct ggml_threadpool * threadpool);

GGML_BACKEND_API enum ggml_status ggml_graph_compute(struct ggml_cgraph * cgraph, struct ggml_cplan * cplan);
GGML_BACKEND_API enum ggml_status ggml_graph_compute_with_ctx(struct ggml_context * ctx, struct ggml_cgraph * cgraph, int n_threads);
```

---

## CPU Feature Detection

### x86 Features

```c
GGML_BACKEND_API int ggml_cpu_has_sse3        (void);
GGML_BACKEND_API int ggml_cpu_has_ssse3       (void);
GGML_BACKEND_API int ggml_cpu_has_avx         (void);
GGML_BACKEND_API int ggml_cpu_has_avx_vnni    (void);
GGML_BACKEND_API int ggml_cpu_has_avx2        (void);
GGML_BACKEND_API int ggml_cpu_has_bmi2        (void);
GGML_BACKEND_API int ggml_cpu_has_f16c        (void);
GGML_BACKEND_API int ggml_cpu_has_fma         (void);
GGML_BACKEND_API int ggml_cpu_has_avx512      (void);
GGML_BACKEND_API int ggml_cpu_has_avx512_vbmi (void);
GGML_BACKEND_API int ggml_cpu_has_avx512_vnni (void);
GGML_BACKEND_API int ggml_cpu_has_avx512_bf16 (void);
GGML_BACKEND_API int ggml_cpu_has_amx_int8    (void);
```

### ARM Features

```c
GGML_BACKEND_API int ggml_cpu_has_neon        (void);
GGML_BACKEND_API int ggml_cpu_has_arm_fma     (void);
GGML_BACKEND_API int ggml_cpu_has_fp16_va     (void);
GGML_BACKEND_API int ggml_cpu_has_dotprod     (void);
GGML_BACKEND_API int ggml_cpu_has_matmul_int8 (void);
GGML_BACKEND_API int ggml_cpu_has_sve         (void);
GGML_BACKEND_API int ggml_cpu_get_sve_cnt     (void);  // SVE vector length in bytes
GGML_BACKEND_API int ggml_cpu_has_sme         (void);
```

### Other Features

```c
GGML_BACKEND_API int ggml_cpu_has_riscv_v     (void);
GGML_BACKEND_API int ggml_cpu_get_rvv_vlen    (void);  // RISC-V vector length
GGML_BACKEND_API int ggml_cpu_has_vsx         (void);
GGML_BACKEND_API int ggml_cpu_has_vxe         (void);
GGML_BACKEND_API int ggml_cpu_has_wasm_simd   (void);
GGML_BACKEND_API int ggml_cpu_has_llamafile   (void);
```

---

## CPU Backend Initialization

```c
GGML_BACKEND_API ggml_backend_t ggml_backend_cpu_init(void);
GGML_BACKEND_API bool ggml_backend_is_cpu(ggml_backend_t backend);
GGML_BACKEND_API void ggml_backend_cpu_set_n_threads(ggml_backend_t backend_cpu, int n_threads);
GGML_BACKEND_API void ggml_backend_cpu_set_threadpool(ggml_backend_t backend_cpu, ggml_threadpool_t threadpool);
GGML_BACKEND_API void ggml_backend_cpu_set_abort_callback(ggml_backend_t backend_cpu, ggml_abort_callback abort_callback, void * abort_callback_data);
GGML_BACKEND_API void ggml_backend_cpu_set_use_ref(ggml_backend_t backend_cpu, bool use_ref);
GGML_BACKEND_API ggml_backend_reg_t ggml_backend_cpu_reg(void);
GGML_BACKEND_API void ggml_cpu_fp32_to_fp32(const float *, float *, int64_t);
GGML_BACKEND_API void ggml_cpu_fp32_to_i32(const float *, int32_t *, int64_t);
GGML_BACKEND_API void ggml_cpu_fp32_to_fp16(const float *, ggml_fp16_t *, int64_t);
GGML_BACKEND_API void ggml_cpu_fp16_to_fp32(const ggml_fp16_t *, float *, int64_t);
GGML_BACKEND_API void ggml_cpu_fp32_to_bf16(const float *, ggml_bf16_t *, int64_t);
GGML_BACKEND_API void ggml_cpu_bf16_to_fp32(const ggml_bf16_t *, float *, int64_t);
```

---

## Type Traits

```c
typedef void (*ggml_vec_dot_t)(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT x, size_t bx, const void * GGML_RESTRICT y, size_t by, int nrc);

struct ggml_type_traits_cpu {
    ggml_from_float_t from_float;
    ggml_vec_dot_t vec_dot;
    enum ggml_type vec_dot_type;
    int64_t nrows;
};

GGML_BACKEND_API const struct ggml_type_traits_cpu * ggml_get_type_traits_cpu(enum ggml_type type);
GGML_BACKEND_API void ggml_cpu_init(void);
```

---

## Dart FFI Notes

### CPU Feature Detection Pattern

```dart
// Check CPU capabilities for UI display
final cpuInfo = {
  'AVX': ggmlCpuHasAvx() == 1,
  'AVX2': ggmlCpuHasAvx2() == 1,
  'AVX512': ggmlCpuHasAvx512() == 1,
  'FMA': ggmlCpuHasFma() == 1,
  'NEON': ggmlCpuHasNeon() == 1,
  'SVE': ggmlCpuHasSve() == 1,
  'LLaMAfile': ggmlCpuHasLlamafile() == 1,
};

// Display in UI
print('CPU Features: ${cpuInfo.entries.where((e) => e.value).map((e) => e.key).join(', ')}');
```

### NUMA Initialization

```dart
// Initialize NUMA for better performance on multi-socket systems
ggmlNumaInit(GgmlNumaStrategy.distribute);
```
