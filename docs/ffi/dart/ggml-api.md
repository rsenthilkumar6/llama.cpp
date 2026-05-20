# ggml.h - Tensor Computation Library API

> **Header**: `ggml/include/ggml.h` (~2700+ lines)
> **Purpose**: Low-level tensor operations, automatic differentiation, computation graphs
> **FFI Priority**: MEDIUM - Needed for advanced use cases, custom operations

## Table of Contents

- [Overview](#overview)
- [Core Types](#core-types)
- [Data Types (ggml_type)](#data-types-ggml_type)
- [Tensor Operations (ggml_op)](#tensor-operations-ggml_op)
- [Context Management](#context-management)
- [Tensor Creation](#tensor-creation)
- [Tensor Access](#tensor-access)
- [Arithmetic Operations](#arithmetic-operations)
- [Activation Functions](#activation-functions)
- [Normalization](#normalization)
- [Matrix Operations](#matrix-operations)
- [Tensor Manipulation](#tensor-manipulation)
- [Convolution Operations](#convolution-operations)
- [Attention Operations](#attention-operations)
- [Pooling & Upscaling](#pooling--upscaling)
- [Sorting & Selection](#sorting--selection)
- [Computation Graphs](#computation-graphs)
- [Graph Execution](#graph-execution)
- [Quantization](#quantization)
- [Type Conversions](#type-conversions)
- [Utility Functions](#utility-functions)
- [Dart FFI Notes](#dart-ffi-notes)

---

## Overview

GGML is a tensor library that provides:
- Multi-dimensional tensor operations (up to 4D)
- Automatic differentiation (backpropagation)
- Computation graph building and execution
- Various quantized data types
- Backend abstraction (CPU, CUDA, Metal, Vulkan, etc.)

## Core Types

### ggml_tensor

The fundamental data structure:

```c
struct ggml_tensor {
    enum ggml_type type;                    // Data type
    struct ggml_backend_buffer * buffer;    // Memory buffer
    int64_t ne[GGML_MAX_DIMS];             // Number of elements per dimension
    size_t  nb[GGML_MAX_DIMS];             // Stride in bytes per dimension
    enum ggml_op op;                        // Operation type
    int32_t op_params[GGML_MAX_OP_PARAMS / sizeof(int32_t)];  // Operation parameters
    int32_t flags;                          // Tensor flags
    struct ggml_tensor * src[GGML_MAX_SRC]; // Source tensors
    struct ggml_tensor * view_src;          // Source tensor for views
    size_t view_offs;                       // Offset for views
    void * data;                            // Data pointer
    char name[GGML_MAX_NAME];               // Tensor name
    void * extra;                           // Backend-specific data
};
```

**Constants**:
- `GGML_MAX_DIMS = 4` - Maximum tensor dimensions
- `GGML_MAX_SRC = 10` - Maximum source tensors per operation
- `GGML_MAX_OP_PARAMS = 64` - Operation parameter storage
- `GGML_MAX_NAME = 64` - Maximum name length

### ggml_context

Memory pool for tensors:

```c
struct ggml_init_params {
    size_t mem_size;    // Memory pool size in bytes
    void * mem_buffer;  // Pre-allocated buffer (NULL = allocate internally)
    bool no_alloc;      // Don't allocate tensor data
};
```

### ggml_cgraph

Computation graph:

```c
struct ggml_cgraph {
    // Internal structure - use API functions to manipulate
};
```

### ggml_cplan

Compute plan for graph execution:

```c
struct ggml_cplan {
    size_t work_size;     // Work buffer size
    uint8_t * work_data;  // Work buffer
    int n_threads;
    struct ggml_threadpool * threadpool;
    ggml_abort_callback abort_callback;
    void * abort_callback_data;
    bool use_ref;         // Use reference implementations only
};
```

---

## Data Types (ggml_type)

| Value | Name | Description | Size |
|-------|------|-------------|------|
| 0 | `GGML_TYPE_F32` | 32-bit float | 4 bytes |
| 1 | `GGML_TYPE_F16` | 16-bit float (IEEE) | 2 bytes |
| 2 | `GGML_TYPE_Q4_0` | 4-bit quantized (type 0) | 0.5 bytes/element |
| 3 | `GGML_TYPE_Q4_1` | 4-bit quantized (type 1) | ~0.56 bytes/element |
| 6 | `GGML_TYPE_Q5_0` | 5-bit quantized (type 0) | 0.625 bytes/element |
| 7 | `GGML_TYPE_Q5_1` | 5-bit quantized (type 1) | ~0.69 bytes/element |
| 8 | `GGML_TYPE_Q8_0` | 8-bit quantized | 1 byte/element |
| 10 | `GGML_TYPE_Q2_K` | 2-bit K-quant | ~0.28 bytes/element |
| 11 | `GGML_TYPE_Q3_K` | 3-bit K-quant | ~0.34 bytes/element |
| 12 | `GGML_TYPE_Q4_K` | 4-bit K-quant | ~0.5 bytes/element |
| 13 | `GGML_TYPE_Q5_K` | 5-bit K-quant | ~0.62 bytes/element |
| 14 | `GGML_TYPE_Q6_K` | 6-bit K-quant | ~0.75 bytes/element |
| 16-23 | IQ types | Various IQ quantizations | Varies |
| 24 | `GGML_TYPE_I8` | 8-bit signed int | 1 byte |
| 25 | `GGML_TYPE_I16` | 16-bit signed int | 2 bytes |
| 26 | `GGML_TYPE_I32` | 32-bit signed int | 4 bytes |
| 27 | `GGML_TYPE_I64` | 64-bit signed int | 8 bytes |
| 28 | `GGML_TYPE_F64` | 64-bit float | 8 bytes |
| 30 | `GGML_TYPE_BF16` | Brain float 16 | 2 bytes |
| 34 | `GGML_TYPE_TQ1_0` | Ternary quantized | Varies |
| 35 | `GGML_TYPE_TQ2_0` | Ternary quantized | Varies |
| 39 | `GGML_TYPE_MXFP4` | MXFP4 (1 block) | Varies |
| 40 | `GGML_TYPE_NVFP4` | NVIDIA FP4 | Varies |

**Total**: 41 types (`GGML_TYPE_COUNT`)

---

## Tensor Operations (ggml_op)

### Basic Operations (1-50)

| Operation | Description |
|-----------|-------------|
| `GGML_OP_DUP` | Duplicate tensor |
| `GGML_OP_ADD` | Element-wise addition |
| `GGML_OP_ADD1` | Add 1 to each element |
| `GGML_OP_ACC` | Accumulate into view |
| `GGML_OP_SUB` | Element-wise subtraction |
| `GGML_OP_MUL` | Element-wise multiplication |
| `GGML_OP_DIV` | Element-wise division |
| `GGML_OP_SQR` | Square |
| `GGML_OP_SQRT` | Square root |
| `GGML_OP_LOG` | Natural logarithm |
| `GGML_OP_SIN` | Sine |
| `GGML_OP_COS` | Cosine |
| `GGML_OP_SUM` | Sum all elements |
| `GGML_OP_SUM_ROWS` | Sum along rows |
| `GGML_OP_CUMSUM` | Cumulative sum |
| `GGML_OP_MEAN` | Mean |
| `GGML_OP_ARGMAX` | Argmax |
| `GGML_OP_COUNT_EQUAL` | Count equal elements |
| `GGML_OP_REPEAT` | Repeat tensor |
| `GGML_OP_REPEAT_BACK` | Reverse repeat (sum) |
| `GGML_OP_CONCAT` | Concatenate |

### Normalization (51-60)

| Operation | Description |
|-----------|-------------|
| `GGML_OP_NORM` | L2 normalize |
| `GGML_OP_RMS_NORM` | RMS normalize |
| `GGML_OP_RMS_NORM_BACK` | RMS norm backward |
| `GGML_OP_GROUP_NORM` | Group normalize |
| `GGML_OP_L2_NORM` | L2 normalize |

### Matrix Operations (61-65)

| Operation | Description |
|-----------|-------------|
| `GGML_OP_MUL_MAT` | Matrix multiplication |
| `GGML_OP_MUL_MAT_ID` | Indirect matrix multiplication |
| `GGML_OP_OUT_PROD` | Outer product |

### Tensor Manipulation (66-85)

| Operation | Description |
|-----------|-------------|
| `GGML_OP_SCALE` | Scale tensor |
| `GGML_OP_SET` | Set values in view |
| `GGML_OP_CPY` | Copy tensor |
| `GGML_OP_CONT` | Make contiguous |
| `GGML_OP_RESHAPE` | Reshape |
| `GGML_OP_VIEW` | Create view |
| `GGML_OP_PERMUTE` | Permute dimensions |
| `GGML_OP_TRANSPOSE` | Transpose |
| `GGML_OP_GET_ROWS` | Get rows by index |
| `GGML_OP_GET_ROWS_BACK` | Get rows backward |
| `GGML_OP_SET_ROWS` | Set rows by index |
| `GGML_OP_DIAG` | Create diagonal |
| `GGML_OP_DIAG_MASK_INF` | Mask upper triangle with -inf |
| `GGML_OP_DIAG_MASK_ZERO` | Mask upper triangle with 0 |
| `GGML_OP_SOFT_MAX` | Softmax |
| `GGML_OP_SOFT_MAX_BACK` | Softmax backward |
| `GGML_OP_ROPE` | Rotary position embedding |
| `GGML_OP_ROPE_BACK` | RoPE backward |
| `GGML_OP_CLAMP` | Clamp values |

### Convolution (86-100)

| Operation | Description |
|-----------|-------------|
| `GGML_OP_CONV_TRANSPOSE_1D` | 1D transposed convolution |
| `GGML_OP_IM2COL` | Image to column |
| `GGML_OP_IM2COL_BACK` | IM2COL backward |
| `GGML_OP_IM2COL_3D` | 3D IM2COL |
| `GGML_OP_CONV_2D` | 2D convolution |
| `GGML_OP_CONV_3D` | 3D convolution |
| `GGML_OP_CONV_2D_DW` | 2D depthwise convolution |
| `GGML_OP_CONV_TRANSPOSE_2D` | 2D transposed convolution |
| `GGML_OP_POOL_1D` | 1D pooling |
| `GGML_OP_POOL_2D` | 2D pooling |
| `GGML_OP_POOL_2D_BACK` | 2D pooling backward |

### Advanced Operations (101-130)

| Operation | Description |
|-----------|-------------|
| `GGML_OP_UPSCALE` | Upscale |
| `GGML_OP_PAD` | Zero padding |
| `GGML_OP_PAD_REFLECT_1D` | Reflect padding |
| `GGML_OP_ROLL` | Roll elements |
| `GGML_OP_ARANGE` | Range of values |
| `GGML_OP_TIMESTEP_EMBEDDING` | Timestep embedding |
| `GGML_OP_ARGSORT` | Argsort |
| `GGML_OP_TOP_K` | Top-K elements |
| `GGML_OP_LEAKY_RELU` | Leaky ReLU |
| `GGML_OP_TRI` | Triangular matrix |
| `GGML_OP_FILL` | Fill with constant |
| `GGML_OP_FLASH_ATTN_EXT` | Extended Flash Attention |
| `GGML_OP_FLASH_ATTN_BACK` | Flash Attention backward |
| `GGML_OP_SSM_CONV` | SSM convolution (Mamba) |
| `GGML_OP_SSM_SCAN` | SSM scan (Mamba) |
| `GGML_OP_WIN_PART` | Window partition |
| `GGML_OP_WIN_UNPART` | Window unpartition |
| `GGML_OP_GET_REL_POS` | Get relative position |
| `GGML_OP_ADD_REL_POS` | Add relative position |
| `GGML_OP_RWKV_WKV6` | RWKV WKV6 |
| `GGML_OP_GATED_LINEAR_ATTN` | Gated linear attention |
| `GGML_OP_RWKV_WKV7` | RWKV WKV7 |
| `GGML_OP_SOLVE_TRI` | Solve triangular system |
| `GGML_OP_GATED_DELTA_NET` | Gated delta network |

### Custom & Loss (131-145)

| Operation | Description |
|-----------|-------------|
| `GGML_OP_UNARY` | Unary operation |
| `GGML_OP_MAP_CUSTOM1` | Custom 1-operand map |
| `GGML_OP_MAP_CUSTOM2` | Custom 2-operand map |
| `GGML_OP_MAP_CUSTOM3` | Custom 3-operand map |
| `GGML_OP_CUSTOM` | Custom operation |
| `GGML_OP_CROSS_ENTROPY_LOSS` | Cross-entropy loss |
| `GGML_OP_CROSS_ENTROPY_LOSS_BACK` | Cross-entropy backward |
| `GGML_OP_OPT_STEP_ADAMW` | AdamW optimizer step |
| `GGML_OP_OPT_STEP_SGD` | SGD optimizer step |
| `GGML_OP_GLU` | Gated Linear Unit |

### Unary Operations (ggml_unary_op)

| Value | Name | Description |
|-------|------|-------------|
| 0 | `GGML_UNARY_OP_ABS` | Absolute value |
| 1 | `GGML_UNARY_OP_SGN` | Sign |
| 2 | `GGML_UNARY_OP_NEG` | Negation |
| 3 | `GGML_UNARY_OP_STEP` | Step function |
| 4 | `GGML_UNARY_OP_TANH` | Hyperbolic tangent |
| 5 | `GGML_UNARY_OP_ELU` | ELU |
| 6 | `GGML_UNARY_OP_RELU` | ReLU |
| 7 | `GGML_UNARY_OP_SIGMOID` | Sigmoid |
| 8 | `GGML_UNARY_OP_GELU` | GELU |
| 9 | `GGML_UNARY_OP_GELU_QUICK` | Quick GELU |
| 10 | `GGML_UNARY_OP_SILU` | SiLU/Swish |
| 11 | `GGML_UNARY_OP_HARDSWISH` | Hard Swish |
| 12 | `GGML_UNARY_OP_HARDSIGMOID` | Hard Sigmoid |
| 13 | `GGML_UNARY_OP_EXP` | Exponential |
| 14 | `GGML_UNARY_OP_EXPM1` | exp(x) - 1 |
| 15 | `GGML_UNARY_OP_SOFTPLUS` | Softplus |
| 16 | `GGML_UNARY_OP_GELU_ERF` | GELU with erf |
| 17 | `GGML_UNARY_OP_XIELU` | xIELU |
| 18 | `GGML_UNARY_OP_FLOOR` | Floor |
| 19 | `GGML_UNARY_OP_CEIL` | Ceiling |
| 20 | `GGML_UNARY_OP_ROUND` | Round |
| 21 | `GGML_UNARY_OP_TRUNC` | Truncate |

### GLU Operations (ggml_glu_op)

| Value | Name | Description |
|-------|------|-------------|
| 0 | `GGML_GLU_OP_REGLU` | ReLU GLU |
| 1 | `GGML_GLU_OP_GEGLU` | GELU GLU |
| 2 | `GGML_GLU_OP_SWIGLU` | SiLU GLU |
| 3 | `GGML_GLU_OP_SWIGLU_OAI` | OpenAI SiLU GLU |
| 4 | `GGML_GLU_OP_GEGLU_ERF` | GELU-erf GLU |
| 5 | `GGML_GLU_OP_GEGLU_QUICK` | Quick GELU GLU |

### Tensor Flags

| Bit | Flag | Description |
|-----|------|-------------|
| 0 | `GGML_TENSOR_FLAG_INPUT` | Input tensor |
| 1 | `GGML_TENSOR_FLAG_OUTPUT` | Output tensor |
| 2 | `GGML_TENSOR_FLAG_PARAM` | Trainable parameter |
| 3 | `GGML_TENSOR_FLAG_LOSS` | Loss tensor |
| 4 | `GGML_TENSOR_FLAG_COMPUTE` | Must be computed |

---

## Context Management

```c
GGML_API struct ggml_context * ggml_init(struct ggml_init_params params);
GGML_API void ggml_reset(struct ggml_context * ctx);
GGML_API void ggml_free(struct ggml_context * ctx);
GGML_API size_t ggml_used_mem(const struct ggml_context * ctx);
GGML_API bool ggml_get_no_alloc(struct ggml_context * ctx);
GGML_API void ggml_set_no_alloc(struct ggml_context * ctx, bool no_alloc);
GGML_API void * ggml_get_mem_buffer(const struct ggml_context * ctx);
GGML_API size_t ggml_get_mem_size(const struct ggml_context * ctx);
GGML_API size_t ggml_get_max_tensor_size(const struct ggml_context * ctx);
```

---

## Tensor Creation

### New Tensors

```c
GGML_API struct ggml_tensor * ggml_new_tensor(
    struct ggml_context * ctx,
    enum ggml_type type,
    int n_dims,
    const int64_t * ne);

GGML_API struct ggml_tensor * ggml_new_tensor_1d(
    struct ggml_context * ctx,
    enum ggml_type type,
    int64_t ne0);

GGML_API struct ggml_tensor * ggml_new_tensor_2d(
    struct ggml_context * ctx,
    enum ggml_type type,
    int64_t ne0,
    int64_t ne1);

GGML_API struct ggml_tensor * ggml_new_tensor_3d(
    struct ggml_context * ctx,
    enum ggml_type type,
    int64_t ne0,
    int64_t ne1,
    int64_t ne2);

GGML_API struct ggml_tensor * ggml_new_tensor_4d(
    struct ggml_context * ctx,
    enum ggml_type type,
    int64_t ne0,
    int64_t ne1,
    int64_t ne2,
    int64_t ne3);
```

### View and Duplicate

```c
GGML_API struct ggml_tensor * ggml_dup_tensor(struct ggml_context * ctx, const struct ggml_tensor * src);
GGML_API struct ggml_tensor * ggml_view_tensor(struct ggml_context * ctx, struct ggml_tensor * src);
GGML_API struct ggml_tensor * ggml_new_buffer(struct ggml_context * ctx, size_t nbytes);
```

### Context Tensor Lookup

```c
GGML_API struct ggml_tensor * ggml_get_first_tensor(const struct ggml_context * ctx);
GGML_API struct ggml_tensor * ggml_get_next_tensor(const struct ggml_context * ctx, struct ggml_tensor * tensor);
GGML_API struct ggml_tensor * ggml_get_tensor(struct ggml_context * ctx, const char * name);
```

---

## Tensor Access

### Size and Shape

```c
GGML_API int64_t ggml_nelements(const struct ggml_tensor * tensor);
GGML_API int64_t ggml_nrows(const struct ggml_tensor * tensor);
GGML_API size_t ggml_nbytes(const struct ggml_tensor * tensor);
GGML_API size_t ggml_nbytes_pad(const struct ggml_tensor * tensor);  // Padded to alignment
GGML_API int64_t ggml_blck_size(enum ggml_type type);
GGML_API size_t ggml_type_size(enum ggml_type type);
GGML_API size_t ggml_row_size(enum ggml_type type, int64_t ne);
GGML_API size_t ggml_element_size(const struct ggml_tensor * tensor);
GGML_API size_t ggml_tensor_overhead(void);
GGML_API int ggml_n_dims(const struct ggml_tensor * tensor);
```

### Data Access

```c
GGML_API void * ggml_get_data(const struct ggml_tensor * tensor);
GGML_API float * ggml_get_data_f32(const struct ggml_tensor * tensor);
```

### 1D Access

```c
GGML_API int32_t ggml_get_i32_1d(const struct ggml_tensor * tensor, int i);
GGML_API void ggml_set_i32_1d(const struct ggml_tensor * tensor, int i, int32_t value);
GGML_API float ggml_get_f32_1d(const struct ggml_tensor * tensor, int i);
GGML_API void ggml_set_f32_1d(const struct ggml_tensor * tensor, int i, float value);
```

### ND Access

```c
GGML_API int32_t ggml_get_i32_nd(const struct ggml_tensor * tensor, int i0, int i1, int i2, int i3);
GGML_API void ggml_set_i32_nd(const struct ggml_tensor * tensor, int i0, int i1, int i2, int i3, int32_t value);
GGML_API float ggml_get_f32_nd(const struct ggml_tensor * tensor, int i0, int i1, int i2, int i3);
GGML_API void ggml_set_f32_nd(const struct ggml_tensor * tensor, int i0, int i1, int i2, int i3, float value);
```

### Scalar Tensors

```c
GGML_API struct ggml_tensor * ggml_new_i32(struct ggml_context * ctx, int32_t value);
GGML_API struct ggml_tensor * ggml_new_f32(struct ggml_context * ctx, float value);
GGML_API struct ggml_tensor * ggml_set_i32(struct ggml_tensor * tensor, int32_t value);
GGML_API struct ggml_tensor * ggml_set_f32(struct ggml_tensor * tensor, float value);
```

### Name

```c
GGML_API const char * ggml_get_name(const struct ggml_tensor * tensor);
GGML_API struct ggml_tensor * ggml_set_name(struct ggml_tensor * tensor, const char * name);
GGML_API struct ggml_tensor * ggml_format_name(struct ggml_tensor * tensor, const char * fmt, ...);
```

### Tensor Properties

```c
GGML_API bool ggml_is_quantized(enum ggml_type type);
GGML_API bool ggml_is_transposed(const struct ggml_tensor * tensor);
GGML_API bool ggml_is_permuted(const struct ggml_tensor * tensor);
GGML_API bool ggml_is_empty(const struct ggml_tensor * tensor);
GGML_API bool ggml_is_view(const struct ggml_tensor * tensor);
GGML_API bool ggml_is_scalar(const struct ggml_tensor * tensor);
GGML_API bool ggml_is_vector(const struct ggml_tensor * tensor);
GGML_API bool ggml_is_matrix(const struct ggml_tensor * tensor);
GGML_API bool ggml_is_3d(const struct ggml_tensor * tensor);
GGML_API bool ggml_is_contiguous(const struct ggml_tensor * tensor);
GGML_API bool ggml_is_contiguous_0(const struct ggml_tensor * tensor);
GGML_API bool ggml_is_contiguous_1(const struct ggml_tensor * tensor);
GGML_API bool ggml_is_contiguous_2(const struct ggml_tensor * tensor);
GGML_API bool ggml_is_contiguously_allocated(const struct ggml_tensor * tensor);
GGML_API bool ggml_is_contiguous_channels(const struct ggml_tensor * tensor);
GGML_API bool ggml_is_contiguous_rows(const struct ggml_tensor * tensor);
GGML_API bool ggml_are_same_shape(const struct ggml_tensor * t0, const struct ggml_tensor * t1);
GGML_API bool ggml_are_same_stride(const struct ggml_tensor * t0, const struct ggml_tensor * t1);
GGML_API bool ggml_can_repeat(const struct ggml_tensor * t0, const struct ggml_tensor * t1);
```

### Tensor Flags

```c
GGML_API void ggml_set_input(struct ggml_tensor * tensor);
GGML_API void ggml_set_output(struct ggml_tensor * tensor);
GGML_API void ggml_set_param(struct ggml_tensor * tensor);
GGML_API void ggml_set_loss(struct ggml_tensor * tensor);
```

---

## Arithmetic Operations

### Binary Operations

Each operation has a regular and in-place variant:

```c
// Addition
GGML_API struct ggml_tensor * ggml_add(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b);
GGML_API struct ggml_tensor * ggml_add_inplace(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b);
GGML_API struct ggml_tensor * ggml_add_cast(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b, enum ggml_type type);
GGML_API struct ggml_tensor * ggml_add1(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b);

// Subtraction
GGML_API struct ggml_tensor * ggml_sub(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b);
GGML_API struct ggml_tensor * ggml_sub_inplace(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b);

// Multiplication
GGML_API struct ggml_tensor * ggml_mul(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b);
GGML_API struct ggml_tensor * ggml_mul_inplace(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b);

// Division
GGML_API struct ggml_tensor * ggml_div(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b);
GGML_API struct ggml_tensor * ggml_div_inplace(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b);

// Scale
GGML_API struct ggml_tensor * ggml_scale(struct ggml_context * ctx, struct ggml_tensor * a, float s);
GGML_API struct ggml_tensor * ggml_scale_inplace(struct ggml_context * ctx, struct ggml_tensor * a, float s);
GGML_API struct ggml_tensor * ggml_scale_bias(struct ggml_context * ctx, struct ggml_tensor * a, float s, float b);

// Square, Square Root
GGML_API struct ggml_tensor * ggml_sqr(struct ggml_context * ctx, struct ggml_tensor * a);
GGML_API struct ggml_tensor * ggml_sqr_inplace(struct ggml_context * ctx, struct ggml_tensor * a);
GGML_API struct ggml_tensor * ggml_sqrt(struct ggml_context * ctx, struct ggml_tensor * a);
GGML_API struct ggml_tensor * ggml_sqrt_inplace(struct ggml_context * ctx, struct ggml_tensor * a);

// Logarithm
GGML_API struct ggml_tensor * ggml_log(struct ggml_context * ctx, struct ggml_tensor * a);
GGML_API struct ggml_tensor * ggml_log_inplace(struct ggml_context * ctx, struct ggml_tensor * a);
```

### Reduction Operations

```c
GGML_API struct ggml_tensor * ggml_sum(struct ggml_context * ctx, struct ggml_tensor * a);
GGML_API struct ggml_tensor * ggml_sum_rows(struct ggml_context * ctx, struct ggml_tensor * a);
GGML_API struct ggml_tensor * ggml_cumsum(struct ggml_context * ctx, struct ggml_tensor * a);
GGML_API struct ggml_tensor * ggml_mean(struct ggml_context * ctx, struct ggml_tensor * a);
GGML_API struct ggml_tensor * ggml_argmax(struct ggml_context * ctx, struct ggml_tensor * a);
GGML_API struct ggml_tensor * ggml_count_equal(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b);
```

---

## Activation Functions

```c
// ReLU family
GGML_API struct ggml_tensor * ggml_relu(struct ggml_context * ctx, struct ggml_tensor * a);
GGML_API struct ggml_tensor * ggml_leaky_relu(struct ggml_context * ctx, struct ggml_tensor * a, float negative_slope, bool inplace);

// GELU family
GGML_API struct ggml_tensor * ggml_gelu(struct ggml_context * ctx, struct ggml_tensor * a);
GGML_API struct ggml_tensor * ggml_gelu_erf(struct ggml_context * ctx, struct ggml_tensor * a);
GGML_API struct ggml_tensor * ggml_gelu_quick(struct ggml_context * ctx, struct ggml_tensor * a);

// SiLU/Swish
GGML_API struct ggml_tensor * ggml_silu(struct ggml_context * ctx, struct ggml_tensor * a);
GGML_API struct ggml_tensor * ggml_silu_back(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b);

// Sigmoid, Tanh
GGML_API struct ggml_tensor * ggml_sigmoid(struct ggml_context * ctx, struct ggml_tensor * a);
GGML_API struct ggml_tensor * ggml_tanh(struct ggml_context * ctx, struct ggml_tensor * a);

// ELU
GGML_API struct ggml_tensor * ggml_elu(struct ggml_context * ctx, struct ggml_tensor * a);

// Hard variants
GGML_API struct ggml_tensor * ggml_hardswish(struct ggml_context * ctx, struct ggml_tensor * a);
GGML_API struct ggml_tensor * ggml_hardsigmoid(struct ggml_context * ctx, struct ggml_tensor * a);

// Exponential
GGML_API struct ggml_tensor * ggml_exp(struct ggml_context * ctx, struct ggml_tensor * a);
GGML_API struct ggml_tensor * ggml_expm1(struct ggml_context * ctx, struct ggml_tensor * a);
GGML_API struct ggml_tensor * ggml_softplus(struct ggml_context * ctx, struct ggml_tensor * a);

// Trigonometric
GGML_API struct ggml_tensor * ggml_sin(struct ggml_context * ctx, struct ggml_tensor * a);
GGML_API struct ggml_tensor * ggml_cos(struct ggml_context * ctx, struct ggml_tensor * a);

// Other
GGML_API struct ggml_tensor * ggml_abs(struct ggml_context * ctx, struct ggml_tensor * a);
GGML_API struct ggml_tensor * ggml_sgn(struct ggml_context * ctx, struct ggml_tensor * a);
GGML_API struct ggml_tensor * ggml_neg(struct ggml_context * ctx, struct ggml_tensor * a);
GGML_API struct ggml_tensor * ggml_step(struct ggml_context * ctx, struct ggml_tensor * a);
GGML_API struct ggml_tensor * ggml_floor(struct ggml_context * ctx, struct ggml_tensor * a);
GGML_API struct ggml_tensor * ggml_ceil(struct ggml_context * ctx, struct ggml_tensor * a);
GGML_API struct ggml_tensor * ggml_round(struct ggml_context * ctx, struct ggml_tensor * a);
GGML_API struct ggml_tensor * ggml_trunc(struct ggml_context * ctx, struct ggml_tensor * a);
GGML_API struct ggml_tensor * ggml_xielu(struct ggml_context * ctx, struct ggml_tensor * a, float alpha_n, float alpha_p, float beta, float eps);
```

### GLU Operations

```c
GGML_API struct ggml_tensor * ggml_glu(struct ggml_context * ctx, struct ggml_tensor * a, enum ggml_glu_op op, bool swapped);
GGML_API struct ggml_tensor * ggml_reglu(struct ggml_context * ctx, struct ggml_tensor * a);
GGML_API struct ggml_tensor * ggml_geglu(struct ggml_context * ctx, struct ggml_tensor * a);
GGML_API struct ggml_tensor * ggml_swiglu(struct ggml_context * ctx, struct ggml_tensor * a);
GGML_API struct ggml_tensor * ggml_geglu_erf(struct ggml_context * ctx, struct ggml_tensor * a);
GGML_API struct ggml_tensor * ggml_geglu_quick(struct ggml_context * ctx, struct ggml_tensor * a);
GGML_API struct ggml_tensor * ggml_swiglu_oai(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b, float alpha, float limit);
```

---

## Normalization

```c
GGML_API struct ggml_tensor * ggml_norm(struct ggml_context * ctx, struct ggml_tensor * a, float eps);
GGML_API struct ggml_tensor * ggml_rms_norm(struct ggml_context * ctx, struct ggml_tensor * a, float eps);
GGML_API struct ggml_tensor * ggml_group_norm(struct ggml_context * ctx, struct ggml_tensor * a, int n_groups, float eps);
GGML_API struct ggml_tensor * ggml_l2_norm(struct ggml_context * ctx, struct ggml_tensor * a, float eps);
GGML_API struct ggml_tensor * ggml_rms_norm_back(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b, float eps);
```

---

## Matrix Operations

### Matrix Multiplication

```c
GGML_API struct ggml_tensor * ggml_mul_mat(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b);
GGML_API void ggml_mul_mat_set_prec(struct ggml_tensor * a, enum ggml_prec prec);
GGML_API struct ggml_tensor * ggml_mul_mat_id(struct ggml_context * ctx, struct ggml_tensor * as, struct ggml_tensor * b, struct ggml_tensor * ids);
GGML_API struct ggml_tensor * ggml_out_prod(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b);
```

---

## Tensor Manipulation

### Reshape

```c
GGML_API struct ggml_tensor * ggml_reshape(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b);
GGML_API struct ggml_tensor * ggml_reshape_1d(struct ggml_context * ctx, struct ggml_tensor * a, int64_t ne0);
GGML_API struct ggml_tensor * ggml_reshape_2d(struct ggml_context * ctx, struct ggml_tensor * a, int64_t ne0, int64_t ne1);
GGML_API struct ggml_tensor * ggml_reshape_3d(struct ggml_context * ctx, struct ggml_tensor * a, int64_t ne0, int64_t ne1, int64_t ne2);
GGML_API struct ggml_tensor * ggml_reshape_4d(struct ggml_context * ctx, struct ggml_tensor * a, int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3);
```

### View

```c
GGML_API struct ggml_tensor * ggml_view_1d(struct ggml_context * ctx, struct ggml_tensor * a, int64_t ne0, size_t offset);
GGML_API struct ggml_tensor * ggml_view_2d(struct ggml_context * ctx, struct ggml_tensor * a, int64_t ne0, int64_t ne1, size_t nb1, size_t offset);
GGML_API struct ggml_tensor * ggml_view_3d(struct ggml_context * ctx, struct ggml_tensor * a, int64_t ne0, int64_t ne1, int64_t ne2, size_t nb1, size_t nb2, size_t offset);
GGML_API struct ggml_tensor * ggml_view_4d(struct ggml_context * ctx, struct ggml_tensor * a, int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3, size_t nb1, size_t nb2, size_t nb3, size_t offset);
```

### Permute & Transpose

```c
GGML_API struct ggml_tensor * ggml_permute(struct ggml_context * ctx, struct ggml_tensor * a, int axis0, int axis1, int axis2, int axis3);
GGML_API struct ggml_tensor * ggml_transpose(struct ggml_context * ctx, struct ggml_tensor * a);
```

### Copy & Set

```c
GGML_API struct ggml_tensor * ggml_cpy(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b);
GGML_API struct ggml_tensor * ggml_cast(struct ggml_context * ctx, struct ggml_tensor * a, enum ggml_type type);
GGML_API struct ggml_tensor * ggml_cont(struct ggml_context * ctx, struct ggml_tensor * a);
GGML_API struct ggml_tensor * ggml_cont_1d(struct ggml_context * ctx, struct ggml_tensor * a, int64_t ne0);
GGML_API struct ggml_tensor * ggml_set(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b, size_t nb1, size_t nb2, size_t nb3, size_t offset);
GGML_API struct ggml_tensor * ggml_set_1d(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b, size_t offset);
GGML_API struct ggml_tensor * ggml_set_2d(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b, size_t nb1, size_t offset);
```

### Repeat & Concat

```c
GGML_API struct ggml_tensor * ggml_repeat(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b);
GGML_API struct ggml_tensor * ggml_repeat_4d(struct ggml_context * ctx, struct ggml_tensor * a, int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3);
GGML_API struct ggml_tensor * ggml_repeat_back(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b);
GGML_API struct ggml_tensor * ggml_concat(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b, int dim);
```

### Rows

```c
GGML_API struct ggml_tensor * ggml_get_rows(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b);
GGML_API struct ggml_tensor * ggml_get_rows_back(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b, struct ggml_tensor * c);
GGML_API struct ggml_tensor * ggml_set_rows(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b, struct ggml_tensor * c);
GGML_API struct ggml_tensor * ggml_diag(struct ggml_context * ctx, struct ggml_tensor * a);
```

### Softmax & Masking

```c
GGML_API struct ggml_tensor * ggml_soft_max(struct ggml_context * ctx, struct ggml_tensor * a);
GGML_API struct ggml_tensor * ggml_soft_max_ext(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * mask, float scale, float max_bias);
GGML_API struct ggml_tensor * ggml_diag_mask_inf(struct ggml_context * ctx, struct ggml_tensor * a, int n_past);
GGML_API struct ggml_tensor * ggml_diag_mask_zero(struct ggml_context * ctx, struct ggml_tensor * a, int n_past);
```

### RoPE (Rotary Position Embedding)

```c
GGML_API struct ggml_tensor * ggml_rope(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b, int n_dims, int mode);
GGML_API struct ggml_tensor * ggml_rope_ext(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b, struct ggml_tensor * c, int n_dims, int mode, int n_ctx_orig, float freq_base, float freq_scale, float ext_factor, float attn_factor, float beta_fast, float beta_slow);
GGML_API struct ggml_tensor * ggml_rope_multi(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b, struct ggml_tensor * c, int n_dims, int sections[GGML_MROPE_SECTIONS], int mode, int n_ctx_orig, float freq_base, float freq_scale, float ext_factor, float attn_factor, float beta_fast, float beta_slow);
```

### Padding & Fill

```c
GGML_API struct ggml_tensor * ggml_pad(struct ggml_context * ctx, struct ggml_tensor * a, int p0, int p1, int p2, int p3);
GGML_API struct ggml_tensor * ggml_pad_ext(struct ggml_context * ctx, struct ggml_tensor * a, int lp0, int rp0, int lp1, int rp1, int lp2, int rp2, int lp3, int rp3);
GGML_API struct ggml_tensor * ggml_fill(struct ggml_context * ctx, struct ggml_tensor * a, float c);
GGML_API struct ggml_tensor * ggml_set_zero(struct ggml_tensor * tensor);
GGML_API struct ggml_tensor * ggml_clamp(struct ggml_context * ctx, struct ggml_tensor * a, float min, float max);
```

---

## Convolution Operations

```c
GGML_API struct ggml_tensor * ggml_conv_1d(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b, int s0, int p0, int d0);
GGML_API struct ggml_tensor * ggml_conv_1d_ph(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b, int s, int d);  // Half padding
GGML_API struct ggml_tensor * ggml_conv_1d_dw(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b, int s0, int p0, int d0);
GGML_API struct ggml_tensor * ggml_conv_transpose_1d(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b, int s0, int p0, int d0);

GGML_API struct ggml_tensor * ggml_conv_2d(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b, int s0, int s1, int p0, int p1, int d0, int d1);
GGML_API struct ggml_tensor * ggml_conv_2d_dw(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b, int s0, int s1, int p0, int p1, int d0, int d1);
GGML_API struct ggml_tensor * ggml_conv_2d_dw_direct(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b, int stride0, int stride1, int pad0, int pad1, int dilation0, int dilation1);
GGML_API struct ggml_tensor * ggml_conv_2d_direct(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b, int s0, int s1, int p0, int p1, int d0, int d1);
GGML_API struct ggml_tensor * ggml_conv_transpose_2d_p0(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b, int stride);

GGML_API struct ggml_tensor * ggml_conv_3d(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b, int64_t IC, int s0, int s1, int s2, int p0, int p1, int p2, int d0, int d1, int d2);
GGML_API struct ggml_tensor * ggml_conv_3d_direct(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b, int s0, int s1, int s2, int p0, int p1, int p2, int d0, int d1, int d2, int n_channels, int n_batch, int n_channels_out);

GGML_API struct ggml_tensor * ggml_im2col(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b, int s0, int s1, int p0, int p1, int d0, int d1, bool is_2D, enum ggml_type dst_type);
```

---

## Attention Operations

### Flash Attention

```c
GGML_API struct ggml_tensor * ggml_flash_attn_ext(
    struct ggml_context * ctx,
    struct ggml_tensor * q,
    struct ggml_tensor * k,
    struct ggml_tensor * v,
    struct ggml_tensor * mask,
    float scale,
    float max_bias,
    float logit_softcap);

GGML_API void ggml_flash_attn_ext_set_prec(struct ggml_tensor * a, enum ggml_prec prec);
GGML_API enum ggml_prec ggml_flash_attn_ext_get_prec(const struct ggml_tensor * a);
GGML_API void ggml_flash_attn_ext_add_sinks(struct ggml_tensor * a, struct ggml_tensor * sinks);
```

### SSM (State Space Models - Mamba)

```c
GGML_API struct ggml_tensor * ggml_ssm_conv(struct ggml_context * ctx, struct ggml_tensor * sx, struct ggml_tensor * c);
GGML_API struct ggml_tensor * ggml_ssm_scan(struct ggml_context * ctx, struct ggml_tensor * s, struct ggml_tensor * x, struct ggml_tensor * dt, struct ggml_tensor * A, struct ggml_tensor * B, struct ggml_tensor * C, struct ggml_tensor * ids);
```

### RWKV

```c
GGML_API struct ggml_tensor * ggml_rwkv_wkv6(struct ggml_context * ctx, struct ggml_tensor * k, struct ggml_tensor * v, struct ggml_tensor * r, struct ggml_tensor * tf, struct ggml_tensor * td, struct ggml_tensor * state);
GGML_API struct ggml_tensor * ggml_rwkv_wkv7(struct ggml_context * ctx, struct ggml_tensor * r, struct ggml_tensor * w, struct ggml_tensor * k, struct ggml_tensor * v, struct ggml_tensor * a, struct ggml_tensor * b, struct ggml_tensor * state);
GGML_API struct ggml_tensor * ggml_gated_linear_attn(struct ggml_context * ctx, struct ggml_tensor * k, struct ggml_tensor * v, struct ggml_tensor * q, struct ggml_tensor * g, struct ggml_tensor * state, float scale);
GGML_API struct ggml_tensor * ggml_gated_delta_net(struct ggml_context * ctx, struct ggml_tensor * q, struct ggml_tensor * k, struct ggml_tensor * v, struct ggml_tensor * g, struct ggml_tensor * beta, struct ggml_tensor * state);
```

---

## Pooling & Upscaling

```c
GGML_API struct ggml_tensor * ggml_pool_1d(struct ggml_context * ctx, struct ggml_tensor * a, enum ggml_op_pool op, int k0, int s0, int p0);
GGML_API struct ggml_tensor * ggml_pool_2d(struct ggml_context * ctx, struct ggml_tensor * a, enum ggml_op_pool op, int k0, int k1, int s0, int s1, float p0, float p1);

GGML_API struct ggml_tensor * ggml_upscale(struct ggml_context * ctx, struct ggml_tensor * a, int scale_factor, enum ggml_scale_mode mode);
GGML_API struct ggml_tensor * ggml_interpolate(struct ggml_context * ctx, struct ggml_tensor * a, int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3, uint32_t mode);

GGML_API struct ggml_tensor * ggml_win_part(struct ggml_context * ctx, struct ggml_tensor * a, int w);
GGML_API struct ggml_tensor * ggml_win_unpart(struct ggml_context * ctx, struct ggml_tensor * a, int w0, int h0, int w);
```

**Scale modes**:
| Value | Name |
|-------|------|
| 0 | `GGML_SCALE_MODE_NEAREST` |
| 1 | `GGML_SCALE_MODE_BILINEAR` |
| 2 | `GGML_SCALE_MODE_BICUBIC` |

**Pool operations**:
| Value | Name |
|-------|------|
| 0 | `GGML_OP_POOL_MAX` |
| 1 | `GGML_OP_POOL_AVG` |

---

## Sorting & Selection

```c
GGML_API struct ggml_tensor * ggml_argsort(struct ggml_context * ctx, struct ggml_tensor * a, enum ggml_sort_order order);
GGML_API struct ggml_tensor * ggml_argsort_top_k(struct ggml_context * ctx, struct ggml_tensor * a, int k);
GGML_API struct ggml_tensor * ggml_top_k(struct ggml_context * ctx, struct ggml_tensor * a, int k);
GGML_API struct ggml_tensor * ggml_arange(struct ggml_context * ctx, float start, float stop, float step);
```

**Sort order**:
| Value | Name |
|-------|------|
| 0 | `GGML_SORT_ORDER_ASC` |
| 1 | `GGML_SORT_ORDER_DESC` |

---

## Computation Graphs

```c
GGML_API struct ggml_cgraph * ggml_new_graph(struct ggml_context * ctx);
GGML_API struct ggml_cgraph * ggml_new_graph_custom(struct ggml_context * ctx, size_t size, bool grads);
GGML_API struct ggml_cgraph * ggml_graph_dup(struct ggml_context * ctx, struct ggml_cgraph * cgraph, bool force_grads);
GGML_API void ggml_graph_cpy(struct ggml_cgraph * src, struct ggml_cgraph * dst);
GGML_API void ggml_graph_reset(struct ggml_cgraph * cgraph);
GGML_API void ggml_graph_clear(struct ggml_cgraph * cgraph);

GGML_API int ggml_graph_size(struct ggml_cgraph * cgraph);
GGML_API struct ggml_tensor * ggml_graph_node(struct ggml_cgraph * cgraph, int i);
GGML_API struct ggml_tensor ** ggml_graph_nodes(struct ggml_cgraph * cgraph);
GGML_API int ggml_graph_n_nodes(struct ggml_cgraph * cgraph);
GGML_API void ggml_graph_add_node(struct ggml_cgraph * cgraph, struct ggml_tensor * tensor);

GGML_API size_t ggml_graph_overhead(void);
GGML_API size_t ggml_graph_overhead_custom(size_t size, bool grads);

GGML_API struct ggml_tensor * ggml_graph_get_tensor(const struct ggml_cgraph * cgraph, const char * name);
GGML_API struct ggml_tensor * ggml_graph_get_grad(const struct ggml_cgraph * cgraph, const struct ggml_tensor * node);
GGML_API struct ggml_tensor * ggml_graph_get_grad_acc(const struct ggml_cgraph * cgraph, const struct ggml_tensor * node);

GGML_API void ggml_graph_print(const struct ggml_cgraph * cgraph);
GGML_API void ggml_graph_dump_dot(const struct ggml_cgraph * gb, const struct ggml_cgraph * cgraph, const char * filename);

GGML_API struct ggml_tensor * ggml_build_forward_select(struct ggml_cgraph * cgraph, struct ggml_tensor ** tensors, int n_tensors, int idx);
GGML_API void ggml_build_forward_expand(struct ggml_cgraph * cgraph, struct ggml_tensor * tensor);
GGML_API void ggml_build_backward_expand(struct ggml_context * ctx, struct ggml_cgraph * cgraph, struct ggml_tensor ** grad_accs);
```

---

## Graph Execution

```c
GGML_API struct ggml_cplan ggml_graph_plan(
    const struct ggml_cgraph * cgraph,
    int n_threads,
    struct ggml_threadpool * threadpool);

GGML_API enum ggml_status ggml_graph_compute(struct ggml_cgraph * cgraph, struct ggml_cplan * cplan);
GGML_API enum ggml_status ggml_graph_compute_with_ctx(struct ggml_context * ctx, struct ggml_cgraph * cgraph, int n_threads);
```

**Status codes**:
| Value | Name | Description |
|-------|------|-------------|
| -2 | `GGML_STATUS_ALLOC_FAILED` | Memory allocation failed |
| -1 | `GGML_STATUS_FAILED` | General failure |
| 0 | `GGML_STATUS_SUCCESS` | Success |
| 1 | `GGML_STATUS_ABORTED` | Computation aborted |

---

## Quantization

```c
GGML_API void ggml_quantize_init(enum ggml_type type);
GGML_API void ggml_quantize_free(void);
GGML_API bool ggml_quantize_requires_imatrix(enum ggml_type type);
GGML_API size_t ggml_quantize_chunk(enum ggml_type type, const float * src, void * dst, int64_t start, int64_t nrows, int64_t n_per_row, const float * imatrix);
```

---

## Type Conversions

### FP16/BF16 Conversion

```c
GGML_API float ggml_fp16_to_fp32(ggml_fp16_t);
GGML_API ggml_fp16_t ggml_fp32_to_fp16(float);
GGML_API void ggml_fp16_to_fp32_row(const ggml_fp16_t *, float *, int64_t);
GGML_API void ggml_fp32_to_fp16_row(const float *, ggml_fp16_t *, int64_t);

GGML_API ggml_bf16_t ggml_fp32_to_bf16(ggml_bf16_t);
GGML_API float ggml_bf16_to_fp32(ggml_bf16_t);
GGML_API void ggml_bf16_to_fp32_row(const ggml_bf16_t *, float *, int64_t);
GGML_API void ggml_fp32_to_bf16_row_ref(const float *, ggml_bf16_t *, int64_t);
GGML_API void ggml_fp32_to_bf16_row(const float *, ggml_bf16_t *, int64_t);
```

### CPU Conversion Functions

```c
GGML_API void ggml_cpu_fp32_to_fp32(const float *, float *, int64_t);
GGML_API void ggml_cpu_fp32_to_i32(const float *, int32_t *, int64_t);
GGML_API void ggml_cpu_fp32_to_fp16(const float *, ggml_fp16_t *, int64_t);
GGML_API void ggml_cpu_fp16_to_fp32(const ggml_fp16_t *, float *, int64_t);
GGML_API void ggml_cpu_fp32_to_bf16(const float *, ggml_bf16_t *, int64_t);
GGML_API void ggml_cpu_bf16_to_fp32(const ggml_bf16_t *, float *, int64_t);
```

---

## Utility Functions

### Time

```c
GGML_API void ggml_time_init(void);
GGML_API int64_t ggml_time_ms(void);
GGML_API int64_t ggml_time_us(void);
GGML_API int64_t ggml_cycles(void);
GGML_API int64_t ggml_cycles_per_ms(void);
```

### Version

```c
GGML_API const char * ggml_version(void);
GGML_API const char * ggml_commit(void);
```

### File I/O

```c
GGML_API FILE * ggml_fopen(const char * fname, const char * mode);  // UTF-8 path support
```

### Logging

```c
GGML_API void ggml_log_get(ggml_log_callback * log_callback, void ** user_data);
GGML_API void ggml_log_set(ggml_log_callback log_callback, void * user_data);
```

### Abort

```c
GGML_API ggml_abort_callback_t ggml_set_abort_callback(ggml_abort_callback_t callback);
GGML_NORETURN GGML_API void ggml_abort(const char * file, int line, const char * fmt, ...);
```

### Type Info

```c
GGML_API const char * ggml_type_name(enum ggml_type type);
GGML_API const char * ggml_op_name(enum ggml_op op);
GGML_API const char * ggml_op_symbol(enum ggml_op op);
GGML_API const char * ggml_unary_op_name(enum ggml_unary_op op);
GGML_API const char * ggml_glu_op_name(enum ggml_glu_op op);
GGML_API const char * ggml_op_desc(const struct ggml_tensor * t);
GGML_API enum ggml_unary_op ggml_get_unary_op(const struct ggml_tensor * tensor);
GGML_API enum ggml_glu_op ggml_get_glu_op(const struct ggml_tensor * tensor);
GGML_API enum ggml_type ggml_ftype_to_ggml_type(enum ggml_ftype ftype);
```

### Validation

```c
GGML_API bool ggml_validate_row_data(enum ggml_type type, const void * data, size_t nbytes);
GGML_API bool ggml_guid_matches(ggml_guid_t guid_a, ggml_guid_t guid_b);
```

### Custom Operations

```c
typedef void (*ggml_custom1_op_t)(struct ggml_tensor * dst, const struct ggml_tensor * a, int ith, int nth, void * userdata);
typedef void (*ggml_custom2_op_t)(struct ggml_tensor * dst, const struct ggml_tensor * a, const struct ggml_tensor * b, int ith, int nth, void * userdata);
typedef void (*ggml_custom3_op_t)(struct ggml_tensor * dst, const struct ggml_tensor * a, const struct ggml_tensor * b, const struct ggml_tensor * c, int ith, int nth, void * userdata);
typedef void (*ggml_custom_op_t)(struct ggml_tensor * dst, int ith, int nth, void * userdata);

GGML_API struct ggml_tensor * ggml_map_custom1(struct ggml_context * ctx, struct ggml_tensor * a, ggml_custom1_op_t fun, int n_tasks, void * userdata);
GGML_API struct ggml_tensor * ggml_map_custom2(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b, ggml_custom2_op_t fun, int n_tasks, void * userdata);
GGML_API struct ggml_tensor * ggml_map_custom3(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b, struct ggml_tensor * c, ggml_custom3_op_t fun, int n_tasks, void * userdata);
GGML_API struct ggml_tensor * ggml_custom_4d(struct ggml_context * ctx, enum ggml_type type, int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3, struct ggml_tensor ** args, int n_args, ggml_custom_op_t fun, int n_tasks, void * userdata);
```

---

## Dart FFI Notes

### ggml_type Mapping

```dart
enum GgmlType {
  f32(0), f16(1), q4_0(2), q4_1(3), q5_0(6), q5_1(7), q8_0(8),
  q2K(10), q3K(11), q4K(12), q5K(13), q6K(14),
  i8(24), i16(25), i32(26), i64(27), f64(28), bf16(30),
  // ... etc
}
```

### ggml_tensor as Opaque

The `ggml_tensor` struct is complex and should generally be treated as opaque in Dart FFI. Access fields through API functions rather than direct struct access.

### Struct by Value

`ggml_init_params` and `ggml_cplan` are passed by value. In Dart, define them as `Struct` subclasses:

```dart
final class GgmlInitParams extends Struct {
  @Size() external int mem_size;
  external Pointer<Void> mem_buffer;
  @Bool() external bool no_alloc;
}
```
