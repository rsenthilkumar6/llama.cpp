# ggml-alloc.h - Memory Allocator API

> **Header**: `ggml/include/ggml-alloc.h` (85 lines)
> **Purpose**: Graph and tensor memory allocation
> **FFI Priority**: LOW - Used internally, rarely needed directly

## Table of Contents

- [Overview](#overview)
- [Tensor Allocator](#tensor-allocator)
- [Graph Allocator](#graph-allocator)
- [Context Tensor Allocation](#context-tensor-allocation)
- [Dart FFI Notes](#dart-ffi-notes)

---

## Overview

The allocator module provides efficient memory management for tensors and computation graphs, enabling memory reuse across graph nodes.

---

## Tensor Allocator

### ggml_tallocr

A simple tensor allocator that allocates from a single buffer:

```c
struct ggml_tallocr {
    ggml_backend_buffer_t buffer;  // Source buffer
    void * base;                   // Base pointer
    size_t alignment;              // Alignment requirement
    size_t offset;                 // Current offset
};
```

### Operations

```c
GGML_API struct ggml_tallocr ggml_tallocr_new(ggml_backend_buffer_t buffer);
GGML_API enum ggml_status ggml_tallocr_alloc(struct ggml_tallocr * talloc, struct ggml_tensor * tensor);
```

---

## Graph Allocator

### ggml_gallocr

A graph allocator that reuses memory across non-overlapping tensor lifetimes:

```c
typedef struct ggml_gallocr * ggml_gallocr_t;
```

### Creation

```c
GGML_API ggml_gallocr_t ggml_gallocr_new(ggml_backend_buffer_type_t buft);
GGML_API ggml_gallocr_t ggml_gallocr_new_n(ggml_backend_buffer_type_t * bufts, int n_bufs);
GGML_API void ggml_gallocr_free(ggml_gallocr_t galloc);
```

### Reserve Buffers

```c
GGML_API bool ggml_gallocr_reserve(ggml_gallocr_t galloc, struct ggml_cgraph * graph);
GGML_API void ggml_gallocr_reserve_n_size(
    ggml_gallocr_t galloc,
    struct ggml_cgraph * graph,
    const int * node_buffer_ids,
    const int * leaf_buffer_ids,
    size_t * sizes);
GGML_API bool ggml_gallocr_reserve_n(
    ggml_gallocr_t galloc,
    struct ggml_cgraph * graph,
    const int * node_buffer_ids,
    const int * leaf_buffer_ids);
```

### Allocate Graph

```c
GGML_API bool ggml_gallocr_alloc_graph(ggml_gallocr_t galloc, struct ggml_cgraph * graph);
```

### Buffer Size Query

```c
GGML_API size_t ggml_gallocr_get_buffer_size(ggml_gallocr_t galloc, int buffer_id);
```

---

## Context Tensor Allocation

Helper functions to allocate all tensors in a ggml_context:

```c
GGML_API size_t ggml_backend_alloc_ctx_tensors_from_buft_size(struct ggml_context * ctx, ggml_backend_buffer_type_t buft);
GGML_API struct ggml_backend_buffer * ggml_backend_alloc_ctx_tensors_from_buft(struct ggml_context * ctx, ggml_backend_buffer_type_t buft);
GGML_API struct ggml_backend_buffer * ggml_backend_alloc_ctx_tensors(struct ggml_context * ctx, ggml_backend_t backend);
```

---

## Dart FFI Notes

The graph allocator is primarily used internally by llama.cpp. For most FFI use cases, the higher-level llama APIs handle memory management automatically.

### When to Use Directly

- Custom computation graphs outside llama.cpp
- Manual memory optimization
- Multi-backend tensor allocation
