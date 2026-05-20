# ggml-backend.h - Backend Abstraction API

> **Header**: `ggml/include/ggml-backend.h` (373 lines)
> **Purpose**: GPU/CPU backend management, device enumeration, tensor allocation, scheduling
> **FFI Priority**: HIGH - Needed for device selection and GPU offloading

## Table of Contents

- [Overview](#overview)
- [Core Types](#core-types)
- [Backend Buffer Type](#backend-buffer-type)
- [Backend Buffer](#backend-buffer)
- [Backend (Stream)](#backend-stream)
- [Backend Events](#backend-events)
- [Backend Device](#backend-device)
- [Backend Registry](#backend-registry)
- [Backend Scheduler](#backend-scheduler)
- [Utilities](#utilities)
- [Dart FFI Notes](#dart-ffi-notes)

---

## Overview

The backend abstraction provides a unified interface for executing tensor operations on different hardware:
- CPU (system memory)
- CUDA/NVIDIA GPU (device memory)
- Metal/Apple GPU (device memory)
- Vulkan/Cross-platform GPU
- And more (SYCL, OpenCL, RPC, etc.)

---

## Core Types

```c
typedef struct ggml_backend_buffer_type * ggml_backend_buffer_type_t;
typedef struct ggml_backend_buffer * ggml_backend_buffer_t;
typedef struct ggml_backend_event * ggml_backend_event_t;
typedef struct ggml_backend * ggml_backend_t;
typedef void * ggml_backend_graph_plan_t;
typedef struct ggml_backend_reg * ggml_backend_reg_t;
typedef struct ggml_backend_device * ggml_backend_dev_t;
```

---

## Backend Buffer Type

A buffer type defines how memory is allocated for a specific backend.

```c
GGML_API const char * ggml_backend_buft_name(ggml_backend_buffer_type_t buft);
GGML_API ggml_backend_buffer_t ggml_backend_buft_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size);
GGML_API size_t ggml_backend_buft_get_alignment(ggml_backend_buffer_type_t buft);
GGML_API size_t ggml_backend_buft_get_max_size(ggml_backend_buffer_type_t buft);
GGML_API size_t ggml_backend_buft_get_alloc_size(ggml_backend_buffer_type_t buft, const struct ggml_tensor * tensor);
GGML_API bool ggml_backend_buft_is_host(ggml_backend_buffer_type_t buft);
GGML_API ggml_backend_dev_t ggml_backend_buft_get_device(ggml_backend_buffer_type_t buft);
```

---

## Backend Buffer

A buffer is an allocated memory region on a specific backend.

### Buffer Usage

```c
enum ggml_backend_buffer_usage {
    GGML_BACKEND_BUFFER_USAGE_ANY = 0,     // Any purpose
    GGML_BACKEND_BUFFER_USAGE_WEIGHTS = 1, // Model weights
    GGML_BACKEND_BUFFER_USAGE_COMPUTE = 2, // Compute intermediates
};
```

### Buffer Operations

```c
GGML_API const char * ggml_backend_buffer_name(ggml_backend_buffer_t buffer);
GGML_API void ggml_backend_buffer_free(ggml_backend_buffer_t buffer);
GGML_API void * ggml_backend_buffer_get_base(ggml_backend_buffer_t buffer);
GGML_API size_t ggml_backend_buffer_get_size(ggml_backend_buffer_t buffer);
GGML_API enum ggml_status ggml_backend_buffer_init_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor);
GGML_API size_t ggml_backend_buffer_get_alignment(ggml_backend_buffer_t buffer);
GGML_API size_t ggml_backend_buffer_get_max_size(ggml_backend_buffer_t buffer);
GGML_API size_t ggml_backend_buffer_get_alloc_size(ggml_backend_buffer_t buffer, const struct ggml_tensor * tensor);
GGML_API void ggml_backend_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value);
GGML_API bool ggml_backend_buffer_is_host(ggml_backend_buffer_t buffer);
GGML_API void ggml_backend_buffer_set_usage(ggml_backend_buffer_t buffer, enum ggml_backend_buffer_usage usage);
GGML_API enum ggml_backend_buffer_usage ggml_backend_buffer_get_usage(ggml_backend_buffer_t buffer);
GGML_API ggml_backend_buffer_type_t ggml_backend_buffer_get_type(ggml_backend_buffer_t buffer);
GGML_API void ggml_backend_buffer_reset(ggml_backend_buffer_t buffer);
```

### Tensor Copy Between Backends

```c
GGML_API void ggml_backend_tensor_copy(struct ggml_tensor * src, struct ggml_tensor * dst);
```

---

## Backend (Stream)

A backend represents a compute stream (like a CUDA stream).

```c
GGML_API ggml_guid_t ggml_backend_guid(ggml_backend_t backend);
GGML_API const char * ggml_backend_name(ggml_backend_t backend);
GGML_API void ggml_backend_free(ggml_backend_t backend);
```

### Default Buffer

```c
GGML_API ggml_backend_buffer_type_t ggml_backend_get_default_buffer_type(ggml_backend_t backend);
GGML_API ggml_backend_buffer_t ggml_backend_alloc_buffer(ggml_backend_t backend, size_t size);
GGML_API size_t ggml_backend_get_alignment(ggml_backend_t backend);
GGML_API size_t ggml_backend_get_max_size(ggml_backend_t backend);
```

### Tensor Data Transfer

```c
// Async operations
GGML_API void ggml_backend_tensor_set_async(ggml_backend_t backend, struct ggml_tensor * tensor, const void * data, size_t offset, size_t size);
GGML_API void ggml_backend_tensor_get_async(ggml_backend_t backend, const struct ggml_tensor * tensor, void * data, size_t offset, size_t size);

// Sync operations
GGML_API void ggml_backend_tensor_set(struct ggml_tensor * tensor, const void * data, size_t offset, size_t size);
GGML_API void ggml_backend_tensor_get(const struct ggml_tensor * tensor, void * data, size_t offset, size_t size);
GGML_API void ggml_backend_tensor_memset(struct ggml_tensor * tensor, uint8_t value, size_t offset, size_t size);
```

### Synchronization

```c
GGML_API void ggml_backend_synchronize(ggml_backend_t backend);
```

### Graph Computation

```c
GGML_API ggml_backend_graph_plan_t ggml_backend_graph_plan_create(ggml_backend_t backend, struct ggml_cgraph * cgraph);
GGML_API void ggml_backend_graph_plan_free(ggml_backend_t backend, ggml_backend_graph_plan_t plan);
GGML_API enum ggml_status ggml_backend_graph_plan_compute(ggml_backend_t backend, ggml_backend_graph_plan_t plan);
GGML_API enum ggml_status ggml_backend_graph_compute(ggml_backend_t backend, struct ggml_cgraph * cgraph);
GGML_API enum ggml_status ggml_backend_graph_compute_async(ggml_backend_t backend, struct ggml_cgraph * cgraph);
```

### Operation Support (DEPRECATED - use device version)

```c
GGML_API bool ggml_backend_supports_op(ggml_backend_t backend, const struct ggml_tensor * op);
GGML_API bool ggml_backend_supports_buft(ggml_backend_t backend, ggml_backend_buffer_type_t buft);
GGML_API bool ggml_backend_offload_op(ggml_backend_t backend, const struct ggml_tensor * op);
```

### Async Copy

```c
GGML_API void ggml_backend_tensor_copy_async(ggml_backend_t backend_src, ggml_backend_t backend_dst, struct ggml_tensor * src, struct ggml_tensor * dst);
```

### Device Access

```c
GGML_API ggml_backend_dev_t ggml_backend_get_device(ggml_backend_t backend);
```

---

## Backend Events

Events for synchronization between backends.

```c
GGML_API ggml_backend_event_t ggml_backend_event_new(ggml_backend_dev_t device);
GGML_API void ggml_backend_event_free(ggml_backend_event_t event);
GGML_API void ggml_backend_event_record(ggml_backend_event_t event, ggml_backend_t backend);
GGML_API void ggml_backend_event_synchronize(ggml_backend_event_t event);
GGML_API void ggml_backend_event_wait(ggml_backend_t backend, ggml_backend_event_t event);
```

---

## Backend Device

### Device Types

```c
enum ggml_backend_dev_type {
    GGML_BACKEND_DEVICE_TYPE_CPU,     // CPU with system memory
    GGML_BACKEND_DEVICE_TYPE_GPU,     // GPU with dedicated memory
    GGML_BACKEND_DEVICE_TYPE_IGPU,    // Integrated GPU using host memory
    GGML_BACKEND_DEVICE_TYPE_ACCEL,   // Accelerator (BLAS, AMX)
};
```

### Device Capabilities

```c
struct ggml_backend_dev_caps {
    bool async;            // Asynchronous operations
    bool host_buffer;      // Pinned host buffer
    bool buffer_from_host_ptr;  // Create buffers from host ptr
    bool events;           // Event synchronization
};
```

### Device Properties

```c
struct ggml_backend_dev_props {
    const char * name;           // Device name
    const char * description;    // Device description
    size_t memory_free;          // Free memory in bytes
    size_t memory_total;         // Total memory in bytes
    enum ggml_backend_dev_type type;  // Device type
    const char * device_id;      // Device ID (e.g., PCI bus ID)
    struct ggml_backend_dev_caps caps;  // Capabilities
};
```

### Device Operations

```c
GGML_API const char * ggml_backend_dev_name(ggml_backend_dev_t device);
GGML_API const char * ggml_backend_dev_description(ggml_backend_dev_t device);
GGML_API void ggml_backend_dev_memory(ggml_backend_dev_t device, size_t * free, size_t * total);
GGML_API enum ggml_backend_dev_type ggml_backend_dev_type(ggml_backend_dev_t device);
GGML_API void ggml_backend_dev_get_props(ggml_backend_dev_t device, struct ggml_backend_dev_props * props);
GGML_API ggml_backend_reg_t ggml_backend_dev_backend_reg(ggml_backend_dev_t device);
GGML_API ggml_backend_t ggml_backend_dev_init(ggml_backend_dev_t device, const char * params);
GGML_API ggml_backend_buffer_type_t ggml_backend_dev_buffer_type(ggml_backend_dev_t device);
GGML_API ggml_backend_buffer_type_t ggml_backend_dev_host_buffer_type(ggml_backend_dev_t device);
GGML_API ggml_backend_buffer_t ggml_backend_dev_buffer_from_host_ptr(ggml_backend_dev_t device, void * ptr, size_t size, size_t max_tensor_size);
GGML_API bool ggml_backend_dev_supports_op(ggml_backend_dev_t device, const struct ggml_tensor * op);
GGML_API bool ggml_backend_dev_supports_buft(ggml_backend_dev_t device, ggml_backend_buffer_type_t buft);
GGML_API bool ggml_backend_dev_offload_op(ggml_backend_dev_t device, const struct ggml_tensor * op);
```

---

## Backend Registry

### Registry Operations

```c
GGML_API const char * ggml_backend_reg_name(ggml_backend_reg_t reg);
GGML_API size_t ggml_backend_reg_dev_count(ggml_backend_reg_t reg);
GGML_API ggml_backend_dev_t ggml_backend_reg_dev_get(ggml_backend_reg_t reg, size_t index);
GGML_API void * ggml_backend_reg_get_proc_address(ggml_backend_reg_t reg, const char * name);
```

### Common Proc Address Types

```c
// Split buffer type for tensor parallelism
typedef ggml_backend_buffer_type_t (*ggml_backend_split_buffer_type_t)(int main_device, const float * tensor_split);

// Set number of threads
typedef void (*ggml_backend_set_n_threads_t)(ggml_backend_t backend, int n_threads);

// Get extra buffer types
typedef ggml_backend_buffer_type_t * (*ggml_backend_dev_get_extra_bufts_t)(ggml_backend_dev_t device);

// Set abort callback
typedef void (*ggml_backend_set_abort_callback_t)(ggml_backend_t backend, ggml_abort_callback abort_callback, void * abort_callback_data);

// Get features
struct ggml_backend_feature {
    const char * name;
    const char * value;
};
typedef struct ggml_backend_feature * (*ggml_backend_get_features_t)(ggml_backend_reg_t reg);
```

### Backend Registration

```c
GGML_API void ggml_backend_register(ggml_backend_reg_t reg);
GGML_API void ggml_backend_device_register(ggml_backend_dev_t device);
```

### Backend Enumeration

```c
GGML_API size_t ggml_backend_reg_count(void);
GGML_API ggml_backend_reg_t ggml_backend_reg_get(size_t index);
GGML_API ggml_backend_reg_t ggml_backend_reg_by_name(const char * name);
```

### Device Enumeration

```c
GGML_API size_t ggml_backend_dev_count(void);
GGML_API ggml_backend_dev_t ggml_backend_dev_get(size_t index);
GGML_API ggml_backend_dev_t ggml_backend_dev_by_name(const char * name);
GGML_API ggml_backend_dev_t ggml_backend_dev_by_type(enum ggml_backend_dev_type type);
```

### Direct Backend Initialization

```c
GGML_API ggml_backend_t ggml_backend_init_by_name(const char * name, const char * params);
GGML_API ggml_backend_t ggml_backend_init_by_type(enum ggml_backend_dev_type type, const char * params);
GGML_API ggml_backend_t ggml_backend_init_best(void);  // Best available GPU or CPU
```

### Dynamic Loading

```c
GGML_API ggml_backend_reg_t ggml_backend_load(const char * path);
GGML_API void ggml_backend_unload(ggml_backend_reg_t reg);
GGML_API void ggml_backend_load_all(void);
GGML_API void ggml_backend_load_all_from_path(const char * dir_path);
```

---

## Backend Scheduler

The scheduler manages multiple backends for compute, handling:
- Buffer allocation across backends
- Tensor-to-backend assignment
- Cross-backend tensor copying
- Graph splitting

### Scheduler Management

```c
typedef struct ggml_backend_sched * ggml_backend_sched_t;

GGML_API ggml_backend_sched_t ggml_backend_sched_new(
    ggml_backend_t * backends,
    ggml_backend_buffer_type_t * bufts,
    int n_backends,
    size_t graph_size,
    bool parallel,
    bool op_offload);

GGML_API void ggml_backend_sched_free(ggml_backend_sched_t sched);
```

### Reserve Buffers

```c
GGML_API void ggml_backend_sched_reserve_size(ggml_backend_sched_t sched, struct ggml_cgraph * measure_graph, size_t * sizes);
GGML_API bool ggml_backend_sched_reserve(ggml_backend_sched_t sched, struct ggml_cgraph * measure_graph);
```

### Scheduler Info

```c
GGML_API int ggml_backend_sched_get_n_backends(ggml_backend_sched_t sched);
GGML_API ggml_backend_t ggml_backend_sched_get_backend(ggml_backend_sched_t sched, int i);
GGML_API int ggml_backend_sched_get_n_splits(ggml_backend_sched_t sched);
GGML_API int ggml_backend_sched_get_n_copies(ggml_backend_sched_t sched);
GGML_API ggml_backend_buffer_type_t ggml_backend_sched_get_buffer_type(ggml_backend_sched_t sched, ggml_backend_t backend);
GGML_API size_t ggml_backend_sched_get_buffer_size(ggml_backend_sched_t sched, ggml_backend_t backend);
```

### Tensor Backend Assignment

```c
GGML_API void ggml_backend_sched_set_tensor_backend(ggml_backend_sched_t sched, struct ggml_tensor * node, ggml_backend_t backend);
GGML_API ggml_backend_t ggml_backend_sched_get_tensor_backend(ggml_backend_sched_t sched, struct ggml_tensor * node);
```

### Graph Computation

```c
GGML_API void ggml_backend_sched_split_graph(ggml_backend_sched_t sched, struct ggml_cgraph * graph);
GGML_API bool ggml_backend_sched_alloc_graph(ggml_backend_sched_t sched, struct ggml_cgraph * graph);
GGML_API enum ggml_status ggml_backend_sched_graph_compute(ggml_backend_sched_t sched, struct ggml_cgraph * graph);
GGML_API enum ggml_status ggml_backend_sched_graph_compute_async(ggml_backend_sched_t sched, struct ggml_cgraph * graph);
GGML_API void ggml_backend_sched_synchronize(ggml_backend_sched_t sched);
GGML_API void ggml_backend_sched_reset(ggml_backend_sched_t sched);
```

### Evaluation Callback

```c
typedef bool (*ggml_backend_sched_eval_callback)(struct ggml_tensor * t, bool ask, void * user_data);

GGML_API void ggml_backend_sched_set_eval_callback(
    ggml_backend_sched_t sched,
    ggml_backend_sched_eval_callback callback,
    void * user_data);
```

---

## Utilities

### Graph Copy

```c
struct ggml_backend_graph_copy {
    ggml_backend_buffer_t buffer;
    struct ggml_context * ctx_allocated;
    struct ggml_context * ctx_unallocated;
    struct ggml_cgraph * graph;
};

GGML_API struct ggml_backend_graph_copy ggml_backend_graph_copy(ggml_backend_t backend, struct ggml_cgraph * graph);
GGML_API void ggml_backend_graph_copy_free(struct ggml_backend_graph_copy copy);
```

### Backend Comparison

```c
typedef bool (*ggml_backend_eval_callback)(int node_index, struct ggml_tensor * t1, struct ggml_tensor * t2, void * user_data);

GGML_API bool ggml_backend_compare_graph_backend(
    ggml_backend_t backend1,
    ggml_backend_t backend2,
    struct ggml_cgraph * graph,
    ggml_backend_eval_callback callback,
    void * user_data,
    struct ggml_tensor const * const * test_nodes,
    size_t num_test_nodes);
```

### Tensor Initialization

```c
GGML_API enum ggml_status ggml_backend_tensor_alloc(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor, void * addr);
GGML_API enum ggml_status ggml_backend_view_init(struct ggml_tensor * tensor);
```

### CPU Buffer Types (Always Available)

```c
GGML_API ggml_backend_buffer_t ggml_backend_cpu_buffer_from_ptr(void * ptr, size_t size);
GGML_API ggml_backend_buffer_type_t ggml_backend_cpu_buffer_type(void);
```

---

## Dart FFI Notes

### Device Enumeration Pattern

```dart
// Enumerate all available devices
final devCount = ggmlBackendDevCount();
for (int i = 0; i < devCount; i++) {
  final dev = ggmlBackendDevGet(i);
  final name = ggmlBackendDevName(dev).toDartString();
  final props = calloc<ggml_backend_dev_props>();
  ggmlBackendDevGetProps(dev, props.ref);
  
  print('Device: ${name}');
  print('  Type: ${props.ref.type}');
  print('  Free: ${props.ref.memory_free}');
  print('  Total: ${props.ref.memory_total}');
  
  calloc.free(props);
}
```

### Best Backend Selection

```dart
// Simple: use ggml_backend_init_best()
final backend = ggmlBackendInitBest();

// Or manually select:
final gpu = ggmlBackendDevByType(GgmlBackendDevType.gpu);
if (gpu != nullptr) {
  final backend = ggmlBackendDevInit(gpu, nullptr);
}
```

### Memory Monitoring

```dart
// Get device memory info
final free = calloc<Size>();
final total = calloc<Size>();
ggmlBackendDevMemory(device, free, total);
print('Free: ${free.value} bytes');
print('Total: ${total.value} bytes');
```
