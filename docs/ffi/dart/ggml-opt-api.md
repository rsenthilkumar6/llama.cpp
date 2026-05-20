# ggml-opt.h - Optimization/Training API

> **Header**: `ggml/include/ggml-opt.h` (256 lines)
> **Purpose**: Model training, datasets, optimizers
> **FFI Priority**: LOW - Only needed for fine-tuning features

## Table of Contents

- [Overview](#overview)
- [Loss Types](#loss-types)
- [Dataset API](#dataset-api)
- [Optimizer Types](#optimizer-types)
- [Context API](#context-api)
- [Result API](#result-api)
- [Computation](#computation)
- [High-Level Training](#high-level-training)
- [Dart FFI Notes](#dart-ffi-notes)

---

## Overview

The optimization module provides training capabilities including dataset management, optimizers (AdamW, SGD), and high-level training loops.

---

## Loss Types

```c
enum ggml_opt_loss_type {
    GGML_OPT_LOSS_TYPE_MEAN,              // Mean reduction
    GGML_OPT_LOSS_TYPE_SUM,               // Sum reduction
    GGML_OPT_LOSS_TYPE_CROSS_ENTROPY,     // Cross-entropy loss
    GGML_OPT_LOSS_TYPE_MEAN_SQUARED_ERROR, // MSE loss
};
```

---

## Dataset API

### Types

```c
typedef struct ggml_opt_dataset * ggml_opt_dataset_t;
typedef struct ggml_opt_context * ggml_opt_context_t;
typedef struct ggml_opt_result * ggml_opt_result_t;
```

### Dataset Management

```c
GGML_API ggml_opt_dataset_t ggml_opt_dataset_init(
    enum ggml_type type_data,     // Data tensor type
    enum ggml_type type_label,    // Label tensor type
    int64_t ne_datapoint,         // Elements per datapoint
    int64_t ne_label,             // Elements per label
    int64_t ndata,                // Total datapoints
    int64_t ndata_shard);         // Shard size for shuffle/copy

GGML_API void ggml_opt_dataset_free(ggml_opt_dataset_t dataset);
```

### Dataset Access

```c
GGML_API int64_t ggml_opt_dataset_ndata(ggml_opt_dataset_t dataset);
GGML_API struct ggml_tensor * ggml_opt_dataset_data(ggml_opt_dataset_t dataset);   // [ne_datapoint, ndata]
GGML_API struct ggml_tensor * ggml_opt_dataset_labels(ggml_opt_dataset_t dataset); // [ne_label, ndata]
```

### Dataset Operations

```c
GGML_API void ggml_opt_dataset_shuffle(ggml_opt_context_t opt_ctx, ggml_opt_dataset_t dataset, int64_t idata);
GGML_API void ggml_opt_dataset_get_batch(
    ggml_opt_dataset_t dataset,
    struct ggml_tensor * data_batch,   // [ne_datapoint, ndata_batch]
    struct ggml_tensor * labels_batch, // [ne_label, ndata_batch]
    int64_t ibatch);
GGML_API void ggml_opt_dataset_get_batch_host(
    ggml_opt_dataset_t dataset,
    void * data_batch,
    size_t nb_data_batch,
    void * labels_batch,
    int64_t ibatch);
```

---

## Optimizer Types

### Build Type

```c
enum ggml_opt_build_type {
    GGML_OPT_BUILD_TYPE_FORWARD = 10,  // Forward pass only
    GGML_OPT_BUILD_TYPE_GRAD = 20,     // Forward + gradient
    GGML_OPT_BUILD_TYPE_OPT = 30,      // Forward + gradient + optimizer step
};
```

### Optimizer Type

```c
enum ggml_opt_optimizer_type {
    GGML_OPT_OPTIMIZER_TYPE_ADAMW,  // AdamW optimizer
    GGML_OPT_OPTIMIZER_TYPE_SGD,    // SGD optimizer
    GGML_OPT_OPTIMIZER_TYPE_COUNT
};
```

### Optimizer Parameters

```c
struct ggml_opt_optimizer_params {
    struct {
        float alpha;  // Learning rate
        float beta1;  // First AdamW momentum
        float beta2;  // Second AdamW momentum
        float eps;    // Epsilon for stability
        float wd;     // Weight decay (0.0f to disable)
    } adamw;
    struct {
        float alpha;  // Learning rate
        float wd;     // Weight decay
    } sgd;
};
```

### Optimizer Callback

```c
typedef struct ggml_opt_optimizer_params (*ggml_opt_get_optimizer_params)(void * userdata);

GGML_API struct ggml_opt_optimizer_params ggml_opt_get_default_optimizer_params(void * userdata);
GGML_API struct ggml_opt_optimizer_params ggml_opt_get_constant_optimizer_params(void * userdata);
```

---

## Context API

### Context Parameters

```c
struct ggml_opt_params {
    ggml_backend_sched_t backend_sched;       // Backend scheduler
    struct ggml_context * ctx_compute;        // Compute context (for static graphs)
    struct ggml_tensor * inputs;              // Input tensor
    struct ggml_tensor * outputs;             // Output tensor
    enum ggml_opt_loss_type loss_type;        // Loss type
    enum ggml_opt_build_type build_type;      // Build type
    int32_t opt_period;                       // Gradient accumulation period
    ggml_opt_get_optimizer_params get_opt_pars; // Optimizer params callback
    void * get_opt_pars_ud;                   // Callback userdata
    enum ggml_opt_optimizer_type optimizer;   // Optimizer type
};
```

### Context Management

```c
GGML_API struct ggml_opt_params ggml_opt_default_params(
    ggml_backend_sched_t backend_sched,
    enum ggml_opt_loss_type loss_type);

GGML_API ggml_opt_context_t ggml_opt_init(struct ggml_opt_params params);
GGML_API void ggml_opt_free(ggml_opt_context_t opt_ctx);
GGML_API void ggml_opt_reset(ggml_opt_context_t opt_ctx, bool optimizer);
GGML_API bool ggml_opt_static_graphs(ggml_opt_context_t opt_ctx);
GGML_API enum ggml_opt_optimizer_type ggml_opt_context_optimizer_type(ggml_opt_context_t);
GGML_API const char * ggml_opt_optimizer_name(enum ggml_opt_optimizer_type);
```

### Context Access

```c
GGML_API struct ggml_tensor * ggml_opt_inputs(  ggml_opt_context_t opt_ctx);
GGML_API struct ggml_tensor * ggml_opt_outputs( ggml_opt_context_t opt_ctx);
GGML_API struct ggml_tensor * ggml_opt_labels(  ggml_opt_context_t opt_ctx);
GGML_API struct ggml_tensor * ggml_opt_loss(    ggml_opt_context_t opt_ctx);
GGML_API struct ggml_tensor * ggml_opt_pred(    ggml_opt_context_t opt_ctx);
GGML_API struct ggml_tensor * ggml_opt_ncorrect(ggml_opt_context_t opt_ctx);
GGML_API struct ggml_tensor * ggml_opt_grad_acc(ggml_opt_context_t opt_ctx, struct ggml_tensor * node);
```

---

## Result API

```c
GGML_API ggml_opt_result_t ggml_opt_result_init(void);
GGML_API void ggml_opt_result_free(ggml_opt_result_t result);
GGML_API void ggml_opt_result_reset(ggml_opt_result_t result);

GGML_API void ggml_opt_result_ndata(ggml_opt_result_t result, int64_t * ndata);
GGML_API void ggml_opt_result_loss(ggml_opt_result_t result, double * loss, double * unc);
GGML_API void ggml_opt_result_pred(ggml_opt_result_t result, int32_t * pred);
GGML_API void ggml_opt_result_accuracy(ggml_opt_result_t result, double * accuracy, double * unc);
```

---

## Computation

```c
GGML_API void ggml_opt_prepare_alloc(
    ggml_opt_context_t opt_ctx,
    struct ggml_context * ctx_compute,
    struct ggml_cgraph * gf,
    struct ggml_tensor * inputs,
    struct ggml_tensor * outputs);

GGML_API void ggml_opt_alloc(ggml_opt_context_t opt_ctx, bool backward);
GGML_API void ggml_opt_eval(ggml_opt_context_t opt_ctx, ggml_opt_result_t result);
```

---

## High-Level Training

### Epoch Callback

```c
typedef void (*ggml_opt_epoch_callback)(
    bool train,              // true = training, false = validation
    ggml_opt_context_t opt_ctx,
    ggml_opt_dataset_t dataset,
    ggml_opt_result_t result,
    int64_t ibatch,          // Current batch
    int64_t ibatch_max,      // Total batches
    int64_t t_start_us);     // Start time
```

### Epoch Training

```c
GGML_API void ggml_opt_epoch(
    ggml_opt_context_t opt_ctx,
    ggml_opt_dataset_t dataset,
    ggml_opt_result_t result_train,
    ggml_opt_result_t result_eval,
    int64_t idata_split,
    ggml_opt_epoch_callback callback_train,
    ggml_opt_epoch_callback callback_eval);
```

### Progress Bar Callback

```c
GGML_API void ggml_opt_epoch_callback_progress_bar(
    bool train,
    ggml_opt_context_t opt_ctx,
    ggml_opt_dataset_t dataset,
    ggml_opt_result_t result,
    int64_t ibatch,
    int64_t ibatch_max,
    int64_t t_start_us);
```

### High-Level Fit Function

```c
GGML_API void ggml_opt_fit(
    ggml_backend_sched_t backend_sched,     // Backend scheduler
    struct ggml_context * ctx_compute,      // Compute context
    struct ggml_tensor * inputs,            // [ne_datapoint, ndata_batch]
    struct ggml_tensor * outputs,           // [ne_label, ndata_batch]
    ggml_opt_dataset_t dataset,             // Dataset
    enum ggml_opt_loss_type loss_type,      // Loss to minimize
    enum ggml_opt_optimizer_type optimizer, // AdamW or SGD
    ggml_opt_get_optimizer_params get_opt_pars, // Optimizer params callback
    int64_t nepoch,                         // Number of epochs
    int64_t nbatch_logical,                 // Logical batch size
    float val_split,                        // Validation split [0.0, 1.0)
    bool silent);                           // Suppress output
```

---

## Dart FFI Notes

The training API is complex and rarely needed for inference-only applications. For most UI applications, focus on the llama.h inference API instead.

### Typical Training Flow

```dart
// 1. Create dataset
final dataset = ggmlOptDatasetInit(
  GgmlType.f32,  // data type
  GgmlType.f32,  // label type
  neDatapoint,   // elements per datapoint
  neLabel,       // elements per label
  nData,         // total datapoints
  1,             // shard size
);

// 2. Set data
// Copy training data to dataset.data() tensor

// 3. Create model graph
// Build your model graph with inputs/outputs

// 4. Train
ggmlOptFit(
  backendSched,
  ctxCompute,
  inputs,
  outputs,
  dataset,
  GgmlOptLossType.crossEntropy,
  GgmlOptOptimizerType.adamw,
  getOptParamsCallback,
  nEpochs,
  nBatchLogical,
  0.1,  // 10% validation split
  false, // show progress
);
```
