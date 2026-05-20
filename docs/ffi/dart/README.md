# llama.cpp FFI API Documentation

This directory contains comprehensive documentation of the llama.cpp public APIs for FFI (Foreign Function Interface) integration, specifically targeting Dart FFI bindings.

## Overview

llama.cpp exposes a clean C API through several header files. All public functions are marked with `LLAMA_API` or `GGML_API` export macros, making them suitable for dynamic linking and FFI bindings.

## API Modules

### 1. llama.h - Main LLM Inference API
- **File**: [`llama-api.md`](./llama-api.md)
- **Header**: `include/llama.h`
- **Purpose**: High-level LLM inference including model loading, tokenization, sampling, KV cache management, and chat templates
- **Key Types**: `llama_model`, `llama_context`, `llama_vocab`, `llama_sampler`, `llama_batch`
- **~200+ public functions**

### 2. ggml.h - Tensor Computation Library
- **File**: [`ggml-api.md`](./ggml-api.md)
- **Header**: `ggml/include/ggml.h`
- **Purpose**: Low-level tensor operations, automatic differentiation, computation graphs
- **Key Types**: `ggml_tensor`, `ggml_context`, `ggml_cgraph`
- **~400+ public functions**
- **~100+ tensor operations**

### 3. gguf.h - GGUF File Format
- **File**: [`gguf-api.md`](./gguf-api.md)
- **Header**: `ggml/include/gguf.h`
- **Purpose**: Reading/writing GGUF model files (the binary format used by llama.cpp)
- **Key Types**: `gguf_context`
- **~40+ public functions**

### 4. ggml-backend.h - Backend Abstraction
- **File**: [`ggml-backend-api.md`](./ggml-backend-api.md)
- **Header**: `ggml/include/ggml-backend.h`
- **Purpose**: GPU/CPU backend management, device enumeration, tensor allocation, scheduling
- **Key Types**: `ggml_backend_t`, `ggml_backend_dev_t`, `ggml_backend_sched_t`, `ggml_backend_buffer_t`
- **~80+ public functions**

### 5. ggml-alloc.h - Memory Allocator
- **File**: [`ggml-alloc-api.md`](./ggml-alloc-api.md)
- **Header**: `ggml/include/ggml-alloc.h`
- **Purpose**: Graph and tensor memory allocation
- **Key Types**: `ggml_gallocr_t`, `ggml_tallocr`
- **~15+ public functions**

### 6. ggml-cpu.h - CPU Backend
- **File**: [`ggml-cpu-api.md`](./ggml-cpu-api.md)
- **Header**: `ggml/include/ggml-cpu.h`
- **Purpose**: CPU-specific operations, NUMA, threading, SIMD detection
- **~40+ public functions**

### 7. ggml-opt.h - Optimization/Training
- **File**: [`ggml-opt-api.md`](./ggml-opt-api.md)
- **Header**: `ggml/include/ggml-opt.h`
- **Purpose**: Model training, datasets, optimizers (AdamW, SGD)
- **Key Types**: `ggml_opt_dataset_t`, `ggml_opt_context_t`, `ggml_opt_result_t`
- **~30+ public functions**

### 8. GPU Backend APIs
- **File**: [`ggml-gpu-api.md`](./ggml-gpu-api.md)
- **Headers**: `ggml-cuda.h`, `ggml-metal.h`, `ggml-vulkan.h`
- **Purpose**: GPU device initialization, buffer types, device info
- **~20+ public functions total**

## C++ Wrapper Headers

These headers provide RAII smart pointer wrappers (for C++ only, not needed for Dart FFI):

| Header | Purpose |
|--------|---------|
| `include/llama-cpp.h` | Smart pointers for llama types (`llama_model_ptr`, `llama_context_ptr`, etc.) |
| `ggml/include/ggml-cpp.h` | Smart pointers for ggml types (`ggml_context_ptr`, `ggml_backend_ptr`, etc.) |

## Data Flow for FFI Integration

```
┌─────────────────────────────────────────────────────────────────┐
│                        Dart Application                         │
├─────────────────────────────────────────────────────────────────┤
│                     Dart FFI Bindings                           │
│  (Generated from these docs or via ffigen)                      │
├─────────────────────────────────────────────────────────────────┤
│                     llama.cpp Shared Library                    │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────┐   │
│  │ llama.h  │  │ ggml.h   │  │ gguf.h   │  │ ggml-backend │   │
│  │ (LLM API)│  │(Tensors) │  │(Format)  │  │  (Devices)   │   │
│  └──────────┘  └──────────┘  └──────────┘  └──────────────┘   │
├─────────────────────────────────────────────────────────────────┤
│                    Backend Implementations                      │
│  ┌────────┐  ┌────────┐  ┌────────┐  ┌────────┐  ┌────────┐  │
│  │  CPU   │  │ CUDA   │  │ Metal  │  │ Vulkan │  │  ...   │  │
│  └────────┘  └────────┘  └────────┘  └────────┘  └────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

## Typical Usage Flow

### 1. Initialization
```
llama_backend_init() → Initialize the library
ggml_backend_load_all() → Load all available backends (optional, auto-loaded)
```

### 2. Model Loading
```
llama_model_default_params() → Get default model params
llama_model_load_from_file() → Load model from GGUF file
llama_model_get_vocab() → Get vocabulary for tokenization
```

### 3. Context Creation
```
llama_context_default_params() → Get default context params
llama_init_from_model() → Create inference context
```

### 4. Tokenization
```
llama_tokenize() → Convert text to tokens
llama_token_to_piece() → Convert token to text piece
llama_detokenize() → Convert tokens back to text
```

### 5. Inference
```
llama_batch_init() → Allocate batch
llama_decode() → Process tokens through model
llama_get_logits_ith() → Get output logits
llama_sampler_sample() → Sample next token
llama_batch_free() → Free batch
```

### 6. Cleanup
```
llama_model_free() → Free model
llama_free() → Free context
llama_backend_free() → Free backend
```

## Key Design Notes for FFI

### Memory Management
- All opaque types (`llama_model*`, `llama_context*`, etc.) are pointers to internal structures
- Each `*_init`/`*_load` function has a corresponding `*_free` function
- Dart FFI should use `Pointer<Void>` or `Opaque` types for these

### String Handling
- C strings are null-terminated UTF-8
- Functions returning strings either:
  - Return a pointer to internal static strings (don't free)
  - Write to a caller-provided buffer and return the length
- Functions that write to buffers return `-1` on failure or the required buffer size

### Error Handling
- Most functions return status codes (0 = success, negative = error)
- Some functions return `bool` for success/failure
- Use `ggml_log_set()` / `llama_log_set()` to capture error messages

### Thread Safety
- The tokenization API (`llama_tokenize`, `llama_token_to_piece`, `llama_detokenize`) is thread-safe
- `llama_context` operations are NOT thread-safe (use one context per thread or synchronize)
- Logger functions (`llama_log_set`, `ggml_log_set`) are NOT thread-safe

### Deprecated Functions
- Functions marked with `DEPRECATED()` should be avoided in new bindings
- Each deprecated function has a replacement noted in the hint

## Constants and Macros

| Constant | Value | Purpose |
|----------|-------|---------|
| `LLAMA_DEFAULT_SEED` | `0xFFFFFFFF` | Default random seed |
| `LLAMA_TOKEN_NULL` | `-1` | Null token value |
| `LLAMA_SESSION_MAGIC` | `0x6767736e` | Session file magic |
| `LLAMA_SESSION_VERSION` | `9` | Session file version |
| `GGML_MAX_DIMS` | `4` | Maximum tensor dimensions |
| `GGML_FILE_MAGIC` | `0x67676d6c` | GGML file magic ("ggml") |
| `GGUF_VERSION` | `3` | Current GGUF version |

## Build Configuration

For FFI integration, build llama.cpp as a shared library:

```bash
cmake -B build -DBUILD_SHARED_LIBS=ON
cmake --build build --config Release
```

This produces:
- macOS: `libllama.dylib`
- Linux: `libllama.so`
- Windows: `llama.dll`

## Related Documentation

- [CONTRIBUTING.md](../CONTRIBUTING.md) - Contribution guidelines
- [docs/build.md](../build.md) - Build instructions
- [docs/backend/](../backend/) - Backend-specific documentation
