# gguf.h - GGUF File Format API

> **Header**: `ggml/include/gguf.h` (204 lines)
> **Purpose**: Reading and writing GGUF model files
> **FFI Priority**: MEDIUM - Needed for model inspection and creation

## Table of Contents

- [Overview](#overview)
- [GGUF File Structure](#gguf-file-structure)
- [Types](#types)
- [Context Management](#context-management)
- [Reading Metadata](#reading-metadata)
- [Reading Tensors](#reading-tensors)
- [Writing Metadata](#writing-metadata)
- [Writing Files](#writing-files)
- [Dart FFI Notes](#dart-ffi-notes)

---

## Overview

GGUF (GGML Universal Format) is the binary file format used by llama.cpp for storing models. Version 3 is the current format.

**Constants**:
```c
#define GGUF_MAGIC   "GGUF"
#define GGUF_VERSION 3
#define GGUF_KEY_GENERAL_ALIGNMENT "general.alignment"
#define GGUF_DEFAULT_ALIGNMENT 32
```

## GGUF File Structure

```
1. Magic "GGUF" (4 bytes)
2. Version (uint32_t)
3. Number of tensors (int64_t)
4. Number of KV pairs (int64_t)
5. KV pairs (key + type + value)
6. Tensor info (name + dimensions + type + offset)
7. Tensor data (binary blob)
```

Strings are serialized as: length (uint64_t) + bytes (no null terminator).
Enums and bools are stored as int32_t and int8_t respectively.

---

## Types

### gguf_type

Types for GGUF metadata values:

| Value | Name | C Type |
|-------|------|--------|
| 0 | `GGUF_TYPE_UINT8` | uint8_t |
| 1 | `GGUF_TYPE_INT8` | int8_t |
| 2 | `GGUF_TYPE_UINT16` | uint16_t |
| 3 | `GGUF_TYPE_INT16` | int16_t |
| 4 | `GGUF_TYPE_UINT32` | uint32_t |
| 5 | `GGUF_TYPE_INT32` | int32_t |
| 6 | `GGUF_TYPE_FLOAT32` | float |
| 7 | `GGUF_TYPE_BOOL` | bool (stored as int8_t) |
| 8 | `GGUF_TYPE_STRING` | string |
| 9 | `GGUF_TYPE_ARRAY` | array |
| 10 | `GGUF_TYPE_UINT64` | uint64_t |
| 11 | `GGUF_TYPE_INT64` | int64_t |
| 12 | `GGUF_TYPE_FLOAT64` | double |

### gguf_context

Opaque handle to a GGUF file in memory.

### gguf_init_params

```c
struct gguf_init_params {
    bool no_alloc;                        // Don't allocate tensor data
    struct ggml_context ** ctx;           // If not NULL, create ggml_context for tensors
};
```

---

## Context Management

### Initialize

```c
GGML_API struct gguf_context * gguf_init_empty(void);
```

Create an empty GGUF context for writing.

```c
GGML_API struct gguf_context * gguf_init_from_file(const char * fname, struct gguf_init_params params);
```

Load a GGUF file from disk.

```c
GGML_API struct gguf_context * gguf_init_from_file_ptr(FILE * file, struct gguf_init_params params);
```

Load from an already-opened FILE pointer.

### Free

```c
GGML_API void gguf_free(struct gguf_context * ctx);
```

Free a GGUF context.

---

## Reading Metadata

### General Info

```c
GGML_API uint32_t gguf_get_version(const struct gguf_context * ctx);
GGML_API size_t gguf_get_alignment(const struct gguf_context * ctx);
GGML_API size_t gguf_get_data_offset(const struct gguf_context * ctx);
GGML_API const char * gguf_type_name(enum gguf_type type);
```

### KV Pair Enumeration

```c
GGML_API int64_t gguf_get_n_kv(const struct gguf_context * ctx);
GGML_API int64_t gguf_find_key(const struct gguf_context * ctx, const char * key);  // Returns -1 if not found
GGML_API const char * gguf_get_key(const struct gguf_context * ctx, int64_t key_id);
GGML_API enum gguf_type gguf_get_kv_type(const struct gguf_context * ctx, int64_t key_id);
GGML_API enum gguf_type gguf_get_arr_type(const struct gguf_context * ctx, int64_t key_id);
```

### Reading Values

All value getters will abort if the wrong type is used:

```c
// Scalar values
GGML_API uint8_t      gguf_get_val_u8  (const struct gguf_context * ctx, int64_t key_id);
GGML_API int8_t       gguf_get_val_i8  (const struct gguf_context * ctx, int64_t key_id);
GGML_API uint16_t     gguf_get_val_u16 (const struct gguf_context * ctx, int64_t key_id);
GGML_API int16_t      gguf_get_val_i16 (const struct gguf_context * ctx, int64_t key_id);
GGML_API uint32_t     gguf_get_val_u32 (const struct gguf_context * ctx, int64_t key_id);
GGML_API int32_t      gguf_get_val_i32 (const struct gguf_context * ctx, int64_t key_id);
GGML_API float        gguf_get_val_f32 (const struct gguf_context * ctx, int64_t key_id);
GGML_API uint64_t     gguf_get_val_u64 (const struct gguf_context * ctx, int64_t key_id);
GGML_API int64_t      gguf_get_val_i64 (const struct gguf_context * ctx, int64_t key_id);
GGML_API double       gguf_get_val_f64 (const struct gguf_context * ctx, int64_t key_id);
GGML_API bool         gguf_get_val_bool(const struct gguf_context * ctx, int64_t key_id);
GGML_API const char * gguf_get_val_str (const struct gguf_context * ctx, int64_t key_id);
GGML_API const void * gguf_get_val_data(const struct gguf_context * ctx, int64_t key_id);  // Raw pointer
```

### Reading Arrays

```c
GGML_API size_t       gguf_get_arr_n   (const struct gguf_context * ctx, int64_t key_id);
GGML_API const void * gguf_get_arr_data(const struct gguf_context * ctx, int64_t key_id);  // Raw pointer to first element
GGML_API const char * gguf_get_arr_str (const struct gguf_context * ctx, int64_t key_id, size_t i);  // ith string
```

---

## Reading Tensors

```c
GGML_API int64_t        gguf_get_n_tensors    (const struct gguf_context * ctx);
GGML_API int64_t        gguf_find_tensor      (const struct gguf_context * ctx, const char * name);  // Returns -1 if not found
GGML_API size_t         gguf_get_tensor_offset(const struct gguf_context * ctx, int64_t tensor_id);
GGML_API const char *   gguf_get_tensor_name  (const struct gguf_context * ctx, int64_t tensor_id);
GGML_API enum ggml_type gguf_get_tensor_type  (const struct gguf_context * ctx, int64_t tensor_id);
GGML_API size_t         gguf_get_tensor_size  (const struct gguf_context * ctx, int64_t tensor_id);
```

---

## Writing Metadata

### Remove Key

```c
GGML_API int64_t gguf_remove_key(struct gguf_context * ctx, const char * key);
```

Returns the prior key ID, or -1 if not found.

### Setting Scalar Values

```c
GGML_API void gguf_set_val_u8  (struct gguf_context * ctx, const char * key, uint8_t val);
GGML_API void gguf_set_val_i8  (struct gguf_context * ctx, const char * key, int8_t val);
GGML_API void gguf_set_val_u16 (struct gguf_context * ctx, const char * key, uint16_t val);
GGML_API void gguf_set_val_i16 (struct gguf_context * ctx, const char * key, int16_t val);
GGML_API void gguf_set_val_u32 (struct gguf_context * ctx, const char * key, uint32_t val);
GGML_API void gguf_set_val_i32 (struct gguf_context * ctx, const char * key, int32_t val);
GGML_API void gguf_set_val_f32 (struct gguf_context * ctx, const char * key, float val);
GGML_API void gguf_set_val_u64 (struct gguf_context * ctx, const char * key, uint64_t val);
GGML_API void gguf_set_val_i64 (struct gguf_context * ctx, const char * key, int64_t val);
GGML_API void gguf_set_val_f64 (struct gguf_context * ctx, const char * key, double val);
GGML_API void gguf_set_val_bool(struct gguf_context * ctx, const char * key, bool val);
GGML_API void gguf_set_val_str (struct gguf_context * ctx, const char * key, const char * val);
```

### Setting Arrays

```c
GGML_API void gguf_set_arr_data(struct gguf_context * ctx, const char * key, enum gguf_type type, const void * data, size_t n);
GGML_API void gguf_set_arr_str (struct gguf_context * ctx, const char * key, const char ** data, size_t n);
```

### Copy KV Pairs

```c
GGML_API void gguf_set_kv(struct gguf_context * ctx, const struct gguf_context * src);
```

### Tensor Management

```c
GGML_API void gguf_add_tensor(struct gguf_context * ctx, const struct ggml_tensor * tensor);
GGML_API void gguf_set_tensor_type(struct gguf_context * ctx, const char * name, enum ggml_type type);
GGML_API void gguf_set_tensor_data(struct gguf_context * ctx, const char * name, const void * data);
```

---

## Writing Files

### Write to File

```c
GGML_API bool gguf_write_to_file(const struct gguf_context * ctx, const char * fname, bool only_meta);
GGML_API bool gguf_write_to_file_ptr(const struct gguf_context * ctx, FILE * file, bool only_meta);
```

**Parameters**:
- `only_meta`: If true, write only metadata (no tensor data)

### Meta Size

```c
GGML_API size_t gguf_get_meta_size(const struct gguf_context * ctx);
GGML_API void   gguf_get_meta_data(const struct gguf_context * ctx, void * data);
```

Get the size of metadata and write it to a buffer. Useful for writing tensor data first, then prepending metadata.

---

## Dart FFI Notes

### Common GGUF Keys for Model Inspection

When inspecting a model file, these are common keys to read:

| Key Pattern | Type | Description |
|-------------|------|-------------|
| `general.architecture` | string | Model architecture (llama, gemma, etc.) |
| `general.name` | string | Model name |
| `general.file_type` | uint32 | File type (quantization) |
| `llama.embedding_length` | uint32 | Embedding dimension |
| `llama.block_count` | uint32 | Number of layers |
| `llama.context_length` | uint32 | Context window size |
| `llama.attention.head_count` | uint32 | Number of attention heads |
| `llama.attention.head_count_kv` | uint32 | KV heads (for GQA) |
| `llama.attention.layer_norm_rms_epsilon` | float32 | RMS norm epsilon |
| `tokenizer.ggml.tokens` | array[string] | Token strings |
| `tokenizer.ggml.scores` | array[float32] | Token scores |
| `tokenizer.ggml.token_type` | array[int32] | Token types |
| `tokenizer.ggml.bos_token_id` | uint32 | BOS token ID |
| `tokenizer.ggml.eos_token_id` | uint32 | EOS token ID |
| `tokenizer.chat_template` | string | Chat template |

### Dart FFI Pattern for Reading GGUF

```dart
// Load GGUF file
final params = calloc<gguf_init_params>()..ref.no_alloc = true;
final gguf = ggufInitFromFile(path, params.value);

// Get metadata
final kvCount = ggufGetNKv(gguf);
for (int i = 0; i < kvCount; i++) {
  final key = ggufGetKey(gguf, i);
  final type = ggufGetKvType(gguf, i);
  
  switch (type) {
    case GgufType.uint32:
      final val = ggufGetValU32(gguf, i);
      print('$key: $val');
      break;
    case GgufType.string:
      final val = ggufGetValStr(gguf, i);
      print('$key: ${val.toDartString()}');
      break;
    // ... etc
  }
}

// Get tensor info
final tensorCount = ggufGetNTensors(gguf);
for (int i = 0; i < tensorCount; i++) {
  final name = ggufGetTensorName(gguf, i);
  final type = ggufGetTensorType(gguf, i);
  final size = ggufGetTensorSize(gguf, i);
  print('Tensor $name: type=$type, size=$size');
}

ggufFree(gguf);
```
