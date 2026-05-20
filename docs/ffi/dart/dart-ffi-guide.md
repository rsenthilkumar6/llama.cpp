# Dart FFI Binding Guide for llama.cpp

> **Purpose**: Practical guide for creating Dart FFI bindings to llama.cpp
> **Audience**: Dart developers creating FFI bindings

## Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Building the Shared Library](#building-the-shared-library)
- [Setting Up FFI](#setting-up-ffi)
- [Type Mappings](#type-mappings)
- [Binding Generation](#binding-generation)
- [Core Binding Patterns](#core-binding-patterns)
- [Memory Management](#memory-management)
- [String Handling](#string-handling)
- [Callback Handling](#callback-handling)
- [Struct Definitions](#struct-definitions)
- [Enum Definitions](#enum-definitions)
- [Error Handling](#error-handling)
- [Complete Example](#complete-example)
- [Performance Tips](#performance-tips)
- [Common Pitfalls](#common-pitfalls)

---

## Overview

This guide covers creating Dart FFI bindings for llama.cpp, enabling you to run LLMs from a Dart/Flutter application.

**Architecture**:
```
Dart App → FFI Bindings → libllama.dylib/so/dll → llama.cpp → GPU/CPU
```

---

## Prerequisites

1. Dart SDK 3.0+ (for improved FFI features)
2. CMake and build tools
3. llama.cpp source code
4. `package:ffi` for string utilities

```yaml
dependencies:
  ffi: ^2.1.0
```

---

## Building the Shared Library

### macOS

```bash
cmake -B build -DBUILD_SHARED_LIBS=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
# Output: build/bin/libllama.dylib
```

### Linux

```bash
cmake -B build -DBUILD_SHARED_LIBS=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
# Output: build/bin/libllama.so
```

### Windows

```bash
cmake -B build -DBUILD_SHARED_LIBS=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
# Output: build/bin/llama.dll
```

---

## Setting Up FFI

### Library Loading

```dart
import 'dart:ffi';
import 'dart:io';
import 'package:ffi/ffi.dart';

class LlamaLibrary {
  late final DynamicLibrary _lib;

  LlamaLibrary() {
    if (Platform.isMacOS) {
      _lib = DynamicLibrary.open('libllama.dylib');
    } else if (Platform.isLinux) {
      _lib = DynamicLibrary.open('libllama.so');
    } else if (Platform.isWindows) {
      _lib = DynamicLibrary.open('llama.dll');
    } else {
      throw UnsupportedError('Unsupported platform');
    }
  }

  DynamicLibrary get lib => _lib;
}
```

---

## Type Mappings

| C Type | Dart FFI Type | Notes |
|--------|---------------|-------|
| `int32_t` | `Int32` | Direct mapping |
| `uint32_t` | `Uint32` | Direct mapping |
| `int64_t` | `Int64` | Direct mapping |
| `uint64_t` | `Uint64` | Direct mapping |
| `size_t` | `Size` | Platform-dependent |
| `float` | `Float` | Direct mapping |
| `double` | `Double` | Direct mapping |
| `bool` | `Bool` | Direct mapping |
| `char *` | `Pointer<Utf8>` | Use `package:ffi` |
| `void *` | `Pointer<Void>` | Opaque pointer |
| `llama_model *` | `Pointer<llama_model>` | Opaque struct |
| `llama_context *` | `Pointer<llama_context>` | Opaque struct |
| `enum` | `Int32` | Enums are integers |
| Function pointer | `Pointer<NativeFunction<...>>` | See callbacks |

---

## Binding Generation

### Using ffigen (Recommended)

Create `ffigen.yaml`:

```yaml
name: LlamaCppBindings
description: FFI bindings for llama.cpp
output: 'lib/llama_bindings.dart'
headers:
  entry-points:
    - '../include/llama.h'
    - '../ggml/include/ggml.h'
    - '../ggml/include/gguf.h'
    - '../ggml/include/ggml-backend.h'
  include-directives:
    - '**llama.h'
    - '**ggml.h'
    - '**gguf.h'
    - '**ggml-backend.h'
preamble: |
  // ignore_for_file: non_constant_identifier_names
  // ignore_for_file: camel_case_types
  // ignore_for_file: unused_element
comments:
  style: any
  length: full
```

Run:
```bash
dart run ffigen --config ffigen.yaml
```

### Manual Bindings (For Fine Control)

For complex APIs, manual bindings provide better control:

```dart
typedef llama_backend_init_native = Void Function();
typedef LlamaBackendInit = void Function();

class LlamaCpp {
  final DynamicLibrary lib;
  
  LlamaCpp(this.lib);

  void llamaBackendInit() {
    lib
        .lookup<NativeFunction<llama_backend_init_native>>('llama_backend_init')
        .asFunction<LlamaBackendInit>()();
  }
}
```

---

## Core Binding Patterns

### Opaque Type Declarations

```dart
final class llama_model extends Opaque {}
final class llama_context extends Opaque {}
final class llama_vocab extends Opaque {}
final class llama_sampler extends Opaque {}
final class llama_adapter_lora extends Opaque {}
final class llama_memory_i extends Opaque {}
```

### Function Bindings

```dart
// void llama_backend_init(void)
typedef llama_backend_init_native = Void Function();
typedef LlamaBackendInit = void Function();

// llama_model* llama_model_load_from_file(const char* path, llama_model_params params)
typedef llama_model_load_from_file_native = Pointer<llama_model> Function(
  Pointer<Utf8> path_model,
  LlamaModelParams params,
);
typedef LlamaModelLoadFromFile = Pointer<llama_model> Function(
  Pointer<Utf8> path_model,
  LlamaModelParams params,
);

// void llama_model_free(llama_model* model)
typedef llama_model_free_native = Void Function(Pointer<llama_model>);
typedef LlamaModelFree = void Function(Pointer<llama_model>);
```

---

## Memory Management

### Using try-finally

```dart
Pointer<llama_model>? model;
try {
  model = llamaModelLoadFromFile(path.toNativeUtf8(), defaultParams);
  if (model == nullptr) {
    throw Exception('Failed to load model');
  }
  
  // Use model...
} finally {
  if (model != null && model != nullptr) {
    llamaModelFree(model);
  }
  calloc.free(path.toNativeUtf8());
}
```

### Using a Wrapper Class

```dart
class LlamaModel implements Finalizable {
  Pointer<llama_model> _ptr;
  final LlamaCpp _lib;
  
  LlamaModel._(this._ptr, this._lib);
  
  static LlamaModel loadFromFile(LlamaCpp lib, String path, LlamaModelParams params) {
    final pathPtr = path.toNativeUtf8();
    try {
      final ptr = lib.llamaModelLoadFromFile(pathPtr, params);
      if (ptr == nullptr) {
        throw Exception('Failed to load model: $path');
      }
      return LlamaModel._(ptr, lib);
    } finally {
      calloc.free(pathPtr);
    }
  }
  
  void free() {
    if (_ptr != nullptr) {
      _lib.llamaModelFree(_ptr);
      _ptr = nullptr;
    }
  }
}
```

---

## String Handling

### Getting Strings from C

```dart
// For static strings (don't free):
String llamaPrintSystemInfo() {
  final ptr = lib.llamaPrintSystemInfo();
  return ptr.cast<Utf8>().toDartString();
}

// For buffer-based strings:
String getModelDescription(Pointer<llama_model> model) {
  final bufSize = 256;
  final buf = calloc<Char>(bufSize);
  try {
    final len = lib.llamaModelDesc(model, buf, bufSize);
    if (len < 0) return 'Unknown';
    return buf.cast<Utf8>().toDartString();
  } finally {
    calloc.free(buf);
  }
}
```

### Two-Call Pattern for Variable-Length Strings

```dart
String tokenToPiece(Pointer<llama_vocab> vocab, int token) {
  // First call to get size
  final size = lib.llamaTokenToPiece(vocab, token, nullptr, 0, 0, false);
  if (size <= 0) return '';
  
  // Second call to get actual string
  final buf = calloc<Char>(size + 1);
  try {
    lib.llamaTokenToPiece(vocab, token, buf, size, 0, false);
    return buf.cast<Utf8>().toDartString();
  } finally {
    calloc.free(buf);
  }
}
```

---

## Callback Handling

### Logging Callback

```dart
// C: typedef void (*ggml_log_callback)(enum ggml_log_level level, const char * text, void * user_data);
typedef GgmlLogCallbackNative = Void Function(Int32 level, Pointer<Utf8> text, Pointer<Void> userData);
typedef GgmlLogCallback = void Function(int level, Pointer<Utf8> text, Pointer<Void> userData);

class LlamaLogger {
  late final NativeCallable<GgmlLogCallbackNative> _callable;
  late final Pointer<NativeFunction<GgmlLogCallbackNative>> _callbackPtr;
  
  LlamaLogger() {
    _callable = NativeCallable<GgmlLogCallbackNative>.listener(_logHandler);
    _callbackPtr = _callable.nativeFunction;
  }
  
  void _logHandler(int level, Pointer<Utf8> text, Pointer<Void> userData) {
    final message = text.toDartString();
    switch (level) {
      case 0: // NONE
        break;
      case 1: // DEBUG
        print('[DEBUG] $message');
        break;
      case 2: // INFO
        print('[INFO] $message');
        break;
      case 3: // WARN
        print('[WARN] $message');
        break;
      case 4: // ERROR
        print('[ERROR] $message');
        break;
    }
  }
  
  void dispose() {
    _callable.close();
  }
}

// Usage:
final logger = LlamaLogger();
lib.llamaLogSet(logger._callbackPtr.cast(), nullptr);
```

### Progress Callback

```dart
// C: typedef bool (*llama_progress_callback)(float progress, void * user_data);
typedef LlamaProgressCallbackNative = Bool Function(Float progress, Pointer<Void> userData);

class ModelLoadProgress {
  late final NativeCallable<LlamaProgressCallbackNative> _callable;
  
  ModelLoadProgress({void Function(double progress)? onProgress}) {
    _callable = NativeCallable<LlamaProgressCallbackNative>.listener(
      (progress, userData) {
        onProgress?.call(progress);
        return true; // Continue loading
      },
    );
  }
  
  Pointer<NativeFunction<LlamaProgressCallbackNative>> get ptr => _callable.nativeFunction;
  
  void dispose() => _callable.close();
}
```

---

## Struct Definitions

### Simple Struct

```dart
final class LlamaModelParams extends Struct {
  external Pointer<Pointer<Void>> devices;
  external Pointer<Void> tensor_buft_overrides;
  @Int32() external int n_gpu_layers;
  @Int32() external int split_mode;
  @Int32() external int main_gpu;
  external Pointer<Float> tensor_split;
  external Pointer<Void> progress_callback;
  external Pointer<Void> progress_callback_user_data;
  external Pointer<Void> kv_overrides;
  @Bool() external bool vocab_only;
  @Bool() external bool use_mmap;
  @Bool() external bool use_direct_io;
  @Bool() external bool use_mlock;
  @Bool() external bool check_tensors;
  @Bool() external bool use_extra_bufts;
  @Bool() external bool no_host;
  @Bool() external bool no_alloc;
}
```

### Batch Struct

```dart
final class LlamaBatch extends Struct {
  @Int32() external int n_tokens;
  external Pointer<Int32> token;
  external Pointer<Float> embd;
  external Pointer<Int32> pos;
  external Pointer<Pointer<Int32>> n_seq_id;
  external Pointer<Pointer<Pointer<Int32>>> seq_id;
  external Pointer<Int8> logits;
}
```

---

## Enum Definitions

```dart
enum LlamaVocabType {
  none(0),
  spm(1),
  bpe(2),
  wpm(3),
  ugm(4),
  rwkv(5),
  plamo2(6);
  
  final int value;
  const LlamaVocabType(this.value);
}

enum LlamaSplitMode {
  none(0),
  layer(1),
  row(2);
  
  final int value;
  const LlamaSplitMode(this.value);
}

enum GgmlType {
  f32(0),
  f16(1),
  q4_0(2),
  q4_1(3),
  q5_0(6),
  q5_1(7),
  q8_0(8),
  q2K(10),
  q3K(11),
  q4K(12),
  q5K(13),
  q6K(14),
  i8(24),
  i16(25),
  i32(26),
  bf16(30);
  
  final int value;
  const GgmlType(this.value);
}
```

---

## Error Handling

```dart
class LlamaException implements Exception {
  final String message;
  final int? code;
  
  LlamaException(this.message, {this.code});
  
  @override
  String toString() => 'LlamaException: $message (code: $code)';
}

// Decode return values
int checkDecodeResult(int result, String operation) {
  if (result == 0) return result;
  if (result == 1) throw LlamaException('KV cache full', code: result);
  if (result == 2) throw LlamaException('Operation aborted', code: result);
  if (result < 0) throw LlamaException('$operation failed with code $result', code: result);
  return result;
}
```

---

## Complete Example

```dart
import 'dart:ffi';
import 'dart:io';
import 'package:ffi/ffi.dart';

class LlamaCpp {
  final DynamicLibrary lib;
  
  LlamaCpp(this.lib);
  
  // Backend
  void llamaBackendInit() =>
      lib.lookupFunction<Void Function(), void Function()>('llama_backend_init')();
  
  void llamaBackendFree() =>
      lib.lookupFunction<Void Function(), void Function()>('llama_backend_free')();
  
  // Model loading
  Pointer<llama_model> llamaModelLoadFromFile(String path, LlamaModelParams params) {
    final pathPtr = path.toNativeUtf8();
    try {
      return lib
          .lookupFunction<
              Pointer<llama_model> Function(Pointer<Utf8>, LlamaModelParams),
              Pointer<llama_model> Function(Pointer<Utf8>, LlamaModelParams)
          >('llama_model_load_from_file')(pathPtr, params);
    } finally {
      calloc.free(pathPtr);
    }
  }
  
  void llamaModelFree(Pointer<llama_model> model) =>
      lib.lookupFunction<Void Function(Pointer<llama_model>), void Function(Pointer<llama_model>)>('llama_model_free')(model);
  
  // Context
  Pointer<llama_context> llamaInitFromModel(Pointer<llama_model> model, LlamaContextParams params) =>
      lib.lookupFunction<
          Pointer<llama_context> Function(Pointer<llama_model>, LlamaContextParams),
          Pointer<llama_context> Function(Pointer<llama_model>, LlamaContextParams)
      >('llama_init_from_model')(model, params);
  
  void llamaFree(Pointer<llama_context> ctx) =>
      lib.lookupFunction<Void Function(Pointer<llama_context>), void Function(Pointer<llama_context>)>('llama_free')(ctx);
  
  // Tokenization
  int llamaTokenize(Pointer<llama_vocab> vocab, String text, Pointer<Int32> tokens, int maxTokens, bool addSpecial, bool parseSpecial) {
    final textPtr = text.toNativeUtf8();
    try {
      return lib.lookupFunction<
          Int32 Function(Pointer<llama_vocab>, Pointer<Utf8>, Int32, Pointer<Int32>, Int32, Bool, Bool),
          int Function(Pointer<llama_vocab>, Pointer<Utf8>, int, Pointer<Int32>, int, bool, bool)
      >('llama_tokenize')(vocab, textPtr, text.length, tokens, maxTokens, addSpecial, parseSpecial);
    } finally {
      calloc.free(textPtr);
    }
  }
  
  // Inference
  int llamaDecode(Pointer<llama_context> ctx, LlamaBatch batch) =>
      lib.lookupFunction<Int32 Function(Pointer<llama_context>, LlamaBatch), int Function(Pointer<llama_context>, LlamaBatch)>('llama_decode')(ctx, batch);
  
  Pointer<Float> llamaGetLogitsIth(Pointer<llama_context> ctx, int i) =>
      lib.lookupFunction<Pointer<Float> Function(Pointer<llama_context>, Int32), Pointer<Float> Function(Pointer<llama_context>, int)>('llama_get_logits_ith')(ctx, i);
}

// Usage
void main() {
  final lib = LlamaCpp(DynamicLibrary.open('libllama.dylib'));
  
  // Initialize
  lib.llamaBackendInit();
  
  try {
    // Load model
    final modelParams = calloc<LlamaModelParams>()
      ..ref.n_gpu_layers = -1  // All layers to GPU
      ..ref.split_mode = 1     // Layer split
      ..ref.main_gpu = 0
      ..ref.use_mmap = true
      ..ref.use_mlock = false;
    
    final model = lib.llamaModelLoadFromFile('/path/to/model.gguf', modelParams.ref);
    if (model == nullptr) {
      throw Exception('Failed to load model');
    }
    
    try {
      // Create context
      final ctxParams = calloc<LlamaContextParams>()
        ..ref.n_ctx = 4096
        ..ref.n_batch = 512
        ..ref.n_threads = 4
        ..ref.n_threads_batch = 4;
      
      final ctx = lib.llamaInitFromModel(model, ctxParams.ref);
      
      try {
        // Get vocabulary
        final vocab = lib.llamaModelGetVocab(model);
        
        // Tokenize prompt
        final prompt = 'Hello, world!';
        final maxTokens = prompt.length * 2;
        final tokens = calloc<Int32>(maxTokens);
        try {
          final nTokens = lib.llamaTokenize(vocab, prompt, tokens, maxTokens, true, false);
          
          // Create batch
          final batch = calloc<LlamaBatch>()
            ..ref.n_tokens = nTokens
            ..ref.token = tokens.cast()
            ..ref.pos = nullptr
            ..ref.n_seq_id = nullptr
            ..ref.seq_id = nullptr
            ..ref.logits = nullptr;
          
          // Decode
          final result = lib.llamaDecode(ctx, batch.ref);
          
          // Get logits for last token
          final logits = lib.llamaGetLogitsIth(ctx, -1);
          // Process logits...
          
        } finally {
          calloc.free(tokens);
        }
        
      } finally {
        lib.llamaFree(ctx);
      }
      
    } finally {
      lib.llamaModelFree(model);
    }
    
  } finally {
    lib.llamaBackendFree();
  }
}
```

---

## Performance Tips

1. **Reuse token buffers**: Allocate token arrays once and reuse them
2. **Batch processing**: Use larger `n_batch` values for faster prompt processing
3. **GPU offloading**: Set `n_gpu_layers = -1` to offload all layers
4. **Thread count**: Set `n_threads` to match your CPU core count
5. **Memory mapping**: Enable `use_mmap = true` for faster model loading
6. **Avoid string conversions**: Minimize `toNativeUtf8()` / `toDartString()` calls in hot paths
7. **Use `Finalizable`**: Mark wrapper classes as `Finalizable` for better GC behavior

---

## Common Pitfalls

1. **Forgetting to free memory**: Always match init/load with free
2. **String lifetime**: C strings returned by llama.cpp are internal - don't free them
3. **Null checks**: Always check for `nullptr` after load/init functions
4. **Thread safety**: `llama_context` is NOT thread-safe - use one per thread or synchronize
5. **Logger callbacks**: Keep `NativeCallable` references alive - they get GC'd otherwise
6. **Struct alignment**: Use `@Int32()`, `@Bool()` annotations for correct struct layout
7. **Boolean size**: C `bool` is 1 byte, not 4 - use `@Bool()` annotation
