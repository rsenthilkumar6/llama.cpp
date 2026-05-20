# GPU Backend APIs - CUDA, Metal, Vulkan

> **Headers**: `ggml-cuda.h`, `ggml-metal.h`, `ggml-vulkan.h`
> **Purpose**: GPU device initialization, buffer types, device info
> **FFI Priority**: MEDIUM - Needed for GPU device selection and info

## Table of Contents

- [Overview](#overview)
- [CUDA Backend](#cuda-backend)
- [Metal Backend (Apple)](#metal-backend-apple)
- [Vulkan Backend](#vulkan-backend)
- [Common Patterns](#common-patterns)
- [Dart FFI Notes](#dart-ffi-notes)

---

## Overview

Each GPU backend follows a similar pattern:
1. Initialize backend device
2. Get buffer types for memory allocation
3. Query device info (count, memory, description)
4. Register host memory for faster transfers

---

## CUDA Backend

> **Header**: `ggml/include/ggml-cuda.h` (47 lines)

### Constants

```c
#define GGML_CUDA_NAME "CUDA"        // or "ROCm" for HIP, "MUSA" for MUSA
#define GGML_CUBLAS_NAME "cuBLAS"    // or "hipBLAS" / "muBLAS"
#define GGML_CUDA_MAX_DEVICES 16
```

### Backend Initialization

```c
GGML_BACKEND_API ggml_backend_t ggml_backend_cuda_init(int device);
GGML_BACKEND_API bool ggml_backend_is_cuda(ggml_backend_t backend);
```

### Buffer Types

```c
// Standard device buffer
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_cuda_buffer_type(int device);

// Split tensor buffer for tensor parallelism
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_cuda_split_buffer_type(int main_device, const float * tensor_split);

// Pinned host buffer for faster CPU-GPU transfers
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_cuda_host_buffer_type(void);
```

### Device Info

```c
GGML_BACKEND_API int ggml_backend_cuda_get_device_count(void);
GGML_BACKEND_API void ggml_backend_cuda_get_device_description(int device, char * description, size_t description_size);
GGML_BACKEND_API void ggml_backend_cuda_get_device_memory(int device, size_t * free, size_t * total);
```

### Host Memory Registration

```c
GGML_BACKEND_API bool ggml_backend_cuda_register_host_buffer(void * buffer, size_t size);
GGML_BACKEND_API void ggml_backend_cuda_unregister_host_buffer(void * buffer);
```

### Registry

```c
GGML_BACKEND_API ggml_backend_reg_t ggml_backend_cuda_reg(void);
```

---

## Metal Backend (Apple)

> **Header**: `ggml/include/ggml-metal.h` (61 lines)

### Backend Initialization

```c
GGML_BACKEND_API ggml_backend_t ggml_backend_metal_init(void);
GGML_BACKEND_API bool ggml_backend_is_metal(ggml_backend_t backend);
```

### Advanced Features

```c
// Set abort callback
GGML_BACKEND_API void ggml_backend_metal_set_abort_callback(ggml_backend_t backend, ggml_abort_callback abort_callback, void * user_data);

// Check Metal feature set support
GGML_BACKEND_API bool ggml_backend_metal_supports_family(ggml_backend_t backend, int family);

// Capture next compute for debugging
GGML_BACKEND_API void ggml_backend_metal_capture_next_compute(ggml_backend_t backend);
```

### Registry

```c
GGML_BACKEND_API ggml_backend_reg_t ggml_backend_metal_reg(void);
```

---

## Vulkan Backend

> **Header**: `ggml/include/ggml-vulkan.h` (29 lines)

### Constants

```c
#define GGML_VK_NAME "Vulkan"
#define GGML_VK_MAX_DEVICES 16
```

### Backend Initialization

```c
GGML_BACKEND_API ggml_backend_t ggml_backend_vk_init(size_t dev_num);
GGML_BACKEND_API bool ggml_backend_is_vk(ggml_backend_t backend);
```

### Buffer Types

```c
// Standard device buffer
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_vk_buffer_type(size_t dev_num);

// Pinned host buffer for faster CPU-GPU transfers
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_vk_host_buffer_type(void);
```

### Device Info

```c
GGML_BACKEND_API int ggml_backend_vk_get_device_count(void);
GGML_BACKEND_API void ggml_backend_vk_get_device_description(int device, char * description, size_t description_size);
GGML_BACKEND_API void ggml_backend_vk_get_device_memory(int device, size_t * free, size_t * total);
```

### Registry

```c
GGML_BACKEND_API ggml_backend_reg_t ggml_backend_vk_reg(void);
```

---

## Common Patterns

### Device Enumeration (All Backends)

All GPU backends follow this pattern:

| Function | CUDA | Vulkan |
|----------|------|--------|
| Init backend | `ggml_backend_cuda_init(device)` | `ggml_backend_vk_init(device)` |
| Check type | `ggml_backend_is_cuda(backend)` | `ggml_backend_is_vk(backend)` |
| Buffer type | `ggml_backend_cuda_buffer_type(device)` | `ggml_backend_vk_buffer_type(device)` |
| Host buffer | `ggml_backend_cuda_host_buffer_type()` | `ggml_backend_vk_host_buffer_type()` |
| Device count | `ggml_backend_cuda_get_device_count()` | `ggml_backend_vk_get_device_count()` |
| Description | `ggml_backend_cuda_get_device_description()` | `ggml_backend_vk_get_device_description()` |
| Memory info | `ggml_backend_cuda_get_device_memory()` | `ggml_backend_vk_get_device_memory()` |

---

## Dart FFI Notes

### GPU Device Discovery

```dart
class GpuDeviceInfo {
  final String name;
  final String backend;
  final int index;
  final int freeMemory;
  final int totalMemory;
}

List<GpuDeviceInfo> discoverGpuDevices() {
  final devices = <GpuDeviceInfo>[];
  
  // CUDA devices
  final cudaCount = ggmlBackendCudaGetDeviceCount();
  for (int i = 0; i < cudaCount; i++) {
    final desc = calloc<Char>(256);
    final free = calloc<Size>();
    final total = calloc<Size>();
    
    ggmlBackendCudaGetDeviceDescription(i, desc, 256);
    ggmlBackendCudaGetDeviceMemory(i, free, total);
    
    devices.add(GpuDeviceInfo(
      name: desc.toDartString(),
      backend: 'CUDA',
      index: i,
      freeMemory: free.value,
      totalMemory: total.value,
    ));
    
    calloc.free(desc);
    calloc.free(free);
    calloc.free(total);
  }
  
  // Vulkan devices
  final vkCount = ggmlBackendVkGetDeviceCount();
  for (int i = 0; i < vkCount; i++) {
    final desc = calloc<Char>(256);
    final free = calloc<Size>();
    final total = calloc<Size>();
    
    ggmlBackendVkGetDeviceDescription(i, desc, 256);
    ggmlBackendVkGetDeviceMemory(i, free, total);
    
    devices.add(GpuDeviceInfo(
      name: desc.toDartString(),
      backend: 'Vulkan',
      index: i,
      freeMemory: free.value,
      totalMemory: total.value,
    ));
    
    calloc.free(desc);
    calloc.free(free);
    calloc.free(total);
  }
  
  return devices;
}
```

### Platform-Specific Backend Selection

```dart
ggml_backend_t selectBestBackend() {
  if (Platform.isMacOS) {
    // Metal is best on Apple
    return ggmlBackendMetalInit();
  }
  
  // Try CUDA first
  if (ggmlBackendCudaGetDeviceCount() > 0) {
    return ggmlBackendCudaInit(0);
  }
  
  // Try Vulkan
  if (ggmlBackendVkGetDeviceCount() > 0) {
    return ggmlBackendVkInit(0);
  }
  
  // Fall back to CPU
  return ggmlBackendCpuInit();
}
```

### Memory Monitoring

```dart
// Monitor GPU memory usage
void printGpuMemory(int device) {
  final free = calloc<Size>();
  final total = calloc<Size>();
  
  ggmlBackendCudaGetDeviceMemory(device, free, total);
  
  final used = total.value - free.value;
  final percent = (used / total.value * 100).toStringAsFixed(1);
  
  print('GPU $device: ${used ~/ (1024*1024)}MB / ${total.value ~/ (1024*1024)}MB ($percent%)');
}
```
