# llama.h - Main LLM Inference API

> **Header**: `include/llama.h` (~1576 lines)
> **Purpose**: High-level API for loading and running LLMs in GGUF format
> **FFI Priority**: **HIGHEST** - This is the primary API for Dart bindings

## Table of Contents

- [Core Types](#core-types)
- [Enums](#enums)
- [Parameter Structs](#parameter-structs)
- [Backend Initialization](#backend-initialization)
- [Model Loading](#model-loading)
- [Context Management](#context-management)
- [Model Information](#model-information)
- [Vocabulary API](#vocabulary-api)
- [Tokenization](#tokenization)
- [Batch Processing](#batch-processing)
- [Inference (Decode/Encode)](#inference-decodeencode)
- [Sampling API](#sampling-api)
- [KV Cache / Memory Management](#kv-cache--memory-management)
- [State / Sessions](#state--sessions)
- [LoRA Adapters](#lora-adapters)
- [Chat Templates](#chat-templates)
- [Performance Utilities](#performance-utilities)
- [Logging](#logging)
- [Model Quantization](#model-quantization)
- [Training API](#training-api)
- [Dart FFI Notes](#dart-ffi-notes)

---

## Core Types

All types are opaque pointers - Dart FFI should use `Opaque` or `Pointer<Void>`.

| Type | Description |
|------|-------------|
| `llama_vocab` | Vocabulary (tokenizer) |
| `llama_model` | Loaded model |
| `llama_context` | Inference context |
| `llama_sampler` | Token sampler |
| `llama_adapter_lora` | LoRA adapter |
| `llama_memory_i` | Memory interface (KV cache) |

### Basic Types

```c
typedef int32_t llama_pos;     // Position in sequence
typedef int32_t llama_token;   // Token ID
typedef int32_t llama_seq_id;  // Sequence ID
typedef struct llama_memory_i * llama_memory_t;  // Memory handle
```

### Token Data

```c
typedef struct llama_token_data {
    llama_token id;   // Token ID
    float logit;      // Log-odds of the token
    float p;          // Probability of the token
} llama_token_data;

typedef struct llama_token_data_array {
    llama_token_data * data;  // Array of token data
    size_t size;              // Array size
    int64_t selected;         // Selected token index
    bool sorted;              // Whether data is sorted
} llama_token_data_array;
```

### Batch Input

```c
typedef struct llama_batch {
    int32_t n_tokens;         // Number of tokens
    llama_token  * token;     // Token IDs (used when embd is NULL)
    float        * embd;      // Token embeddings (used when token is NULL)
    llama_pos    * pos;       // Positions in sequence
    int32_t      * n_seq_id;  // Number of sequence IDs per token
    llama_seq_id ** seq_id;   // Sequence IDs per token
    int8_t       * logits;    // Whether to output logits for each token
} llama_batch;
```

### Chat Message

```c
typedef struct llama_chat_message {
    const char * role;     // "system", "user", "assistant", etc.
    const char * content;  // Message content
} llama_chat_message;
```

### Logit Bias

```c
typedef struct llama_logit_bias {
    llama_token token;
    float bias;
} llama_logit_bias;
```

---

## Enums

### llama_vocab_type

Tokenizer type:

| Value | Name | Description |
|-------|------|-------------|
| 0 | `LLAMA_VOCAB_TYPE_NONE` | No vocabulary |
| 1 | `LLAMA_VOCAB_TYPE_SPM` | SentencePiece (LLaMA) |
| 2 | `LLAMA_VOCAB_TYPE_BPE` | Byte-Pair Encoding (GPT-2) |
| 3 | `LLAMA_VOCAB_TYPE_WPM` | WordPiece (BERT) |
| 4 | `LLAMA_VOCAB_TYPE_UGM` | Unigram (T5) |
| 5 | `LLAMA_VOCAB_TYPE_RWKV` | RWKV tokenizer |
| 6 | `LLAMA_VOCAB_TYPE_PLAMO2` | PLaMo-2 tokenizer |

### llama_token_type / llama_token_attr

Token attributes (bitmask for attr):

| Bit | Flag | Description |
|-----|------|-------------|
| 0 | `LLAMA_TOKEN_ATTR_UNKNOWN` | Unknown token |
| 1 | `LLAMA_TOKEN_ATTR_UNUSED` | Unused token |
| 2 | `LLAMA_TOKEN_ATTR_NORMAL` | Normal token |
| 3 | `LLAMA_TOKEN_ATTR_CONTROL` | Control/special token |
| 4 | `LLAMA_TOKEN_ATTR_USER_DEFINED` | User-defined token |
| 5 | `LLAMA_TOKEN_ATTR_BYTE` | Byte token |
| 6 | `LLAMA_TOKEN_ATTR_NORMALIZED` | Normalized |
| 7 | `LLAMA_TOKEN_ATTR_LSTRIP` | Left-strip spaces |
| 8 | `LLAMA_TOKEN_ATTR_RSTRIP` | Right-strip spaces |
| 9 | `LLAMA_TOKEN_ATTR_SINGLE_WORD` | Single word token |

### llama_ftype (Model File Types)

Quantization types:

| Value | Name | Description |
|-------|------|-------------|
| 0 | `LLAMA_FTYPE_ALL_F32` | All F32 |
| 1 | `LLAMA_FTYPE_MOSTLY_F16` | Mostly F16 |
| 2-3 | Q4_0, Q4_1 | 4-bit quantization |
| 7 | `LLAMA_FTYPE_MOSTLY_Q8_0` | 8-bit quantization |
| 8-9 | Q5_0, Q5_1 | 5-bit quantization |
| 10-18 | Q2_K through Q6_K | K-quants |
| 19-31 | IQ2_XXS through IQ4_XS | IQ quants |
| 32 | `LLAMA_FTYPE_MOSTLY_BF16` | Mostly BF16 |
| 36-37 | TQ1_0, TQ2_0 | Ternary quants |
| 38 | `LLAMA_FTYPE_MOSTLY_MXFP4_MOE` | MXFP4 for MoE |
| 39 | `LLAMA_FTYPE_MOSTLY_NVFP4` | NVIDIA FP4 |
| 1024 | `LLAMA_FTYPE_GUESSED` | Not specified in file |

### llama_rope_scaling_type

| Value | Name | Description |
|-------|------|-------------|
| -1 | `LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED` | Unspecified |
| 0 | `LLAMA_ROPE_SCALING_TYPE_NONE` | No scaling |
| 1 | `LLAMA_ROPE_SCALING_TYPE_LINEAR` | Linear scaling |
| 2 | `LLAMA_ROPE_SCALING_TYPE_YARN` | YaRN scaling |
| 3 | `LLAMA_ROPE_SCALING_TYPE_LONGROPE` | LongRoPE scaling |

### llama_pooling_type

| Value | Name | Description |
|-------|------|-------------|
| -1 | `LLAMA_POOLING_TYPE_UNSPECIFIED` | Unspecified |
| 0 | `LLAMA_POOLING_TYPE_NONE` | No pooling |
| 1 | `LLAMA_POOLING_TYPE_MEAN` | Mean pooling |
| 2 | `LLAMA_POOLING_TYPE_CLS` | CLS token pooling |
| 3 | `LLAMA_POOLING_TYPE_LAST` | Last token pooling |
| 4 | `LLAMA_POOLING_TYPE_RANK` | Ranking (for reranking models) |

### llama_attention_type

| Value | Name | Description |
|-------|------|-------------|
| -1 | `LLAMA_ATTENTION_TYPE_UNSPECIFIED` | Unspecified |
| 0 | `LLAMA_ATTENTION_TYPE_CAUSAL` | Causal attention |
| 1 | `LLAMA_ATTENTION_TYPE_NON_CAUSAL` | Non-causal attention |

### llama_flash_attn_type

| Value | Name | Description |
|-------|------|-------------|
| -1 | `LLAMA_FLASH_ATTN_TYPE_AUTO` | Auto-detect |
| 0 | `LLAMA_FLASH_ATTN_TYPE_DISABLED` | Disabled |
| 1 | `LLAMA_FLASH_ATTN_TYPE_ENABLED` | Enabled |

### llama_split_mode

GPU model splitting:

| Value | Name | Description |
|-------|------|-------------|
| 0 | `LLAMA_SPLIT_MODE_NONE` | Single GPU |
| 1 | `LLAMA_SPLIT_MODE_LAYER` | Split layers across GPUs |
| 2 | `LLAMA_SPLIT_MODE_ROW` | Split rows across GPUs (tensor parallelism) |

### llama_model_kv_override_type

| Value | Name | Description |
|-------|------|-------------|
| 0 | `LLAMA_KV_OVERRIDE_TYPE_INT` | Integer override |
| 1 | `LLAMA_KV_OVERRIDE_TYPE_FLOAT` | Float override |
| 2 | `LLAMA_KV_OVERRIDE_TYPE_BOOL` | Boolean override |
| 3 | `LLAMA_KV_OVERRIDE_TYPE_STR` | String override |

### llama_model_meta_key

Sampling metadata keys:

| Value | Name |
|-------|------|
| 0 | `LLAMA_MODEL_META_KEY_SAMPLING_SEQUENCE` |
| 1 | `LLAMA_MODEL_META_KEY_SAMPLING_TOP_K` |
| 2 | `LLAMA_MODEL_META_KEY_SAMPLING_TOP_P` |
| 3 | `LLAMA_MODEL_META_KEY_SAMPLING_MIN_P` |
| 4 | `LLAMA_MODEL_META_KEY_SAMPLING_XTC_PROBABILITY` |
| 5 | `LLAMA_MODEL_META_KEY_SAMPLING_XTC_THRESHOLD` |
| 6 | `LLAMA_MODEL_META_KEY_SAMPLING_TEMP` |
| 7 | `LLAMA_MODEL_META_KEY_SAMPLING_PENALTY_LAST_N` |
| 8 | `LLAMA_MODEL_META_KEY_SAMPLING_PENALTY_REPEAT` |
| 9 | `LLAMA_MODEL_META_KEY_SAMPLING_MIROSTAT` |
| 10 | `LLAMA_MODEL_META_KEY_SAMPLING_MIROSTAT_TAU` |
| 11 | `LLAMA_MODEL_META_KEY_SAMPLING_MIROSTAT_ETA` |

### llama_params_fit_status

| Value | Name | Description |
|-------|------|-------------|
| 0 | `LLAMA_PARAMS_FIT_STATUS_SUCCESS` | Parameters fit |
| 1 | `LLAMA_PARAMS_FIT_STATUS_FAILURE` | Could not fit |
| 2 | `LLAMA_PARAMS_FIT_STATUS_ERROR` | Hard error |

---

## Parameter Structs

### llama_model_params

Model loading configuration:

```c
struct llama_model_params {
    ggml_backend_dev_t * devices;                    // NULL-terminated device list
    const struct llama_model_tensor_buft_override * tensor_buft_overrides;  // Buffer type overrides
    int32_t n_gpu_layers;                            // Layers to offload to GPU (-1 = all)
    enum llama_split_mode split_mode;                // GPU split mode
    int32_t main_gpu;                                // Main GPU when split_mode = NONE
    const float * tensor_split;                      // Per-GPU tensor split ratios
    llama_progress_callback progress_callback;       // Loading progress callback
    void * progress_callback_user_data;              // User data for callback
    const struct llama_model_kv_override * kv_overrides;  // KV metadata overrides
    bool vocab_only;                                 // Load only vocabulary
    bool use_mmap;                                   // Use memory mapping
    bool use_direct_io;                              // Use direct I/O
    bool use_mlock;                                  // Lock memory in RAM
    bool check_tensors;                              // Validate tensor data
    bool use_extra_bufts;                            // Use extra buffer types
    bool no_host;                                    // Bypass host buffer
    bool no_alloc;                                   // Only simulate allocation
};
```

**Key FFI fields**:
- `n_gpu_layers`: Integer, -1 means all layers to GPU
- `split_mode`: enum (0=none, 1=layer, 2=row)
- `main_gpu`: Integer, GPU index
- `vocab_only`: Boolean, for vocab-only loading
- `use_mmap`: Boolean, enable memory-mapped files
- `use_mlock`: Boolean, prevent swapping

### llama_context_params

Inference context configuration:

```c
struct llama_context_params {
    uint32_t n_ctx;                    // Text context (0 = from model)
    uint32_t n_batch;                  // Logical batch size for llama_decode
    uint32_t n_ubatch;                 // Physical batch size
    uint32_t n_seq_max;                // Max sequences
    int32_t  n_threads;                // Threads for generation
    int32_t  n_threads_batch;          // Threads for batch processing
    enum llama_rope_scaling_type rope_scaling_type;
    enum llama_pooling_type      pooling_type;
    enum llama_attention_type    attention_type;
    enum llama_flash_attn_type   flash_attn_type;
    float    rope_freq_base;           // RoPE base frequency (0 = from model)
    float    rope_freq_scale;          // RoPE frequency scaling (0 = from model)
    float    yarn_ext_factor;          // YaRN extrapolation mix
    float    yarn_attn_factor;         // YaRN magnitude scaling
    float    yarn_beta_fast;           // YaRN low correction
    float    yarn_beta_slow;           // YaRN high correction
    uint32_t yarn_orig_ctx;            // YaRN original context
    float    defrag_thold;             // [DEPRECATED] KV defrag threshold
    ggml_backend_sched_eval_callback cb_eval;
    void * cb_eval_user_data;
    enum ggml_type type_k;             // KV cache K type
    enum ggml_type type_v;             // KV cache V type
    ggml_abort_callback abort_callback;
    void * abort_callback_data;
    bool embeddings;                   // Extract embeddings
    bool offload_kqv;                  // Offload KQV to GPU
    bool no_perf;                      // Disable perf measurements
    bool op_offload;                   // Offload host ops to device
    bool swa_full;                     // Use full SWA cache
    bool kv_unified;                   // Unified KV cache across sequences
    struct llama_sampler_seq_config * samplers;  // [EXPERIMENTAL] Backend samplers
    size_t n_samplers;                 // Number of sampler configs
};
```

**Key FFI fields for UI**:
- `n_ctx`: Context window size (default: from model)
- `n_batch`: Batch size for prompt processing
- `n_ubatch`: Physical batch size (micro-batch)
- `n_threads`: CPU threads for generation
- `n_threads_batch`: CPU threads for prompt
- `type_k`/`type_v`: KV cache quantization (ggml_type enum)
- `embeddings`: Enable embedding extraction
- `offload_kqv`: Offload KV cache to GPU

### llama_model_quantize_params

```c
typedef struct llama_model_quantize_params {
    int32_t nthread;                       // Threads for quantization
    enum llama_ftype ftype;                // Target quantization type
    enum ggml_type output_tensor_type;     // Output tensor type override
    enum ggml_type token_embedding_type;   // Embedding tensor type override
    bool allow_requantize;                 // Allow requantizing
    bool quantize_output_tensor;           // Quantize output layer
    bool only_copy;                        // Only copy, no quantization
    bool pure;                             // Quantize all to default type
    bool keep_split;                       // Keep split shard count
    bool dry_run;                          // Calculate size only
    void * imatrix;                        // Importance matrix
    void * kv_overrides;                   // KV overrides
    void * tensor_types;                   // Tensor type overrides
    void * prune_layers;                   // Layer indices to prune
} llama_model_quantize_params;
```

### llama_sampler_chain_params

```c
typedef struct llama_sampler_chain_params {
    bool no_perf;  // Disable performance measurements
} llama_sampler_chain_params;
```

---

## Backend Initialization

### llama_backend_init

```c
LLAMA_API void llama_backend_init(void);
```

Initialize the llama + ggml backend. Call once at program start.

**Dart FFI**: Call this before any other llama function.

### llama_backend_free

```c
LLAMA_API void llama_backend_free(void);
```

Call once at program end. Currently only used for MPI.

### llama_numa_init

```c
LLAMA_API void llama_numa_init(enum ggml_numa_strategy numa);
```

Initialize NUMA optimizations. Optional, call after `llama_backend_init`.

**NUMA Strategies** (from ggml-cpu.h):
| Value | Name | Description |
|-------|------|-------------|
| 0 | `GGML_NUMA_STRATEGY_DISABLED` | Disabled |
| 1 | `GGML_NUMA_STRATEGY_DISTRIBUTE` | Distribute across NUMA nodes |
| 2 | `GGML_NUMA_STRATEGY_ISOLATE` | Isolate to NUMA nodes |
| 3 | `GGML_NUMA_STRATEGY_NUMACTL` | Use numactl |
| 4 | `GGML_NUMA_STRATEGY_MIRROR` | Mirror across nodes |

### llama_attach_threadpool

```c
LLAMA_API void llama_attach_threadpool(
    struct llama_context * ctx,
    ggml_threadpool_t threadpool,
    ggml_threadpool_t threadpool_batch);
```

Attach external thread pools to a context.

### llama_detach_threadpool

```c
LLAMA_API void llama_detach_threadpool(struct llama_context * ctx);
```

Detach external thread pools.

---

## Model Loading

### llama_model_default_params

```c
LLAMA_API struct llama_model_params llama_model_default_params(void);
```

Returns default model parameters. **Call this first** and modify fields as needed.

### llama_model_load_from_file

```c
LLAMA_API struct llama_model * llama_model_load_from_file(
    const char * path_model,
    struct llama_model_params params);
```

Load a model from a GGUF file. Returns NULL on failure.

**Dart FFI**: Returns `Pointer<llama_model>` or `nullptr`.

### llama_model_load_from_file_ptr

```c
LLAMA_API struct llama_model * llama_model_load_from_file_ptr(
    FILE * file,
    struct llama_model_params params);
```

Load from an already-opened FILE pointer.

### llama_model_load_from_splits

```c
LLAMA_API struct llama_model * llama_model_load_from_splits(
    const char ** paths,
    size_t n_paths,
    struct llama_model_params params);
```

Load from multiple split files. Paths must be in correct order.

### llama_model_free

```c
LLAMA_API void llama_model_free(struct llama_model * model);
```

Free a loaded model. **Must call** when done.

### llama_model_save_to_file

```c
LLAMA_API void llama_model_save_to_file(
    const struct llama_model * model,
    const char * path_model);
```

Save a model to a file.

### DEPRECATED: llama_load_model_from_file

```c
// Use llama_model_load_from_file instead
```

---

## Context Management

### llama_context_default_params

```c
LLAMA_API struct llama_context_params llama_context_default_params(void);
```

Returns default context parameters. **Call this first** and modify fields.

### llama_init_from_model

```c
LLAMA_API struct llama_context * llama_init_from_model(
    struct llama_model * model,
    struct llama_context_params params);
```

Create an inference context from a loaded model. Returns NULL on failure.

### llama_free

```c
LLAMA_API void llama_free(struct llama_context * ctx);
```

Free a context. Frees all allocated memory.

### DEPRECATED: llama_new_context_with_model

```c
// Use llama_init_from_model instead
```

### llama_get_model

```c
LLAMA_API const struct llama_model * llama_get_model(const struct llama_context * ctx);
```

Get the model associated with a context.

### llama_get_memory

```c
LLAMA_API llama_memory_t llama_get_memory(const struct llama_context * ctx);
```

Get the memory (KV cache) handle from a context.

### llama_pooling_type

```c
LLAMA_API enum llama_pooling_type llama_pooling_type(const struct llama_context * ctx);
```

Get the pooling type of a context.

### llama_params_fit

```c
LLAMA_API enum llama_params_fit_status llama_params_fit(
    const char * path_model,
    struct llama_model_params * mparams,
    struct llama_context_params * cparams,
    float * tensor_split,
    struct llama_model_tensor_buft_override * tensor_buft_overrides,
    size_t * margins,
    uint32_t n_ctx_min,
    enum ggml_log_level log_level);
```

Automatically fit model and context parameters to available device memory.

**Dart FFI**: Useful for "auto-configure" feature in UI.

---

## Model Information

### Query Functions

```c
LLAMA_API uint32_t llama_n_ctx      (const struct llama_context * ctx);
LLAMA_API uint32_t llama_n_ctx_seq  (const struct llama_context * ctx);
LLAMA_API uint32_t llama_n_batch    (const struct llama_context * ctx);
LLAMA_API uint32_t llama_n_ubatch   (const struct llama_context * ctx);
LLAMA_API uint32_t llama_n_seq_max  (const struct llama_context * ctx);
```

Get actual context parameters (may differ from requested).

### Model Dimensions

```c
LLAMA_API int32_t llama_model_n_ctx_train(const struct llama_model * model);  // Training context
LLAMA_API int32_t llama_model_n_embd     (const struct llama_model * model);  // Embedding size
LLAMA_API int32_t llama_model_n_embd_inp (const struct llama_model * model);  // Input embedding size
LLAMA_API int32_t llama_model_n_embd_out (const struct llama_model * model);  // Output embedding size
LLAMA_API int32_t llama_model_n_layer    (const struct llama_model * model);  // Number of layers
LLAMA_API int32_t llama_model_n_head     (const struct llama_model * model);  // Number of heads
LLAMA_API int32_t llama_model_n_head_kv  (const struct llama_model * model);  // KV heads (for GQA)
LLAMA_API int32_t llama_model_n_swa      (const struct llama_model * model);  // SWA window size
```

### Model Properties

```c
LLAMA_API float llama_model_rope_freq_scale_train(const struct llama_model * model);
LLAMA_API uint32_t llama_model_n_cls_out(const struct llama_model * model);
LLAMA_API const char * llama_model_cls_label(const struct llama_model * model, uint32_t i);
LLAMA_API enum llama_rope_type llama_model_rope_type(const struct llama_model * model);
```

### Model Metadata

```c
LLAMA_API int32_t llama_model_meta_val_str(const struct llama_model * model, const char * key, char * buf, size_t buf_size);
LLAMA_API int32_t llama_model_meta_count(const struct llama_model * model);
LLAMA_API const char * llama_model_meta_key_str(enum llama_model_meta_key key);
LLAMA_API int32_t llama_model_meta_key_by_index(const struct llama_model * model, int32_t i, char * buf, size_t buf_size);
LLAMA_API int32_t llama_model_meta_val_str_by_index(const struct llama_model * model, int32_t i, char * buf, size_t buf_size);
LLAMA_API int32_t llama_model_desc(const struct llama_model * model, char * buf, size_t buf_size);
```

### Model Size and Parameters

```c
LLAMA_API uint64_t llama_model_size(const struct llama_model * model);     // Total size in bytes
LLAMA_API uint64_t llama_model_n_params(const struct llama_model * model); // Total parameter count
```

### Model Type Detection

```c
LLAMA_API bool llama_model_has_encoder(const struct llama_model * model);
LLAMA_API bool llama_model_has_decoder(const struct llama_model * model);
LLAMA_API llama_token llama_model_decoder_start_token(const struct llama_model * model);
LLAMA_API bool llama_model_is_recurrent(const struct llama_model * model);  // Mamba, RWKV
LLAMA_API bool llama_model_is_hybrid(const struct llama_model * model);     // Jamba, Granite
LLAMA_API bool llama_model_is_diffusion(const struct llama_model * model);  // LLaDA, Dream
```

### Chat Template

```c
LLAMA_API const char * llama_model_chat_template(const struct llama_model * model, const char * name);
```

Get the default chat template. Returns NULL if not available.

### Vocabulary Access

```c
LLAMA_API const struct llama_vocab * llama_model_get_vocab(const struct llama_model * model);
```

### System Info

```c
LLAMA_API const char * llama_print_system_info(void);
```

Returns build info string (CPU features, backends, etc.).

### Capability Checks

```c
LLAMA_API size_t llama_max_devices(void);
LLAMA_API size_t llama_max_parallel_sequences(void);
LLAMA_API size_t llama_max_tensor_buft_overrides(void);
LLAMA_API bool llama_supports_mmap(void);
LLAMA_API bool llama_supports_mlock(void);
LLAMA_API bool llama_supports_gpu_offload(void);
LLAMA_API bool llama_supports_rpc(void);
LLAMA_API int64_t llama_time_us(void);
```

---

## Vocabulary API

### Token Properties

```c
LLAMA_API const char * llama_vocab_get_text(const struct llama_vocab * vocab, llama_token token);
LLAMA_API float llama_vocab_get_score(const struct llama_vocab * vocab, llama_token token);
LLAMA_API enum llama_token_attr llama_vocab_get_attr(const struct llama_vocab * vocab, llama_token token);
LLAMA_API bool llama_vocab_is_eog(const struct llama_vocab * vocab, llama_token token);  // End of generation
LLAMA_API bool llama_vocab_is_control(const struct llama_vocab * vocab, llama_token token);
```

### Special Tokens

```c
LLAMA_API llama_token llama_vocab_bos(const struct llama_vocab * vocab);  // Beginning of sentence
LLAMA_API llama_token llama_vocab_eos(const struct llama_vocab * vocab);  // End of sentence
LLAMA_API llama_token llama_vocab_eot(const struct llama_vocab * vocab);  // End of turn
LLAMA_API llama_token llama_vocab_sep(const struct llama_vocab * vocab);  // Separator
LLAMA_API llama_token llama_vocab_nl (const struct llama_vocab * vocab);  // Newline
LLAMA_API llama_token llama_vocab_pad(const struct llama_vocab * vocab);  // Padding
LLAMA_API llama_token llama_vocab_mask(const struct llama_vocab * vocab); // Mask
```

### FIM (Fill-In-Middle) Tokens

```c
LLAMA_API llama_token llama_vocab_fim_pre(const struct llama_vocab * vocab);
LLAMA_API llama_token llama_vocab_fim_suf(const struct llama_vocab * vocab);
LLAMA_API llama_token llama_vocab_fim_mid(const struct llama_vocab * vocab);
LLAMA_API llama_token llama_vocab_fim_pad(const struct llama_vocab * vocab);
LLAMA_API llama_token llama_vocab_fim_rep(const struct llama_vocab * vocab);
LLAMA_API llama_token llama_vocab_fim_sep(const struct llama_vocab * vocab);
```

### Vocab Settings

```c
LLAMA_API bool llama_vocab_get_add_bos(const struct llama_vocab * vocab);
LLAMA_API bool llama_vocab_get_add_eos(const struct llama_vocab * vocab);
LLAMA_API bool llama_vocab_get_add_sep(const struct llama_vocab * vocab);
LLAMA_API int32_t llama_vocab_n_tokens(const struct llama_vocab * vocab);
LLAMA_API enum llama_vocab_type llama_vocab_type(const struct llama_vocab * vocab);
```

---

## Tokenization

### llama_tokenize

```c
LLAMA_API int32_t llama_tokenize(
    const struct llama_vocab * vocab,
    const char * text,
    int32_t text_len,
    llama_token * tokens,
    int32_t n_tokens_max,
    bool add_special,
    bool parse_special);
```

Convert text to tokens.

**Parameters**:
- `text`: Input text (UTF-8)
- `text_len`: Length of text in bytes
- `tokens`: Output buffer (can be NULL to get required size)
- `n_tokens_max`: Max tokens to write
- `add_special`: Add BOS/EOS if model is configured to
- `parse_special`: Tokenize special/control tokens

**Returns**: Number of tokens on success, negative on failure (absolute value = needed size), `INT32_MIN` on overflow.

**Dart FFI Pattern**:
```dart
// First call to get size
int needed = llama_tokenize(vocab, text, textLen, nullptr, 0, true, false);
// Allocate buffer
final tokens = calloc<llama_token>(needed.abs());
// Second call to actually tokenize
int count = llama_tokenize(vocab, text, textLen, tokens, needed.abs(), true, false);
```

### llama_token_to_piece

```c
LLAMA_API int32_t llama_token_to_piece(
    const struct llama_vocab * vocab,
    llama_token token,
    char * buf,
    int32_t length,
    int32_t lstrip,
    bool special);
```

Convert a token to its text piece.

**Parameters**:
- `lstrip`: Skip leading spaces (useful for BPE)
- `special`: Include special token markers

**Returns**: Number of bytes written, negative on failure (absolute value = needed size).

### llama_detokenize

```c
LLAMA_API int32_t llama_detokenize(
    const struct llama_vocab * vocab,
    const llama_token * tokens,
    int32_t n_tokens,
    char * text,
    int32_t text_len_max,
    bool remove_special,
    bool unparse_special);
```

Convert tokens back to text (inverse of tokenize).

**Returns**: Number of chars written, negative on failure.

---

## Batch Processing

### llama_batch_init

```c
LLAMA_API struct llama_batch llama_batch_init(
    int32_t n_tokens,
    int32_t embd,
    int32_t n_seq_max);
```

Allocate a batch. Returns by value (struct).

**Parameters**:
- `n_tokens`: Max tokens
- `embd`: If non-zero, allocate embedding buffer of size `n_tokens * embd * sizeof(float)`
- `n_seq_max`: Max sequence IDs per token

### llama_batch_free

```c
LLAMA_API void llama_batch_free(struct llama_batch batch);
```

Free a batch allocated with `llama_batch_init`.

### llama_batch_get_one

```c
LLAMA_API struct llama_batch llama_batch_get_one(
    llama_token * tokens,
    int32_t n_tokens);
```

Helper to create a batch for a single sequence. Position tracking is automatic.

**Dart FFI**: Avoid using - this is a transitional helper.

---

## Inference (Decode/Encode)

### llama_decode

```c
LLAMA_API int32_t llama_decode(
    struct llama_context * ctx,
    struct llama_batch batch);
```

Process a batch through the decoder (uses KV cache).

**Return values**:
| Value | Meaning |
|-------|---------|
| 0 | Success |
| 1 | KV cache full (reduce batch or increase context) |
| 2 | Aborted |
| -1 | Invalid input batch |
| < -1 | Fatal error |

### llama_encode

```c
LLAMA_API int32_t llama_encode(
    struct llama_context * ctx,
    struct llama_batch batch);
```

Process a batch through the encoder (no KV cache). For encoder-decoder models.

**Returns**: 0 = success, < 0 = error.

### llama_synchronize

```c
LLAMA_API void llama_synchronize(struct llama_context * ctx);
```

Wait for all computations to finish. Called automatically by output functions.

### Thread Control

```c
LLAMA_API void llama_set_n_threads(struct llama_context * ctx, int32_t n_threads, int32_t n_threads_batch);
LLAMA_API int32_t llama_n_threads(struct llama_context * ctx);
LLAMA_API int32_t llama_n_threads_batch(struct llama_context * ctx);
```

### Context Settings

```c
LLAMA_API void llama_set_embeddings(struct llama_context * ctx, bool embeddings);
LLAMA_API void llama_set_causal_attn(struct llama_context * ctx, bool causal_attn);
LLAMA_API void llama_set_warmup(struct llama_context * ctx, bool warmup);
LLAMA_API void llama_set_abort_callback(struct llama_context * ctx, ggml_abort_callback abort_callback, void * abort_callback_data);
```

### Getting Output

```c
LLAMA_API float * llama_get_logits(struct llama_context * ctx);
LLAMA_API float * llama_get_logits_ith(struct llama_context * ctx, int32_t i);
LLAMA_API float * llama_get_embeddings(struct llama_context * ctx);
LLAMA_API float * llama_get_embeddings_ith(struct llama_context * ctx, int32_t i);
LLAMA_API float * llama_get_embeddings_seq(struct llama_context * ctx, llama_seq_id seq_id);
```

**Dart FFI Pattern for logits**:
```dart
// Get logits for last token (index -1)
Pointer<Float> logitsPtr = llama_get_logits_ith(ctx, -1);
// Read n_vocab floats
final nVocab = llama_model_n_vocab(model);
final logits = logitsPtr.asTypedList(nVocab);
```

### Backend Sampling API [EXPERIMENTAL]

```c
LLAMA_API llama_token llama_get_sampled_token_ith(struct llama_context * ctx, int32_t i);
LLAMA_API float *  llama_get_sampled_probs_ith      (struct llama_context * ctx, int32_t i);
LLAMA_API uint32_t llama_get_sampled_probs_count_ith(struct llama_context * ctx, int32_t i);
LLAMA_API float *  llama_get_sampled_logits_ith      (struct llama_context * ctx, int32_t i);
LLAMA_API uint32_t llama_get_sampled_logits_count_ith(struct llama_context * ctx, int32_t i);
LLAMA_API llama_token * llama_get_sampled_candidates_ith      (struct llama_context * ctx, int32_t i);
LLAMA_API uint32_t      llama_get_sampled_candidates_count_ith(struct llama_context * ctx, int32_t i);
```

---

## Sampling API

### Sampler Interface (for custom samplers)

```c
struct llama_sampler_i {
    const char *           (*name)  (const struct llama_sampler * smpl);
    void                   (*accept)(struct llama_sampler * smpl, llama_token token);
    void                   (*apply) (struct llama_sampler * smpl, llama_token_data_array * cur_p);
    void                   (*reset) (struct llama_sampler * smpl);
    struct llama_sampler * (*clone) (const struct llama_sampler * smpl);
    void                   (*free)  (struct llama_sampler * smpl);
    // Backend sampling interface (experimental)...
};
```

### Sampler Chain

```c
LLAMA_API struct llama_sampler_chain_params llama_sampler_chain_default_params(void);
LLAMA_API struct llama_sampler * llama_sampler_chain_init(struct llama_sampler_chain_params params);
LLAMA_API void llama_sampler_chain_add(struct llama_sampler * chain, struct llama_sampler * smpl);
LLAMA_API struct llama_sampler * llama_sampler_chain_get(struct llama_sampler * chain, int32_t i);
LLAMA_API int llama_sampler_chain_n(const struct llama_sampler * chain);
LLAMA_API struct llama_sampler * llama_sampler_chain_remove(struct llama_sampler * chain, int32_t i);
```

### Generic Sampler Functions

```c
LLAMA_API struct llama_sampler * llama_sampler_init(struct llama_sampler_i * iface, llama_sampler_context_t ctx);
LLAMA_API const char * llama_sampler_name(const struct llama_sampler * smpl);
LLAMA_API void llama_sampler_accept(struct llama_sampler * smpl, llama_token token);
LLAMA_API void llama_sampler_apply(struct llama_sampler * smpl, llama_token_data_array * cur_p);
LLAMA_API void llama_sampler_reset(struct llama_sampler * smpl);
LLAMA_API struct llama_sampler * llama_sampler_clone(const struct llama_sampler * smpl);
LLAMA_API void llama_sampler_free(struct llama_sampler * smpl);
LLAMA_API llama_token llama_sampler_sample(struct llama_sampler * smpl, struct llama_context * ctx, int32_t idx);
LLAMA_API uint32_t llama_sampler_get_seed(const struct llama_sampler * smpl);
LLAMA_API bool llama_set_sampler(struct llama_context * ctx, llama_seq_id seq_id, struct llama_sampler * smpl);
```

### Built-in Samplers

| Function | Description | Parameters |
|----------|-------------|------------|
| `llama_sampler_init_greedy()` | Always pick highest probability | None |
| `llama_sampler_init_dist(seed)` | Categorical sampling | `seed`: Random seed |
| `llama_sampler_init_top_k(k)` | Top-K filtering | `k`: Keep top K tokens (≤0 = noop) |
| `llama_sampler_init_top_p(p, min_keep)` | Nucleus sampling | `p`: Cumulative probability threshold, `min_keep`: Min tokens |
| `llama_sampler_init_min_p(p, min_keep)` | Minimum P sampling | `p`: Minimum probability threshold |
| `llama_sampler_init_typical(p, min_keep)` | Locally Typical Sampling | `p`: Typical probability |
| `llama_sampler_init_temp(t)` | Temperature scaling | `t`: Temperature (≤0 = max only) |
| `llama_sampler_init_temp_ext(t, delta, exponent)` | Dynamic temperature | `t`, `delta`, `exponent` |
| `llama_sampler_init_xtc(p, t, min_keep, seed)` | XTC sampler | `p`, `t`, `min_keep`, `seed` |
| `llama_sampler_init_top_n_sigma(n)` | Top-n-sigma sampling | `n`: Sigma threshold |
| `llama_sampler_init_mirostat(n_vocab, seed, tau, eta, m)` | Mirostat 1.0 | See params below |
| `llama_sampler_init_mirostat_v2(seed, tau, eta)` | Mirostat 2.0 | See params below |
| `llama_sampler_init_grammar(vocab, grammar_str, grammar_root)` | Grammar-constrained | Grammar string + root rule |
| `llama_sampler_init_grammar_lazy_patterns(...)` | Lazy grammar | Trigger patterns/tokens |
| `llama_sampler_init_penalties(penalty_last_n, penalty_repeat, penalty_freq, penalty_present)` | Repetition penalty | See params below |
| `llama_sampler_init_dry(...)` | DRY sampler | Multiple params |
| `llama_sampler_init_adaptive_p(target, decay, seed)` | Adaptive-P sampling | `target`, `decay`, `seed` |
| `llama_sampler_init_logit_bias(n_vocab, n_logit_bias, logit_bias)` | Logit bias | Bias array |
| `llama_sampler_init_infill(vocab)` | Fill-in-the-middle | Vocabulary |

**Mirostat Parameters**:
- `tau`: Target cross-entropy (surprise)
- `eta`: Learning rate for mu updates
- `m`: Number of tokens for s_hat calculation (typically 100)

**Penalty Parameters**:
- `penalty_last_n`: Last N tokens to penalize (0 = disabled, -1 = context size)
- `penalty_repeat`: Repeat penalty (1.0 = disabled)
- `penalty_freq`: Frequency penalty (0.0 = disabled)
- `penalty_present`: Presence penalty (0.0 = disabled)

---

## KV Cache / Memory Management

### Memory Operations

```c
LLAMA_API void llama_memory_clear(llama_memory_t mem, bool data);
LLAMA_API bool llama_memory_seq_rm(llama_memory_t mem, llama_seq_id seq_id, llama_pos p0, llama_pos p1);
LLAMA_API void llama_memory_seq_cp(llama_memory_t mem, llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1);
LLAMA_API void llama_memory_seq_keep(llama_memory_t mem, llama_seq_id seq_id);
LLAMA_API void llama_memory_seq_add(llama_memory_t mem, llama_seq_id seq_id, llama_pos p0, llama_pos p1, llama_pos delta);
LLAMA_API void llama_memory_seq_div(llama_memory_t mem, llama_seq_id seq_id, llama_pos p0, llama_pos p1, int d);
LLAMA_API llama_pos llama_memory_seq_pos_min(llama_memory_t mem, llama_seq_id seq_id);
LLAMA_API llama_pos llama_memory_seq_pos_max(llama_memory_t mem, llama_seq_id seq_id);
LLAMA_API bool llama_memory_can_shift(llama_memory_t mem);
```

**Usage**:
- `llama_memory_seq_rm`: Remove tokens from a sequence (for context shifting)
- `llama_memory_seq_cp`: Copy tokens between sequences
- `llama_memory_seq_keep`: Keep only one sequence, remove others
- `llama_memory_seq_add`: Shift positions (for sliding window)
- `llama_memory_seq_div`: Divide positions by factor

---

## State / Sessions

### Full State Save/Load

```c
LLAMA_API size_t llama_state_get_size(struct llama_context * ctx);
LLAMA_API size_t llama_state_get_data(struct llama_context * ctx, uint8_t * dst, size_t size);
LLAMA_API size_t llama_state_set_data(struct llama_context * ctx, const uint8_t * src, size_t size);
```

### Session File Save/Load

```c
LLAMA_API bool llama_state_load_file(
    struct llama_context * ctx,
    const char * path_session,
    llama_token * tokens_out,
    size_t n_token_capacity,
    size_t * n_token_count_out);

LLAMA_API bool llama_state_save_file(
    struct llama_context * ctx,
    const char * path_session,
    const llama_token * tokens,
    size_t n_token_count);
```

### Single Sequence State

```c
LLAMA_API size_t llama_state_seq_get_size(struct llama_context * ctx, llama_seq_id seq_id);
LLAMA_API size_t llama_state_seq_get_data(struct llama_context * ctx, uint8_t * dst, size_t size, llama_seq_id seq_id);
LLAMA_API size_t llama_state_seq_set_data(struct llama_context * ctx, const uint8_t * src, size_t size, llama_seq_id dest_seq_id);
LLAMA_API size_t llama_state_seq_save_file(struct llama_context * ctx, const char * filepath, llama_seq_id seq_id, const llama_token * tokens, size_t n_token_count);
LLAMA_API size_t llama_state_seq_load_file(struct llama_context * ctx, const char * filepath, llama_seq_id dest_seq_id, llama_token * tokens_out, size_t n_token_capacity, size_t * n_token_count_out);
```

### Extended State Operations (with flags)

```c
typedef uint32_t llama_state_seq_flags;
#define LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY 1  // Work with partial states (SWA, recurrent)

LLAMA_API size_t llama_state_seq_get_size_ext(struct llama_context * ctx, llama_seq_id seq_id, llama_state_seq_flags flags);
LLAMA_API size_t llama_state_seq_get_data_ext(struct llama_context * ctx, uint8_t * dst, size_t size, llama_seq_id seq_id, llama_state_seq_flags flags);
LLAMA_API size_t llama_state_seq_set_data_ext(struct llama_context * ctx, const uint8_t * src, size_t size, llama_seq_id dest_seq_id, llama_state_seq_flags flags);
```

---

## LoRA Adapters

### Load/Free

```c
LLAMA_API struct llama_adapter_lora * llama_adapter_lora_init(
    struct llama_model * model,
    const char * path_lora);

LLAMA_API void llama_adapter_lora_free(struct llama_adapter_lora * adapter);
```

### Adapter Metadata

```c
LLAMA_API int32_t llama_adapter_meta_val_str(const struct llama_adapter_lora * adapter, const char * key, char * buf, size_t buf_size);
LLAMA_API int32_t llama_adapter_meta_count(const struct llama_adapter_lora * adapter);
LLAMA_API int32_t llama_adapter_meta_key_by_index(const struct llama_adapter_lora * adapter, int32_t i, char * buf, size_t buf_size);
LLAMA_API int32_t llama_adapter_meta_val_str_by_index(const struct llama_adapter_lora * adapter, int32_t i, char * buf, size_t buf_size);
```

### Alora (Adaptive LoRA)

```c
LLAMA_API uint64_t llama_adapter_get_alora_n_invocation_tokens(const struct llama_adapter_lora * adapter);
LLAMA_API const llama_token * llama_adapter_get_alora_invocation_tokens(const struct llama_adapter_lora * adapter);
```

### Apply Adapters

```c
LLAMA_API int32_t llama_set_adapters_lora(
    struct llama_context * ctx,
    struct llama_adapter_lora ** adapters,
    size_t n_adapters,
    float * scales);
```

### Control Vector

```c
LLAMA_API int32_t llama_set_adapter_cvec(
    struct llama_context * ctx,
    const float * data,
    size_t len,
    int32_t n_embd,
    int32_t il_start,
    int32_t il_end);
```

---

## Chat Templates

### Apply Template

```c
LLAMA_API int32_t llama_chat_apply_template(
    const char * tmpl,
    const struct llama_chat_message * chat,
    size_t n_msg,
    bool add_ass,
    char * buf,
    int32_t length);
```

**Parameters**:
- `tmpl`: Jinja template string (NULL for model default)
- `chat`: Array of chat messages
- `n_msg`: Number of messages
- `add_ass`: End with assistant token
- `buf`: Output buffer
- `length`: Buffer size

**Returns**: Total bytes needed. If > length, reallocate and retry.

### Built-in Templates

```c
LLAMA_API int32_t llama_chat_builtin_templates(const char ** output, size_t len);
```

Get list of built-in template names.

---

## Performance Utilities

### Context Performance

```c
struct llama_perf_context_data {
    double t_start_ms;   // Absolute start time
    double t_load_ms;    // Model load time
    double t_p_eval_ms;  // Prompt processing time
    double t_eval_ms;    // Token generation time
    int32_t n_p_eval;    // Prompt tokens processed
    int32_t n_eval;      // Generated tokens
    int32_t n_reused;    // Reused compute graphs
};

LLAMA_API struct llama_perf_context_data llama_perf_context(const struct llama_context * ctx);
LLAMA_API void llama_perf_context_print(const struct llama_context * ctx);
LLAMA_API void llama_perf_context_reset(struct llama_context * ctx);
```

### Sampler Performance

```c
struct llama_perf_sampler_data {
    double t_sample_ms;  // Sampling time
    int32_t n_sample;    // Number of samples
};

LLAMA_API struct llama_perf_sampler_data llama_perf_sampler(const struct llama_sampler * chain);
LLAMA_API void llama_perf_sampler_print(const struct llama_sampler * chain);
LLAMA_API void llama_perf_sampler_reset(struct llama_sampler * chain);
```

### Memory Breakdown

```c
LLAMA_API void llama_memory_breakdown_print(const struct llama_context * ctx);
```

---

## Logging

```c
LLAMA_API void llama_log_get(ggml_log_callback * log_callback, void ** user_data);
LLAMA_API void llama_log_set(ggml_log_callback log_callback, void * user_data);
```

**Callback type** (from ggml.h):
```c
typedef void (*ggml_log_callback)(enum ggml_log_level level, const char * text, void * user_data);
```

**Log levels**:
| Value | Name |
|-------|------|
| 0 | `GGML_LOG_LEVEL_NONE` |
| 1 | `GGML_LOG_LEVEL_DEBUG` |
| 2 | `GGML_LOG_LEVEL_INFO` |
| 3 | `GGML_LOG_LEVEL_WARN` |
| 4 | `GGML_LOG_LEVEL_ERROR` |
| 5 | `GGML_LOG_LEVEL_CONT` |

---

## Model Quantization

```c
LLAMA_API uint32_t llama_model_quantize(
    const char * fname_inp,
    const char * fname_out,
    const llama_model_quantize_params * params);
```

Quantize a model file. Returns 0 on success.

---

## Training API

```c
typedef bool (*llama_opt_param_filter)(const struct ggml_tensor * tensor, void * userdata);

LLAMA_API bool llama_opt_param_filter_all(const struct ggml_tensor * tensor, void * userdata);

struct llama_opt_params {
    uint32_t n_ctx_train;
    llama_opt_param_filter param_filter;
    void * param_filter_ud;
    ggml_opt_get_optimizer_params get_opt_pars;
    void * get_opt_pars_ud;
    enum ggml_opt_optimizer_type optimizer_type;
};

LLAMA_API void llama_opt_init(struct llama_context * lctx, struct llama_model * model, struct llama_opt_params lopt_params);
LLAMA_API void llama_opt_epoch(
    struct llama_context * lctx,
    ggml_opt_dataset_t dataset,
    ggml_opt_result_t result_train,
    ggml_opt_result_t result_eval,
    int64_t idata_split,
    ggml_opt_epoch_callback callback_train,
    ggml_opt_epoch_callback callback_eval);
```

---

## Model Split Helpers

```c
LLAMA_API int32_t llama_split_path(char * split_path, size_t maxlen, const char * path_prefix, int32_t split_no, int32_t split_count);
LLAMA_API int32_t llama_split_prefix(char * split_prefix, size_t maxlen, const char * split_path, int32_t split_no, int32_t split_count);
```

---

## Dart FFI Notes

### Type Mappings

| C Type | Dart FFI Type |
|--------|---------------|
| `int32_t` | `Int32` |
| `uint32_t` | `Uint32` |
| `int64_t` | `Int64` |
| `uint64_t` | `Uint64` |
| `size_t` | `Size` (or `Uint64` on 64-bit) |
| `float` | `Float` |
| `double` | `Double` |
| `bool` | `Bool` |
| `char *` | `Pointer<Utf8>` (package:ffi) |
| `void *` | `Pointer<Void>` |
| `llama_model *` | `Pointer<llama_model>` (Opaque) |
| `llama_context *` | `Pointer<llama_context>` (Opaque) |
| `llama_token` | `Int32` |
| `llama_seq_id` | `Int32` |
| `llama_pos` | `Int32` |
| `enum` | `Int32` |

### Struct Handling

For structs passed by value (like `llama_batch`, `llama_model_params`):
- Use `Struct` subclasses in Dart
- For return-by-value structs, Dart FFI supports returning them directly
- For pointer-to-struct params, use `Pointer<T>`

### Memory Management Pattern

```dart
// 1. Initialize
llamaBackendInit();

// 2. Load model
final modelParams = llamaModelDefaultParams();
final model = llamaModelLoadFromFile(path, modelParams);
if (model == nullptr) throw Exception('Failed to load model');

try {
  // 3. Create context
  final ctxParams = llamaContextDefaultParams();
  final ctx = llamaInitFromModel(model, ctxParams);
  
  try {
    // 4. Use context...
  } finally {
    llamaFree(ctx);
  }
} finally {
  llamaModelFree(model);
}

// 5. Cleanup
llamaBackendFree();
```

### Callback Handling

For callbacks (progress, logging, abort):
- Use `NativeCallable` in Dart FFI
- Keep references alive for the duration of use
- Clean up with `.close()` when done
