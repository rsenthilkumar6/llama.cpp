#define GGML_COMMON_IMPL_C
#include "ggml-common.h"

#include "ggml-turboq.h"
#include "ggml-turboq-tables.h"
#include "ggml-quants.h"
#include "ggml-impl.h"
#include "ggml.h"

#include <math.h>
#include <string.h>
#include <assert.h>
#include <stdlib.h>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

#if defined(__GNUC__) || defined(__clang__)
#define TURBOQ_TLS __thread
#elif defined(_MSC_VER)
#define TURBOQ_TLS __declspec(thread)
#elif defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L && !defined(__STDC_NO_THREADS__)
#define TURBOQ_TLS _Thread_local
#else
#define TURBOQ_TLS
#endif

#define TURBOQ_KV_DIM 128

static inline uint64_t splitmix64_next(uint64_t * state) {
    uint64_t z = (*state += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

static void turboq_generate_gaussian(float * out, int64_t n, uint64_t seed) {
    uint64_t state = seed;
    int64_t i = 0;
    for (; i + 1 < n; i += 2) {
        double u1 = ((double)(splitmix64_next(&state) >> 11) + 0.5) / (double)(1ULL << 53);
        double u2 = ((double)(splitmix64_next(&state) >> 11) + 0.5) / (double)(1ULL << 53);
        double r  = sqrt(-2.0 * log(u1));
        double th = 2.0 * 3.14159265358979323846 * u2;
        out[i]     = (float)(r * cos(th));
        out[i + 1] = (float)(r * sin(th));
    }
    if (i < n) {
        double u1 = ((double)(splitmix64_next(&state) >> 11) + 0.5) / (double)(1ULL << 53);
        double u2 = ((double)(splitmix64_next(&state) >> 11) + 0.5) / (double)(1ULL << 53);
        double r  = sqrt(-2.0 * log(u1));
        double th = 2.0 * 3.14159265358979323846 * u2;
        out[i] = (float)(r * cos(th));
    }
}

static void turboq_householder_qr(float * A, float * Q_out, int64_t d) {
    float * tau = (float *)malloc(d * sizeof(float));
    float * r_sign = (float *)malloc(d * sizeof(float));

    for (int64_t k = 0; k < d; k++) {
        float norm_sq = 0.0f;
        for (int64_t i = k; i < d; i++) {
            float val = A[i + k * d];
            norm_sq += val * val;
        }
        float norm = sqrtf(norm_sq);

        float alpha = A[k + k * d];
        float sign_alpha = (alpha >= 0.0f) ? 1.0f : -1.0f;
        float u1 = alpha + sign_alpha * norm;

        r_sign[k] = -sign_alpha;

        float vtv = u1 * u1 + (norm_sq - alpha * alpha);
        if (vtv < 1e-30f) {
            tau[k] = 0.0f;
            continue;
        }
        tau[k] = 2.0f / vtv;

        A[k + k * d] = u1;

        for (int64_t j = k + 1; j < d; j++) {
            float dot = 0.0f;
            dot += u1 * A[k + j * d];
            for (int64_t i = k + 1; i < d; i++) {
                dot += A[i + k * d] * A[i + j * d];
            }
            dot *= tau[k];
            A[k + j * d] -= dot * u1;
            for (int64_t i = k + 1; i < d; i++) {
                A[i + j * d] -= dot * A[i + k * d];
            }
        }
    }

    memset(Q_out, 0, d * d * sizeof(float));
    for (int64_t i = 0; i < d; i++) {
        Q_out[i + i * d] = 1.0f;
    }

    for (int64_t k = d - 1; k >= 0; k--) {
        if (tau[k] == 0.0f) continue;
        float u1 = A[k + k * d];
        for (int64_t j = 0; j < d; j++) {
            float dot = 0.0f;
            dot += u1 * Q_out[k + j * d];
            for (int64_t i = k + 1; i < d; i++) {
                dot += A[i + k * d] * Q_out[i + j * d];
            }
            dot *= tau[k];
            Q_out[k + j * d] -= dot * u1;
            for (int64_t i = k + 1; i < d; i++) {
                Q_out[i + j * d] -= dot * A[i + k * d];
            }
        }
    }

    for (int64_t j = 0; j < d; j++) {
        if (r_sign[j] < 0.0f) {
            for (int64_t i = 0; i < d; i++) {
                Q_out[i + j * d] = -Q_out[i + j * d];
            }
        }
    }

    free(tau);
    free(r_sign);
}

static TURBOQ_TLS float * tl_Q = NULL;
static TURBOQ_TLS float * tl_Q_row = NULL;
static TURBOQ_TLS int64_t tl_Q_dim = 0;
static TURBOQ_TLS uint64_t tl_Q_seed = 0;

static const float * turboq_get_rotation(int64_t d, uint64_t seed) {
    if (tl_Q != NULL && tl_Q_dim == d && tl_Q_seed == seed) {
        return tl_Q;
    }
    free(tl_Q);
    free(tl_Q_row);
    tl_Q = (float *)malloc(d * d * sizeof(float));
    tl_Q_row = (float *)malloc(d * d * sizeof(float));
    tl_Q_dim = d;
    tl_Q_seed = seed;

    float * A = (float *)malloc(d * d * sizeof(float));
    turboq_generate_gaussian(A, d * d, seed);
    turboq_householder_qr(A, tl_Q, d);

    for (int64_t i = 0; i < d; ++i) {
        for (int64_t j = 0; j < d; ++j) {
            tl_Q_row[i * d + j] = tl_Q[i + j * d];
        }
    }

    free(A);
    return tl_Q;
}

static const float * turboq_get_rotation_row(int64_t d, uint64_t seed) {
    turboq_get_rotation(d, seed);
    return tl_Q_row;
}

static TURBOQ_TLS float * tl_S = NULL;
static TURBOQ_TLS float * tl_S_row = NULL;
static TURBOQ_TLS int64_t tl_S_dim = 0;
static TURBOQ_TLS uint64_t tl_S_seed = 0;

static const float * turboq_get_projection(int64_t d, uint64_t seed) {
    uint64_t s_seed = seed ^ 0x1234567890abcdefULL;
    if (tl_S != NULL && tl_S_dim == d && tl_S_seed == s_seed) {
        return tl_S;
    }
    free(tl_S);
    free(tl_S_row);
    tl_S = (float *)malloc(d * d * sizeof(float));
    tl_S_row = (float *)malloc(d * d * sizeof(float));
    tl_S_dim = d;
    tl_S_seed = s_seed;

    turboq_generate_gaussian(tl_S, d * d, s_seed);

    for (int64_t i = 0; i < d; ++i) {
        for (int64_t j = 0; j < d; ++j) {
            tl_S_row[i * d + j] = tl_S[i + j * d];
        }
    }

    return tl_S;
}

static const float * turboq_get_projection_row(int64_t d, uint64_t seed) {
    turboq_get_projection(d, seed);
    return tl_S_row;
}

static void matvec(const float * A, const float * x, float * y, int64_t n, int64_t m) {
    for (int64_t i = 0; i < n; ++i) {
        float sum = 0.0f;
        for (int64_t j = 0; j < m; ++j) {
            sum += A[i + j * n] * x[j];
        }
        y[i] = sum;
    }
}

static void matvec_row(const float * A, const float * x, float * y, int64_t n, int64_t m) {
    for (int64_t i = 0; i < n; ++i) {
        float sum = 0.0f;
        for (int64_t j = 0; j < m; ++j) {
            sum += A[i * m + j] * x[j];
        }
        y[i] = sum;
    }
}

static void matvec_t(const float * A, const float * x, float * y, int64_t n, int64_t m) {
    for (int64_t j = 0; j < m; ++j) {
        float sum = 0.0f;
        for (int64_t i = 0; i < n; ++i) {
            sum += A[i + j * n] * x[i];
        }
        y[j] = sum;
    }
}

uint64_t turboq_seed_from_row(int64_t row_idx) {
    return (uint64_t)(row_idx) * 0x9e3779b97f4a7c15ULL;
}

void turboq_rotate_forward(float * y, const float * x, int64_t d, uint64_t seed) {
    const float * Q = turboq_get_rotation(d, seed);
    matvec(Q, x, y, d, d);
}

void turboq_rotate_inverse(float * x, const float * y, int64_t d, uint64_t seed) {
    const float * Q = turboq_get_rotation_row(d, seed);
    matvec_row(Q, y, x, d, d);
}

static void rotate_block_forward(float * y, const float * x, int64_t d, int64_t row_idx) {
    uint64_t seed = turboq_seed_from_row(row_idx);
    turboq_rotate_forward(y, x, d, seed);
}

static void rotate_block_inverse(float * x, const float * y, int64_t d, int64_t row_idx) {
    uint64_t seed = turboq_seed_from_row(row_idx);
    turboq_rotate_inverse(x, y, d, seed);
}

static void project_block_forward(float * y, const float * x, int64_t d, int64_t row_idx) {
    uint64_t seed = turboq_seed_from_row(row_idx);
    const float * S = turboq_get_projection(d, seed);
    matvec(S, x, y, d, d);
}

static void project_block_inverse(float * x, const float * y, int64_t d, int64_t row_idx) {
    uint64_t seed = turboq_seed_from_row(row_idx);
    const float * S = turboq_get_projection_row(d, seed);
    matvec_row(S, y, x, d, d);
}

static inline float scale_up(float x) {
    return x / (float)TURBOQ_KV_DIM;
}

static inline float scale_down(float x) {
    return x * (float)TURBOQ_KV_DIM;
}

// Scratch buffer for TurboQuant operations
static TURBOQ_TLS float * tl_scratch = NULL;
static TURBOQ_TLS int64_t tl_scratch_size = 0;

static float * turboq_get_scratch(int64_t n) {
    if (tl_scratch_size < n) {
        free(tl_scratch);
        tl_scratch = (float *)malloc(n * sizeof(float));
        tl_scratch_size = n;
    }
    return tl_scratch;
}

// ============== Quantization helpers ==============

static inline int quantize_2bit_scalar(float x) {
    if (x < turboq_boundaries_2bit[0]) return 0;
    if (x < turboq_boundaries_2bit[1]) return 1;
    if (x < turboq_boundaries_2bit[2]) return 2;
    return 3;
}

static inline int quantize_3bit_scalar(float x) {
    if (x < turboq_boundaries_3bit[0]) return 0;
    if (x < turboq_boundaries_3bit[1]) return 1;
    if (x < turboq_boundaries_3bit[2]) return 2;
    if (x < turboq_boundaries_3bit[3]) return 3;
    if (x < turboq_boundaries_3bit[4]) return 4;
    if (x < turboq_boundaries_3bit[5]) return 5;
    if (x < turboq_boundaries_3bit[6]) return 6;
    return 7;
}

static inline int quantize_4bit_scalar(float x) {
    if (x < turboq_boundaries_4bit[0]) return 0;
    if (x < turboq_boundaries_4bit[1]) return 1;
    if (x < turboq_boundaries_4bit[2]) return 2;
    if (x < turboq_boundaries_4bit[3]) return 3;
    if (x < turboq_boundaries_4bit[4]) return 4;
    if (x < turboq_boundaries_4bit[5]) return 5;
    if (x < turboq_boundaries_4bit[6]) return 6;
    if (x < turboq_boundaries_4bit[7]) return 7;
    if (x < turboq_boundaries_4bit[8]) return 8;
    if (x < turboq_boundaries_4bit[9]) return 9;
    if (x < turboq_boundaries_4bit[10]) return 10;
    if (x < turboq_boundaries_4bit[11]) return 11;
    if (x < turboq_boundaries_4bit[12]) return 12;
    if (x < turboq_boundaries_4bit[13]) return 13;
    if (x < turboq_boundaries_4bit[14]) return 14;
    return 15;
}

// ============== 3-bit pack/unpack helpers ==============

// Pack 8x 3-bit values into 3 bytes (24 bits)
static inline void pack_3bit(uint8_t * dst, const int * src) {
    dst[0] = (uint8_t)( src[0]        | (src[1] << 3) | ((src[2] & 0x3) << 6));
    dst[1] = (uint8_t)((src[2] >> 2)  | (src[3] << 1) | (src[4] << 4) | ((src[5] & 0x1) << 7));
    dst[2] = (uint8_t)((src[5] >> 1)  | (src[6] << 2) | (src[7] << 5));
}

static inline void unpack_3bit(int * dst, const uint8_t * src) {
    dst[0] =  src[0]        & 0x7;
    dst[1] = (src[0] >> 3)  & 0x7;
    dst[2] = (src[0] >> 6)  | ((src[1] & 0x1) << 2);
    dst[3] = (src[1] >> 1)  & 0x7;
    dst[4] = (src[1] >> 4)  & 0x7;
    dst[5] = (src[1] >> 7)  | ((src[2] & 0x3) << 1);
    dst[6] = (src[2] >> 2)  & 0x7;
    dst[7] = (src[2] >> 5)  & 0x7;
}

// ============== TBQ3_0 (3.0625 bpw TurboQuant) ==============

// Rotate, project, quantize + scale
void quantize_row_tbq3_0_ref(const float * GGML_RESTRICT x, block_tbq3_0 * GGML_RESTRICT y, int64_t k) {
    const int64_t nb = k / QK_K;
    float * rotated = turboq_get_scratch(QK_K);

    for (int64_t i = 0; i < nb; ++i) {
        const float * row = x + i * QK_K;
        block_tbq3_0 * block = y + i;

        // Find max absolute value for scale
        float max_abs = 0.0f;
        for (int j = 0; j < QK_K; ++j) {
            float abs_val = fabsf(row[j]);
            if (abs_val > max_abs) max_abs = abs_val;
        }
        float scale = max_abs / 3.5f;
        if (scale < 1e-10f) scale = 1.0f;

        // Scale, rotate, project
        float scaled[QK_K];
        for (int j = 0; j < QK_K; ++j) {
            scaled[j] = row[j] / scale;
        }

        // Rotate
        rotate_block_forward(rotated, scaled, TURBOQ_KV_DIM, i);

        // Project
        float projected[TURBOQ_KV_DIM];
        project_block_forward(projected, rotated, TURBOQ_KV_DIM, i);

        // Quantize to 3-bit
        int q[QK_K];
        for (int j = 0; j < QK_K; ++j) {
            q[j] = quantize_3bit_scalar(projected[j]);
        }

        // Pack
        for (int j = 0; j < QK_K; j += 8) {
            pack_3bit(block->qs + j * 3 / 8, q + j);
        }

        block->d = GGML_FP32_TO_FP16(scale_up(scale));
    }
}

static float dequantize_tbq3_0_value(const block_tbq3_0 * block, int j) {
    int idx[8];
    int base = (j / 8) * 3;
    unpack_3bit(idx, block->qs + base);
    return turboq_codebook_3bit[idx[j % 8]];
}

void dequantize_row_tbq3_0(const block_tbq3_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    const int64_t nb = k / QK_K;
    float * rotated = turboq_get_scratch(QK_K);

    for (int64_t i = 0; i < nb; ++i) {
        const block_tbq3_0 * block = x + i;
        float * row = y + i * QK_K;
        float scale = scale_down(GGML_FP16_TO_FP32(block->d));

        // Dequantize codebook values
        float projected[QK_K];
        for (int j = 0; j < QK_K; ++j) {
            projected[j] = dequantize_tbq3_0_value(block, j);
        }

        // Inverse project
        project_block_inverse(rotated, projected, TURBOQ_KV_DIM, i);

        // Inverse rotate
        rotate_block_inverse(row, rotated, TURBOQ_KV_DIM, i);

        // Scale back
        for (int j = 0; j < QK_K; ++j) {
            row[j] *= scale;
        }
    }
}

size_t quantize_tbq3_0(const float * src, void * dst, int64_t nrows, int64_t n_per_row, const float * imatrix) {
    (void)imatrix;
    const int64_t nb = n_per_row / QK_K;
    for (int64_t i = 0; i < nrows; ++i) {
        quantize_row_tbq3_0_ref(src + i * n_per_row, (block_tbq3_0 *)dst + i * nb, n_per_row);
    }
    return nrows * nb * sizeof(block_tbq3_0);
}

// ============== TBQ4_0 (4.0625 bpw TurboQuant) ==============

void quantize_row_tbq4_0_ref(const float * GGML_RESTRICT x, block_tbq4_0 * GGML_RESTRICT y, int64_t k) {
    const int64_t nb = k / QK_K;
    float * rotated = turboq_get_scratch(QK_K);

    for (int64_t i = 0; i < nb; ++i) {
        const float * row = x + i * QK_K;
        block_tbq4_0 * block = y + i;

        float max_abs = 0.0f;
        for (int j = 0; j < QK_K; ++j) {
            float abs_val = fabsf(row[j]);
            if (abs_val > max_abs) max_abs = abs_val;
        }
        float scale = max_abs / 7.5f;
        if (scale < 1e-10f) scale = 1.0f;

        float scaled[QK_K];
        for (int j = 0; j < QK_K; ++j) {
            scaled[j] = row[j] / scale;
        }

        rotate_block_forward(rotated, scaled, TURBOQ_KV_DIM, i);

        float projected[QK_K];
        project_block_forward(projected, rotated, TURBOQ_KV_DIM, i);

        // Quantize to 4-bit and pack (2 per byte)
        for (int j = 0; j < QK_K; j += 2) {
            int q0 = quantize_4bit_scalar(projected[j]);
            int q1 = quantize_4bit_scalar(projected[j + 1]);
            block->qs[j / 2] = (uint8_t)(q0 | (q1 << 4));
        }

        block->d = GGML_FP32_TO_FP16(scale_up(scale));
    }
}

void dequantize_row_tbq4_0(const block_tbq4_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    const int64_t nb = k / QK_K;
    float * rotated = turboq_get_scratch(QK_K);

    for (int64_t i = 0; i < nb; ++i) {
        const block_tbq4_0 * block = x + i;
        float * row = y + i * QK_K;
        float scale = scale_down(GGML_FP16_TO_FP32(block->d));

        float projected[QK_K];
        for (int j = 0; j < QK_K; j += 2) {
            uint8_t byte = block->qs[j / 2];
            projected[j]     = turboq_codebook_4bit[byte & 0xf];
            projected[j + 1] = turboq_codebook_4bit[byte >> 4];
        }

        project_block_inverse(rotated, projected, TURBOQ_KV_DIM, i);
        rotate_block_inverse(row, rotated, TURBOQ_KV_DIM, i);

        for (int j = 0; j < QK_K; ++j) {
            row[j] *= scale;
        }
    }
}

size_t quantize_tbq4_0(const float * src, void * dst, int64_t nrows, int64_t n_per_row, const float * imatrix) {
    (void)imatrix;
    const int64_t nb = n_per_row / QK_K;
    for (int64_t i = 0; i < nrows; ++i) {
        quantize_row_tbq4_0_ref(src + i * n_per_row, (block_tbq4_0 *)dst + i * nb, n_per_row);
    }
    return nrows * nb * sizeof(block_tbq4_0);
}

// ============== Vec dot (dequantize-then-dot) ==============

void ggml_vec_dot_tbq3_0_q8_K_generic(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc) {
    assert(nrc == 1);
    (void)nrc;
    (void)bx;
    (void)by;
    (void)bs;

    const block_tbq3_0 * GGML_RESTRICT x = vx;
    const block_q8_K  * GGML_RESTRICT y = vy;

    const int nb = n / QK_K;
    float * deq = (float *)malloc(QK_K * sizeof(float));
    float sumf = 0.0f;

    for (int i = 0; i < nb; ++i) {
        dequantize_row_tbq3_0(x + i, deq, QK_K);
        const float d = y[i].d;
        const int8_t * q8 = y[i].qs;
        float sumi = 0.0f;
        for (int j = 0; j < QK_K; ++j) {
            sumi += deq[j] * q8[j];
        }
        sumf += sumi * d;
    }

    *s = sumf;
    free(deq);
}

void ggml_vec_dot_tbq4_0_q8_K_generic(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc) {
    assert(nrc == 1);
    (void)nrc;
    (void)bx;
    (void)by;
    (void)bs;

    const block_tbq4_0 * GGML_RESTRICT x = vx;
    const block_q8_K  * GGML_RESTRICT y = vy;

    const int nb = n / QK_K;
    float * deq = (float *)malloc(QK_K * sizeof(float));
    float sumf = 0.0f;

    for (int i = 0; i < nb; ++i) {
        dequantize_row_tbq4_0(x + i, deq, QK_K);
        const float d = y[i].d;
        const int8_t * q8 = y[i].qs;
        float sumi = 0.0f;
        for (int j = 0; j < QK_K; ++j) {
            sumi += deq[j] * q8[j];
        }
        sumf += sumi * d;
    }

    *s = sumf;
    free(deq);
}
