#include <assert.h>
#include <errno.h>
#include <getopt.h>
#include <inttypes.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <arm_neon.h>

#include "common.h"
#include "tables.h"

#include "profiling.h"

#define ISQRT2 0.70710678118654f

static float32x4_t precalcIdct[MACROBLOCK_SIZE][MACROBLOCK_SIZE][MACROBLOCK_SIZE][2];
static float32x4_t precalcDct[MACROBLOCK_SIZE][MACROBLOCK_SIZE][MACROBLOCK_SIZE][2];

void precompute_dctlookup_values() {
  for (int v = 0; v < MACROBLOCK_SIZE; v++) {
    for (int u = 0; u < MACROBLOCK_SIZE; u++) {
      for (int y = 0; y < MACROBLOCK_SIZE; y++) {
        // idct
        float32x4_t idct_y_vec = vdupq_n_f32(dctlookup[v][y]);
        precalcIdct[y][u][v][0] = vmulq_f32(*((float32x4_t *) dctlookup[u]), idct_y_vec);
        precalcIdct[y][u][v][1] = vmulq_f32(*((float32x4_t *) &dctlookup[u][4]), idct_y_vec);

        // dct (notice the transpose)
        float32x4_t dct_y_vec = vdupq_n_f32(dctlookup[y][v]);
        float32x4_t dct_x_vec1 = {dctlookup[0][u], dctlookup[1][u], dctlookup[2][u], dctlookup[3][u]};
        float32x4_t dct_x_vec2 = {dctlookup[4][u], dctlookup[5][u], dctlookup[6][u], dctlookup[7][u]};

        precalcDct[y][u][v][0] = vmulq_f32(dct_x_vec1, dct_y_vec);
        precalcDct[y][u][v][1] = vmulq_f32(dct_x_vec2, dct_y_vec);
      }
    }
  }
}

static void dct_2d(const float *in, float *out)
{
  startTrace8("dct 2d");
  memset(out, 0, MACROBLOCK_SIZE*MACROBLOCK_SIZE*sizeof(float));

  for (int y = 0; y < MACROBLOCK_SIZE; y++) {
    float32x4_t in_vec1 = vld1q_f32((const float32_t *)&in[y * MACROBLOCK_SIZE]);
    float32x4_t in_vec2 = vld1q_f32((const float32_t *)&in[y * MACROBLOCK_SIZE + 4]);

    for (int v = 0; v < MACROBLOCK_SIZE; v++) {
      for (int u = 0; u < MACROBLOCK_SIZE; u++) {
        float32x4_t dct = { 0.0f, 0.0f, 0.0f, 0.0f };

        dct = vmlaq_f32(dct, in_vec1, precalcDct[y][u][v][0]);
        dct = vmlaq_f32(dct, in_vec2, precalcDct[y][u][v][1]);

        out[v * MACROBLOCK_SIZE + u] += vaddvq_f32(dct);
      }
    }
  }
  endTrace();
}


static void idct_2d(const float *in, float *out)
{
  startTrace8("idct 2d");
  for (int v = 0; v < MACROBLOCK_SIZE; v++) {
    for (int u = 0; u < MACROBLOCK_SIZE; u++) {
      float dct = 0.0f;

      for (int y = 0; y < MACROBLOCK_SIZE; y++) {

        float32x4_t in_vec1 = vld1q_f32(&in[y * MACROBLOCK_SIZE]);
        float32x4_t in_vec2 = vld1q_f32(&in[y * MACROBLOCK_SIZE + 4]);

        float32x4_t mul_sum1 = vmulq_f32(in_vec1, precalcIdct[y][u][v][0]);
        float32x4_t mul_sum2 = vmulq_f32(in_vec2, precalcIdct[y][u][v][1]);

        float32x4_t mul_sum = vaddq_f32(mul_sum1, mul_sum2);
        dct += vaddvq_f32(mul_sum);
      }

      out[v * MACROBLOCK_SIZE + u] = dct;
    }
  }
  endTrace();
}

static void scale_block(const float32_t *in, float32_t *out)
{
  startTrace8("scaleblk");
  int u, v;

  for (v = 0; v < MACROBLOCK_SIZE; ++v) {
    for (u = 0; u < MACROBLOCK_SIZE; ++u) {
      float pixel = in[v * MACROBLOCK_SIZE + u];

      float a1 = !u ? ISQRT2 : 1.0f;
      float a2 = !v ? ISQRT2 : 1.0f;
      float scale = a1*a2;

      float scaled = pixel * scale;
      out[v * MACROBLOCK_SIZE + u] = scaled;
    }
  }
  endTrace();
}

static void quantize_block(const float *in, float *out, const uint8_t *quant_tbl)
{
  startTrace8("quant blk");
  int zigzag;

  for (zigzag = 0; zigzag < 64; ++zigzag) {
    uint8_t u = zigzag_U[zigzag];
    uint8_t v = zigzag_V[zigzag];

    float dct = in[v * 8 + u];

    /* Zig-zag and quantize */
    out[zigzag] = (float)round((dct / 4.0) / quant_tbl[zigzag]);
  }
  endTrace();
}

static void dequantize_block(const float *in, float *out, const uint8_t *quant_tbl)
{
  startTrace8("deq blk");
  int zigzag;

  for (zigzag = 0; zigzag < 64; ++zigzag) {
    uint8_t u = zigzag_U[zigzag];
    uint8_t v = zigzag_V[zigzag];

    float dct = in[zigzag];

    /* Zig-zag and de-quantize */
    out[v * 8 + u] = (float)round((dct * quant_tbl[zigzag]) / 4.0);
  }
  endTrace();
}

static void dct_quant_block_8x8(const int16_t *in, int16_t *out, const uint8_t *quant_tbl)
{
  startTrace7("quant 8x8");

  float mb[MACROBLOCK_SIZE * MACROBLOCK_SIZE] __attribute((aligned(16)));
  float mb2[MACROBLOCK_SIZE * MACROBLOCK_SIZE] __attribute((aligned(16)));

  for (int i = 0; i < MACROBLOCK_SIZE * MACROBLOCK_SIZE; i++) {
    mb[i] = in[i];
  }

  dct_2d(mb, mb2);
  scale_block(mb2, mb);
  quantize_block(mb, mb2, quant_tbl);

  for (int i = 0; i < MACROBLOCK_SIZE * MACROBLOCK_SIZE; i++) {
    out[i] = mb2[i];
  }

  endTrace();
}

static void dequant_idct_block_8x8(const int16_t *in, int16_t *out, const uint8_t *quant_tbl)
{
  startTrace7("deq 8x8");

  float mb[MACROBLOCK_SIZE * MACROBLOCK_SIZE] __attribute((aligned(16)));
  float mb2[MACROBLOCK_SIZE * MACROBLOCK_SIZE] __attribute((aligned(16)));

  for (int i = 0; i < MACROBLOCK_SIZE * MACROBLOCK_SIZE; i++) {
    mb[i] = in[i];
  }

  dequantize_block(mb, mb2, quant_tbl);
  scale_block(mb2, mb);
  idct_2d(mb, mb2);

  for (int i = 0; i < MACROBLOCK_SIZE * MACROBLOCK_SIZE; i++) {
    out[i] = mb2[i];
  }

  endTrace();
}

static void dequantize_idct_row(const int16_t *in, const uint8_t *prediction, int w, int h, int y, uint8_t *out, const uint8_t *quant_tbl)
{
  startTrace6("deq row");
  int x;

  int16_t block[MACROBLOCK_SIZE * MACROBLOCK_SIZE];

  /* Perform the dequantization and iDCT */
  for (x = 0; x < w; x += MACROBLOCK_SIZE) {
    int i, j;

    dequant_idct_block_8x8(in + (x * MACROBLOCK_SIZE), block, quant_tbl);

    for (i = 0; i < MACROBLOCK_SIZE; ++i) {
      for (j = 0; j < MACROBLOCK_SIZE; ++j) {
        /* Add prediction block. Note: DCT is not precise -
           Clamp to legal values */
        int16_t tmp = block[i * MACROBLOCK_SIZE + j] + (int16_t)prediction[i * w + j + x];

        if (tmp < 0) {
          tmp = 0;
        } else if (tmp > 255) {
          tmp = 255;
        }

        out[i * w + j + x] = tmp;
      }
    }
  }
  endTrace();
}

static void dct_quantize_row(const uint8_t *in, uint8_t *prediction, int w, int h, int16_t *out, const uint8_t *quant_tbl)
{
  startTrace6("quant row");
  int x;

  int16_t block[MACROBLOCK_SIZE * MACROBLOCK_SIZE];

  /* Perform the DCT and quantization */
  for (x = 0; x < w; x += MACROBLOCK_SIZE) {
    int i, j;

    for (i = 0; i < MACROBLOCK_SIZE; ++i) {
      for (j = 0; j < MACROBLOCK_SIZE; ++j) {
        block[i * MACROBLOCK_SIZE + j] =
            ((int16_t)in[i * w + j + x] - prediction[i * w + j + x]);
      }
    }

    /* Store MBs linear in memory, i.e. the 64 coefficients are stored
       continous. This allows us to ignore stride in DCT/iDCT and other
       functions. */
    dct_quant_block_8x8(block, out + (x * MACROBLOCK_SIZE), quant_tbl);
  }
  endTrace();
}

void dequantize_idct(const int16_t *in, uint8_t *prediction, uint32_t width, uint32_t height, uint8_t *out, const uint8_t *quant_tbl)
{
  int y;

  for (y = 0; y < height; y += MACROBLOCK_SIZE) {
    dequantize_idct_row(in + y * width, prediction + y * width, width,
                        height, y, out + y * width, quant_tbl);
  }
}


void dct_quantize(const uint8_t *in, uint8_t *prediction, uint32_t width, uint32_t height, int16_t *out, const uint8_t *quant_tbl)
{
  int y;

  for (y = 0; y < height; y += MACROBLOCK_SIZE) {
    dct_quantize_row(in + y * width, prediction + y * width, width, height,
                     out + y * width, quant_tbl);
  }
}
