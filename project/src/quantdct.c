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

#define DCT_SCALE_BITS 8
#define ISQRT2 0.70710678118654f

typedef int16_t quant16_t;
typedef int32_t quant32_t;
typedef int64_t quant64_t;

static quant16_t precalcIdct_q16[MACROBLOCK_SIZE][MACROBLOCK_SIZE][MACROBLOCK_SIZE][MACROBLOCK_SIZE];
static quant16_t precalcDct_q16[MACROBLOCK_SIZE][MACROBLOCK_SIZE][MACROBLOCK_SIZE][MACROBLOCK_SIZE];

void precompute_dctlookup_values() {
  quant16_t dctlookup_q16[MACROBLOCK_SIZE][MACROBLOCK_SIZE];

  for (int u = 0; u < MACROBLOCK_SIZE; u++) {
    for (int v = 0; v < MACROBLOCK_SIZE; v++) {
      dctlookup_q16[u][v] = (quant16_t)roundf(dctlookup[u][v] * (1 << DCT_SCALE_BITS));
    }
  }

  for (int y = 0; y < MACROBLOCK_SIZE; y++) {
    for (int v = 0; v < MACROBLOCK_SIZE; v++) {
      for (int u = 0; u < MACROBLOCK_SIZE; u++) {
        for (int x = 0; x < MACROBLOCK_SIZE; x++) {
          quant16_t cxu = dctlookup_q16[x][u];
          quant16_t cyv = dctlookup_q16[y][v];

          // q16 * q16 = q32 → shift down to q16
          quant16_t dct_coeff_q16 = ((quant32_t)cxu * cyv) >> DCT_SCALE_BITS;
          precalcDct_q16[y][v][u][x] = dct_coeff_q16;

          quant16_t cux = dctlookup_q16[u][x];
          quant16_t cvy = dctlookup_q16[v][y];

          // q16 * q16 = q32 → shift down to q16
          quant16_t idct_coeff_q16 = ((quant32_t)cux * cvy) >> DCT_SCALE_BITS;
          precalcIdct_q16[y][v][u][x] = idct_coeff_q16;
        }
      }
    }
  }
}

static void dct_2d(const quant32_t *in, quant32_t *out)
{
  startTrace8("dct 2d");
  for (int v = 0; v < MACROBLOCK_SIZE; v++) {
    for (int u = 0; u < MACROBLOCK_SIZE; u++) {
      quant64_t dct_q64 = 0;

      for (int y = 0; y < MACROBLOCK_SIZE; y++) {
        for (int x = 0; x < MACROBLOCK_SIZE; x++) {
          quant32_t pixel_q32 = in[y*MACROBLOCK_SIZE + x];
          quant16_t coeff_q16 = precalcDct_q16[y][v][u][x];

          // q32 * q16 = q48, accumulate in q64
          dct_q64 += (quant64_t)pixel_q32 * coeff_q16;
        }
      }
      out[v * MACROBLOCK_SIZE + u] = MIN(dct_q64, INT32_MAX);
    }
  }
  endTrace();
}

static void idct_2d(const quant32_t *in, quant32_t *out)
{
  startTrace8("idct 2d");
  for (int v = 0; v < MACROBLOCK_SIZE; v++) {
    for (int u = 0; u < MACROBLOCK_SIZE; u++) {
      quant64_t idct_q64 = 0;

      for (int y = 0; y < MACROBLOCK_SIZE; y++) {
        for (int x = 0; x < MACROBLOCK_SIZE; x++) {
          quant32_t pixel_q32 = in[y*MACROBLOCK_SIZE + x];
          quant16_t coeff_q16 = precalcIdct_q16[y][v][u][x];

          // q32 * q16 = q48, accumulate in q64
          idct_q64 += (quant64_t)pixel_q32 * coeff_q16;
        }
      }
      out[v * MACROBLOCK_SIZE + u] = MIN(idct_q64, INT32_MAX);
    }
  }
  endTrace();
}

static void scale_block(const quant32_t *in, quant32_t *out)
{
  startTrace8("scaleblk");
  int u, v;

  for (v = 0; v < MACROBLOCK_SIZE; ++v) {
    for (u = 0; u < MACROBLOCK_SIZE; ++u) {
      quant32_t pixel_q32 = in[v * MACROBLOCK_SIZE + u];

      float a1 = !u ? ISQRT2 : 1.0f;
      float a2 = !v ? ISQRT2 : 1.0f;
      float scale = a1*a2;
      quant16_t scale_q16 = (quant16_t)roundf(scale * (1 << DCT_SCALE_BITS));

      // q32 * q32 = q64 → store in q64, scale down to q32
      quant64_t scaled_q64 = (quant64_t)pixel_q32 * scale_q16;
      quant32_t scaled_q32 = scaled_q64 >> DCT_SCALE_BITS;

      out[v * MACROBLOCK_SIZE + u] = scaled_q32;
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
  quant32_t mb_q32[MACROBLOCK_SIZE * MACROBLOCK_SIZE];
  quant32_t mb2_q32[MACROBLOCK_SIZE * MACROBLOCK_SIZE] = {0};

  for (int i = 0; i < MACROBLOCK_SIZE * MACROBLOCK_SIZE; i++) {
    mb[i] = in[i];
    mb_q32[i] = (quant32_t)roundf(in[i] * (1 << DCT_SCALE_BITS));
  }

  dct_2d(mb_q32, mb2_q32);
  scale_block(mb2_q32, mb_q32);

  for (int i = 0; i < MACROBLOCK_SIZE * MACROBLOCK_SIZE; i++) {
    mb[i] = (float32_t)mb_q32[i] / (float)(1 << (2 * DCT_SCALE_BITS));
  }

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
  quant32_t mb_q32[MACROBLOCK_SIZE * MACROBLOCK_SIZE];
  quant32_t mb2_q32[MACROBLOCK_SIZE * MACROBLOCK_SIZE] = {0};

  for (int i = 0; i < MACROBLOCK_SIZE * MACROBLOCK_SIZE; i++) {
    mb[i] = in[i];
  }

  dequantize_block(mb, mb2, quant_tbl);

  for (int i = 0; i < MACROBLOCK_SIZE * MACROBLOCK_SIZE; i++) {
    float32_t pixel = (float32_t) mb2[i];
    mb2_q32[i] = (quant32_t)roundf(pixel * (1 << DCT_SCALE_BITS));
  }

  scale_block(mb2_q32, mb_q32);
  idct_2d(mb_q32, mb2_q32);

  for (int i = 0; i < MACROBLOCK_SIZE * MACROBLOCK_SIZE; i++) {
    mb2[i] = (float32_t)mb2_q32[i] / (float)(1 << (2 * DCT_SCALE_BITS));
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
