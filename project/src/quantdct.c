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

static float16x8_t precalcIdct[MACROBLOCK_SIZE][MACROBLOCK_SIZE][MACROBLOCK_SIZE];
static float16x8_t precalcDct[MACROBLOCK_SIZE][MACROBLOCK_SIZE][MACROBLOCK_SIZE];

void precompute_dctlookup_values() {
  startTrace3("precompute dctlookup");
  for (int v = 0; v < MACROBLOCK_SIZE; v++) {
    for (int u = 0; u < MACROBLOCK_SIZE; u++) {
      for (int y = 0; y < MACROBLOCK_SIZE; y++) {
        // idct
        float32x4_t idct_y_vec = vdupq_n_f32(dctlookup[v][y]);
        float16x4_t lower_idct = vcvt_f16_f32(vmulq_f32(*((float32x4_t *) dctlookup[u]), idct_y_vec));
        float32x4_t upper_idct =              vmulq_f32(*((float32x4_t *) &dctlookup[u][4]), idct_y_vec);

        precalcIdct[v][u][y] = vcvt_high_f16_f32(lower_idct, upper_idct);

        // dct (notice the transpose)
        float32x4_t dct_y_vec = vdupq_n_f32(dctlookup[y][v]);
        float32x4_t dct_x_vec1 = {dctlookup[0][u], dctlookup[1][u], dctlookup[2][u], dctlookup[3][u]};
        float32x4_t dct_x_vec2 = {dctlookup[4][u], dctlookup[5][u], dctlookup[6][u], dctlookup[7][u]};

        float16x4_t lower_dct = vcvt_f16_f32(vmulq_f32(dct_x_vec1, dct_y_vec));
        float32x4_t upper_dct =              vmulq_f32(dct_x_vec2, dct_y_vec);

        precalcDct[v][u][y] = vcvt_high_f16_f32(lower_dct, upper_dct);
      }
    }
  }
  endTrace();
}

static void dct_2d(const float *in, float *out)
{
  startTrace8("dct 2d");
  for (int v = 0; v < MACROBLOCK_SIZE; v++) {
    for (int u = 0; u < MACROBLOCK_SIZE; u++) {
      float16x8_t dct = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };

      for (int y = 0; y < MACROBLOCK_SIZE; y++) {

        float16x4_t in_vec_low = vcvt_f16_f32(vld1q_f32((const float32_t *)&in[y * MACROBLOCK_SIZE]));
        float16x8_t in_vec = vcvt_high_f16_f32(in_vec_low, vld1q_f32((const float32_t *)&in[y * MACROBLOCK_SIZE + 4]));

        dct = vaddq_f16(dct,vmulq_f16(in_vec, precalcDct[v][u][y])); //  dct += (in * precalc)
      }

      out[v * MACROBLOCK_SIZE + u] = vaddvq_f32(vcvt_high_f32_f16(dct)) +
                                     vaddvq_f32(vcvt_f32_f16(*(float16x4_t*)&dct));
      }
    }
  endTrace();
}


static void idct_2d(const float *in, float *out)
{
  startTrace8("idct 2d");
  for (int v = 0; v < MACROBLOCK_SIZE; v++) {
    for (int u = 0; u < MACROBLOCK_SIZE; u++) {
      float16x8_t idct = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };

      for (int y = 0; y < MACROBLOCK_SIZE; y++) {

        float16x4_t in_vec_low = vcvt_f16_f32(vld1q_f32((const float32_t *)&in[y * MACROBLOCK_SIZE]));
        float16x8_t in_vec = vcvt_high_f16_f32(in_vec_low, vld1q_f32((const float32_t *)&in[y * MACROBLOCK_SIZE + 4]));

        idct = vaddq_f16(idct,vmulq_f16(in_vec, precalcIdct[v][u][y])); //  idct += (in * precalc)
      }

      out[v * MACROBLOCK_SIZE + u] = vaddvq_f32(vcvt_high_f32_f16(idct)) +
                                     vaddvq_f32(vcvt_f32_f16(*(float16x4_t*)&idct));
    }
    }

  endTrace();
}

static void scale_block(float *in_data, float *out_data)
{
  startTrace8("scaleblk");
  int u, v;

  for (v = 0; v < MACROBLOCK_SIZE; ++v) {
    for (u = 0; u < MACROBLOCK_SIZE; ++u) {
      float a1 = !u ? ISQRT2 : 1.0f;
      float a2 = !v ? ISQRT2 : 1.0f;

      /* Scale according to normalizing function */
      out_data[v * MACROBLOCK_SIZE + u] =
          in_data[v * MACROBLOCK_SIZE + u] * a1 * a2;
    }
  }
  endTrace();
}

static void quantize_block(float *in_data, float *out_data, uint8_t *quant_tbl)
{
  startTrace8("quant blk");
  int zigzag;

  for (zigzag = 0; zigzag < 64; ++zigzag) {
    uint8_t u = zigzag_U[zigzag];
    uint8_t v = zigzag_V[zigzag];

    float dct = in_data[v * 8 + u];

    /* Zig-zag and quantize */
    out_data[zigzag] = (float)round((dct / 4.0) / quant_tbl[zigzag]);
  }
  endTrace();
}

static void dequantize_block(float *in_data, float *out_data, uint8_t *quant_tbl)
{
  startTrace8("deq blk");
  int zigzag;

  for (zigzag = 0; zigzag < 64; ++zigzag) {
    uint8_t u = zigzag_U[zigzag];
    uint8_t v = zigzag_V[zigzag];

    float dct = in_data[zigzag];

    /* Zig-zag and de-quantize */
    out_data[v * 8 + u] = (float)round((dct * quant_tbl[zigzag]) / 4.0);
  }
  endTrace();
}

static void dct_quant_block_8x8(int16_t *in_data, int16_t *out_data, uint8_t *quant_tbl)
{
  startTrace7("quant 8x8");

  float mb[MACROBLOCK_SIZE * MACROBLOCK_SIZE] __attribute((aligned(16)));
  float mb2[MACROBLOCK_SIZE * MACROBLOCK_SIZE] __attribute((aligned(16)));

  for (int i = 0; i < MACROBLOCK_SIZE * MACROBLOCK_SIZE; i++) {
    mb[i] = in_data[i];
  }

  dct_2d(mb, mb2);
  scale_block(mb2, mb);
  quantize_block(mb, mb2, quant_tbl);

  for (int i = 0; i < MACROBLOCK_SIZE * MACROBLOCK_SIZE; i++) {
    out_data[i] = mb2[i];
  }

  endTrace();
}

static void dequant_idct_block_8x8(int16_t *in_data, int16_t *out_data, uint8_t *quant_tbl)
{
  startTrace7("deq 8x8");

  float mb[MACROBLOCK_SIZE * MACROBLOCK_SIZE] __attribute((aligned(16)));
  float mb2[MACROBLOCK_SIZE * MACROBLOCK_SIZE] __attribute((aligned(16)));

  for (int i = 0; i < MACROBLOCK_SIZE * MACROBLOCK_SIZE; i++) {
    mb[i] = in_data[i];
  }

  dequantize_block(mb, mb2, quant_tbl);
  scale_block(mb2, mb);
  idct_2d(mb, mb2);

  for (int i = 0; i < MACROBLOCK_SIZE * MACROBLOCK_SIZE; i++) {
    out_data[i] = mb2[i];
  }

  endTrace();
}

static void dequantize_idct_row(int16_t *in_data, uint8_t *prediction, int w, int h, int y, uint8_t *out_data, uint8_t *quantization)
{
  startTrace6("deq row");
  int x;

  int16_t block[MACROBLOCK_SIZE * MACROBLOCK_SIZE];

  /* Perform the dequantization and iDCT */
  for (x = 0; x < w; x += MACROBLOCK_SIZE) {
    int i, j;

    dequant_idct_block_8x8(in_data + (x * MACROBLOCK_SIZE), block, quantization);

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

        out_data[i * w + j + x] = tmp;
      }
    }
  }
  endTrace();
}

static void dct_quantize_row(uint8_t *in_data, uint8_t *prediction, int w, int h, int16_t *out_data, uint8_t *quantization)
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
            ((int16_t)in_data[i * w + j + x] - prediction[i * w + j + x]);
      }
    }

    /* Store MBs linear in memory, i.e. the 64 coefficients are stored
       continous. This allows us to ignore stride in DCT/iDCT and other
       functions. */
    dct_quant_block_8x8(block, out_data + (x * MACROBLOCK_SIZE), quantization);
  }
  endTrace();
}

void dequantize_idct(int16_t *in_data, uint8_t *prediction, uint32_t width, uint32_t height, uint8_t *out_data, uint8_t *quantization)
{
  int y;

  for (y = 0; y < height; y += MACROBLOCK_SIZE) {
    dequantize_idct_row(in_data + y * width, prediction + y * width, width,
                        height, y, out_data + y * width, quantization);
  }
}


void dct_quantize(uint8_t *in_data, uint8_t *prediction, uint32_t width,
                  uint32_t height, int16_t *out_data, uint8_t *quantization) {
  int y;

  for (y = 0; y < height; y += MACROBLOCK_SIZE) {
    dct_quantize_row(in_data + y * width, prediction + y * width, width, height,
                     out_data + y * width, quantization);
  }
}
