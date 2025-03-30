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
#include <threads.h>

#include "c63.h"
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
static quant16_t ISQRT2_Q16;

thread_local static quant16_t QUANT_TBL_q16[MACROBLOCK_SIZE*MACROBLOCK_SIZE];
thread_local static quant16_t DEQUANT_TBL_q16[MACROBLOCK_SIZE*MACROBLOCK_SIZE];
thread_local static uint32_t HEIGHT, WIDTH;

void initialize_dctlookup_values() {
  quant16_t dctlookup_q16[MACROBLOCK_SIZE][MACROBLOCK_SIZE];
  ISQRT2_Q16 = (quant16_t)(ISQRT2 * (1 << DCT_SCALE_BITS) + 0.5f);

  for (uint8_t u = 0; u < MACROBLOCK_SIZE; u++) {
    for (uint8_t v = 0; v < MACROBLOCK_SIZE; v++) {
      dctlookup_q16[u][v] = (quant16_t)roundf(dctlookup[u][v] * (1 << DCT_SCALE_BITS));
    }
  }

  for (uint8_t y = 0; y < MACROBLOCK_SIZE; y++) {
    for (uint8_t v = 0; v < MACROBLOCK_SIZE; v++) {
      for (uint8_t u = 0; u < MACROBLOCK_SIZE; u++) {
        for (uint8_t x = 0; x < MACROBLOCK_SIZE; x++) {
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

void initialize_quantization_values(const uint8_t *tbl, uint32_t padw, uint32_t padh)
{
  WIDTH = padw;
  HEIGHT = padh;

  for (uint8_t i = 0; i < MACROBLOCK_SIZE*MACROBLOCK_SIZE; i++) {
    // out[zigzag] = (float)round((dct / 4.0) / QUANT_TBL[zigzag]);
    float32_t quant =  1.0f / (4.0f * tbl[i]);
    QUANT_TBL_q16[i] = (quant16_t)roundf(quant * (1 << DCT_SCALE_BITS));

    // out[v * 8 + u] = (float)round((dct * QUANT_TBL[zigzag]) / 4.0);
    float32_t dequant = tbl[i] / 4.0f;
    DEQUANT_TBL_q16[i] = (quant16_t)roundf(dequant * (1 << DCT_SCALE_BITS));
  }
}

static void idct_2d(const quant32_t *in, quant32_t *out)
{
  startTrace8("idct 2d");
  for (uint8_t v = 0; v < MACROBLOCK_SIZE; v++) {
    for (uint8_t u = 0; u < MACROBLOCK_SIZE; u++) {
      quant64_t idct_q64 = 0;

      for (uint8_t y = 0; y < MACROBLOCK_SIZE; y++) {
        for (uint8_t x = 0; x < MACROBLOCK_SIZE; x++) {
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
  for (uint8_t v = 0; v < MACROBLOCK_SIZE; ++v) {
    for (uint8_t u = 0; u < MACROBLOCK_SIZE; ++u) {
      quant32_t pixel_q32 = in[v * MACROBLOCK_SIZE + u];

      quant16_t a1_q16 = u ? (1 << DCT_SCALE_BITS) : ISQRT2_Q16;
      quant16_t a2_q16 = v ? (1 << DCT_SCALE_BITS) : ISQRT2_Q16;
      quant32_t scale_q32 = ((quant32_t)a1_q16 * a2_q16) >> DCT_SCALE_BITS;

      // q32 * q32 = q64 → q64, scale down to q32
      quant32_t scaled_q32 = ((quant64_t)pixel_q32 * scale_q32) >> DCT_SCALE_BITS;

      out[v * MACROBLOCK_SIZE + u] = scaled_q32;
    }
  }
  endTrace();
}

static void dequantize_block(const quant32_t *in, quant32_t *out)
{
  startTrace8("deq blk");
  for (uint8_t zigzag = 0; zigzag < MACROBLOCK_SIZE * MACROBLOCK_SIZE; ++zigzag) {
    uint8_t u = zigzag_U[zigzag];
    uint8_t v = zigzag_V[zigzag];

    /* Zig-zag and de-quantize */
    quant32_t dct_q32 = in[zigzag];

    quant32_t dequantized_q32 = ((quant32_t)dct_q32 * DEQUANT_TBL_q16[zigzag]) >> DCT_SCALE_BITS;

    out[v * 8 + u] = dequantized_q32;
  }
  endTrace();
}

static void dct_quant_block_8x8(const int16_t *in, int16_t *out)
{
  startTrace7("quant 8x8");

  float mb[MACROBLOCK_SIZE * MACROBLOCK_SIZE] __attribute((aligned(16)));
  float mb2[MACROBLOCK_SIZE * MACROBLOCK_SIZE] __attribute((aligned(16)));
  quant32_t mb_q32[MACROBLOCK_SIZE * MACROBLOCK_SIZE];
  quant32_t mb2_q32[MACROBLOCK_SIZE * MACROBLOCK_SIZE] = {0};

  for (uint8_t i = 0; i < MACROBLOCK_SIZE * MACROBLOCK_SIZE; i++) {
    mb[i] = in[i];
    mb_q32[i] = (quant32_t)roundf(in[i] * (1 << DCT_SCALE_BITS));
  }

  // static void dct_2d(const quant32_t *in, quant_32t *out)
  startTrace8("dct 2d");
  for (uint8_t v = 0; v < MACROBLOCK_SIZE; v++) {
    for (uint8_t u = 0; u < MACROBLOCK_SIZE; u++) {
      quant64_t dct_q64 = 0;

      for (uint8_t y = 0; y < MACROBLOCK_SIZE; y++) {
        for (uint8_t x = 0; x < MACROBLOCK_SIZE; x++) {
          quant32_t pixel_q32 = mb_q32[y*MACROBLOCK_SIZE + x];
          quant16_t coeff_q16 = precalcDct_q16[y][v][u][x];

          // q32 * q16 = q48, accumulate in q64
          dct_q64 += (quant64_t)pixel_q32 * coeff_q16;
        }
      }
      mb2_q32[v * MACROBLOCK_SIZE + u] = MIN(dct_q64, INT32_MAX);
    }
  }
  endTrace();

  // static void scale_block(quant32_t *in_data, quant32_t *out_data)
  startTrace8("scaleblk");
  for (uint8_t v = 0; v < MACROBLOCK_SIZE; ++v) {
    for (uint8_t u = 0; u < MACROBLOCK_SIZE; ++u) {
      quant32_t pixel_q32 = mb2_q32[v * MACROBLOCK_SIZE + u];

      quant16_t a1_q16 = u ? (1 << DCT_SCALE_BITS) : ISQRT2_Q16;
      quant16_t a2_q16 = v ? (1 << DCT_SCALE_BITS) : ISQRT2_Q16;
      quant32_t scale_q32 = ((quant32_t)a1_q16 * a2_q16) >> DCT_SCALE_BITS;

      // q32 * q32 = q64 → q64, scale down to q32
      quant32_t scaled_q32 = ((quant64_t)pixel_q32 * scale_q32) >> DCT_SCALE_BITS;

      mb_q32[v * MACROBLOCK_SIZE + u] = scaled_q32;
    }
  }
  endTrace();

  // static void quantize_block(const quant32_t *in_data, quant32_t *out_data)
  startTrace8("quant blk");
  for (uint8_t zigzag = 0; zigzag < MACROBLOCK_SIZE * MACROBLOCK_SIZE; ++zigzag) {
    uint8_t u = zigzag_U[zigzag];
    uint8_t v = zigzag_V[zigzag];

    /* Zig-zag and quantize */
    quant32_t dct_q32 = mb_q32[v * 8 + u];

    quant32_t quantized_q32 = ((quant64_t)dct_q32 * QUANT_TBL_q16[zigzag]) >> DCT_SCALE_BITS;

    mb2_q32[zigzag] = quantized_q32;
  }
  endTrace();

  for (uint8_t i = 0; i < MACROBLOCK_SIZE * MACROBLOCK_SIZE; i++) {
    mb2[i] = (float32_t)mb2_q32[i] / (float)(1 << (2 * DCT_SCALE_BITS));
    out[i] = mb2[i];
  }

  endTrace();
}

static void dequant_idct_block_8x8(const int16_t *in, int16_t *out)
{
  startTrace7("deq 8x8");

  float mb[MACROBLOCK_SIZE * MACROBLOCK_SIZE] __attribute((aligned(16)));
  float mb2[MACROBLOCK_SIZE * MACROBLOCK_SIZE] __attribute((aligned(16)));
  quant32_t mb_q32[MACROBLOCK_SIZE * MACROBLOCK_SIZE];
  quant32_t mb2_q32[MACROBLOCK_SIZE * MACROBLOCK_SIZE] = {0};

  for (uint8_t i = 0; i < MACROBLOCK_SIZE * MACROBLOCK_SIZE; i++) {
    mb[i] = in[i];
    mb_q32[i] = (quant32_t)roundf(mb[i] * (1 << DCT_SCALE_BITS));
  }

  dequantize_block(mb_q32, mb2_q32);
  scale_block(mb2_q32, mb_q32);
  idct_2d(mb_q32, mb2_q32);

  for (uint8_t i = 0; i < MACROBLOCK_SIZE * MACROBLOCK_SIZE; i++) {
    mb2[i] = (float32_t)mb2_q32[i] / (float)(1 << (2 * DCT_SCALE_BITS));
    out[i] = mb2[i];
  }

  endTrace();
}

static void dequantize_idct_row(const int16_t *in, const uint8_t *prediction, uint8_t *out)
{
  startTrace6("deq row");
  int16_t block[MACROBLOCK_SIZE * MACROBLOCK_SIZE];

  /* Perform the dequantization and iDCT */
  for (uint x = 0; x < WIDTH; x += MACROBLOCK_SIZE) {
    dequant_idct_block_8x8(in + (x * MACROBLOCK_SIZE), block);

    for (uint8_t i = 0; i < MACROBLOCK_SIZE; ++i) {
      for (uint8_t j = 0; j < MACROBLOCK_SIZE; ++j) {
        /* Add prediction block. Note: DCT is not precise -
           Clamp to legal values */
        int16_t tmp = block[i * MACROBLOCK_SIZE + j] + (int16_t)prediction[i * WIDTH + j + x];

        if (tmp < 0) {
          tmp = 0;
        } else if (tmp > 255) {
          tmp = 255;
        }

        out[i * WIDTH + j + x] = tmp;
      }
    }
  }
  endTrace();
}

static void dct_quantize_row(const uint8_t *in, uint8_t *prediction, int16_t *out)
{
  startTrace6("quant row");
  int16_t block[MACROBLOCK_SIZE * MACROBLOCK_SIZE];

  /* Perform the DCT and quantization */
  for (uint x = 0; x < WIDTH; x += MACROBLOCK_SIZE) {
    for (uint8_t i = 0; i < MACROBLOCK_SIZE; ++i) {
      for (uint8_t j = 0; j < MACROBLOCK_SIZE; ++j) {
        block[i * MACROBLOCK_SIZE + j] = ((int16_t)in[i * WIDTH + j + x] - prediction[i * WIDTH + j + x]);
      }
    }

    /* Store MBs linear in memory, i.e. the 64 coefficients are stored
       continous. This allows us to ignore stride in DCT/iDCT and other
       functions. */
    dct_quant_block_8x8(block, out + (x * MACROBLOCK_SIZE));
  }
  endTrace();
}

void dequantize_idct(const int16_t *in, uint8_t *prediction, uint8_t *out)
{
  for (uint y = 0; y < HEIGHT; y += MACROBLOCK_SIZE) {
    dequantize_idct_row(in + y * WIDTH, prediction + y * WIDTH, out + y * WIDTH);
  }
}


void dct_quantize(const uint8_t *in, uint8_t *prediction, int16_t *out)
{
  for (uint y = 0; y < HEIGHT; y += MACROBLOCK_SIZE) {
    dct_quantize_row(in + y * WIDTH, prediction + y * WIDTH, out + y * WIDTH);
  }
}
