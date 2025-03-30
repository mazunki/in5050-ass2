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

#define ISQRT2 0.70710678118654f

static float16x8_t precalcIdct[MACROBLOCK_SIZE][MACROBLOCK_SIZE][MACROBLOCK_SIZE];
static float16x8_t precalcDct[MACROBLOCK_SIZE][MACROBLOCK_SIZE][MACROBLOCK_SIZE];
static uint8_t linearZigzag[MACROBLOCK_SIZE*MACROBLOCK_SIZE];

thread_local static float32_t QUANT_TBL_f32[MACROBLOCK_SIZE*MACROBLOCK_SIZE];
thread_local static float32_t DEQUANT_TBL_f32[MACROBLOCK_SIZE*MACROBLOCK_SIZE];
thread_local static uint32_t HEIGHT, WIDTH;

void initialize_dctlookup_values() {
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

   startTrace3("precompute zigzag");
   for (int i = 0; i < MACROBLOCK_SIZE*MACROBLOCK_SIZE; ++i) {
     int u = zigzag_U[i];
     int v = zigzag_V[i];
     linearZigzag[v * MACROBLOCK_SIZE + u] = i;
   }
   endTrace();
}

void initialize_quantization_values(const uint8_t *tbl, uint32_t padw, uint32_t padh)
{
  WIDTH = padw;
  HEIGHT = padh;

  for (uint8_t i = 0; i < MACROBLOCK_SIZE*MACROBLOCK_SIZE; i++) {
    // out[zigzag] = (float)round((dct / 4.0) / QUANT_TBL[zigzag]);
    QUANT_TBL_f32[i] =  1.0f / (4.0f * tbl[i]);

    // out[v * 8 + u] = (float)round((dct * QUANT_TBL[zigzag]) / 4.0);
    DEQUANT_TBL_f32[i] = (float)tbl[i] / 4.0f;
  }
}

static void dct_quant_block_8x8(const int16_t *in, int16_t *out)
{
  startTrace7("quant 8x8");

  float mb[MACROBLOCK_SIZE * MACROBLOCK_SIZE] __attribute((aligned(16)));
  float mb2[MACROBLOCK_SIZE * MACROBLOCK_SIZE] __attribute((aligned(16))) = {0};

  for (uint8_t i = 0; i < MACROBLOCK_SIZE * MACROBLOCK_SIZE; i++) {
    mb[i] = in[i];
  }

  // static void dct_2d(const float *in, float *out)
  startTrace8("dct 2d");
  for (uint8_t v = 0; v < MACROBLOCK_SIZE; v++) {
    for (uint8_t u = 0; u < MACROBLOCK_SIZE; u++) {
      float16x8_t dct = vdupq_n_f16(0.0f);

      for (uint8_t y = 0; y < MACROBLOCK_SIZE; y++) {
        float32x4_t lo = vld1q_f32(&mb[y * MACROBLOCK_SIZE]);
        float32x4_t hi = vld1q_f32(&mb[y * MACROBLOCK_SIZE + 4]);
        float16x8_t in_vec = vcvt_high_f16_f32(vcvt_f16_f32(lo), hi);

        dct = vaddq_f16(dct,vmulq_f16(in_vec, precalcDct[v][u][y])); //  dct += (in * precalc)
      }

      mb2[v * MACROBLOCK_SIZE + u] = vaddvq_f32(vcvt_high_f32_f16(dct)) +
                                      vaddvq_f32(vcvt_f32_f16(*(float16x4_t*)&dct));
    }
  }
  endTrace();

  // static void scale_block(float *in_data, float *out_data)
  startTrace8("scaleblk");
  for (uint8_t v = 0; v < MACROBLOCK_SIZE; ++v) {
    for (uint8_t u = 0; u < MACROBLOCK_SIZE; ++u) {
      float a1 = !u ? ISQRT2 : 1.0f;
      float a2 = !v ? ISQRT2 : 1.0f;

      float32_t scale = a1 * a2;

      /* Scale according to normalizing function */
      mb[v * MACROBLOCK_SIZE + u] = mb2[v * MACROBLOCK_SIZE + u] * scale;
    }
  }
  endTrace();

  // static void quantize_block(float *in_data, float *out_data)
  startTrace8("quant blk");
  for (uint8_t i = 0; i < MACROBLOCK_SIZE*MACROBLOCK_SIZE; ++i) {
    uint8_t z = linearZigzag[i];
    out[z] = mb[i] * QUANT_TBL_f32[z];
  }
  endTrace();

  endTrace();
}

static void dequant_idct_block_8x8(const int16_t *in, int16_t *out)
{
  startTrace7("deq 8x8");

  float mb[MACROBLOCK_SIZE * MACROBLOCK_SIZE] __attribute((aligned(16)));
  float mb2[MACROBLOCK_SIZE * MACROBLOCK_SIZE] __attribute((aligned(16))) = {0};

  for (uint8_t i = 0; i < MACROBLOCK_SIZE * MACROBLOCK_SIZE; i++) {
    mb[i] = in[i];
  }

  // static void dequantize_block(float *in_data, float *out_data)
  startTrace8("deq blk");
  for (uint8_t zigzag = 0; zigzag < MACROBLOCK_SIZE * MACROBLOCK_SIZE; ++zigzag) {
    uint8_t u = zigzag_U[zigzag];
    uint8_t v = zigzag_V[zigzag];

    float idct = mb[zigzag];
    float32_t dequantized = idct * DEQUANT_TBL_f32[zigzag];

    /* Zig-zag and de-quantize */
    mb2[v * 8 + u] = dequantized;
  }
  endTrace();

  // static void scale_block(float *in_data, float *out_data)
  startTrace8("scaleblk");
  for (uint8_t v = 0; v < MACROBLOCK_SIZE; ++v) {
    for (uint8_t u = 0; u < MACROBLOCK_SIZE; ++u) {
      float a1 = !u ? ISQRT2 : 1.0f;
      float a2 = !v ? ISQRT2 : 1.0f;

      float32_t scale = a1 * a2;

      /* Scale according to normalizing function */
      mb[v * MACROBLOCK_SIZE + u] = mb2[v * MACROBLOCK_SIZE + u] * scale;
    }
  }
  endTrace();

  // static void idct_2d(const float *in, float *out)
  startTrace8("idct 2d");
  for (uint8_t v = 0; v < MACROBLOCK_SIZE; v++) {
    for (uint8_t u = 0; u < MACROBLOCK_SIZE; u++) {
      float16x8_t idct = vdupq_n_f16(0.0f);

      for (uint8_t y = 0; y < MACROBLOCK_SIZE; y++) {
        float32x4_t lo = vld1q_f32(&mb[y * MACROBLOCK_SIZE]);
        float32x4_t hi = vld1q_f32(&mb[y * MACROBLOCK_SIZE + 4]);
        float16x8_t in_vec = vcvt_high_f16_f32(vcvt_f16_f32(lo), hi);

        idct = vaddq_f16(idct, vmulq_f16(in_vec, precalcIdct[v][u][y])); //  idct += (in * precalc)
      }

      out[v * MACROBLOCK_SIZE + u] = vaddvq_f32(vcvt_high_f32_f16(idct)) +
                                      vaddvq_f32(vcvt_f32_f16(*(float16x4_t*)&idct));
    }
  }
  endTrace();

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
      int16x8_t row = vld1q_s16(block + i * MACROBLOCK_SIZE);
      int16x8_t pred = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(prediction + i * WIDTH + x)));
      int16x8_t sum = vaddq_s16(row, pred);

      // clamp result to [0, 255]
      int16x8_t clamped = vmaxq_s16(vdupq_n_s16(0), vminq_s16(sum, vdupq_n_s16(255)));

      uint8x8_t out_u8 = vmovn_u16(vreinterpretq_u16_s16(clamped));
      vst1_u8(out + i * WIDTH + x, out_u8);
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
      uint8x8_t in_u8  = vld1_u8(in + i * WIDTH + x);
      uint8x8_t pred_u8 = vld1_u8(prediction + i * WIDTH + x);

      int16x8_t in_s16  = vreinterpretq_s16_u16(vmovl_u8(in_u8));
      int16x8_t pred_s16 = vreinterpretq_s16_u16(vmovl_u8(pred_u8));
      int16x8_t diff = vsubq_s16(in_s16, pred_s16);

      vst1q_s16(block + i * MACROBLOCK_SIZE, diff);
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
