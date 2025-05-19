#include <assert.h>
#include <errno.h>
#include <getopt.h>
#include <inttypes.h>
#include <math.h>
#include <stdalign.h>
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
static uint8_t quantizeZigzag[MACROBLOCK_SIZE*MACROBLOCK_SIZE];
static uint8_t dequantizeZigzag[MACROBLOCK_SIZE*MACROBLOCK_SIZE];
static float scaleFactors[MACROBLOCK_SIZE*MACROBLOCK_SIZE];

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
   endTrace3();

   startTrace3("precompute zigzag");
   for (int i = 0; i < MACROBLOCK_SIZE*MACROBLOCK_SIZE; ++i) {
     int u = zigzag_U[i];
     int v = zigzag_V[i];
     quantizeZigzag[v * MACROBLOCK_SIZE + u] = i;
     dequantizeZigzag[i] = v * MACROBLOCK_SIZE + u;
     scaleFactors[v * MACROBLOCK_SIZE + u] = (!u && !v) ? 0.5f : (u || v) ? ISQRT2 : 1.0f;
   }
   endTrace3();
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

  startTrace8("load mb");
  float16x8_t mb_row_f16[MACROBLOCK_SIZE];
  for (uint8_t y = 0; y < MACROBLOCK_SIZE; y++) {
    alignas(16) float temp[8];

    for (uint8_t x = 0; x < MACROBLOCK_SIZE; x++) {
      uint8_t idx = y * MACROBLOCK_SIZE + x;
      temp[x] = (float)in[idx];
    }

    float32x4_t lo = vld1q_f32(temp);
    float32x4_t hi = vld1q_f32(temp + 4);

    mb_row_f16[y] = vcvt_high_f16_f32(vcvt_f16_f32(lo), hi);
  }
  endTrace8();

  startTrace8("dct 2d");
  for (uint8_t v = 0; v < MACROBLOCK_SIZE; v++) {
    for (uint8_t u = 0; u < MACROBLOCK_SIZE; u++) {
      float16x8_t dct = vdupq_n_f16(0.0f);

      for (uint8_t y = 0; y < MACROBLOCK_SIZE; y++) {
        dct = vfmaq_f16(dct, mb_row_f16[y], precalcDct[v][u][y]);
      }

      float dct_sum = vaddvq_f32(vcvt_high_f32_f16(dct)) + vaddvq_f32(vcvt_f32_f16(*(float16x4_t*)&dct));

      uint8_t i = v * MACROBLOCK_SIZE + u;
      uint8_t z = quantizeZigzag[i];

      out[z] = (dct_sum * scaleFactors[i]) * QUANT_TBL_f32[z];
    }
  }
  endTrace8();

  endTrace7();
}


static void dequant_idct_block_8x8(const int16_t *in, int16_t *out)
{
  startTrace7("deq 8x8");

  startTrace8("load mb");
  float16x8_t mb_row_f16[MACROBLOCK_SIZE];

  for (uint8_t y = 0; y < MACROBLOCK_SIZE; y++) {
      alignas(16) float temp[8];

      for (uint8_t x = 0; x < 8; x++) {
          uint8_t zigzag_idx = quantizeZigzag[y * MACROBLOCK_SIZE + x];
          temp[x] = (float)in[zigzag_idx] * DEQUANT_TBL_f32[zigzag_idx] * scaleFactors[y * MACROBLOCK_SIZE + x];
      }

      float32x4_t lo = vld1q_f32(temp);
      float32x4_t hi = vld1q_f32(temp + 4);

      mb_row_f16[y] = vcvt_high_f16_f32(vcvt_f16_f32(lo), hi);
  }
  endTrace8();

  startTrace8("dequantize");
  for (uint8_t v = 0; v < MACROBLOCK_SIZE; v++) {
    for (uint8_t u = 0; u < MACROBLOCK_SIZE; u++) {
      float16x8_t idct = vdupq_n_f16(0.0f);

      for (uint8_t y = 0; y < MACROBLOCK_SIZE; y++) {
        idct = vfmaq_f16(idct, mb_row_f16[y], precalcIdct[v][u][y]); //  idct += (in * precalc)
      }

      out[v * MACROBLOCK_SIZE + u] = vaddvq_f32(vcvt_high_f32_f16(idct)) + vaddvq_f32(vcvt_f32_f16(*(float16x4_t*)&idct));
    }
  }
  endTrace8();

  endTrace7();
}

void dequantize_idct_row(const int16_t *in, const uint8_t *prediction, uint8_t *out)
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
  endTrace6();
}

void dct_quantize_row(const uint8_t *in, uint8_t *prediction, int16_t *out)
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
  endTrace6();
}
