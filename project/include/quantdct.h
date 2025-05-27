#ifndef QUANT_DCT_H
#define QUANT_DCT_H

#include <assert.h>
#include <errno.h>
#include <getopt.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "c63.h"

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

struct worker_ctx {
  struct c63_encoder *enc;
  int component;
};

//void dequantize_idct(int16_t *in_data, uint8_t *prediction, uint8_t *out_data);
void dequantize_idct_row(const int16_t *in, const uint8_t *prediction, uint8_t *out);

void dequantize_idct_Y(struct c63_encoder *enc);
void dequantize_idct_U(struct c63_encoder *enc);
void dequantize_idct_V(struct c63_encoder *enc);

//void dct_quantize(uint8_t *in_data, uint8_t *prediction, int16_t *out_data);
void dct_quantize_row(const uint8_t *in, uint8_t *prediction, int16_t *out);

void dct_quantize_Y(struct c63_encoder *enc);
void dct_quantize_U(struct c63_encoder *enc);
void dct_quantize_V(struct c63_encoder *enc);

void dct_idct_Y(struct c63_encoder *enc);
void dct_idct_U(struct c63_encoder *enc);
void dct_idct_V(struct c63_encoder *enc);

void *pthread_dct_idct(void *ptr);
void *pthread_idct(void *ptr);

void initialize_dctlookup_values();
void initialize_quantization_values(const uint8_t *tbl, uint32_t padw, uint32_t padh);

#ifdef __cplusplus
}
#endif // __cplusplus


#endif
