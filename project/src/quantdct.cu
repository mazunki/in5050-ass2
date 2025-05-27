
#include "c63.h"
#include "c63enc.h"
#include "common.h"
#include "quantdct.h"
#include "profiling.h"
#include <pthread.h>


static inline void dequantize_idct(const int16_t *in, uint8_t *prediction, uint8_t *out, uintptr_t offset)
{
  dequantize_idct_row(in + offset, prediction + offset, out + offset);
}


static inline void dct_quantize(const uint8_t *in, uint8_t *prediction, int16_t *out, uintptr_t offset)
{
  dct_quantize_row(in + offset, prediction + offset, out + offset);
}

/** quantize (slow CPU-only function)
 *   @param[in]  orig
 *   @param[in]  predicted
 *   @param[out] residuals
 */
static inline void dct_quantize_Y(struct c63_encoder *enc, uintptr_t offset) {
  startTrace5("wait pred");
  CUDA_ASSERT(cudaEventSynchronize(enc->pipe->event_compensate_Y));
  endTrace();

  startTrace5("dct Y");
  dct_quantize(enc->curframe->orig->Y, enc->curframe->predicted->Y, enc->curframe->residuals->Ydct, offset);
  endTrace();
}
static inline void dct_quantize_U(struct c63_encoder *enc, uintptr_t offset) {
  startTrace5("wait pred");
  CUDA_ASSERT(cudaEventSynchronize(enc->pipe->event_compensate_U));
  endTrace();

  startTrace5("dct U");
  dct_quantize(enc->curframe->orig->U, enc->curframe->predicted->U, enc->curframe->residuals->Udct, offset);
  endTrace();
}
static inline void dct_quantize_V(struct c63_encoder *enc, uintptr_t offset) {
  startTrace5("wait pred");
  CUDA_ASSERT(cudaEventSynchronize(enc->pipe->event_compensate_V));
  endTrace();

  startTrace5("dct V");
  dct_quantize(enc->curframe->orig->V, enc->curframe->predicted->V, enc->curframe->residuals->Vdct, offset);
  endTrace();
}

/** dequantize (slow CPU-only function)
 *   @param[in]  residuals
 *   @param[in]  predicted
 *   @param[out] recons
 */
static inline void dequantize_idct_Y(struct c63_encoder *enc, uintptr_t offset) {
  startTrace5("idct Y");
  dequantize_idct(enc->curframe->residuals->Ydct, enc->curframe->predicted->Y, enc->curframe->recons->Y, offset);
  endTrace5();
}
static inline void dequantize_idct_U(struct c63_encoder *enc, uintptr_t offset) {
  startTrace5("idct U");
  dequantize_idct(enc->curframe->residuals->Udct, enc->curframe->predicted->U, enc->curframe->recons->U, offset);
  endTrace5();
}
static inline void dequantize_idct_V(struct c63_encoder *enc, uintptr_t offset) {
  startTrace5("idct V");
  dequantize_idct(enc->curframe->residuals->Vdct, enc->curframe->predicted->V, enc->curframe->recons->V, offset);
  endTrace5();
}

static void *dct_idct_worker(void *arg)
{
  struct worker_ctx *ctx = (struct worker_ctx *)arg;
  struct c63_encoder *enc  = ctx->enc;
  struct c63_common *cm  = enc->cm;
  const int pool   = ctx->component;

  uint32_t width, height;
  switch (pool) {
    case TASK_LUMA:
      width = cm->ypw;
      height = cm->yph;
      initialize_quantization_values(cm->quanttbl[Y_COMPONENT], cm->ypw, cm->yph);
      break;

    case TASK_CHROMA:
      width = cm->upw; // =vpw
      height = cm->uph; // =upw
      initialize_quantization_values(cm->quanttbl[U_COMPONENT], width, height); // quanttbl[u == v]
      break;
  }

  while (true) {
    pthread_barrier_wait(&enc->pth_barrier_dct_idct_start);
    if (!enc->pthreads_run) {
      pthread_barrier_wait(&enc->pth_barrier_dct_idct_end);
      break; // shutdown
    }

    pthread_mutex_lock(&enc->pth_mutex_next_row[pool]);
    enc->pth_next_row[pool] = 0;
    pthread_mutex_unlock(&enc->pth_mutex_next_row[pool]);

    for (;;) {
      // claim a task
      pthread_mutex_lock  (&enc->pth_mutex_next_row[pool]);
      uint32_t row = enc->pth_next_row[pool];
      enc->pth_next_row[pool] += 1;
      pthread_mutex_unlock(&enc->pth_mutex_next_row[pool]);

      if (row >= height / MACROBLOCK_SIZE)
        break; // nothing more to do

      uintptr_t offset = row * width * MACROBLOCK_SIZE;

      // dct
      switch (pool) {
        case TASK_LUMA: dct_quantize_Y(enc, offset); break;
        case TASK_CHROMA: dct_quantize_U(enc, offset); dct_quantize_V(enc, offset); break;
      }

      // idct
      switch (pool) {
        case TASK_LUMA: dequantize_idct_Y(enc, offset); break;
        case TASK_CHROMA: dequantize_idct_U(enc, offset); dequantize_idct_V(enc, offset); break;
      }
    }

    pthread_barrier_wait(&enc->pth_barrier_dct_idct_end);
  }
  return NULL;
}
static void *idct_worker(void *arg)
{
  struct worker_ctx *ctx = (struct worker_ctx *)arg;
  struct c63_encoder *enc  = ctx->enc;
  struct c63_common *cm  = enc->cm;
  const int  pool   = ctx->component;

  uint32_t width, height;
  switch (pool) {
    case TASK_LUMA:
      width = cm->ypw;
      height = cm->yph;
      initialize_quantization_values(cm->quanttbl[Y_COMPONENT], cm->ypw, cm->yph);
      break;

    case TASK_CHROMA:
      width = cm->upw; // =vpw
      height = cm->uph; // =upw
      initialize_quantization_values(cm->quanttbl[U_COMPONENT], width, height); // quanttbl[u == v]
      break;
  }

  while (true) {
    pthread_barrier_wait(&enc->pth_barrier_idct_start);
    if (!enc->pthreads_run) {
      pthread_barrier_wait(&enc->pth_barrier_idct_end);
      break; // shutdown
    }

    pthread_mutex_lock(&enc->pth_mutex_next_row[pool]);
    enc->pth_next_row[pool] = 0;
    pthread_mutex_unlock(&enc->pth_mutex_next_row[pool]);

    for (;;) {
      // claim a task
      pthread_mutex_lock(&enc->pth_mutex_next_row[pool]);
      uint32_t row = enc->pth_next_row[pool];
      enc->pth_next_row[pool] += 1;
      pthread_mutex_unlock(&enc->pth_mutex_next_row[pool]);

      if (row >= height / MACROBLOCK_SIZE)
        break; // nothing more to do

      uintptr_t offset = row * width * MACROBLOCK_SIZE;

      // idct
      switch (pool) {
        case TASK_LUMA: dequantize_idct_Y(enc, offset); break;
        case TASK_CHROMA: dequantize_idct_U(enc, offset); dequantize_idct_V(enc, offset); break;
      }
    }

    pthread_barrier_wait(&enc->pth_barrier_idct_end);
  }
  return NULL;
}

void *pthread_dct_idct(void *ptr) {
  return dct_idct_worker(ptr);
}

void *pthread_idct(void *ptr) {
  return idct_worker(ptr);
}

