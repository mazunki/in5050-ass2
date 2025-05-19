
#include "c63.h"
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
static inline void dct_quantize_Y(struct c63_common *cm, uintptr_t offset) {
  startTrace5("wait pred");
  CUDA_ASSERT(cudaEventSynchronize(cm->pipe->event_compensate_Y));
  endTrace();

  startTrace5("dct Y");
  dct_quantize(cm->curframe->orig->Y, cm->curframe->predicted->Y, cm->curframe->residuals->Ydct, offset);
  endTrace();
}
static inline void dct_quantize_U(struct c63_common *cm, uintptr_t offset) {
  startTrace5("wait pred");
  CUDA_ASSERT(cudaEventSynchronize(cm->pipe->event_compensate_U));
  endTrace();

  startTrace5("dct U");
  dct_quantize(cm->curframe->orig->U, cm->curframe->predicted->U, cm->curframe->residuals->Udct, offset);
  endTrace();
}
static inline void dct_quantize_V(struct c63_common *cm, uintptr_t offset) {
  startTrace5("wait pred");
  CUDA_ASSERT(cudaEventSynchronize(cm->pipe->event_compensate_V));
  endTrace();

  startTrace5("dct V");
  dct_quantize(cm->curframe->orig->V, cm->curframe->predicted->V, cm->curframe->residuals->Vdct, offset);
  endTrace();
}

/** dequantize (slow CPU-only function)
 *   @param[in]  residuals
 *   @param[in]  predicted
 *   @param[out] recons
 */
static inline void dequantize_idct_Y(struct c63_common *cm, uintptr_t offset) {
  startTrace5("idct Y");
  dequantize_idct(cm->curframe->residuals->Ydct, cm->curframe->predicted->Y, cm->curframe->recons->Y, offset);
  endTrace5();
}
static inline void dequantize_idct_U(struct c63_common *cm, uintptr_t offset) {
  startTrace5("idct U");
  dequantize_idct(cm->curframe->residuals->Udct, cm->curframe->predicted->U, cm->curframe->recons->U, offset);
  endTrace5();
}
static inline void dequantize_idct_V(struct c63_common *cm, uintptr_t offset) {
  startTrace5("idct V");
  dequantize_idct(cm->curframe->residuals->Vdct, cm->curframe->predicted->V, cm->curframe->recons->V, offset);
  endTrace5();
}

static void *dct_idct_worker(void *arg)
{
    struct worker_ctx *ctx = (struct worker_ctx *)arg;
    struct c63_common *cm  = ctx->cm;
    const int  component   = ctx->component;

    uint32_t width, height;
    switch (component) {
        case Y_COMPONENT: width = cm->ypw; height = cm->yph; break;
        case U_COMPONENT: width = cm->upw; height = cm->uph; break;
        case V_COMPONENT: width = cm->vpw; height = cm->vph; break;
    }
    initialize_quantization_values(cm->quanttbl[component], width, height);

    while (true) {
        pthread_barrier_wait(&cm->pth_barrier_dct_idct_start);
        if (!cm->pthreads_run) {
            pthread_barrier_wait(&cm->pth_barrier_dct_idct_end);
            break; // shutdown
        }

        // first worker of each component resets task counter
        if (ctx->worker_id == 0)
            cm->pth_next_row[component] = 0;

        for (;;) {
            // claim a task
            pthread_mutex_lock  (&cm->pth_mutex_next_row[component]);
            uint32_t row = cm->pth_next_row[component];
            cm->pth_next_row[component] += 1;
            pthread_mutex_unlock(&cm->pth_mutex_next_row[component]);

            if (row >= height / MACROBLOCK_SIZE)
                break; // nothing more to do

            uintptr_t offset = row * width * MACROBLOCK_SIZE;

            // dct
            switch (component) {
                case Y_COMPONENT: dct_quantize_Y(cm, offset); break;
                case U_COMPONENT: dct_quantize_U(cm, offset); break;
                case V_COMPONENT: dct_quantize_V(cm, offset); break;
            }

            // idct
            switch (component) {
                case Y_COMPONENT: dequantize_idct_Y(cm, offset); break;
                case U_COMPONENT: dequantize_idct_U(cm, offset); break;
                case V_COMPONENT: dequantize_idct_V(cm, offset); break;
            }
        }

        pthread_barrier_wait(&cm->pth_barrier_dct_idct_end);
    }
    return NULL;
}
static void *idct_worker(void *arg)
{
    struct worker_ctx *ctx = (struct worker_ctx *)arg;
    struct c63_common *cm  = ctx->cm;
    const int  component   = ctx->component;

    uint32_t width, height;
    switch (component) {
        case Y_COMPONENT: width = cm->ypw; height = cm->yph; break;
        case U_COMPONENT: width = cm->upw; height = cm->uph; break;
        case V_COMPONENT: width = cm->vpw; height = cm->vph; break;
    }
    initialize_quantization_values(cm->quanttbl[component], width, height);

    while (true) {
        pthread_barrier_wait(&cm->pth_barrier_idct_start);
        if (!cm->pthreads_run) {
            pthread_barrier_wait(&cm->pth_barrier_idct_end);
            break; // shutdown
        }

        // first worker of each component resets task counter
        if (ctx->worker_id == 0)
            cm->pth_next_row[component] = 0;

        for (;;) {
            // claim a task
            pthread_mutex_lock(&cm->pth_mutex_next_row[component]);
            uint32_t row = cm->pth_next_row[component];
            cm->pth_next_row[component] += 1;
            pthread_mutex_unlock(&cm->pth_mutex_next_row[component]);

            if (row >= height / MACROBLOCK_SIZE)
                break; // nothing more to do

            uintptr_t offset = row * width * MACROBLOCK_SIZE;

            // idct
            switch (component) {
                case Y_COMPONENT: dequantize_idct_Y(cm, offset); break;
                case U_COMPONENT: dequantize_idct_U(cm, offset); break;
                case V_COMPONENT: dequantize_idct_V(cm, offset); break;
            }
        }

        pthread_barrier_wait(&cm->pth_barrier_idct_end);
    }
    return NULL;
}

void *pthread_dct_idct(void *ptr) {
  return dct_idct_worker(ptr);
}

void *pthread_idct(void *ptr) {
  return idct_worker(ptr);
}

