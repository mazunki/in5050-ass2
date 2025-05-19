
#include "c63.h"
#include "common.h"
#include "quantdct.h"
#include "profiling.h"
#include <pthread.h>

void dequantize_idct(const int16_t *in, uint8_t *prediction, uint8_t *out, uintptr_t offset)
{
  dequantize_idct_row(in + offset, prediction + offset, out + offset);
}


void dct_quantize(const uint8_t *in, uint8_t *prediction, int16_t *out, uintptr_t offset)
{
  dct_quantize_row(in + offset, prediction + offset, out + offset);
}

/** quantize (slow CPU-only function)
 *   @param[in]  orig
 *   @param[in]  predicted
 *   @param[out] residuals
 */
void dct_quantize_Y(struct c63_common *cm, uint32_t HEIGHT, uint32_t WIDTH) {
  startTrace5("wait pred");
  CUDA_ASSERT(cudaEventSynchronize(cm->pipe->event_compensate_Y));
  endTrace();

  startTrace5("dct Y");
  for (uint y = 0; y < HEIGHT; y += MACROBLOCK_SIZE) {
    uintptr_t offset = y * WIDTH;
    dct_quantize(cm->curframe->orig->Y, cm->curframe->predicted->Y, cm->curframe->residuals->Ydct, offset);
  }
  endTrace();
}
void dct_quantize_U(struct c63_common *cm, uint32_t HEIGHT, uint32_t WIDTH) {
  startTrace5("wait pred");
  CUDA_ASSERT(cudaEventSynchronize(cm->pipe->event_compensate_U));
  endTrace();

  startTrace5("dct U");
  for (uint y = 0; y < HEIGHT; y += MACROBLOCK_SIZE) {
    uintptr_t offset = y * WIDTH;
    dct_quantize(cm->curframe->orig->U, cm->curframe->predicted->U, cm->curframe->residuals->Udct, offset);
  }
  endTrace();
}
void dct_quantize_V(struct c63_common *cm, uint32_t HEIGHT, uint32_t WIDTH) {
  startTrace5("wait pred");
  CUDA_ASSERT(cudaEventSynchronize(cm->pipe->event_compensate_V));
  endTrace();

  startTrace5("dct V");
  for (uint y = 0; y < HEIGHT; y += MACROBLOCK_SIZE) {
    uintptr_t offset = y * WIDTH;
    dct_quantize(cm->curframe->orig->V, cm->curframe->predicted->V, cm->curframe->residuals->Vdct, offset);
  }
  endTrace();
}

/** dequantize (slow CPU-only function)
 *   @param[in]  residuals
 *   @param[in]  predicted
 *   @param[out] recons
 */
void dequantize_idct_Y(struct c63_common *cm, uint32_t HEIGHT, uint32_t WIDTH) {
  startTrace5("idct Y");
  for (uint y = 0; y < HEIGHT; y += MACROBLOCK_SIZE) {
    uintptr_t offset = y * WIDTH;
    dequantize_idct(cm->curframe->residuals->Ydct, cm->curframe->predicted->Y, cm->curframe->recons->Y, offset);
  }
  endTrace5();
}
void dequantize_idct_U(struct c63_common *cm, uint32_t HEIGHT, uint32_t WIDTH) {
  startTrace5("idct U");
  for (uint y = 0; y < HEIGHT; y += MACROBLOCK_SIZE) {
    uintptr_t offset = y * WIDTH;
    dequantize_idct(cm->curframe->residuals->Udct, cm->curframe->predicted->U, cm->curframe->recons->U, offset);
  }
  endTrace5();
}
void dequantize_idct_V(struct c63_common *cm, uint32_t HEIGHT, uint32_t WIDTH) {
  startTrace5("idct V");
  for (uint y = 0; y < HEIGHT; y += MACROBLOCK_SIZE) {
    uintptr_t offset = y * WIDTH;
    dequantize_idct(cm->curframe->residuals->Vdct, cm->curframe->predicted->V, cm->curframe->recons->V, offset);
  }
  endTrace5();
}

// pthread wrappers
void *dct_worker(struct c63_common *cm, intptr_t component)
{
  uint32_t WIDTH, HEIGHT;
  switch (component) {
    case Y_COMPONENT: WIDTH=cm->ypw; HEIGHT=cm->yph; break;
    case U_COMPONENT: WIDTH=cm->upw; HEIGHT=cm->uph; break;
    case V_COMPONENT: WIDTH=cm->vpw; HEIGHT=cm->vph; break;
  }

  switch (component) {
    case Y_COMPONENT: initialize_quantization_values(cm->quanttbl[Y_COMPONENT], WIDTH, HEIGHT); break;
    case U_COMPONENT: initialize_quantization_values(cm->quanttbl[U_COMPONENT], WIDTH, HEIGHT); break;
    case V_COMPONENT: initialize_quantization_values(cm->quanttbl[V_COMPONENT], WIDTH, HEIGHT); break;
  }

  while (true) {
    pthread_barrier_wait(&cm->pth_barrier_dct_start);

    // shutdown from main thread
    if (!cm->pthreads_run) {
      pthread_barrier_wait(&cm->pth_barrier_dct_end);
      break;
    }

    switch (component) {
      case Y_COMPONENT: dct_quantize_Y(cm, HEIGHT, WIDTH); break;
      case U_COMPONENT: dct_quantize_U(cm, HEIGHT, WIDTH); break;
      case V_COMPONENT: dct_quantize_V(cm, HEIGHT, WIDTH); break;
    }

    pthread_barrier_wait(&cm->pth_barrier_dct_end);
  }

  return NULL;
}
void *idct_worker(struct c63_common *cm, intptr_t component)
{
  uint32_t WIDTH, HEIGHT;
  switch (component) {
    case Y_COMPONENT: WIDTH=cm->ypw; HEIGHT=cm->yph; break;
    case U_COMPONENT: WIDTH=cm->upw; HEIGHT=cm->uph; break;
    case V_COMPONENT: WIDTH=cm->vpw; HEIGHT=cm->vph; break;
  }

  switch (component) {
    case Y_COMPONENT: initialize_quantization_values(cm->quanttbl[Y_COMPONENT], WIDTH, HEIGHT); break;
    case U_COMPONENT: initialize_quantization_values(cm->quanttbl[U_COMPONENT], WIDTH, HEIGHT); break;
    case V_COMPONENT: initialize_quantization_values(cm->quanttbl[V_COMPONENT], WIDTH, HEIGHT); break;
  }

  while (true) {
    pthread_barrier_wait(&cm->pth_barrier_idct_start);

    // shutdown from main thread
    if (!cm->pthreads_run) {
      pthread_barrier_wait(&cm->pth_barrier_idct_end);
      break;
    }

    switch (component) {
      case Y_COMPONENT: dequantize_idct_Y(cm, HEIGHT, WIDTH); break;
      case U_COMPONENT: dequantize_idct_U(cm, HEIGHT, WIDTH); break;
      case V_COMPONENT: dequantize_idct_V(cm, HEIGHT, WIDTH); break;
    }

    pthread_barrier_wait(&cm->pth_barrier_idct_end);
  }

  return NULL;
}

void *pthread_dct_Y(void *ptr) {
  return dct_worker((struct c63_common *) ptr, Y_COMPONENT);
}
void *pthread_dct_U(void *ptr) {
  return dct_worker((struct c63_common *) ptr, U_COMPONENT);
}
void *pthread_dct_V(void *ptr) {
  return dct_worker((struct c63_common *) ptr, V_COMPONENT);
}


void *pthread_idct_Y(void *ptr) {
  return idct_worker((struct c63_common *) ptr, Y_COMPONENT);
}
void *pthread_idct_U(void *ptr) {
  return idct_worker((struct c63_common *) ptr, U_COMPONENT);
}
void *pthread_idct_V(void *ptr) {
  return idct_worker((struct c63_common *) ptr, V_COMPONENT);
}
