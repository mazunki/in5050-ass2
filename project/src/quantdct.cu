
#include "c63.h"
#include "common.h"
#include "quantdct.h"
#include "profiling.h"
#include <pthread.h>

/** quantize (slow CPU-only function)
 *   @param[in]  orig
 *   @param[in]  predicted
 *   @param[out] residuals
 */
void dct_quantize_Y(struct c63_common *cm) {
  startTrace5("wait pred");
  CUDA_ASSERT(cudaEventSynchronize(cm->pipe->event_compensate_Y));
  endTrace();

  startTrace5("dct Y");
  dct_quantize(cm->pipe->shm_orig_Y, cm->pipe->shm_predicted_Y, cm->pipe->residuals_Y);
  endTrace();
}
void dct_quantize_U(struct c63_common *cm) {
  startTrace5("wait pred");
  CUDA_ASSERT(cudaEventSynchronize(cm->pipe->event_compensate_U));
  endTrace();

  startTrace5("dct U");
  dct_quantize(cm->pipe->shm_orig_U, cm->pipe->shm_predicted_U, cm->pipe->residuals_U);
  endTrace();
}
void dct_quantize_V(struct c63_common *cm) {
  startTrace5("wait pred");
  CUDA_ASSERT(cudaEventSynchronize(cm->pipe->event_compensate_V));
  endTrace();

  startTrace5("dct V");
  dct_quantize(cm->pipe->shm_orig_V, cm->pipe->shm_predicted_V, cm->pipe->residuals_V);
  endTrace();
}

/** dequantize (slow CPU-only function)
 *   @param[in]  residuals
 *   @param[in]  predicted
 *   @param[out] recons
 */
void dequantize_idct_Y(struct c63_common *cm) {
  startTrace5("idct Y");
  dequantize_idct(cm->pipe->residuals_Y, cm->pipe->shm_predicted_Y, cm->pipe->shm_recons_Y);
  endTrace();
}
void dequantize_idct_U(struct c63_common *cm) {
  startTrace5("idct U");
  dequantize_idct(cm->pipe->residuals_U, cm->pipe->shm_predicted_U, cm->pipe->shm_recons_U);
  endTrace();
}
void dequantize_idct_V(struct c63_common *cm) {
  startTrace5("idct V");
  dequantize_idct(cm->pipe->residuals_U, cm->pipe->shm_predicted_V, cm->pipe->shm_recons_V);
  endTrace();
}

// pthread wrappers
void *dct_idct_worker(struct c63_common *cm, intptr_t component, bool dequantize_only)
{
  switch (component) {
    case Y_COMPONENT: initialize_quantization_values(cm->quanttbl[Y_COMPONENT], cm->ypw, cm->yph); break;
    case U_COMPONENT: initialize_quantization_values(cm->quanttbl[U_COMPONENT], cm->upw, cm->uph); break;
    case V_COMPONENT: initialize_quantization_values(cm->quanttbl[V_COMPONENT], cm->vpw, cm->vph); break;
  }

  while (true) {
    pthread_barrier_wait(&cm->pth_barrier_dct_idct_start);

    // shutdown from main thread
    if (!cm->pthreads_run) {
      pthread_barrier_wait(&cm->pth_barrier_dct_idct_end);
      break;
    }

    if (!dequantize_only) {
      switch (component) {
        case Y_COMPONENT: dct_quantize_Y(cm); break;
        case U_COMPONENT: dct_quantize_U(cm); break;
        case V_COMPONENT: dct_quantize_V(cm); break;
      }
    }

    switch (component) {
      case Y_COMPONENT: dequantize_idct_Y(cm); break;
      case U_COMPONENT: dequantize_idct_U(cm); break;
      case V_COMPONENT: dequantize_idct_V(cm); break;
    }

    pthread_barrier_wait(&cm->pth_barrier_dct_idct_end);
  }

  return NULL;
}

void *pthread_dct_idct_Y(void *ptr) {
  return dct_idct_worker((struct c63_common *) ptr, Y_COMPONENT, 0);
}
void *pthread_dct_idct_U(void *ptr) {
  return dct_idct_worker((struct c63_common *) ptr, U_COMPONENT, 0);
}
void *pthread_dct_idct_V(void *ptr) {
  return dct_idct_worker((struct c63_common *) ptr, V_COMPONENT, 0);
}


void *pthread_idct_Y(void *ptr) {
  return dct_idct_worker((struct c63_common *) ptr, Y_COMPONENT, 1);
}
void *pthread_idct_U(void *ptr) {
  return dct_idct_worker((struct c63_common *) ptr, U_COMPONENT, 1);
}
void *pthread_idct_V(void *ptr) {
  return dct_idct_worker((struct c63_common *) ptr, V_COMPONENT, 1);
}
