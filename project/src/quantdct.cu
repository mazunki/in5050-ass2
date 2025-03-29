
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
  startTrace5("stream pred");
  CUDA_ASSERT(cudaStreamSynchronize(cm->pipe->stream_predictions_Y));
  endTrace();

  startTrace5("dct Y");
  dct_quantize(cm->curframe->orig->Y, cm->curframe->predicted->Y, cm->padw[Y_COMPONENT], cm->padh[Y_COMPONENT], cm->curframe->residuals->Ydct, cm->quanttbl[Y_COMPONENT]);
  endTrace();
}
void dct_quantize_U(struct c63_common *cm) {
  startTrace5("stream pred");
  CUDA_ASSERT(cudaStreamSynchronize(cm->pipe->stream_predictions_U));
  endTrace();

  startTrace5("dct U");
  dct_quantize(cm->curframe->orig->U, cm->curframe->predicted->U, cm->padw[U_COMPONENT], cm->padh[U_COMPONENT], cm->curframe->residuals->Udct, cm->quanttbl[U_COMPONENT]);
  endTrace();
}
void dct_quantize_V(struct c63_common *cm) {
  startTrace5("stream pred");
  CUDA_ASSERT(cudaStreamSynchronize(cm->pipe->stream_predictions_V));
  endTrace();

  startTrace5("dct V");
  dct_quantize(cm->curframe->orig->V, cm->curframe->predicted->V, cm->padw[V_COMPONENT], cm->padh[V_COMPONENT], cm->curframe->residuals->Vdct, cm->quanttbl[V_COMPONENT]);
  endTrace();
}

/** dequantize (slow CPU-only function)
 *   @param[in]  residuals
 *   @param[in]  predicted
 *   @param[out] recons
 */
void dequantize_idct_Y(struct c63_common *cm) {
  startTrace5("idct Y");
  dequantize_idct(cm->curframe->residuals->Ydct, cm->curframe->predicted->Y, cm->ypw, cm->yph, cm->curframe->recons->Y, cm->quanttbl[Y_COMPONENT]);
  endTrace();

  if (cm->frame_buffer[(cm->fb_curr_index+1) % FRAMEBUFFER_SIZE] != NULL) {
    CUDA_ASSERT(cudaMemcpyAsync(cm->pipe->d_recons_Y, cm->pipe->h_recons->Y, cm->luma_size, cudaMemcpyHostToDevice, cm->pipe->stream_image));
  }
}
void dequantize_idct_U(struct c63_common *cm) {
  startTrace5("idct U");
  dequantize_idct(cm->curframe->residuals->Udct, cm->curframe->predicted->U, cm->upw, cm->uph, cm->curframe->recons->U, cm->quanttbl[U_COMPONENT]);
  endTrace();

  if (cm->frame_buffer[(cm->fb_curr_index+1) % FRAMEBUFFER_SIZE] != NULL) {
    CUDA_ASSERT(cudaMemcpyAsync(cm->pipe->d_recons_U, cm->pipe->h_recons->U, cm->chroma_size, cudaMemcpyHostToDevice, cm->pipe->stream_image));
  }
}
void dequantize_idct_V(struct c63_common *cm) {
  startTrace5("idct V");
  dequantize_idct(cm->curframe->residuals->Vdct, cm->curframe->predicted->V, cm->vpw, cm->vph, cm->curframe->recons->V, cm->quanttbl[V_COMPONENT]);
  endTrace();

  if (cm->frame_buffer[(cm->fb_curr_index+1) % FRAMEBUFFER_SIZE] != NULL) {
    CUDA_ASSERT(cudaMemcpyAsync(cm->pipe->d_recons_V, cm->pipe->h_recons->V, cm->chroma_size, cudaMemcpyHostToDevice, cm->pipe->stream_image));
  }
}

// pthread wrappers
void *dct_idct_worker(struct c63_common *cm, intptr_t component, bool dequantize_only)
{
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
