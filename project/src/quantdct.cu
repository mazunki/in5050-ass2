
#include <nvToolsExt.h>
#include "c63.h"
#include "quantdct.h"

/** quantize (slow CPU-only function)
 *   @param[in]  orig
 *   @param[in]  predicted
 *   @param[out] residuals
 */
void dct_quantize_Y(struct c63_common *cm) {
  dct_quantize(cm->curframe->orig->Y, cm->curframe->predicted->Y, cm->padw[Y_COMPONENT], cm->padh[Y_COMPONENT], cm->curframe->residuals->Ydct, cm->quanttbl[Y_COMPONENT]);
}
void dct_quantize_U(struct c63_common *cm) {
  dct_quantize(cm->curframe->orig->U, cm->curframe->predicted->U, cm->padw[U_COMPONENT], cm->padh[U_COMPONENT], cm->curframe->residuals->Udct, cm->quanttbl[U_COMPONENT]);
}
void dct_quantize_V(struct c63_common *cm) {
  dct_quantize(cm->curframe->orig->V, cm->curframe->predicted->V, cm->padw[V_COMPONENT], cm->padh[V_COMPONENT], cm->curframe->residuals->Vdct, cm->quanttbl[V_COMPONENT]);
}

/** dequantize (slow CPU-only function)
 *   @param[in]  residuals
 *   @param[in]  predicted
 *   @param[out] recons
 */
void dequantize_idct_Y(struct c63_common *cm) {
  dequantize_idct(cm->curframe->residuals->Ydct, cm->curframe->predicted->Y, cm->ypw, cm->yph, cm->curframe->recons->Y, cm->quanttbl[Y_COMPONENT]);
}
void dequantize_idct_U(struct c63_common *cm) {
  dequantize_idct(cm->curframe->residuals->Udct, cm->curframe->predicted->U, cm->upw, cm->uph, cm->curframe->recons->U, cm->quanttbl[U_COMPONENT]);
}
void dequantize_idct_V(struct c63_common *cm) {
  dequantize_idct(cm->curframe->residuals->Vdct, cm->curframe->predicted->V, cm->vpw, cm->vph, cm->curframe->recons->V, cm->quanttbl[V_COMPONENT]);
}

void dct_idct_Y(struct c63_common *cm) {
  nvtxRangePush("Y");

  nvtxRangePush("dct");
  dct_quantize_Y(cm);
  nvtxRangePop(); // dct

  nvtxRangePush("idct");
  dequantize_idct_Y(cm);
  nvtxRangePop(); // idct

  nvtxRangePop();
}

void dct_idct_U(struct c63_common *cm) {
  nvtxRangePush("U");

  nvtxRangePush("dct");
  dct_quantize_U(cm);
  nvtxRangePop(); // dct

  nvtxRangePush("idct");
  dequantize_idct_U(cm);
  nvtxRangePop(); // idct

  nvtxRangePop();
}

void dct_idct_V(struct c63_common *cm) {
  nvtxRangePush("V");

  nvtxRangePush("dct");
  dct_quantize_V(cm);
  nvtxRangePop(); // dct

  nvtxRangePush("idct");
  dequantize_idct_V(cm);
  nvtxRangePop(); // idct

  nvtxRangePop();
}

// pthread wrappers
void *dct_idct_worker(struct c63_common *cm, intptr_t component) {
  yuv_t *next_frame;
  do {
    next_frame = cm->frame_buffer[(cm->fb_curr_index+1) % FRAMEBUFFER_SIZE];

    pthread_mutex_lock(&cm->pth_mutex_dct_idct);
    while (!cm->pth_pending_dct_idct[component] && next_frame != NULL) {
      pthread_cond_wait(&cm->pth_cond_dct_idct_ready, &cm->pth_mutex_dct_idct);
    }

    if (next_frame == NULL) {
      pthread_mutex_unlock(&cm->pth_mutex_dct_idct);
      break;
    }

    cm->pth_pending_dct_idct[component] = 0;
    pthread_mutex_unlock(&cm->pth_mutex_dct_idct);

    switch (component) {
      case Y_COMPONENT: dct_idct_Y(cm); break;
      case U_COMPONENT: dct_idct_U(cm); break;
      case V_COMPONENT: dct_idct_V(cm); break;
    }

    pthread_mutex_lock(&cm->pth_mutex_dct_idct);
    cm->pth_barrier_dct_idct--;
    if (cm->pth_barrier_dct_idct == 0) {
      pthread_cond_signal(&cm->pth_cond_dct_idct_done);
    }
    pthread_mutex_unlock(&cm->pth_mutex_dct_idct);

  } while (next_frame != NULL);

  return NULL;
}


void *pthread_dct_idct_Y(void *ptr) {
  return dct_idct_worker((struct c63_common *) ptr, Y_COMPONENT);
}

void *pthread_dct_idct_U(void *ptr) {
  return dct_idct_worker((struct c63_common *) ptr, U_COMPONENT);
}

void *pthread_dct_idct_V(void *ptr) {
  return dct_idct_worker((struct c63_common *) ptr, V_COMPONENT);
}
