
#include <nvToolsExt.h>
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
  nvtxRangePush("dct");
  dct_quantize_Y(cm);
  nvtxRangePop(); // dct

  nvtxRangePush("idct");
  dequantize_idct_Y(cm);
  nvtxRangePop(); // idct
}

void dct_idct_U(struct c63_common *cm) {
  nvtxRangePush("dct");
  dct_quantize_U(cm);
  nvtxRangePop(); // dct

  nvtxRangePush("idct");
  dequantize_idct_U(cm);
  nvtxRangePop(); // idct

}

void dct_idct_V(struct c63_common *cm) {
  nvtxRangePush("dct");
  dct_quantize_V(cm);
  nvtxRangePop(); // dct

  nvtxRangePush("idct");
  dequantize_idct_V(cm);
  nvtxRangePop(); // idct
}

void *pthread_dct_idct_Y(void *ptr) {
  dct_idct_Y((struct c63_common *) ptr);

  return NULL;
}

void *pthread_dct_idct_U(void *ptr) {
  dct_idct_U((struct c63_common *) ptr);

  return NULL;
}

void *pthread_dct_idct_V(void *ptr) {
  dct_idct_V((struct c63_common *) ptr);

  return NULL;
}
