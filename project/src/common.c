#include <assert.h>
#include <errno.h>
#include <getopt.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "c63.h"
#include "common.h"
#include "tables.h"

struct c63_common* c63_common_init(int width, int height)
{
  struct c63_common *cm = (struct c63_common*)calloc(1, sizeof(struct c63_common));
  cm->width = width;
  cm->height = height;

  // compute padded dimensions
  cm->padw[Y_COMPONENT] = cm->ypw = (uint32_t)(ceil(width/16.0f)*16);
  cm->padh[Y_COMPONENT] = cm->yph = (uint32_t)(ceil(height/16.0f)*16);
  cm->padw[U_COMPONENT] = cm->upw = (uint32_t)(ceil(width*UX/(YX*8.0f))*8);
  cm->padh[U_COMPONENT] = cm->uph = (uint32_t)(ceil(height*UY/(YY*8.0f))*8);
  cm->padw[V_COMPONENT] = cm->vpw = (uint32_t)(ceil(width*VX/(YX*8.0f))*8);
  cm->padh[V_COMPONENT] = cm->vph = (uint32_t)(ceil(height*VY/(YY*8.0f))*8);

  // macroblock layout
  cm->mb_cols_luma   = cm->ypw / MACROBLOCK_SIZE;
  cm->mb_rows_luma   = cm->yph / MACROBLOCK_SIZE;
  cm->mb_cols_chroma = cm->upw / MACROBLOCK_SIZE;
  cm->mb_rows_chroma = cm->uph / MACROBLOCK_SIZE;

  // rate control defaults
  cm->qp               = 25;
  cm->me_search_range  = 16;
  cm->keyframe_interval = 100;

  // init quantization tables
  for (int i = 0; i < 64; ++i) {
    cm->quanttbl[Y_COMPONENT][i] = yquanttbl_def[i] / (cm->qp / 10.0);
    cm->quanttbl[U_COMPONENT][i] = uvquanttbl_def[i] / (cm->qp / 10.0);
    cm->quanttbl[V_COMPONENT][i] = uvquanttbl_def[i] / (cm->qp / 10.0);
  }

  // size counts
  cm->luma_size       = cm->ypw * cm->yph;
  cm->chroma_size     = cm->upw * cm->uph;
  cm->num_mbs_luma    = cm->mb_rows_luma * cm->mb_cols_luma;
  cm->num_mbs_chroma  = cm->mb_rows_chroma * cm->mb_cols_chroma;

  return cm;
}

void c63_common_free(struct c63_common *cm) {
  free(cm);
}

struct frame* create_frame(struct c63_pipeline *pipe, int role)
{
  struct frame *f = (struct frame *)calloc(1, sizeof(struct frame));

  f->orig = (yuv_t *) malloc(sizeof(yuv_t));
  f->recons = (yuv_t *) malloc(sizeof(yuv_t));
  f->predicted = (yuv_t *) malloc(sizeof(yuv_t));
  f->residuals = (dct_t *) malloc(sizeof(dct_t));

  switch (role) {
    case FRAME_CUR:
      f->orig->Y = pipe->shm_orig_Y;
      f->orig->U = pipe->shm_orig_U;
      f->orig->V = pipe->shm_orig_V;

      f->residuals->Ydct = pipe->residuals_Y;
      f->residuals->Udct = pipe->residuals_U;
      f->residuals->Vdct = pipe->residuals_V;

      f->predicted->Y = pipe->shm_predicted_Y;
      f->predicted->U = pipe->shm_predicted_U;
      f->predicted->V = pipe->shm_predicted_V;

      f->mbs[Y_COMPONENT] = pipe->shm_mbs_Y;
      f->mbs[U_COMPONENT] = pipe->shm_mbs_U;
      f->mbs[V_COMPONENT] = pipe->shm_mbs_V;
      break;

    case FRAME_NEXT:
      f->orig->Y = pipe->shm_next_Y;
      f->orig->U = pipe->shm_next_U;
      f->orig->V = pipe->shm_next_V;

      f->residuals->Ydct = pipe->prev_residuals_Y;
      f->residuals->Udct = pipe->prev_residuals_U;
      f->residuals->Vdct = pipe->prev_residuals_V;

      f->predicted->Y = pipe->shm_next_predicted_Y;
      f->predicted->U = pipe->shm_next_predicted_U;
      f->predicted->V = pipe->shm_next_predicted_V;

      f->mbs[Y_COMPONENT] = pipe->shm_prev_mbs_Y;
      f->mbs[U_COMPONENT] = pipe->shm_prev_mbs_U;
      f->mbs[V_COMPONENT] = pipe->shm_prev_mbs_V;
      break;

    case FRAME_REF:
      // not used for orig/residuals
      // old predicted becomes new recons
      break;
  }

  f->recons->Y = pipe->shm_recons_Y;
  f->recons->U = pipe->shm_recons_U;
  f->recons->V = pipe->shm_recons_V;

  return f;
}


void destroy_frame(struct frame *f)
{
  if (!f) return;
  free(f->orig);
  free(f->recons);
  free(f->predicted);
  free(f);
}

void dump_image(yuv_t *image, int w, int h, FILE *fp)
{
  fwrite(image->Y, 1, w*h, fp);
  fwrite(image->U, 1, w*h/4, fp);
  fwrite(image->V, 1, w*h/4, fp);
}

int fpeek(FILE *stream)
{
  int c = fgetc(stream);
  ungetc(c, stream);
  return c;
}
