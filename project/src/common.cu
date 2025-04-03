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

#define CUDA_ALLOC_SHARED(host_ptr, size) \
  do { \
    CUDA_ASSERT(cudaHostAlloc((void **) &(host_ptr), (size), cudaHostAllocMapped)); \
    /* CUDA_ASSERT(cudaHostGetDevicePointer((void **) &(device_ptr), (void *) (host_ptr), 0)); */ \
} while (0)

struct c63_pipeline* c63_pipeline_init(size_t frame_size, size_t chroma_size, size_t num_blocks_luma, size_t num_blocks_chroma)
{
  struct c63_pipeline *pipe = (struct c63_pipeline*) calloc(1, sizeof(struct c63_pipeline));
  if (pipe == NULL) { return NULL; }

  // data
  CUDA_ALLOC_SHARED(pipe->shm_orig_Y, frame_size);
  CUDA_ALLOC_SHARED(pipe->shm_orig_U, chroma_size);
  CUDA_ALLOC_SHARED(pipe->shm_orig_V, chroma_size);

  CUDA_ALLOC_SHARED(pipe->shm_next_Y, frame_size);
  CUDA_ALLOC_SHARED(pipe->shm_next_U, chroma_size);
  CUDA_ALLOC_SHARED(pipe->shm_next_V, chroma_size);

  CUDA_ALLOC_SHARED(pipe->shm_recons_Y, frame_size);
  CUDA_ALLOC_SHARED(pipe->shm_recons_U, chroma_size);
  CUDA_ALLOC_SHARED(pipe->shm_recons_V, chroma_size);

  CUDA_ALLOC_SHARED(pipe->shm_refframe_Y, frame_size);
  CUDA_ALLOC_SHARED(pipe->shm_refframe_U, chroma_size);
  CUDA_ALLOC_SHARED(pipe->shm_refframe_V, chroma_size);

  CUDA_ALLOC_SHARED(pipe->shm_predicted_Y, frame_size);
  CUDA_ALLOC_SHARED(pipe->shm_predicted_U, chroma_size);
  CUDA_ALLOC_SHARED(pipe->shm_predicted_V, chroma_size);

  CUDA_ALLOC_SHARED(pipe->shm_mbs_Y, num_blocks_luma * sizeof(struct macroblock));
  CUDA_ALLOC_SHARED(pipe->shm_mbs_U, num_blocks_chroma * sizeof(struct macroblock));
  CUDA_ALLOC_SHARED(pipe->shm_mbs_V, num_blocks_chroma * sizeof(struct macroblock));

  pipe->residuals_Y = (int16_t *)malloc(frame_size * sizeof(int16_t));
  pipe->residuals_U = (int16_t *)malloc(chroma_size * sizeof(int16_t));
  pipe->residuals_V = (int16_t *)malloc(chroma_size * sizeof(int16_t));

  // background
  pipe->prev_residuals_Y = (int16_t *)malloc(frame_size * sizeof(int16_t));
  pipe->prev_residuals_U = (int16_t *)malloc(chroma_size * sizeof(int16_t));
  pipe->prev_residuals_V = (int16_t *)malloc(chroma_size * sizeof(int16_t));

  CUDA_ALLOC_SHARED(pipe->shm_prev_mbs_Y, num_blocks_luma * sizeof(struct macroblock));
  CUDA_ALLOC_SHARED(pipe->shm_prev_mbs_U, num_blocks_chroma * sizeof(struct macroblock));
  CUDA_ALLOC_SHARED(pipe->shm_prev_mbs_V, num_blocks_chroma * sizeof(struct macroblock));

  // synchronization
  CUDA_ASSERT(cudaEventCreate(&pipe->event_estimate_Y));
  CUDA_ASSERT(cudaEventCreate(&pipe->event_estimate_U));
  CUDA_ASSERT(cudaEventCreate(&pipe->event_estimate_V));

  CUDA_ASSERT(cudaEventCreate(&pipe->event_compensate_Y));
  CUDA_ASSERT(cudaEventCreate(&pipe->event_compensate_U));
  CUDA_ASSERT(cudaEventCreate(&pipe->event_compensate_V));

  CUDA_ASSERT(cudaStreamCreate(&pipe->stream_image));
  CUDA_ASSERT(cudaStreamCreate(&pipe->stream_estimate_Y));
  CUDA_ASSERT(cudaStreamCreate(&pipe->stream_estimate_U));
  CUDA_ASSERT(cudaStreamCreate(&pipe->stream_estimate_V));
  CUDA_ASSERT(cudaStreamCreate(&pipe->stream_compensate_Y));
  CUDA_ASSERT(cudaStreamCreate(&pipe->stream_compensate_U));
  CUDA_ASSERT(cudaStreamCreate(&pipe->stream_compensate_V));

  return pipe;
}

void c63_pipeline_free(struct c63_pipeline *pipe)
{
  cudaFreeHost(pipe->shm_orig_Y);
  cudaFreeHost(pipe->shm_orig_U);
  cudaFreeHost(pipe->shm_orig_V);

  cudaFreeHost(pipe->shm_next_Y);
  cudaFreeHost(pipe->shm_next_U);
  cudaFreeHost(pipe->shm_next_V);

  cudaFreeHost(pipe->shm_recons_Y);
  cudaFreeHost(pipe->shm_recons_U);
  cudaFreeHost(pipe->shm_recons_V);

  cudaFreeHost(pipe->shm_refframe_Y);
  cudaFreeHost(pipe->shm_refframe_U);
  cudaFreeHost(pipe->shm_refframe_V);

  cudaFreeHost(pipe->shm_predicted_Y);
  cudaFreeHost(pipe->shm_predicted_U);
  cudaFreeHost(pipe->shm_predicted_V);

  free(pipe->residuals_Y);
  free(pipe->residuals_U);
  free(pipe->residuals_V);

  cudaFreeHost(pipe->shm_mbs_Y);
  cudaFreeHost(pipe->shm_mbs_U);
  cudaFreeHost(pipe->shm_mbs_V);

  // background
  free(pipe->prev_residuals_Y);
  free(pipe->prev_residuals_U);
  free(pipe->prev_residuals_V);

  cudaFreeHost(pipe->shm_prev_mbs_Y);
  cudaFreeHost(pipe->shm_prev_mbs_U);
  cudaFreeHost(pipe->shm_prev_mbs_V);

  // synchronization
  cudaEventDestroy(pipe->event_estimate_Y);
  cudaEventDestroy(pipe->event_estimate_U);
  cudaEventDestroy(pipe->event_estimate_V);

  cudaEventDestroy(pipe->event_compensate_Y);
  cudaEventDestroy(pipe->event_compensate_U);
  cudaEventDestroy(pipe->event_compensate_V);

  cudaStreamDestroy(pipe->stream_image);
  cudaStreamDestroy(pipe->stream_estimate_Y);
  cudaStreamDestroy(pipe->stream_estimate_U);
  cudaStreamDestroy(pipe->stream_estimate_V);
  cudaStreamDestroy(pipe->stream_compensate_Y);
  cudaStreamDestroy(pipe->stream_compensate_U);
  cudaStreamDestroy(pipe->stream_compensate_V);
}

struct frame* create_frame(struct c63_common *cm, int role)
{
  struct frame *f = (struct frame *)calloc(1, sizeof(struct frame));

  f->orig = (yuv_t *)malloc(sizeof(yuv_t));
  f->recons = (yuv_t *)malloc(sizeof(yuv_t));
  f->predicted = (yuv_t *)malloc(sizeof(yuv_t));
  f->residuals = (dct_t *)malloc(sizeof(dct_t));

  switch (role) {
    case FRAME_CUR:
      f->orig->Y = cm->pipe->shm_orig_Y;
      f->orig->U = cm->pipe->shm_orig_U;
      f->orig->V = cm->pipe->shm_orig_V;

      f->residuals->Ydct = cm->pipe->residuals_Y;
      f->residuals->Udct = cm->pipe->residuals_U;
      f->residuals->Vdct = cm->pipe->residuals_V;

      f->mbs[Y_COMPONENT] = cm->pipe->shm_mbs_Y;
      f->mbs[U_COMPONENT] = cm->pipe->shm_mbs_U;
      f->mbs[V_COMPONENT] = cm->pipe->shm_mbs_V;
      break;

    case FRAME_NEXT:
      f->orig->Y = cm->pipe->shm_next_Y;
      f->orig->U = cm->pipe->shm_next_U;
      f->orig->V = cm->pipe->shm_next_V;

      f->residuals->Ydct = cm->pipe->prev_residuals_Y;
      f->residuals->Udct = cm->pipe->prev_residuals_U;
      f->residuals->Vdct = cm->pipe->prev_residuals_V;

      f->mbs[Y_COMPONENT] = cm->pipe->shm_prev_mbs_Y;
      f->mbs[U_COMPONENT] = cm->pipe->shm_prev_mbs_U;
      f->mbs[V_COMPONENT] = cm->pipe->shm_prev_mbs_V;
      break;

    case FRAME_REF:
      // not used for orig/residuals
      // old predicted becomes new recons
      break;
  }

  // These are shared in all modes
  f->recons->Y = cm->pipe->shm_recons_Y;
  f->recons->U = cm->pipe->shm_recons_U;
  f->recons->V = cm->pipe->shm_recons_V;

  f->predicted->Y = cm->pipe->shm_predicted_Y;
  f->predicted->U = cm->pipe->shm_predicted_U;
  f->predicted->V = cm->pipe->shm_predicted_V;

  if (role != FRAME_REF) {
    cudaMemset(f->mbs[Y_COMPONENT], 0, cm->num_mbs_luma * sizeof(struct macroblock));
    cudaMemset(f->mbs[U_COMPONENT], 0, cm->num_mbs_chroma * sizeof(struct macroblock));
    cudaMemset(f->mbs[V_COMPONENT], 0, cm->num_mbs_chroma * sizeof(struct macroblock));
  }

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

void prepare_next_frame(struct c63_common *cm)
{
  struct frame *f = cm->curframe;
  cm->refframe = cm->curframe;
  cm->curframe = cm->nextframe;
  cm->nextframe = f;

  SWAP_POINTERS(cm->pipe->shm_prev_mbs_Y, cm->pipe->shm_mbs_Y, struct macroblock *);
  SWAP_POINTERS(cm->pipe->shm_prev_mbs_U, cm->pipe->shm_mbs_U, struct macroblock *);
  SWAP_POINTERS(cm->pipe->shm_prev_mbs_V, cm->pipe->shm_mbs_V, struct macroblock *);

  SWAP_POINTERS(cm->pipe->prev_residuals_Y, cm->pipe->residuals_Y, int16_t *);
  SWAP_POINTERS(cm->pipe->prev_residuals_U, cm->pipe->residuals_U, int16_t *);
  SWAP_POINTERS(cm->pipe->prev_residuals_V, cm->pipe->residuals_V, int16_t *);
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
