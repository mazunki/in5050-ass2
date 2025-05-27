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

  CUDA_ALLOC_SHARED(pipe->shm_next_predicted_Y, frame_size);
  CUDA_ALLOC_SHARED(pipe->shm_next_predicted_U, chroma_size);
  CUDA_ALLOC_SHARED(pipe->shm_next_predicted_V, chroma_size);

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

  CUDA_ASSERT(cudaStreamCreate(&pipe->stream_estimate_Y));
  CUDA_ASSERT(cudaStreamCreate(&pipe->stream_estimate_U));
  CUDA_ASSERT(cudaStreamCreate(&pipe->stream_estimate_V));
  CUDA_ASSERT(cudaStreamCreate(&pipe->stream_compensate_Y));
  CUDA_ASSERT(cudaStreamCreate(&pipe->stream_compensate_U));
  CUDA_ASSERT(cudaStreamCreate(&pipe->stream_compensate_V));

  CUDA_ASSERT(cudaStreamCreate(&pipe->stream_dct_Y));
  CUDA_ASSERT(cudaStreamCreate(&pipe->stream_dct_U));
  CUDA_ASSERT(cudaStreamCreate(&pipe->stream_dct_V));
  CUDA_ASSERT(cudaStreamCreate(&pipe->stream_idct_Y));
  CUDA_ASSERT(cudaStreamCreate(&pipe->stream_idct_U));
  CUDA_ASSERT(cudaStreamCreate(&pipe->stream_idct_V));

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

  cudaFreeHost(pipe->shm_next_predicted_Y);
  cudaFreeHost(pipe->shm_next_predicted_U);
  cudaFreeHost(pipe->shm_next_predicted_V);

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

  cudaStreamDestroy(pipe->stream_estimate_Y);
  cudaStreamDestroy(pipe->stream_estimate_U);
  cudaStreamDestroy(pipe->stream_estimate_V);
  cudaStreamDestroy(pipe->stream_compensate_Y);
  cudaStreamDestroy(pipe->stream_compensate_U);
  cudaStreamDestroy(pipe->stream_compensate_V);

  cudaStreamDestroy(pipe->stream_dct_Y);
  cudaStreamDestroy(pipe->stream_dct_U);
  cudaStreamDestroy(pipe->stream_dct_V);
  cudaStreamDestroy(pipe->stream_idct_Y);
  cudaStreamDestroy(pipe->stream_idct_U);
  cudaStreamDestroy(pipe->stream_idct_V);
}

