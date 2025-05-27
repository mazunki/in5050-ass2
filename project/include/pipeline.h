#ifndef C63_PIPELINE_H_
#define C63_PIPELINE_H_

#include <stddef.h>
#include <stdint.h>

struct c63_pipeline {
  uint8_t *shm_orig_Y, *shm_orig_U, *shm_orig_V;
  uint8_t *shm_recons_Y, *shm_recons_U, *shm_recons_V;
  uint8_t *shm_refframe_Y, *shm_refframe_U, *shm_refframe_V;
  uint8_t *shm_predicted_Y, *shm_predicted_U, *shm_predicted_V;
  struct macroblock *shm_mbs_Y, *shm_mbs_U, *shm_mbs_V;
  int16_t *residuals_Y, *residuals_U, *residuals_V;

  // preemptively reading
  uint8_t *shm_next_Y, *shm_next_U, *shm_next_V;
  uint8_t *shm_next_predicted_Y, *shm_next_predicted_U, *shm_next_predicted_V;
  // writing in the background
  struct macroblock *shm_prev_mbs_Y, *shm_prev_mbs_U, *shm_prev_mbs_V;
  int16_t *prev_residuals_Y, *prev_residuals_U, *prev_residuals_V;

#ifdef __CUDACC__ // CUDA contexts
  cudaStream_t stream_estimate_Y, stream_estimate_U, stream_estimate_V;
  cudaStream_t stream_compensate_Y, stream_compensate_U, stream_compensate_V;

  cudaStream_t stream_dct_Y, stream_dct_U, stream_dct_V;
  cudaStream_t stream_idct_Y, stream_idct_U, stream_idct_V;

  cudaEvent_t event_estimate_Y, event_estimate_U, event_estimate_V;
  cudaEvent_t event_compensate_Y, event_compensate_U, event_compensate_V;
#endif
};

struct c63_pipeline* c63_pipeline_init(size_t frame_size, size_t chroma_size, size_t num_blocks_luma, size_t num_blocks_chroma);

void c63_pipeline_free(struct c63_pipeline *pipe);

#endif // C63_PIPELINE_H_

