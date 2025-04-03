#include <assert.h>
#include <errno.h>
#include <getopt.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <inttypes.h>
#include <math.h>
#include <stdlib.h>

#include "c63.h"
#include "common.h"
#include "me.h"
#include "tables.h"

#include "profiling.h"


/** constant memory */
__constant__ int c_padw[COLOR_COMPONENTS];
__constant__ int c_padh[COLOR_COMPONENTS];
__constant__ int c_mb_cols[COLOR_COMPONENTS];
__constant__ int c_mb_rows[COLOR_COMPONENTS];
__constant__ int c_me_search_range;

void c63_initialize_constant_values(struct c63_common *cm)
{
    int padw[3] = {cm->padw[Y_COMPONENT], cm->padw[U_COMPONENT], cm->padw[V_COMPONENT]};
    int padh[3] = {cm->padh[Y_COMPONENT], cm->padh[U_COMPONENT], cm->padh[V_COMPONENT]};
    int mb_cols[3] = {cm->mb_cols_luma, cm->mb_cols_chroma, cm->mb_cols_chroma};
    int mb_rows[3] = {cm->mb_rows_luma, cm->mb_rows_chroma, cm->mb_rows_chroma};
    int me_range = cm->me_search_range;

    cudaMemcpyToSymbol(c_padw, padw, sizeof(padw));
    cudaMemcpyToSymbol(c_padh, padh, sizeof(padh));
    cudaMemcpyToSymbol(c_mb_cols, mb_cols, sizeof(mb_cols));
    cudaMemcpyToSymbol(c_mb_rows, mb_rows, sizeof(mb_rows));
    cudaMemcpyToSymbol(c_me_search_range, &me_range, sizeof(int));
}

 /**
  * @brief Motion estimation
  *
  * Motion estimation calculates the motion vectors using
  * the original image from a reference image (made by
  * reconstructing the previous frame)
  *
  * This is only used during encoding, since the decoder only
  * has the original image during key-frames.
  *
  * @param[in]  curr orig
  * @param[in]  prev recons
  * @param[out] curr mbs
  */

 /**
  * @brief Sums up the Sum of Absolute Difference between two blocks.
  *
  * This value can then be used to pick the best match for any given
  * macroblock during motion estimation.
  */
  // Serial SAD computation — safe for small blocks
  __device__ static int sad_block_8x8(uint8_t *block1, uint8_t *block2, int stride)
  {
    int sad = 0;
    for (int i = 0; i < MACROBLOCK_SIZE; ++i)
    {
      for (int j = 0; j < MACROBLOCK_SIZE; ++j)
      {
        sad += abs((int)block1[i * stride + j] - (int)block2[i * stride + j]);
      }
    }
    return sad;
  }


  // Perform motion estimation for a full macroblock
  __device__ static void me_block_8x8(struct macroblock *mb, int mb_x, int mb_y,
                                      uint8_t *orig, uint8_t *ref, int padw, int padh, int range)
  {
    int left   = MAX(mb_x * MACROBLOCK_SIZE - range, 0);
    int top    = MAX(mb_y * MACROBLOCK_SIZE - range, 0);
    int right  = MIN(mb_x * MACROBLOCK_SIZE + range, padw - MACROBLOCK_SIZE);
    int bottom = MIN(mb_y * MACROBLOCK_SIZE + range, padh - MACROBLOCK_SIZE);

    int mx = mb_x * MACROBLOCK_SIZE;
    int my = mb_y * MACROBLOCK_SIZE;

    __shared__ int s_best_sad;
    __shared__ int s_best_mv_x;
    __shared__ int s_best_mv_y;

    if (threadIdx.x == 0 && threadIdx.y == 0)
    {
      s_best_sad = INT_MAX;
      s_best_mv_x = 0;
      s_best_mv_y = 0;
    }
    __syncthreads();

    int local_best_sad = INT_MAX;
    int local_best_x = 0;
    int local_best_y = 0;

    for (int y = top + threadIdx.y; y < bottom; y += blockDim.y)
    {
      for (int x = left + threadIdx.x; x < right; x += blockDim.x)
      {
        int sad = sad_block_8x8(orig + my * padw + mx, ref + y * padw + x, padw);
        if (sad < local_best_sad)
        {
          local_best_sad = sad;
          local_best_x = x - mx;
          local_best_y = y - my;
        }
      }
    }

    // Safe atomicMin — the thread that finds the min SAD will match below
    int old_sad = atomicMin(&s_best_sad, local_best_sad);

    // Only one thread writes mv_x/mv_y
    if (local_best_sad < old_sad)
    {
      s_best_mv_x = local_best_x;
      s_best_mv_y = local_best_y;
    }

    __syncthreads();

    if (threadIdx.x == 0 && threadIdx.y == 0)
    {
      mb->mv_x = s_best_mv_x;
      mb->mv_y = s_best_mv_y;
      mb->use_mv = 1;
    }
  }


__global__ void c63_motion_estimate_kernel(uint8_t *d_orig, uint8_t *d_recons,
                                           macroblock *d_mbs, int comp)
{
  int mb_x = blockIdx.x;
  int mb_y = blockIdx.y;

  if (mb_x >= c_mb_cols[comp] || mb_y >= c_mb_rows[comp]) return;

  macroblock *mb = &d_mbs[mb_y * c_mb_cols[comp] + mb_x];
  me_block_8x8(mb, mb_x, mb_y, d_orig, d_recons, c_padw[comp], c_padh[comp], c_me_search_range);
}

__host__ void c63_motion_estimate(struct c63_common *cm)
{
  startTrace3("estimate");
  dim3 block_size(MACROBLOCK_SIZE, MACROBLOCK_SIZE);
  dim3 grid_size_luma(cm->mb_cols_luma, cm->mb_rows_luma);
  dim3 grid_size_chroma(cm->mb_cols_chroma, cm->mb_rows_chroma);

  c63_pipeline *pipe = cm->pipe;

  c63_motion_estimate_kernel<<<grid_size_luma,   block_size, 0, pipe->stream_estimate_Y>>>(cm->curframe->orig->Y, cm->refframe->recons->Y, cm->curframe->mbs[Y_COMPONENT], Y_COMPONENT);
  CUDA_CHECK();
  CUDA_ASSERT(cudaEventRecord(pipe->event_estimate_Y, pipe->stream_estimate_Y));

  c63_motion_estimate_kernel<<<grid_size_chroma, block_size, 0, pipe->stream_estimate_U>>>(cm->curframe->orig->U, cm->refframe->recons->U, cm->curframe->mbs[U_COMPONENT], U_COMPONENT);
  CUDA_CHECK();
  CUDA_ASSERT(cudaEventRecord(pipe->event_estimate_U, pipe->stream_estimate_U));

  c63_motion_estimate_kernel<<<grid_size_chroma, block_size, 0, pipe->stream_estimate_V>>>(cm->curframe->orig->V, cm->refframe->recons->V, cm->curframe->mbs[V_COMPONENT], V_COMPONENT);
  CUDA_CHECK();
  CUDA_ASSERT(cudaEventRecord(pipe->event_estimate_V, pipe->stream_estimate_V));
  endTrace3();
}




 /**
  * @brief Motion Compensation
  *
  * Motion compensation predicts what a frame would look like
  * using the datablocks provided to it, and the previous
  * frame's reconstructed frame.
  *
  * This is used both during encoding and decoding.
  *
  * @param[in]  curr mbs
  * @param[out] curr predicted
  * @param[in]  curr ref == prev recons
  */

  __global__ void c63_motion_compensate_kernel(uint8_t *predicted, const uint8_t *ref, struct macroblock *mbs, int w, int mb_width)
  {
    int mb_x = blockIdx.x;
    int mb_y = blockIdx.y;

    struct macroblock mb = mbs[mb_y * mb_width + mb_x];
    if (!mb.use_mv) return;

    int dst_x = mb_x * 8 + threadIdx.x;
    int dst_y = mb_y * 8 + threadIdx.y;

    int src_x = dst_x + mb.mv_x;
    int src_y = dst_y + mb.mv_y;

    predicted[dst_y * w + dst_x] = ref[src_y * w + src_x];
  }


  void c63_motion_compensate(struct c63_common *cm)
  {
    startTrace3("compensate");
    dim3 block_size(8, 8);
    dim3 grid_size_luma(cm->mb_cols_luma, cm->mb_rows_luma);
    dim3 grid_size_chroma(cm->mb_cols_chroma, cm->mb_rows_chroma);

    CUDA_ASSERT(cudaStreamWaitEvent(cm->pipe->stream_compensate_Y, cm->pipe->event_estimate_Y));
    c63_motion_compensate_kernel<<<grid_size_luma,   block_size, 0, cm->pipe->stream_compensate_Y>>>(cm->curframe->predicted->Y, cm->refframe->recons->Y, cm->curframe->mbs[Y_COMPONENT], cm->padw[Y_COMPONENT], cm->mb_cols_luma);
    CUDA_CHECK();
    CUDA_ASSERT(cudaEventRecord(cm->pipe->event_compensate_Y, cm->pipe->stream_compensate_Y));

    CUDA_ASSERT(cudaStreamWaitEvent(cm->pipe->stream_compensate_U, cm->pipe->event_estimate_U));
    c63_motion_compensate_kernel<<<grid_size_chroma, block_size, 0, cm->pipe->stream_compensate_U>>>(cm->curframe->predicted->U, cm->refframe->recons->U, cm->curframe->mbs[U_COMPONENT], cm->padw[U_COMPONENT], cm->mb_cols_chroma);
    CUDA_CHECK();
    CUDA_ASSERT(cudaEventRecord(cm->pipe->event_compensate_U, cm->pipe->stream_compensate_U));

    CUDA_ASSERT(cudaStreamWaitEvent(cm->pipe->stream_compensate_V, cm->pipe->event_estimate_V));
    c63_motion_compensate_kernel<<<grid_size_chroma, block_size, 0, cm->pipe->stream_compensate_V>>>(cm->curframe->predicted->V, cm->refframe->recons->V, cm->curframe->mbs[V_COMPONENT], cm->padw[V_COMPONENT], cm->mb_cols_chroma);
    CUDA_CHECK();
    CUDA_ASSERT(cudaEventRecord(cm->pipe->event_compensate_V, cm->pipe->stream_compensate_V));
    endTrace3();
  }
