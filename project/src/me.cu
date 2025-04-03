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

#include <nvToolsExt.h>


// estimation
__global__ void c63_motion_estimate_kernel(uint8_t *d_orig, uint8_t *d_recons, macroblock *d_mbs, int comp);
__device__ static int sad_block_8x8(uint8_t *block1, uint8_t *block2, int stride);


// compensation
__global__ void c63_motion_compensate_kernel(struct macroblock *d_mbs, uint8_t *d_predicted, uint8_t *d_ref, int comp);
__device__ void mc_block_8x8(struct macroblock *mbs, uint8_t *predicted, uint8_t *ref, int comp);

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
  * reconstructing the previous frame
  *
  * This is only used during encoding, since the decoder only
  * has the original image during key-frames.
  *
  * @param[in]  d_orig
  * @param[in]  d_recons
  * @param[out] d_mbs
  */
 __host__ void c63_motion_estimate(struct c63_common *cm)
 {
   dim3 block_size(MACROBLOCK_SIZE, MACROBLOCK_SIZE);
   dim3 grid_size_luma(cm->mb_cols_luma, cm->mb_rows_luma);
   dim3 grid_size_chroma(cm->mb_cols_chroma, cm->mb_rows_chroma);

   c63_pipeline *pipe = cm->pipe;

   c63_motion_estimate_kernel<<<grid_size_luma, block_size, 0, pipe->stream_estimate_Y>>>(cm->curframe->orig->Y, cm->refframe->recons->Y, cm->curframe->mbs[Y_COMPONENT], Y_COMPONENT);
   CUDA_ASSERT(cudaEventRecord(pipe->event_estimate_Y, pipe->stream_estimate_Y));

   c63_motion_estimate_kernel<<<grid_size_chroma, block_size, 0, pipe->stream_estimate_U>>>(cm->curframe->orig->U, cm->refframe->recons->U, cm->curframe->mbs[U_COMPONENT], U_COMPONENT);
   CUDA_ASSERT(cudaEventRecord(pipe->event_estimate_U, pipe->stream_estimate_U));

   c63_motion_estimate_kernel<<<grid_size_chroma, block_size, 0, pipe->stream_estimate_V>>>(cm->curframe->orig->V, cm->refframe->recons->V, cm->curframe->mbs[V_COMPONENT], V_COMPONENT);
   CUDA_ASSERT(cudaEventRecord(pipe->event_estimate_V, pipe->stream_estimate_V));
 }

 /**
  * @brief Sums up the Sum of Absolute Difference between two blocks.
  *
  * This value can then be used to pick the best match for any given
  * macroblock during motion estimation.
  */
  __device__ int sad_block_8x8(uint8_t *blk1, uint8_t *blk2, int stride)
  {
    int sad = 0;
    for (int i = 0; i < MACROBLOCK_SIZE; ++i)
    {
      for (int j = 0; j < MACROBLOCK_SIZE; ++j)
      {
        sad += abs((int)blk1[i * stride + j] - (int)blk2[i * stride + j]);
      }
    }
    return sad;
  }


 __global__ void c63_motion_estimate_kernel(uint8_t *orig, uint8_t *ref, macroblock *mbs, int comp)
 {
   int mb_x = blockIdx.x;
   int mb_y = blockIdx.y;
   int thread_id = threadIdx.y * blockDim.x + threadIdx.x;

   if (mb_x >= c_mb_cols[comp] || mb_y >= c_mb_rows[comp]) return;

   int padw = c_padw[comp];
   int padh = c_padh[comp];
   int range = c_me_search_range;

   int mx = mb_x * 8;
   int my = mb_y * 8;

   macroblock *mb = &mbs[mb_y * c_mb_cols[comp] + mb_x];

   int best_sad = INT_MAX;
   int best_dx = 0;
   int best_dy = 0;

   for (int dy = -range; dy <= range; ++dy)
   {
     for (int dx = -range; dx <= range; ++dx)
     {
       int ref_x = mx + dx;
       int ref_y = my + dy;

       if (ref_x < 0 || ref_y < 0 || ref_x + MACROBLOCK_SIZE > padw || ref_y + MACROBLOCK_SIZE > padh)
       {
         continue;
       }

       uint8_t *blk_orig = &orig[my * padw + mx];
       uint8_t *blk_ref = &ref[ref_y * padw + ref_x];
       int sad = sad_block_8x8(blk_orig, blk_ref, padw);

       if (sad < best_sad)
       {
         best_sad = sad;
         best_dx = dx;
         best_dy = dy;
       }
     }
   }

   if (thread_id == 0)
   {
     mb->mv_x = best_dx;
     mb->mv_y = best_dy;
     mb->use_mv = 1;
   }
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
  * @param[in]  d_mbs
  * @param[out] d_predicted
  * @param[in]  d_ref
  */

  __global__ void mc_block_8x8_kernel(uint8_t *predicted, const uint8_t *ref, struct macroblock *mbs, int w, int mb_width)
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
    dim3 block_size(8, 8);
    dim3 grid_size_luma(cm->mb_cols_luma, cm->mb_rows_luma);
    dim3 grid_size_chroma(cm->mb_cols_chroma, cm->mb_rows_chroma);

    cudaStreamWaitEvent(cm->pipe->stream_compensate_Y, cm->pipe->event_estimate_Y);
    mc_block_8x8_kernel<<<grid_size_luma,   block_size, 0, cm->pipe->stream_compensate_Y>>>( cm->curframe->predicted->Y, cm->refframe->recons->Y, cm->curframe->mbs[Y_COMPONENT], cm->padw[Y_COMPONENT], cm->mb_cols_luma );
    cudaEventRecord(cm->pipe->event_compensate_Y, cm->pipe->stream_compensate_Y);

    cudaStreamWaitEvent(cm->pipe->stream_compensate_U, cm->pipe->event_estimate_U);
    mc_block_8x8_kernel<<<grid_size_chroma, block_size, 0, cm->pipe->stream_compensate_U>>>( cm->curframe->predicted->U, cm->refframe->recons->U, cm->curframe->mbs[U_COMPONENT], cm->padw[U_COMPONENT], cm->mb_cols_chroma );
    cudaEventRecord(cm->pipe->event_compensate_U, cm->pipe->stream_compensate_U);

    cudaStreamWaitEvent(cm->pipe->stream_compensate_V, cm->pipe->event_estimate_V);
    mc_block_8x8_kernel<<<grid_size_chroma, block_size, 0, cm->pipe->stream_compensate_V>>>( cm->curframe->predicted->V, cm->refframe->recons->V, cm->curframe->mbs[V_COMPONENT], cm->padw[V_COMPONENT], cm->mb_cols_chroma );
    cudaEventRecord(cm->pipe->event_compensate_V, cm->pipe->stream_compensate_V);
  }
