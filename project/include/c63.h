#ifndef C63_C63_H_
#define C63_C63_H_

#include <inttypes.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>

#define MAX_FILELENGTH 200
#define DEFAULT_OUTPUT_FILE "a.mjpg"

#define FRAMEBUFFER_SIZE 2

#define PI 3.14159265358979
#define ILOG2 1.442695040888963 // 1/log(2);

#define COLOR_COMPONENTS 3

#define Y_COMPONENT 0
#define U_COMPONENT 1
#define V_COMPONENT 2

#define YX 2
#define YY 2
#define UX 1
#define UY 1
#define VX 1
#define VY 1

/* The JPEG file format defines several parts and each part is defined by a
 marker. A file always starts with 0xFF and is then followed by a magic number,
 e.g., like 0xD8 in the SOI marker below. Some markers have a payload, and if
 so, the size of the payload is written before the payload itself. */

#define JPEG_DEF_MARKER 0xFF
#define JPEG_SOI_MARKER 0xD8
#define JPEG_DQT_MARKER 0xDB
#define JPEG_SOF_MARKER 0xC0
#define JPEG_DHT_MARKER 0xC4
#define JPEG_SOS_MARKER 0xDA
#define JPEG_EOI_MARKER 0xD9

#define HUFF_AC_ZERO 16
#define HUFF_AC_SIZE 11

#define MIN(a,b) ((a) < (b) ? (a) : (b))
#define MAX(a,b) ((a) > (b) ? (a) : (b))
#define ARRAY_SIZE(x) (sizeof(x) / sizeof((x)[0]))

struct yuv
{
  uint8_t *Y;
  uint8_t *U;
  uint8_t *V;
};

struct dct
{
  int16_t *Ydct;
  int16_t *Udct;
  int16_t *Vdct;
};

typedef struct yuv yuv_t;
typedef struct dct dct_t;

struct entropy_ctx
{
  FILE *fp;
  unsigned int bit_buffer;
  unsigned int bit_buffer_width;
};

struct macroblock
{
  int use_mv;
  int8_t mv_x, mv_y;
};

struct frame
{
  yuv_t *orig;        // Original input image
  yuv_t *recons;      // Reconstructed image
  yuv_t *predicted;   // Predicted frame from intra-prediction

  dct_t *residuals;   // Difference between original image and predicted frame

  struct macroblock *mbs[COLOR_COMPONENTS];
  int keyframe;
};

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

struct c63_common
{
  int width, height;
  int ypw, yph, upw, uph, vpw, vph;

  int padw[COLOR_COMPONENTS], padh[COLOR_COMPONENTS];

  size_t luma_size, chroma_size;
  int mb_cols_luma, mb_rows_luma;
  int mb_cols_chroma, mb_rows_chroma;
  size_t num_mbs_luma, num_mbs_chroma;

  uint8_t qp;                         // Quality parameter

  int me_search_range;

  uint8_t quanttbl[COLOR_COMPONENTS][64];

  struct frame *refframe;
  struct frame *curframe;
  struct frame *nextframe;

  int framenum;

  int keyframe_interval;
  int frames_since_keyframe;

  struct entropy_ctx e_ctx;
  struct c63_pipeline *pipe;

  int pthreads_run;

  //pthread_t pth_dct_idct[COLOR_COMPONENTS];
  pthread_barrier_t pth_barrier_dct_start;
  pthread_barrier_t pth_barrier_dct_end;

  pthread_barrier_t pth_barrier_idct_start;
  pthread_barrier_t pth_barrier_idct_end;

  //pthread_t pth_write_frame;
  pthread_mutex_t pth_mutex_write_frame;
  pthread_cond_t pth_cond_write_frame;
  int pth_pending_write_frame;
  int pth_done_write_frame;

  struct frame *unwritten_frame;
};

#endif  /* C63_C63_H_ */
