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
  uint8_t *d_orig_Y, *d_orig_U, *d_orig_V;
  uint8_t *d_recons_Y, *d_recons_U, *d_recons_V;
  uint8_t *d_refframe_Y, *d_refframe_U, *d_refframe_V;
  uint8_t *d_predicted_Y, *d_predicted_U, *d_predicted_V;
  struct macroblock *d_mbs[COLOR_COMPONENTS];

  yuv_t *h_refframe, *h_recons;  // note that these pointers are swapped each frame
  yuv_t *h_predicted;
  dct_t *h_residuals, *unwritten_residuals;
  struct macroblock *h_mbs[COLOR_COMPONENTS], *unwritten_mbs[COLOR_COMPONENTS];

#ifdef __CUDACC__ // CUDA contexts
  cudaStream_t stream_estimate_Y, stream_estimate_U, stream_estimate_V;
  cudaStream_t stream_compensate_Y, stream_compensate_U, stream_compensate_V;

  cudaStream_t stream_macroblocks_Y, stream_macroblocks_U, stream_macroblocks_V;
  cudaStream_t stream_predictions_Y, stream_predictions_U, stream_predictions_V;
  cudaStream_t stream_image;

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

  int framenum;

  int keyframe_interval;
  int frames_since_keyframe;

  struct entropy_ctx e_ctx;
  struct c63_pipeline *pipe;
  yuv_t *frame_buffer[FRAMEBUFFER_SIZE];
  int fb_curr_index;

  pthread_t pth_dct_idct[COLOR_COMPONENTS];
  pthread_mutex_t pth_mutex_dct_idct;
  pthread_cond_t pth_cond_dct_idct_ready, pth_cond_dct_idct_done;
  int pth_pending_dct_idct[COLOR_COMPONENTS];
  int pth_barrier_dct_idct;

  pthread_t pth_write_frame;
  pthread_mutex_t pth_mutex_write_frame;
  pthread_cond_t pth_cond_write_frame;
  int pth_pending_write_frame;
  int pth_done_write_frame;

  struct frame *unwritten_frame;
};

#endif  /* C63_C63_H_ */
