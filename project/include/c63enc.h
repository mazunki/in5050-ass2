#ifndef C63ENC_H
#define C63ENC_H

#include "c63.h"

#ifdef __cplusplus
extern "C" {
#endif

struct c63_encoder {
  struct c63_common *cm;
  struct c63_pipeline *pipe;

  struct frame *refframe;
  struct frame *curframe;
  struct frame *nextframe;

  int framenum;


  int pthreads_run;
  int pthreads_luma_threads, pthreads_chroma_threads;

  //pthread_t pth_dct_idct[COLOR_COMPONENTS];
  uint32_t        pth_next_row[TASK_POOLS];      // task counters
  pthread_mutex_t pth_mutex_next_row[TASK_POOLS];

  pthread_barrier_t pth_barrier_dct_idct_start;
  pthread_barrier_t pth_barrier_dct_idct_end;

  pthread_barrier_t pth_barrier_idct_start;
  pthread_barrier_t pth_barrier_idct_end;

  //pthread_t pth_write_frame;
  pthread_mutex_t pth_mutex_write_frame;
  pthread_cond_t pth_cond_write_frame;
  int pth_pending_write_frame;
  int pth_done_write_frame;

  struct frame *unwritten_frame;
};


struct c63_encoder *c63_encoder_init(struct c63_common *cm);

void c63_encode_image(struct c63_encoder *enc);
void prepare_next_frame(struct c63_encoder *enc);


void cleanup_cm(void);
void c63_encoder_free(struct c63_encoder *enc);

#ifdef __cplusplus
}
#endif

#endif // C63ENC_H

