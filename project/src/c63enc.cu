#include <assert.h>
#include <complex.h>
#include <errno.h>
#include <limits.h>
#include <math.h>
#include <pthread.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "c63.h"
#include "c63enc.h"
#include "c63_write.h"
#include "c63server.h"
#include "quantdct.h"
#include "common.h"
#include "me.h"
#include "tables.h"
#include "profiling.h"
#include "pipeline.h"

static int pthread_total_threads;
static pthread_t *threads;
static int dirty = 1;

void cleanup_cm(void) {
  if (!dirty) return;
  for (int i = 0; i < pthread_total_threads; i++) {
    pthread_cancel(threads[i]);
  }
}

void prepare_next_frame(struct c63_encoder *enc)
{
  struct frame *f = enc->curframe;
  enc->refframe = enc->curframe;
  enc->curframe = enc->nextframe;
  enc->nextframe = f;
}


struct c63_encoder *c63_encoder_init(struct c63_common *cm)
{
  struct c63_encoder *enc = (struct c63_encoder *) calloc(1, sizeof(struct c63_encoder));
  enc->cm = cm;

  // pipeline buffer init
  enc->pipe = c63_pipeline_init(
    cm->luma_size, cm->chroma_size,
    cm->num_mbs_luma, cm->num_mbs_chroma
  );

  // allocate frames
  enc->curframe  = create_frame(enc->pipe, FRAME_CUR);
  enc->nextframe = create_frame(enc->pipe, FRAME_NEXT);
  enc->refframe  = NULL;


  // tables and constants
  c63_initialize_constant_values(cm);
  initialize_dctlookup_values();

  // threads setup
  enc->pthreads_run           = 1;
  enc->pthreads_luma_threads   = 4;
  enc->pthreads_chroma_threads = 2;

  for (int c = 0; c < TASK_POOLS; ++c) {
    enc->pth_next_row[c] = 0;
    pthread_mutex_init(&enc->pth_mutex_next_row[c], NULL);
  }

  int nworkers = enc->pthreads_luma_threads + enc->pthreads_chroma_threads;
  pthread_total_threads = nworkers;
  threads = (pthread_t*)calloc(pthread_total_threads, sizeof(pthread_t));

  pthread_barrier_init(&enc->pth_barrier_dct_idct_start, NULL, nworkers + 1);
  pthread_barrier_init(&enc->pth_barrier_dct_idct_end,   NULL, nworkers + 1);

  // worker threads
  struct worker_ctx *ctx = (struct worker_ctx*)malloc(nworkers * sizeof(*ctx));
  int t = 0;
  for (int i = 0; i < enc->pthreads_luma_threads; ++i, ++t) {
    ctx[t].enc = enc;
    ctx[t].component = TASK_LUMA;
    pthread_create(&threads[t], NULL, pthread_dct_idct, &ctx[t]);
  }

  for (int i = 0; i < enc->pthreads_chroma_threads; ++i, ++t) {
    ctx[t].enc = enc;
    ctx[t].component = TASK_CHROMA;
    pthread_create(&threads[t], NULL, pthread_dct_idct, &ctx[t]);
  }

  atexit(cleanup_cm);
  return enc;
}

void c63_encoder_free(struct c63_encoder *enc) {
  destroy_frame(enc->curframe);
  destroy_frame(enc->nextframe);

  c63_pipeline_free(enc->pipe);
  c63_common_free(enc->cm);

  free(enc);
  dirty = 0;
}


void c63_encode_image(struct c63_encoder *enc) {
  prepare_next_frame(enc);

  if (!enc->nextframe->keyframe)     c63_motion_estimate(enc);
  if (!enc->curframe->keyframe)      c63_motion_compensate(enc);

  pthread_barrier_wait(&enc->pth_barrier_dct_idct_start);
  pthread_barrier_wait(&enc->pth_barrier_dct_idct_end);
}

