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
#include "quantdct.h"
#include "common.h"
#include "me.h"
#include "tables.h"
#include "profiling.h"

static int pthread_total_threads;
static pthread_t *threads;
static int dirty = 1;

extern "C" void cleanup_cm(void) {
    if (!dirty) return;
    for (int i = 0; i < pthread_total_threads; i++) {
        pthread_cancel(threads[i]);
    }
}

extern "C" struct c63_common* init_c63_enc(int width, int height) {
    c63_common *cm = (c63_common*)calloc(1, sizeof(struct c63_common));
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

    // pipeline buffer init
    cm->pipe = c63_pipeline_init(
        cm->luma_size, cm->chroma_size,
        cm->num_mbs_luma, cm->num_mbs_chroma
    );

    // allocate frames
    cm->curframe  = create_frame(cm, FRAME_CUR);
    cm->nextframe = create_frame(cm, FRAME_NEXT);
    cm->refframe  = NULL;

    // tables and constants
    c63_initialize_constant_values(cm);
    initialize_dctlookup_values();

    // threads setup
    cm->pthreads_run           = 1;
    cm->pthreads_luma_threads   = 4;
    cm->pthreads_chroma_threads = 2;

    for (int c = 0; c < TASK_POOLS; ++c) {
        cm->pth_next_row[c] = 0;
        pthread_mutex_init(&cm->pth_mutex_next_row[c], NULL);
    }

    int nworkers = cm->pthreads_luma_threads + cm->pthreads_chroma_threads;
    pthread_total_threads = nworkers + 1;  // +1 writer
    threads = (pthread_t*)calloc(pthread_total_threads, sizeof(pthread_t));

    pthread_barrier_init(&cm->pth_barrier_dct_idct_start, NULL, nworkers + 1);
    pthread_barrier_init(&cm->pth_barrier_dct_idct_end,   NULL, nworkers + 1);
    pthread_mutex_init(&cm->pth_mutex_write_frame, NULL);
    pthread_cond_init(&cm->pth_cond_write_frame, NULL);

    // writer thread
    pthread_create(&threads[nworkers], NULL, pthread_write_frame, cm);

    // worker threads
    struct worker_ctx *ctx = (struct worker_ctx*)malloc(nworkers * sizeof(*ctx));
    int t = 0;
    for (int i = 0; i < cm->pthreads_luma_threads; ++i, ++t) {
        ctx[t].cm = cm;
        ctx[t].component = TASK_LUMA;
        pthread_create(&threads[t], NULL, pthread_dct_idct, &ctx[t]);
    }
    for (int i = 0; i < cm->pthreads_chroma_threads; ++i, ++t) {
        ctx[t].cm = cm;
        ctx[t].component = TASK_CHROMA;
        pthread_create(&threads[t], NULL, pthread_dct_idct, &ctx[t]);
    }

    atexit(cleanup_cm);
    return cm;
}

extern "C" void c63_encode_image(struct c63_common *cm) {
    prepare_next_frame(cm);
    if (!cm->nextframe->keyframe)     c63_motion_estimate(cm);
    if (!cm->curframe->keyframe)      c63_motion_compensate(cm);
    pthread_barrier_wait(&cm->pth_barrier_dct_idct_start);
    pthread_barrier_wait(&cm->pth_barrier_dct_idct_end);
    pthread_mutex_lock(&cm->pth_mutex_write_frame);
    cm->unwritten_frame = cm->curframe;
    pthread_cond_signal(&cm->pth_cond_write_frame);
    pthread_mutex_unlock(&cm->pth_mutex_write_frame);
}

extern "C" void free_c63_enc(struct c63_common* cm) {
    c63_pipeline_free(cm->pipe);
    destroy_frame(cm->curframe);
    destroy_frame(cm->nextframe);
    free(cm);
    dirty = 0;
}

