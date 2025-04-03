#include <assert.h>
#include <complex.h>
#include <errno.h>
#include <getopt.h>
#include <limits.h>
#include <math.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "c63.h"
#include "c63_write.h"
#include "quantdct.h"
#include "common.h"
#include "me.h"
#include "tables.h"

#include "profiling.h"

static char *output_file, *input_file;
FILE *outfile;

static int limit_numframes = 0;

static uint32_t width;
static uint32_t height;

/* getopt */
extern int optind;
extern char *optarg;

#define N_THREADS (COLOR_COMPONENTS + 1)
static pthread_t threads[N_THREADS];
void cleanup_cm(void) {
  for (int i=0; i < N_THREADS; i++) {
    pthread_cancel(threads[i]);
  }
}

static yuv_t* read_yuv(FILE *file, struct c63_common *cm)
{
  size_t len = 0;

  yuv_t *image = cm->nextframe->orig;
  uint8_t *Y = image->Y;
  uint8_t *U = image->U;
  uint8_t *V = image->V;

  len += fread(Y, 1, cm->width*cm->height, file);
  len += fread(U, 1, (cm->width*cm->height)/4, file);
  len += fread(V, 1, (cm->width*cm->height)/4, file);

  if (ferror(file)) {
    perror("ferror");
    exit(EXIT_FAILURE);
  } else if (len != cm->width*cm->height*1.5) {
    fprintf(stderr, "Reached end of file, but incorrect bytes read.\n");
    fprintf(stderr, "Wrong input? (height: %d width: %d)\n", cm->height, cm->width);
    return NULL;
  }

  return image;
}

static void c63_encode_image(struct c63_common *cm)
{
  startTrace1("Encode image");

  c63_pipeline *pipe = cm->pipe;

  prepare_next_frame(cm);

  if (cm->framenum == 0 || cm->frames_since_keyframe == cm->keyframe_interval) {
    cm->curframe->keyframe = 1;
    cm->frames_since_keyframe = 0;
  } else {
    cm->curframe->keyframe = 0;
  }

  if (!cm->curframe->keyframe) {
    startTrace2("stream image");
    CUDA_ASSERT(cudaStreamSynchronize(pipe->stream_image));
    pthread_mutex_lock(&cm->pth_mutex_write_frame);
    pthread_mutex_unlock(&cm->pth_mutex_write_frame);
    endTrace();

    c63_motion_estimate(cm);

    c63_motion_compensate(cm);
  }

  startTrace2("quantize+dequantize");
  pthread_barrier_wait(&cm->pth_barrier_dct_idct_start);
  pthread_barrier_wait(&cm->pth_barrier_dct_idct_end);
  endTrace();

  startTrace2("Writing to Disk");

  pthread_mutex_lock(&cm->pth_mutex_write_frame);
  cm->unwritten_frame = cm->curframe;
  pthread_cond_signal(&cm->pth_cond_write_frame);
  pthread_mutex_unlock(&cm->pth_mutex_write_frame);

  endTrace();

  ++cm->framenum;
  ++cm->frames_since_keyframe;
  endTrace();
}

struct c63_common* init_c63_enc(int width, int height)
{
  int i;

  c63_common *cm = (c63_common*)calloc(1, sizeof(struct c63_common));

  cm->width = width;
  cm->height = height;

  cm->padw[Y_COMPONENT] = cm->ypw = (uint32_t)(ceil(width/16.0f)*16);
  cm->padh[Y_COMPONENT] = cm->yph = (uint32_t)(ceil(height/16.0f)*16);
  cm->padw[U_COMPONENT] = cm->upw = (uint32_t)(ceil(width*UX/(YX*8.0f))*8);
  cm->padh[U_COMPONENT] = cm->uph = (uint32_t)(ceil(height*UY/(YY*8.0f))*8);
  cm->padw[V_COMPONENT] = cm->vpw = (uint32_t)(ceil(width*VX/(YX*8.0f))*8);
  cm->padh[V_COMPONENT] = cm->vph = (uint32_t)(ceil(height*VY/(YY*8.0f))*8);

  cm->mb_cols_luma = cm->ypw / MACROBLOCK_SIZE;
  cm->mb_rows_luma = cm->yph / MACROBLOCK_SIZE;
  cm->mb_cols_chroma = cm->upw / MACROBLOCK_SIZE;
  cm->mb_rows_chroma = cm->uph / MACROBLOCK_SIZE;

  cm->qp = 25;
  cm->me_search_range = 16;
  cm->keyframe_interval = 100;

  for (i = 0; i < 64; ++i)
  {
    cm->quanttbl[Y_COMPONENT][i] = yquanttbl_def[i] / (cm->qp / 10.0);
    cm->quanttbl[U_COMPONENT][i] = uvquanttbl_def[i] / (cm->qp / 10.0);
    cm->quanttbl[V_COMPONENT][i] = uvquanttbl_def[i] / (cm->qp / 10.0);
  }

  cm->luma_size = cm->ypw * cm->yph;
  cm->chroma_size = cm->upw * cm->uph;
  cm->num_mbs_luma = cm->mb_rows_luma * cm->mb_cols_luma;
  cm->num_mbs_chroma = cm->mb_rows_chroma * cm->mb_cols_chroma;

  cm->pipe = c63_pipeline_init(cm->luma_size, cm->chroma_size, cm->num_mbs_luma, cm->num_mbs_chroma);

  cm->curframe = create_frame(cm, FRAME_CUR);
  cm->nextframe = create_frame(cm, FRAME_NEXT);
  cm->refframe = NULL;

  c63_initialize_constant_values(cm);
  initialize_dctlookup_values();

  cm->pthreads_run = 1;

  pthread_barrier_init(&cm->pth_barrier_dct_idct_start, NULL, COLOR_COMPONENTS + 1);
  pthread_barrier_init(&cm->pth_barrier_dct_idct_end, NULL, COLOR_COMPONENTS + 1);

  pthread_create(&threads[0], NULL, pthread_dct_idct_Y, (void *) cm);
  pthread_create(&threads[1], NULL, pthread_dct_idct_U, (void *) cm);
  pthread_create(&threads[2], NULL, pthread_dct_idct_V, (void *) cm);

  pthread_mutex_init(&cm->pth_mutex_write_frame, NULL);
  pthread_cond_init(&cm->pth_cond_write_frame, NULL);
  pthread_create(&threads[3], NULL, pthread_write_frame, (void *) cm);

  atexit(cleanup_cm);

  return cm;
}

void free_c63_enc(struct c63_common* cm)
{
  c63_pipeline_free(cm->pipe);
  destroy_frame(cm->curframe);
  destroy_frame(cm->nextframe);
  free(cm);
}

static void print_help()
{
  printf("Usage: ./c63enc [options] input_file\n");
  printf("Commandline options:\n");
  printf("  -h                             Height of images to compress\n");
  printf("  -w                             Width of images to compress\n");
  printf("  -o                             Output file (.c63)\n");
  printf("  [-f]                           Limit number of frames to encode\n");
  printf("\n");

  exit(EXIT_FAILURE);
}

int main(int argc, char **argv)
{
  int c;

  if (argc == 1) { print_help(); }

  while ((c = getopt(argc, argv, "h:w:o:f:i:")) != -1)
  {
    switch (c)
    {
      case 'h':
        height = atoi(optarg);
        break;
      case 'w':
        width = atoi(optarg);
        break;
      case 'o':
        output_file = optarg;
        break;
      case 'f':
        limit_numframes = atoi(optarg);
        break;
      default:
        print_help();
        break;
    }
  }

  if (optind >= argc)
  {
    fprintf(stderr, "Error getting program options, try --help.\n");
    exit(EXIT_FAILURE);
  }

  outfile = fopen(output_file, "wb");
  if (outfile == NULL) {
    perror("fopen");
    exit(EXIT_FAILURE);
  }

  struct c63_common *cm = init_c63_enc(width, height);
  cm->e_ctx.fp = outfile;

  input_file = argv[optind];
  if (limit_numframes) {
    fprintf(stderr, "Limited to %d frames.\n", limit_numframes);
  }

  FILE *infile = fopen(input_file, "rb");
  if (infile == NULL) {
    perror("fopen");
    exit(EXIT_FAILURE);
  }

  int numframes = 0;
  if (read_yuv(infile, cm) == NULL) {
    exit(EXIT_FAILURE);
  }

  int done = 0;
  while (!done) {
    if (fpeek(infile) == EOF) {
      done = 1;
    } else if (read_yuv(infile, cm) == NULL) {
      exit(EXIT_FAILURE);
    }

    printf("Encoding frame %d...", numframes+1);
    c63_encode_image(cm);
    printf(" done!\n");

    ++numframes;

    if (limit_numframes && numframes >= limit_numframes) {
      break;
    }
  }

  destroy_frame(cm->nextframe);
  cm->nextframe = NULL;

  cm->unwritten_frame = cm->curframe;
  cm->curframe = NULL;
  pthread_cond_broadcast(&cm->pth_cond_write_frame);

  cm->pthreads_run = 0;

  pthread_barrier_wait(&cm->pth_barrier_dct_idct_start);
  pthread_barrier_wait(&cm->pth_barrier_dct_idct_end);

  for (int i = 0; i < N_THREADS; i++) {
    pthread_join(threads[i], NULL);
  }

  pthread_mutex_destroy(&cm->pth_mutex_write_frame);
  pthread_cond_destroy(&cm->pth_cond_write_frame);

  pthread_barrier_destroy(&cm->pth_barrier_dct_idct_start);
  pthread_barrier_destroy(&cm->pth_barrier_dct_idct_end);
  destroy_frame(cm->curframe);
  cm->curframe = NULL;

  printf("Completed encoding! Encoded %d frames\n", numframes);

  free_c63_enc(cm);
  fclose(outfile);
  fclose(infile);

  return EXIT_SUCCESS;
}
