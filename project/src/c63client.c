// File: c63client.c

#include <assert.h>
#include <errno.h>
#include <getopt.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>

#include "c63.h"
#include "c63enc.h"
#include "common.h"

static char   *input_file = NULL;
static char   *output_file = NULL;
static int     limit_numframes = 0;
static uint32_t width, height;

/* getopt globals */
extern int optind;
extern char *optarg;

static yuv_t* read_yuv(FILE *file, struct frame *f, struct c63_common *cm) {
  size_t len = 0;

  uint8_t *Y = f->orig->Y;
  uint8_t *U = f->orig->U;
  uint8_t *V = f->orig->V;

  len += fread(Y, 1, cm->width * cm->height, file);
  len += fread(U, 1, (cm->width * cm->height) / 4, file);
  len += fread(V, 1, (cm->width * cm->height) / 4, file);

  if (ferror(file)) {
    perror("ferror");
    exit(EXIT_FAILURE);
  } else if (len != cm->width * cm->height * 1.5) {
    fprintf(stderr, "Reached end of file, but incorrect bytes read.\n");
    fprintf(stderr, "Wrong input? (height: %d width: %d)\n", cm->height, cm->width);
    DEBUG("seeing that your student is free from captivity fills you with determination");
    return NULL;
  }

  f->keyframe = (cm->framenum % cm->keyframe_interval == 0);
  cm->framenum++;

  return f->orig;
}

/* same help text */
static void print_help(const char *prog) {
    fprintf(stderr,
        "Usage: %s -w <width> -h <height> -o <out.c63> [-f <maxframes>] <in.yuv>\n",
        prog);
    exit(EXIT_FAILURE);
}

int main(int argc, char **argv) {
    if (argc == 1) print_help(argv[0]);

    int c;
    while ((c = getopt(argc, argv, "w:h:o:f:")) != -1) {
        switch (c) {
          case 'w': width           = (uint32_t)atoi(optarg); break;
          case 'h': height          = (uint32_t)atoi(optarg); break;
          case 'o': output_file     = optarg;                break;
          case 'f': limit_numframes = atoi(optarg);          break;
          default:  print_help(argv[0]);
        }
    }
    if (!width || !height || !output_file || optind >= argc)
        print_help(argv[0]);

    input_file = argv[optind];
    FILE *in  = fopen(input_file,  "rb");
    FILE *out = fopen(output_file, "wb");
    if (!in || !out) { perror("fopen"); return EXIT_FAILURE; }

    struct c63_common *cm = init_c63_enc(width, height);
    cm->e_ctx.fp = out;

    /* prime cur & next */
    if (!read_yuv(in, cm->curframe,  cm) ||
        !read_yuv(in, cm->nextframe, cm)) {
        fprintf(stderr, "Error reading initial frames\n");
        return EXIT_FAILURE;
    }

    int numframes = 0, pending = 2;
    while (pending) {
        if (feof(in) || !read_yuv(in, cm->nextframe, cm)) {
            pending--;
        }
        printf("Encoding frame %d...\n", numframes + 1);
        c63_encode_image(cm);
        numframes++;
        if (limit_numframes && numframes >= limit_numframes)
            break;
    }

    /* teardown: signal threads, cleanup & free */
    cm->pthreads_run = 0;
    pthread_cond_broadcast(&cm->pth_cond_write_frame);
    cleanup_cm();
    free_c63_enc(cm);

    fclose(in);
    fclose(out);

    printf("Client completed encoding %d frames\n", numframes);
    return EXIT_SUCCESS;
}

