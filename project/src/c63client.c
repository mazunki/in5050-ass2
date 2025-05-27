// File: c63_client.c

#include <assert.h>
#include <errno.h>
#include <getopt.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <sisci_error.h>
#include <sisci_api.h>

#include "c63client.h"
#include "c63.h"
#include "c63sisci.h"
#include "common.h"
#include "c63_write.h"

static char *input_file, *output_file;

static uint32_t remote_node = 0;
static int limit_numframes = 0;

static uint32_t width;
static uint32_t height;

/* getopt globals */
extern int optind;
extern char *optarg;

/*-----------------------------------------------
 * c63_client_* implementations
 *----------------------------------------------*/

struct c63_client * c63_client_init(uint32_t adapter_no, uint32_t remote_node, struct c63_common *cm)
{
  sci_error_t err;
  struct c63_client *cl = calloc(1, sizeof(*cl));
  if (!cl) return NULL;

  cl->cm              = cm;
  cl->adapter_no      = adapter_no;
  cl->remote_node     = remote_node;
  cl->seg_sz_frame_hdr = sizeof(*cm);

  /* 0) SISCI init & open */
  DEBUG("c63_client_init %d", 0);
  SCIInitialize(NO_FLAGS, &err);
  SISCI_ASSERT();

  SCIOpen(&cl->v_dev, NO_FLAGS, &err);
  SISCI_ASSERT();


  /* 1) macroblock & residual segments (client creates these, server will connect and push to us) */
  DEBUG("c63_client_init %d", 1);
  for (int c = 0; c < COLOR_COMPONENTS; c++) {
    /* macroblock segment for server → client */
    cl->seg_sz_mbs_data[c] = (c == Y_COMPONENT ? cm->num_mbs_luma : cm->num_mbs_chroma) * sizeof(struct macroblock);

    SCICreateSegment(cl->v_dev, &cl->seg_in_mbs_data[c], MBS_SEG_ID + c, cl->seg_sz_mbs_data[c], NO_CALLBACK, NO_CALLBACK_ARGS, NO_FLAGS, &err);
    SISCI_ASSERT();

    SCIPrepareSegment(cl->seg_in_mbs_data[c], cl->adapter_no, NO_FLAGS, &err);
    SISCI_ASSERT();

    cl->mbs[c] = (struct macroblock *) SCIMapLocalSegment(cl->seg_in_mbs_data[c], &cl->seg_map_mbs_data[c], NO_OFFSET, cl->seg_sz_mbs_data[c], AUTO_ADDRESS, NO_FLAGS, &err);
    SISCI_ASSERT();

    SCISetSegmentAvailable(cl->seg_in_mbs_data[c], cl->adapter_no, NO_FLAGS, &err);
    SISCI_ASSERT();
  }


  for (int c = 0; c < COLOR_COMPONENTS; c++) {
    /* residuals segment for server → client */
    cl->seg_sz_residuals_data[c] = (c == Y_COMPONENT) ? (cm->luma_size  / 8) : (cm->chroma_size / 8);

    SCICreateSegment(cl->v_dev, &cl->seg_in_residuals_data[c], DCT_SEG_ID + c, cl->seg_sz_residuals_data[c], NO_CALLBACK, NO_CALLBACK_ARGS, NO_FLAGS, &err);
    SISCI_ASSERT();

    SCIPrepareSegment(cl->seg_in_residuals_data[c], cl->adapter_no, NO_FLAGS, &err);
    SISCI_ASSERT();

    cl->residuals[c] = (dct_t *) SCIMapLocalSegment(cl->seg_in_residuals_data[c], &cl->seg_map_residuals_data[c], NO_OFFSET, cl->seg_sz_residuals_data[c], AUTO_ADDRESS, NO_FLAGS, &err);
    SISCI_ASSERT();

    SCISetSegmentAvailable(cl->seg_in_residuals_data[c], cl->adapter_no, NO_FLAGS, &err);
    SISCI_ASSERT();
  }


  /* 2) set up synchronization segment for server computation to tell us if they're done */
    DEBUG("c63_client_init %d", 2);
    SCICreateSegment(cl->v_dev, &cl->seg_sync_compute_framenum, SYNC_FRAME_COMPUTE_SEG_ID, sizeof(struct c63_frame_sync), NO_CALLBACK, NO_CALLBACK_ARGS, NO_FLAGS, &err);
    SISCI_ASSERT();

    SCIPrepareSegment(cl->seg_sync_compute_framenum, cl->adapter_no, NO_FLAGS, &err);
    SISCI_ASSERT();

    SCISetSegmentAvailable(cl->seg_sync_compute_framenum, cl->adapter_no, NO_FLAGS, &err);
    SISCI_ASSERT();

    cl->sync_compute_framenum = (struct c63_frame_sync *) SCIMapLocalSegment(cl->seg_sync_compute_framenum, &cl->seg_map_compute_framenum, NO_OFFSET, sizeof(struct c63_frame_sync), AUTO_ADDRESS, NO_FLAGS, &err);
    SISCI_ASSERT();


  /* 3) connect to header segment of server, and inform it of frame size */
    DEBUG("c63_client_init %d", 3);
    do {
      SCIConnectSegment(cl->v_dev, &cl->seg_in_frame_hdr, cl->remote_node, FRAME_HDR_SEG_ID, cl->adapter_no, NO_CALLBACK, NO_CALLBACK_ARGS, SCI_INFINITE_TIMEOUT, NO_FLAGS, &err);
      // SISCI_ASSERT();
    } while (err == SCI_ERR_NO_SUCH_SEGMENT);
    DEBUG("c63_client_init %d connected", 3);

    cl->cm_hdr = (struct c63_common*) SCIMapRemoteSegment(cl->seg_in_frame_hdr, &cl->seg_map_frame_hdr, NO_OFFSET, cl->seg_sz_frame_hdr, AUTO_ADDRESS, NO_FLAGS, &err);
    SISCI_ASSERT();

    SCIMemCpy(NO_SEQUENCE, cm, cl->seg_map_frame_hdr, NO_OFFSET, sizeof(struct c63_common), NO_FLAGS, &err);
    DEBUG("cm->width=%d, cm_hdr->width=%d", cm->width, cl->cm_hdr->width);


  /* 4) attach to server's synchronization segment so we can update the frame numbers being sent */
    DEBUG("c63_client_init %d", 4);
    do {
      SCIConnectSegment(cl->v_dev, &cl->seg_sync_io_framenum, cl->remote_node, SYNC_FRAME_IO_SEG_ID, cl->adapter_no, NO_CALLBACK, NO_CALLBACK_ARGS, SCI_INFINITE_TIMEOUT, NO_FLAGS, &err);
      // SISCI_ASSERT();
    } while (err == SCI_ERR_NO_SUCH_SEGMENT);

    cl->sync_io_framenum = (struct c63_frame_sync*) SCIMapRemoteSegment(cl->seg_sync_io_framenum, &cl->seg_map_io_framenum, NO_OFFSET, sizeof(struct c63_frame_sync), AUTO_ADDRESS, NO_FLAGS, &err);
    SISCI_ASSERT();

    cl->sync_io_framenum->started = -1;
    cl->sync_io_framenum->finished = -1;

  /* 5) attach to frame segments on server, so we can push input frames to be processed */
    DEBUG("c63_client_init %d", 5);
    for (int c = 0; c < COLOR_COMPONENTS; c++) {
      cl->seg_sz_frame_data[c] = (c == Y_COMPONENT ? cm->luma_size : cm->chroma_size);

      do {
        SCIConnectSegment(cl->v_dev, &cl->seg_out_frame_data[c], cl->remote_node, FRAME_DATA_SEG_ID + c, cl->adapter_no, NO_CALLBACK, NO_CALLBACK_ARGS, SCI_INFINITE_TIMEOUT, NO_FLAGS, &err);
        // SISCI_ASSERT();
      } while (err == SCI_ERR_NO_SUCH_SEGMENT);

      cl->buf_frame_data[c] = (yuv_t *) SCIMapRemoteSegment(cl->seg_out_frame_data[c], &cl->seg_map_frame_data[c], NO_OFFSET, cl->seg_sz_frame_data[c], AUTO_ADDRESS, NO_FLAGS, &err);
      SISCI_ASSERT();
    }


  DEBUG("c63_client_init %d", 6);
  return cl;
}

int c63_client_begin_frame(struct c63_client *cl, uint32_t frameno)
{
  DEBUG("c63_client_begin_frame 1: %d", frameno);
  /* DMA Y/U/V into server buffers */
  memcpy(cl->buf_frame_data[Y_COMPONENT], cl->curframe->orig->Y, cl->seg_sz_frame_data[Y_COMPONENT]);
  memcpy(cl->buf_frame_data[U_COMPONENT], cl->curframe->orig->U, cl->seg_sz_frame_data[U_COMPONENT]);
  memcpy(cl->buf_frame_data[V_COMPONENT], cl->curframe->orig->V, cl->seg_sz_frame_data[V_COMPONENT]);

  /* signal new frame being ready */
  cl->sync_io_framenum->started = frameno;

  DEBUG("c63_client_begin_frame 2: %d", frameno);
  return 0;
}

void c63_client_process_frame(struct c63_client *cl, uint32_t frameno)
{
  DEBUG("c63_client_process_frame 1: %d", frameno);
  /* wait for server to finish computation */
  while (cl->sync_compute_framenum->finished < cl->sync_io_framenum->started) {
    /* spin */
  }
  DEBUG("c63_client_process_frame 2: %d", frameno);
}

void c63_client_end_frame(struct c63_client *cl, uint32_t frameno)
{
  DEBUG("c63_client_end_frame 1: %d", frameno);
  /* pull MBs & residuals from cl->mbs[] and cl->residuals[] */
  write_frame(cl->cm, cl->curframe);

  /* signal frame being finalized */
  cl->sync_io_framenum->finished = frameno;
  DEBUG("c63_client_end_frame 2: %d", frameno);
}

void c63_client_free(struct c63_client *cl)
{
  DEBUG("c63_client_free %d", 1);
  sci_error_t err;

  /* unmap header & sync segments */
  SCIUnmapSegment(cl->seg_map_frame_hdr, NO_FLAGS, &err);
  SISCI_ASSERT();

  SCIUnmapSegment(cl->seg_map_io_framenum, NO_FLAGS, &err);
  SISCI_ASSERT();

  SCIUnmapSegment(cl->seg_map_compute_framenum, NO_FLAGS, &err);
  SISCI_ASSERT();

  SISCI_ASSERT();

  /* unmap all remote segments */
  for (int c = 0; c < COLOR_COMPONENTS; c++) {
    SCIUnmapSegment(cl->seg_map_frame_data[c], NO_FLAGS, &err);
    SISCI_ASSERT();

    SCIUnmapSegment(cl->seg_map_mbs_data[c], NO_FLAGS, &err);
    SISCI_ASSERT();

    SCIUnmapSegment(cl->seg_map_residuals_data[c], NO_FLAGS, &err);
    SISCI_ASSERT();
  }

  /* close SISCI */
  SCIClose(cl->v_dev, NO_FLAGS, &err);
  SISCI_ASSERT();

  SCITerminate();

  free(cl);
  DEBUG("c63_client_free %d", 2);
}


/*-----------------------------------------------
 * client and file reading stuff
 *----------------------------------------------*/


/* Read planar YUV frames with 4:2:0 chroma sub-sampling */
static yuv_t * read_yuv(FILE *file, struct c63_common *cm)
{
  size_t wantY = cm->width * cm->height;
  size_t wantC = wantY / 4;

  yuv_t *img = malloc(sizeof(*img));
  if (!img) {
    perror("malloc");
    exit(EXIT_FAILURE);
  }
  img->Y = calloc(1, cm->padw[Y_COMPONENT] * cm->padh[Y_COMPONENT]);
  img->U = calloc(1, cm->padw[U_COMPONENT] * cm->padh[U_COMPONENT]);
  img->V = calloc(1, cm->padw[V_COMPONENT] * cm->padh[V_COMPONENT]);
  if (!img->Y || !img->U || !img->V) {
    perror("calloc");
    exit(EXIT_FAILURE);
  }

  size_t len = 0;
  len += fread(img->Y, 1, wantY, file);
  len += fread(img->U, 1, wantC, file);
  len += fread(img->V, 1, wantC, file);

  if (ferror(file)) {
    perror("fread");
    exit(EXIT_FAILURE);
  }
  if (feof(file)) {
    free(img->Y);
    free(img->U);
    free(img->V);
    free(img);

    return NULL;
  }
  if (len != wantY + 2*wantC) {
    fprintf(stderr, "Unexpected frame length: got %zu bytes, expected %zu\n", len, wantY + 2*wantC);

    free(img->Y);
    free(img->U);
    free(img->V);
    free(img);

    return NULL;
  }
  return img;
}



static void print_help(void)
{
  printf("Usage: ./c63client -r nodeid [options] input_file\n");
  printf("Commandline options:\n");
  printf("  -r                             Node id of client\n");
  printf("  -h                             Height of images to compress\n");
  printf("  -w                             Width of images to compress\n");
  printf("  -o                             Output file (.c63)\n");
  printf("  [-f]                           Limit number of frames to encode\n");
  printf("\n");

  exit( EXIT_FAILURE );
}

int main(int argc, char **argv)
{
  int c;
  while ((c = getopt(argc, argv, "r:w:h:o:f:")) != -1) {
    switch (c) {
      case 'r': remote_node     = (uint32_t)atoi(optarg); break;
      case 'w': width           = (uint32_t)atoi(optarg); break;
      case 'h': height          = (uint32_t)atoi(optarg); break;
      case 'o': output_file     = optarg;                 break;
      case 'f': limit_numframes = atoi(optarg);           break;
      default:  print_help();                             break;
    }
  }
  if (!remote_node || !width || !height || !output_file || optind >= argc) {
    print_help();
  }
  input_file = argv[optind];

  /* open input/output files */
  FILE *infile = fopen(input_file, "rb");
  if (!infile) {
    perror("fopen input");
    return EXIT_FAILURE;
  }

  FILE *outfile = fopen(output_file, "wb");
  if (!outfile) {
    perror("fopen output");
    fclose(infile);
    return EXIT_FAILURE;
  }


  /* initialize common state */
  struct c63_common *cm = c63_common_init(width, height);
  cm->e_ctx.fp = outfile;


  /* set up SISCI client */
  struct c63_client *client = c63_client_init(0, remote_node, cm);
  if (!client) {
    fprintf(stderr, "c63_client_init failed\n");
    fclose(infile);
    fclose(outfile);
    return EXIT_FAILURE;
  }

  // frame allocation
  client->curframe = calloc(1, sizeof(struct frame));
  client->curframe->orig = calloc(1, sizeof(yuv_t));
  client->curframe->residuals = calloc(1, sizeof(dct_t));

  client->curframe->mbs[Y_COMPONENT] = (struct macroblock *)malloc(cm->num_mbs_luma *   sizeof(struct macroblock));
  client->curframe->mbs[U_COMPONENT] = (struct macroblock *)malloc(cm->num_mbs_chroma * sizeof(struct macroblock));
  client->curframe->mbs[V_COMPONENT] = (struct macroblock *)malloc(cm->num_mbs_chroma * sizeof(struct macroblock));

  client->curframe->residuals->Ydct = (int16_t *)malloc(cm->luma_size * sizeof(int16_t));
  client->curframe->residuals->Udct = (int16_t *)malloc(cm->chroma_size * sizeof(int16_t));
  client->curframe->residuals->Vdct = (int16_t *)malloc(cm->chroma_size * sizeof(int16_t));


  /* encode frames remotely */
  int frameno = 0;
  while (1) {
    yuv_t *image = read_yuv(infile, cm);
    if (!image) break;

    printf("Sending frame %d...\n", frameno);

    /* point cm->curframe to our new image */
    client->curframe->orig->Y = image->Y;
    client->curframe->orig->U = image->U;
    client->curframe->orig->V = image->V;

    /* handshake with server */
    if (c63_client_begin_frame(client, frameno)) {
      fprintf(stderr, "begin_frame failed\n");
      break;
    }
    c63_client_process_frame(client, frameno);
    c63_client_end_frame(client, frameno);

    /* free staging image */
    free(image->Y);
    free(image->U);
    free(image->V);
    free(image);

    frameno++;
    if (limit_numframes && frameno >= limit_numframes) {
      break;
    }
  }

  client->sync_io_framenum->started = -69;

  /* cleanup */
  c63_client_free(client);
  fclose(infile);
  fclose(outfile);
  SCITerminate();
  return EXIT_SUCCESS;
}


