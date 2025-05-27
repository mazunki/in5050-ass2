// File: c63server.c

#include <assert.h>
#include <errno.h>
#include <getopt.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include <sisci_error.h>
#include <sisci_api.h>

#include "c63.h"
#include "c63server.h"
#include "common.h"
#include "c63enc.h"
#include "c63sisci.h"

extern int optind;
extern char *optarg;

static void print_help(void)
{
  printf("Usage: ./c63server -r <nodeid>\n"
         "  -r  Node ID of client\n");
  exit(EXIT_FAILURE);
}

struct c63_server* c63_server_init(uint32_t remote_node)
{
  sci_error_t err;
  struct c63_server *srv = calloc(1, sizeof(*srv));
  if (!srv) return NULL;

  srv->adapter_no      = 0;
  srv->remote_node     = remote_node;
  srv->seg_sz_frame_hdr = sizeof(struct c63_common);

  // 0) initialize SISCI
    DEBUG("c63_server_init %d", 0);
    SCIInitialize(NO_FLAGS,&err);
    SISCI_ASSERT();

    SCIOpen(&srv->v_dev,NO_FLAGS,&err);
    SISCI_ASSERT();


  // 1) create & map the IO‐sync segment (client → server) so we can be informed of input frames being ready
    DEBUG("c63_server_init %d", 1);
    SCICreateSegment(srv->v_dev, &srv->seg_sync_io_framenum, SYNC_FRAME_IO_SEG_ID, sizeof(struct c63_frame_sync), NO_CALLBACK, NO_CALLBACK_ARGS, NO_FLAGS, &err);
    SISCI_ASSERT();

    SCIPrepareSegment(srv->seg_sync_io_framenum, srv->adapter_no, NO_FLAGS, &err);
    SISCI_ASSERT();

    srv->sync_io_framenum = (struct c63_frame_sync*) SCIMapLocalSegment(srv->seg_sync_io_framenum, &srv->seg_map_io_framenum, NO_OFFSET, sizeof(struct c63_frame_sync), AUTO_ADDRESS, NO_FLAGS, &err);
    SISCI_ASSERT();

    SCISetSegmentAvailable(srv->seg_sync_io_framenum, srv->adapter_no, NO_FLAGS, &err);
    SISCI_ASSERT();

    srv->sync_io_framenum->started = -1;


  // 2) create & prepare the header segment (PIO) so client can write c63_common
    DEBUG("c63_server_init %d", 2);
    SCICreateSegment(srv->v_dev, &srv->seg_in_frame_hdr, FRAME_HDR_SEG_ID, srv->seg_sz_frame_hdr, NO_CALLBACK, NO_CALLBACK_ARGS, NO_FLAGS, &err);
    SISCI_ASSERT();

    SCIPrepareSegment(srv->seg_in_frame_hdr, srv->adapter_no, NO_FLAGS, &err);
    SISCI_ASSERT();

    srv->cm = (struct c63_common*) SCIMapLocalSegment(srv->seg_in_frame_hdr, &srv->seg_map_frame_hdr, NO_OFFSET, srv->seg_sz_frame_hdr, AUTO_ADDRESS, NO_FLAGS, &err);
    SISCI_ASSERT();

    SCISetSegmentAvailable(srv->seg_in_frame_hdr, srv->adapter_no, NO_FLAGS, &err);
    SISCI_ASSERT();


  // 3) connect & map the compute‐sync segment (server → client) so we can inform of compute state
    DEBUG("c63_server_init %d", 3);
    do {
      SCIConnectSegment(srv->v_dev, &srv->seg_sync_compute_framenum, srv->remote_node, SYNC_FRAME_COMPUTE_SEG_ID, srv->adapter_no, NO_CALLBACK, NO_CALLBACK_ARGS, SCI_INFINITE_TIMEOUT, NO_FLAGS, &err);
      // SISCI_ASSERT();
    } while (err == SCI_ERR_NO_SUCH_SEGMENT);

    srv->sync_compute_framenum = (struct c63_frame_sync*) SCIMapRemoteSegment(srv->seg_sync_compute_framenum, &srv->seg_map_compute_framenum, NO_OFFSET, sizeof(struct c63_frame_sync), AUTO_ADDRESS, NO_FLAGS, &err);
    SISCI_ASSERT();

    srv->sync_compute_framenum->started = -1;
    srv->sync_compute_framenum->finished = -1;


  // 4) wait for client to write the header (width==0 means "not yet")
    DEBUG("c63_server_init %d", 4);
    DEBUG("srv->cm->width=%d", srv->cm->width);
    while (srv->cm->width == 0) { /* spin */ }

    srv->seg_sz_frame_data[Y_COMPONENT] = srv->cm->luma_size;
    srv->seg_sz_frame_data[U_COMPONENT] = srv->cm->chroma_size;
    srv->seg_sz_frame_data[V_COMPONENT] = srv->cm->chroma_size;


  // 5) create & map per‐component frame buffers (DMA)
    DEBUG("c63_server_init %d", 5);
    for (int c = 0; c < COLOR_COMPONENTS; c++) {
      SCICreateSegment(srv->v_dev, &srv->seg_in_frame_data[c], FRAME_DATA_SEG_ID + c, srv->seg_sz_frame_data[c], NO_CALLBACK, NO_CALLBACK_ARGS, NO_FLAGS, &err);
      SISCI_ASSERT();

      SCIPrepareSegment(srv->seg_in_frame_data[c], srv->adapter_no, NO_FLAGS, &err);
      SISCI_ASSERT();

      srv->buf_frame_data[c] = SCIMapLocalSegment(srv->seg_in_frame_data[c], &srv->seg_map_frame_data[c], NO_OFFSET, srv->seg_sz_frame_data[c], AUTO_ADDRESS, NO_FLAGS, &err);
      SISCI_ASSERT();

      SCISetSegmentAvailable(srv->seg_in_frame_data[c], srv->adapter_no, NO_FLAGS, &err);
      SISCI_ASSERT();
    }


  // 6) connect & map the macroblock segments (server → client)
    DEBUG("c63_server_init %d", 6);
    srv->seg_sz_mbs_data[Y_COMPONENT] = srv->cm->num_mbs_luma * sizeof(struct macroblock);
    srv->seg_sz_mbs_data[U_COMPONENT] = srv->cm->num_mbs_chroma * sizeof(struct macroblock);
    srv->seg_sz_mbs_data[V_COMPONENT] = srv->cm->num_mbs_chroma * sizeof(struct macroblock);

    for (int c = 0; c < COLOR_COMPONENTS; c++) {
      do {
        SCIConnectSegment(srv->v_dev, &srv->seg_out_mbs_data[c], srv->remote_node, MBS_SEG_ID + c, srv->adapter_no, NO_CALLBACK, NO_CALLBACK_ARGS, SCI_INFINITE_TIMEOUT, NO_FLAGS, &err);
        // SISCI_ASSERT();
      } while (err == SCI_ERR_NO_SUCH_SEGMENT);

      SCIMapRemoteSegment(srv->seg_out_mbs_data[c], &srv->seg_map_mbs_data[c], NO_OFFSET, srv->seg_sz_mbs_data[c], AUTO_ADDRESS, NO_FLAGS, &err);
      SISCI_ASSERT();

      srv->mbs[c] = (struct macroblock*) srv->seg_map_mbs_data[c];
    }

  // 7) connect & map the residual segments (server → client)
    DEBUG("c63_server_init %d", 7);
    srv->seg_sz_residuals_data[Y_COMPONENT] = sizeof(int16_t) * srv->cm->luma_size;
    srv->seg_sz_residuals_data[U_COMPONENT] = sizeof(int16_t) * srv->cm->chroma_size;
    srv->seg_sz_residuals_data[V_COMPONENT] = sizeof(int16_t) * srv->cm->chroma_size;

    for (int c = 0; c < COLOR_COMPONENTS; c++) {
      do {
        SCIConnectSegment(srv->v_dev, &srv->seg_out_residuals_data[c], srv->remote_node, DCT_SEG_ID + c, srv->adapter_no, NO_CALLBACK, NO_CALLBACK_ARGS, SCI_INFINITE_TIMEOUT, NO_FLAGS, &err);
        // SISCI_ASSERT();
      } while (err == SCI_ERR_NO_SUCH_SEGMENT);

      SCIMapRemoteSegment(srv->seg_out_residuals_data[c], &srv->seg_map_residuals_data[c], NO_OFFSET, srv->seg_sz_residuals_data[c], AUTO_ADDRESS, NO_FLAGS, &err);
      SISCI_ASSERT();

      srv->residuals[c] = (dct_t *) srv->seg_map_residuals_data[c];
    }

  // 8) finally, init the GPU encoder
    DEBUG("c63_server_init %d", 8);
    struct c63_common *cm = c63_common_init(srv->cm->width, srv->cm->height);

    DEBUG("c63_server_init %d cm=%p", 9, cm);
    srv->enc = c63_encoder_init(cm);

  DEBUG("init orig cur=%p", (void *) srv->enc->curframe->orig);
  DEBUG("c63_server_init %d", 10);
  return srv;
}

int c63_server_begin_frame(struct c63_server *srv)
{
  static int last_frame = -1;
  DEBUG("c63_server_begin_frame %d", 1);

  // wait for client to bump io.started
  while (srv->sync_io_framenum->started <= last_frame) {
    if (srv->sync_io_framenum->started == -69) {
      return 1;
    }
    /* spin */
  }
  last_frame = srv->sync_io_framenum->started;

  DEBUG("c63_server_begin_frame %d", 2);

  // copy YUV into encoder
  DEBUG("begin orig cur=%p", (void *) srv->enc->curframe->orig);

  memcpy(srv->enc->curframe->orig->Y, srv->buf_frame_data[Y_COMPONENT], srv->seg_sz_frame_data[Y_COMPONENT]);
  memcpy(srv->enc->curframe->orig->U, srv->buf_frame_data[U_COMPONENT], srv->seg_sz_frame_data[U_COMPONENT]);
  memcpy(srv->enc->curframe->orig->V, srv->buf_frame_data[V_COMPONENT], srv->seg_sz_frame_data[V_COMPONENT]);

  DEBUG("c63_server_begin_frame %d", 3);
  return 0;
}

void c63_server_process_frame(struct c63_server *srv)
{
  DEBUG("c63_server_process_frame %d", 1);
  c63_encode_image(srv->enc);
  DEBUG("c63_server_process_frame %d", 2);
}

void c63_server_end_frame(struct c63_server *srv)
{
  DEBUG("c63_server_end_frame %d", 1);

  sci_error_t err;

  // send macroblocks back
    SCIMemCpy(NO_SEQUENCE, srv->enc->curframe->mbs[Y_COMPONENT], srv->seg_map_mbs_data[Y_COMPONENT], NO_OFFSET, srv->seg_sz_mbs_data[Y_COMPONENT], NO_FLAGS, &err);
    SISCI_ASSERT();

    SCIMemCpy(NO_SEQUENCE, srv->enc->curframe->mbs[U_COMPONENT], srv->seg_map_mbs_data[U_COMPONENT], NO_OFFSET, srv->seg_sz_mbs_data[U_COMPONENT], NO_FLAGS, &err);
    SISCI_ASSERT();

    SCIMemCpy(NO_SEQUENCE, srv->enc->curframe->mbs[V_COMPONENT], srv->seg_map_mbs_data[V_COMPONENT], NO_OFFSET, srv->seg_sz_mbs_data[V_COMPONENT], NO_FLAGS, &err);
    SISCI_ASSERT();


  // send residuals back
    SCIMemCpy(NO_SEQUENCE, srv->enc->curframe->residuals->Ydct, srv->seg_map_residuals_data[Y_COMPONENT], NO_OFFSET, srv->seg_sz_residuals_data[Y_COMPONENT], NO_FLAGS, &err);
    SISCI_ASSERT();

    SCIMemCpy(NO_SEQUENCE, srv->enc->curframe->residuals->Udct, srv->seg_map_residuals_data[U_COMPONENT], NO_OFFSET, srv->seg_sz_residuals_data[U_COMPONENT], NO_FLAGS, &err);
    SISCI_ASSERT();

    SCIMemCpy(NO_SEQUENCE, srv->enc->curframe->residuals->Vdct, srv->seg_map_residuals_data[V_COMPONENT], NO_OFFSET, srv->seg_sz_residuals_data[V_COMPONENT], NO_FLAGS, &err);
    SISCI_ASSERT();


  // signal computation done
    SCIMemCpy(NO_SEQUENCE, srv->sync_io_framenum, srv->seg_map_io_framenum, NO_OFFSET, sizeof(struct c63_frame_sync), NO_FLAGS, &err);

    srv->sync_compute_framenum->finished = srv->sync_io_framenum->started;

  DEBUG("c63_server_end_frame %d", 2);
}

void c63_server_free(struct c63_server *srv)
{
  DEBUG("c63_server_free %d", 1);
  sci_error_t err;

  // 1) free encoder
  c63_encoder_free(srv->enc);

  // 2) unmap & destroy header segment
  SCIUnmapSegment(srv->seg_map_frame_hdr,NO_FLAGS,&err);
  SISCI_ASSERT();
  SCIRemoveSegment(srv->seg_in_frame_hdr,NO_FLAGS,&err);
  SISCI_ASSERT();

  // 3) unmap & destroy IO‐sync segment
  SCIUnmapSegment(srv->seg_map_io_framenum,NO_FLAGS,&err);
  SISCI_ASSERT();
  SCIRemoveSegment(srv->seg_sync_io_framenum,NO_FLAGS,&err);
  SISCI_ASSERT();

  // 4) unmap & destroy compute‐sync segment
  SCIUnmapSegment(srv->seg_map_compute_framenum,NO_FLAGS,&err);
  SISCI_ASSERT();

  // 5) unmap & destroy frame‐data segments
  for (int c = 0; c < COLOR_COMPONENTS; c++) {
    SCIUnmapSegment(srv->seg_map_frame_data[c],NO_FLAGS,&err);
    SISCI_ASSERT();
    SCIRemoveSegment(srv->seg_in_frame_data[c],NO_FLAGS,&err);
    SISCI_ASSERT();
  }

  // 6) unmap remote (client) segments
  for (int c = 0; c < COLOR_COMPONENTS; c++) {
    SCIUnmapSegment(srv->seg_map_mbs_data[c],NO_FLAGS,&err);
    SISCI_ASSERT();
    SCIUnmapSegment(srv->seg_map_residuals_data[c],NO_FLAGS,&err);
    SISCI_ASSERT();
  }

  // 7) close & terminate SISCI
  SCIClose(srv->v_dev,NO_FLAGS,&err);
  SISCI_ASSERT();
  SCITerminate();
  SISCI_ASSERT();

  free(srv);
  DEBUG("c63_server_free %d", 2);
}

int main(int argc, char **argv)
{
  int c;
  uint32_t node = 0;

  while ((c = getopt(argc,argv,"r:")) != -1) {
    if (c == 'r') {
      node = (uint32_t)atoi(optarg);
    } else {
      print_help();
    }
  }
  if (!node) print_help();

  struct c63_server *srv = c63_server_init(node);
  if (!srv) {
    fprintf(stderr,"server initialization failed\n");
    return EXIT_FAILURE;
  }

  while (!c63_server_begin_frame(srv)) {
    c63_server_process_frame(srv);
    c63_server_end_frame(srv);
  }

  c63_server_free(srv);
  return EXIT_SUCCESS;
}

