#ifndef C63_CLIENT_H_
#define C63_CLIENT_H_

#include <stdint.h>
#include <stdlib.h>

#include <sisci_error.h>
#include <sisci_api.h>

#include "c63.h"
#include "c63sisci.h"

#ifdef __cplusplus
extern "C" {
#endif

struct c63_client {
  struct c63_common *cm;     // shared header + encode params
  struct frame *curframe;

  uint32_t adapter_no;       // SISCI adapter
  uint32_t remote_node;      // server’s node ID
  sci_desc_t v_dev;          // SISCI virtual device descriptor

  /* synchronization: from us to the server, frame counter */
  sci_local_segment_t    seg_sync_compute_framenum;
  sci_map_t              seg_map_compute_framenum;
  struct c63_frame_sync *sync_compute_framenum;

  sci_remote_segment_t   seg_sync_io_framenum;
  sci_map_t              seg_map_io_framenum;
  struct c63_frame_sync *sync_io_framenum;


  /* header segment: one‐time PIO of c63_common */
  size_t               seg_sz_frame_hdr;
  sci_remote_segment_t seg_in_frame_hdr;
  sci_map_t            seg_map_frame_hdr;
  struct c63_common   *cm_hdr;

  /* input frames, one per luma/chroma channel (DMA) */
  size_t               seg_sz_frame_data[COLOR_COMPONENTS];
  sci_remote_segment_t seg_out_frame_data[COLOR_COMPONENTS];
  sci_map_t            seg_map_frame_data[COLOR_COMPONENTS];
  yuv_t               *buf_frame_data[COLOR_COMPONENTS];

  /* output macroblocks from server (PIO) */
  struct macroblock   *mbs[COLOR_COMPONENTS];
  size_t               seg_sz_mbs_data[COLOR_COMPONENTS];
  sci_map_t            seg_map_mbs_data[COLOR_COMPONENTS];
  sci_local_segment_t  seg_in_mbs_data[COLOR_COMPONENTS];

  /* output residuals from server (PIO) */
  dct_t               *residuals[COLOR_COMPONENTS];
  size_t               seg_sz_residuals_data[COLOR_COMPONENTS];
  sci_map_t            seg_map_residuals_data[COLOR_COMPONENTS];
  sci_local_segment_t  seg_in_residuals_data[COLOR_COMPONENTS];
};


/**
 * initializes a session with a compute server. we need to transfer a
 * c63_common header during initialization, and then we will transfer
 * new frames to the remote every round
 */
struct c63_client *c63_client_init(uint32_t adapter_no, uint32_t remote_node, struct c63_common *cm);

/*
 * clean up after ourselves, reverses c63_client_init
 */
void  c63_client_free(struct c63_client *client);

/*
 * reads file from disk, sends it to compute server, and informs
 * them that they should start processing
 */
int c63_client_begin_frame(struct c63_client *client, uint32_t frameno);

/*
 * waits for remote processing to complete
 */
void c63_client_process_frame(struct c63_client *client, uint32_t frameno);

/*
 * finalize a frame by writing it to disk
 */
void  c63_client_end_frame(struct c63_client *client, uint32_t frameno);

#ifdef __cplusplus
}
#endif

#endif // C63_CLIENT_H_

