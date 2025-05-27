#ifndef C63_SERVER_H_
#define C63_SERVER_H_

#include <stdint.h>
#include <stdlib.h>

#include <sisci_error.h>
#include <sisci_api.h>

#include "c63.h"
#include "c63sisci.h"

#ifdef __cplusplus
extern "C" {
#endif

struct c63_server {
  struct c63_encoder *enc;

  uint32_t adapter_no;
  uint32_t remote_node;
  sci_desc_t v_dev;

  /* synchronization stuff */
  sci_local_segment_t    seg_sync_io_framenum; // client pushes THEIR io data into OUR segment
  sci_map_t              seg_map_io_framenum;
  struct c63_frame_sync *sync_io_framenum;

  sci_remote_segment_t   seg_sync_compute_framenum;
  sci_map_t              seg_map_compute_framenum;
  struct c63_frame_sync *sync_compute_framenum;


  /* input segments. received by client */
  // c63_common contains frame size information, and is received ONCE during initialization over PIO
  size_t               seg_sz_frame_hdr;
  sci_local_segment_t  seg_in_frame_hdr;
  sci_map_t            seg_map_frame_hdr;
  struct c63_common   *cm;

  // incoming frames, one per luma/chroma channel. populated by client every frame
  size_t               seg_sz_frame_data[COLOR_COMPONENTS];
  sci_local_segment_t  seg_in_frame_data[COLOR_COMPONENTS];
  sci_map_t            seg_map_frame_data[COLOR_COMPONENTS];
  yuv_t               *buf_frame_data[COLOR_COMPONENTS];

  /* output segments. for now, these are also sent over PIO, for the sake of simplicity */
  struct macroblock   *mbs[COLOR_COMPONENTS];
  size_t               seg_sz_mbs_data[COLOR_COMPONENTS];
  sci_map_t            seg_map_mbs_data[COLOR_COMPONENTS];
  sci_remote_segment_t seg_out_mbs_data[COLOR_COMPONENTS];

  dct_t               *residuals[COLOR_COMPONENTS];
  size_t               seg_sz_residuals_data[COLOR_COMPONENTS];
  sci_map_t            seg_map_residuals_data[COLOR_COMPONENTS];
  sci_remote_segment_t seg_out_residuals_data[COLOR_COMPONENTS];
};


/*
 * sets up a new encoder session with a client
 */
struct c63_server* c63_server_init(uint32_t remote_node);

/*
 * cleans up after encoder session, reverses c63_server_init
 */
void c63_server_free(struct c63_server *ctx);

/*
 * wait for frame to appear and prepare it for encoder
 * returns 0 on ok, returns 1 on failure
 */
int c63_server_begin_frame(struct c63_server *srv);

/*
 * inform the encoder that there's a new frame ready to be encoded
 */
void c63_server_process_frame(struct c63_server *srv);

/*
 * send frame back to client
 */
void c63_server_end_frame(struct c63_server *srv);


#ifdef __cplusplus
}
#endif

#endif // C63_SERVER_H_

