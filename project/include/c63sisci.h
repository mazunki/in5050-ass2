#ifndef C63_SISCI_H_
#define C63_SISCI_H_

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <sisci_error.h>
#include <sisci_api.h>

/* Maximum possible YUV frame (up to 4K×4K) */
#define FRAME_MAX_SIZE  (4096*4096)

/* segment IDs must match server */
enum {
  FRAME_HDR_SEG_ID          = 10, // header client -> server
  FRAME_DATA_SEG_ID         = 20, // data client -> server
  //
  MBS_SEG_ID                = 30, // data server -> client
  DCT_SEG_ID                = 40, // data server -> client
  //
  SYNC_FRAME_IO_SEG_ID      = 50, // synchronization client => server
  SYNC_FRAME_COMPUTE_SEG_ID = 60  // synhronization server => client
};

struct c63_frame_sync {
  volatile int started;
  volatile int finished;
};

/* SISCI helper macros */
#define NO_FLAGS           0
#define NO_OFFSET          0
#define NO_CALLBACK        NULL
#define NO_CALLBACK_ARGS   NULL
#define NO_SEQUENCE   (sci_sequence_t) NULL
#define AUTO_ADDRESS       NULL

#define SISCI_ASSERT()                                                \
    do {                                                              \
        if (err != SCI_ERR_OK) {                                      \
            fprintf(stderr,                                           \
                    "SISCI Error: %s (file %s:%d) (n=%d)\n",          \
                    SCIGetErrorString(err), __FILE__, __LINE__, err); \
            exit(err);                                                \
        }                                                             \
    } while (0);

#define SISCI_CALL(call)                                              \
    do {                                                              \
        err = (call);                                                 \
        SISCI_ASSERT(ctx);                                            \
    } while (0);

#endif // C63_SISCI_H_

