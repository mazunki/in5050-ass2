#ifndef C63_COMMON_H_
#define C63_COMMON_H_

#include <inttypes.h>
#include "c63.h"
#include "pipeline.h"

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus


#define MACROBLOCK_SIZE 8


#define CUDA_ASSERT(call)                                                     \
    {                                                                         \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA Error: %s (file %s, line %d)\n",            \
                    cudaGetErrorString(err), __FILE__, __LINE__);             \
            exit(err);                                                        \
        }                                                                     \
    }

#define CUDA_CHECK()                                                          \
    {                                                                         \
        cudaError_t err = cudaGetLastError();                                 \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA Error: %s (file %s, line %d)\n",            \
                    cudaGetErrorString(err), __FILE__, __LINE__);             \
            exit(err);                                                        \
        }                                                                     \
    }

#ifdef NDEBUG
#define DEBUG(fmt, ...) fprintf(stderr, "[DEBUG] %s:%d: " fmt "\n", __FILE__, __LINE__, ##__VA_ARGS__)
#else
#define DEBUG(fmt, ...)
#undef CUDA_CHECK
#define CUDA_CHECK()
#undef CUDA_ASSERT
#define CUDA_ASSERT(call)
#endif

#define SWAP_POINTERS(x, y, T) do { T SWAP = x; x = y; y = SWAP; } while (0)

// Declarations
enum {
  FRAME_CUR,
  FRAME_NEXT,
  FRAME_REF,
};

struct c63_common* c63_common_init(int width, int height);
void c63_common_free(struct c63_common *cm);

struct frame* create_frame(struct c63_pipeline *pipe, int role);

void destroy_frame(struct frame *f);

void dump_image(yuv_t *image, int w, int h, FILE *fp);

int fpeek(FILE *stream);

#ifdef __cplusplus
}
#endif // __cplusplus


#endif  /* C63_COMMON_H_ */
