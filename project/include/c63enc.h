#ifndef C63ENC_H
#define C63ENC_H

#include "c63.h"

#ifdef __cplusplus
extern "C" {
#endif

struct c63_common* init_c63_enc(int width, int height);

void c63_encode_image(struct c63_common *cm);

void cleanup_cm(void);
void free_c63_enc(struct c63_common *cm);

#ifdef __cplusplus
}
#endif

#endif // C63ENC_H

