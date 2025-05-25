#ifndef C63_WRITE_H_
#define C63_WRITE_H_

#include "c63.h"

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

// Declaration
void write_frame(struct c63_common *cm, struct frame *f);
void *pthread_write_frame(void *ptr);

#ifdef __cplusplus
}
#endif // __cplusplus

#endif  /* C63_WRITE_H_ */
