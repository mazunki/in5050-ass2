#ifndef C63_WRITE_H_
#define C63_WRITE_H_

#include "c63.h"

// Declaration
void write_frame(struct c63_common *cm);
void *pthread_write_frame(void *ptr);

#endif  /* C63_WRITE_H_ */
