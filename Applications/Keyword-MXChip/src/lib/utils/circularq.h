#pragma once
/*
 * WARNING 1: This is not a true circular queue. You can't pop from
 * the front unless the queue is full. I haven't implemented
 * a pop front method
 *
 * WARNING 2: Not thread-safe
 */

// @todo: reimplement with just size and not function
// pointers and memcpy.

#include <stddef.h>
#include <arm_math.h>

typedef struct StructFIFOCircularQ {
    void *queue;
    int front;
    int back;
    int maxSize;
    // (container to edit, index, value)
    void (*cb_write)(void *, int, void *);
    // (container to read, index)
    void* (*cb_read)(void *, int);
} FIFOCircularQ;

#ifdef __cplusplus
    extern "C" {
#endif 

// Since we are being compied as C++
// Returns the length of the queue (vacuous)
int q_init(
    FIFOCircularQ *q, 
    void* container, 
    int length,
    void (*cb_write)(void *, int, void*),
    void* (*cb_read)(void *, int)
    );
//
// Returns 0 if q is not full, 1 otherwise
//
int q_is_full(FIFOCircularQ *q);

//
// Resets the q
//
void q_reset(FIFOCircularQ *q);

//
// deletes the at(0) element. Does not return it.
// Use q_oldest + q_del_oldest for that behaviour.
//
int q_del_oldest(FIFOCircularQ *q);

// 
// Pushes element to the back of the buffer
// Return 0 on success, -1 if buffer full
//
int q_enqueue(FIFOCircularQ *q, void *obj);

//
// Batch version of q_enqueue. Pushes len elements or nothing
// Specify @obj_size of each object in bytes 
// For a batch [0, 1, 2, 3], 0 is enqueued first, followed by
// 1, 2, 3 in that order.
// Return 0 on success, -1 if buffer full, -2 if q corrupted
//
int q_enqueue_batch(FIFOCircularQ *q, void *obj, size_t obj_size, int len);

//
// q_enqueue and overwrite earliest element if buffer full
//
void q_force_enqueue(FIFOCircularQ *q, void *obj);
//
// Similar to q_enqueue_batch but forcefully removed earlier elements
// if queue does not have enough space
// TODO: Test cases
void q_force_enqueue_batch(FIFOCircularQ *q, void *obj,
    size_t obj_size, int len);
//
// get element at N-th position in circular buffer
// 0 - earliest pushed element
// size - 1: last pushed element
// Fails if N > size
//
void* q_atN(FIFOCircularQ *q, int N);

//
// get earliest element inserted into the circular buffer
//
void* q_oldest(FIFOCircularQ *q);

//
// set element at N-th position in circular buffer
// Fails if N > size
//
void q_setN(FIFOCircularQ *q, int N, void *obj);

//
// Returns number of elements in queue
//
int q_getSize(FIFOCircularQ *q);


void q_flatten_int(FIFOCircularQ *q, int *dst);
void q_flatten_float(FIFOCircularQ *q, float *dst);
// data type specific call back
void cb_write_int(void *container, int index, void *val);
void cb_write_int16(void *container, int index, void *val);
void cb_write_float(void *container, int index, void *val);
void cb_write_char(void *container, int index, void *val);
void* cb_read_int(void *container, int index);
void* cb_read_int16(void *container, int index);
void* cb_read_float(void *container, int index);
void* cb_read_char(void *container, int index);

#ifdef __TEST_CIRCULAR_Q__
    You also have to manually uncomment this function
    declaration in circularq.c
    int test_circularq();
#endif

#ifdef __cplusplus
} // end extern "C"
#endif
