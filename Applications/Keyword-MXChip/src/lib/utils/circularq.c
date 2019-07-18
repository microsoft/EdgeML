#include "circularq.h"

int q_init(
    FIFOCircularQ *q,
    void *container,
    int length,
    void(*cb_write)(void *, int, void*),
    void* (*cb_read)(void *, int)
) {
    q->front = -1;
    q->back = -1;
    q->queue = container;
    q->maxSize = length;
    q->cb_write = cb_write;
    q->cb_read = cb_read;
    return length;
}

int q_del_oldest(FIFOCircularQ *q) {
    if (q->front == -1) {
        return -1; // FALSE
    }
    q->back++;
    q->back %= q->maxSize;
    if (q->back == q->front) {
        q->back = -1;
        q->front = -1;
    }
    return 0; // True
}

int q_enqueue(
    FIFOCircularQ *q,
    void *obj) {
    if (q->front == -1) {
        // empty
        q->back = 0;
        q->front = 0;
    }
    else if (q->back == q->front) {
        // full
        return -1; // false
    }
    q->cb_write(q->queue, q->front, obj);
    q->front++;
    q->front %= q->maxSize;
    q->back %= q->maxSize;
    return 0;
}

int q_enqueue_batch(FIFOCircularQ *q, void *obj,
    size_t obj_size, int len) {
    if (q_getSize(q) + len > q->maxSize)
        return -1;
    else {
        for (int i = 0; i < len; i++) {
            int ret = q_enqueue(q, obj + obj_size * i);
            if (ret != 0)
                return -2;
        }
    }
    return 0;
}

void q_force_enqueue(FIFOCircularQ *q, void *obj) {
    if (q->front == -1) {
        // empty
        q->back = 0;
        q->front = 0;
    }
    else if (q->back == q->front) {
        // full
        q->back += 1;
    }
    q->cb_write(q->queue, q->front, obj);
    q->front++;
    q->front %= q->maxSize;
    q->back %= q->maxSize;
}

void q_force_enqueue_batch(FIFOCircularQ *q, void *obj,
    size_t obj_size, int len){
    for (int i = 0; i < len; i++){
        q_force_enqueue(q, obj + obj_size * i);
    }
}

void* q_atN(FIFOCircularQ *q, int i) {
    if (q_getSize(q) < i || q_getSize(q) == 0)
        return NULL;

    int index = (q->back + i) % q->maxSize;
    return q->cb_read(q->queue, index);
}

void* q_oldest(FIFOCircularQ *q) {
    return q_atN(q, 0);
}

void q_setN(FIFOCircularQ *q, int i, void *obj) {
    int index = (q->back + i) % q->maxSize;
    q->cb_write(q->queue, index, obj);
}

int q_getSize(FIFOCircularQ *q) {
    if (q->front == -1)
        return 0;
    else if (q->front == q->back)
        return q->maxSize;
    else if (q->front > q->back)
        return q->front - q->back;
    else
        return q->front + (q->maxSize - q->back);
}

int q_is_full(FIFOCircularQ *q){
    return (q_getSize(q) == q->maxSize);
}

void q_reset(FIFOCircularQ *q){
    q->front = -1;
    q->back = -1;
}

void q_flatten_int(FIFOCircularQ *q, int *dst) {
    int size = q_getSize(q);
    for (int i = 0; i < size; i++) {
        dst[i] = *(int*)q_atN(q, i);
    }
}

void q_flatten_float(FIFOCircularQ *q, float *dst) {
    int size = q_getSize(q);
    for (int i = 0; i < size; i++) {
        dst[i] = *(float*)q_atN(q, i);
    }
}

// There is no point inlining these functions. Since we are taking
// the functions address at other places, the compiler might not
// (and cannot in most cases) inline it.
inline void cb_write_int(void *container, int index, void *val) {
    int *_container = (int*)container;
    _container[index] = *((int*)val);
}

inline void cb_write_int16(void *container, int index, void *val) {
    int16_t *_container = (int16_t *)container;
    _container[index] = *((int16_t*)val);
}

inline void cb_write_float(void *container, int index, void *val) {
    float *_container = (float*)container;
    _container[index] = *((float*)val);
}

inline void cb_write_char(void *container, int index, void *val) {
    char *_container = (char*)container;
    _container[index] = *((char*)val);
}

inline void* cb_read_int(void *container, int index) {
    int *_container = (int*)container;
    return &(_container[index]);
}

inline void* cb_read_int16(void *container, int index) {
    int16_t *_container = (int16_t *)container;
    return &(_container[index]);
}

void* cb_read_float(void *container, int index) {
    float *_container = (float*)container;
    return &(_container[index]);
}

void* cb_read_char(void *container, int index) {
    char *_container = (char*)container;
    return &(_container[index]);
}

/*
 * Uncomment the following to perform circularq tests on mxchip.
 * Do not forget to define the __TEST_CIRCULAR_Q__ preprocessor directive
 */

#ifdef __TEST_CIRCULAR_Q__
#include <math.h>
int test_circularq() {
    FIFOCircularQ Q;
    unsigned errorCode = 0;
    int container[200];
    if (!q_init(&Q, container, 200, cb_write_int, cb_read_int))
        errorCode |= 1;

    for (int i = 0; i < 100; i++) {
        int j = i + 100;
        q_force_enqueue(&Q, &(j));
    }

    int size = q_getSize(&Q);
    if (!(size == 100))
        errorCode |= 2;
    for (int i = 0; i < 100; i++) {
        if (!(*(int*)(q_atN(&Q, i)) == i + 100))
            errorCode |= 4;
    }


    for (int i = 100; i < 200; i++) {
        int j = i + 100;
        q_force_enqueue(&Q, &j);
    }

    size = q_getSize(&Q);
    if (!(size == 200))
        errorCode |= 8;
    for (int i = 0; i < 200; i++) {
        if (!(*(int*)q_atN(&Q, i) == i + 100))
            errorCode |= 16;
    }


    for (int i = 0; i < 100; i++) {
        int j = i + -10000;
        q_force_enqueue(&Q, &j);
    }

    size = q_getSize(&Q);
    if (!(size == 200))
        errorCode |= 32;
    for (int i = 0; i < 100; i++) {
        if (!(*(int*)q_atN(&Q, i) == i + 100 + 100)) {
            errorCode |= 64;
        }
    }
    for (int i = 100; i < 200; i++) {
        if (!(*(int*)q_atN(&Q, i) == i - 100 - 10000)) {
            errorCode |= 128;
        }
    }

    FIFOCircularQ Qf;
    float eps = 1e-10;
    float containerf[1000];
    if (!q_init(&Qf, containerf, 1000, cb_write_float, cb_read_float))
        errorCode |= 256;

    for (int i = 0; i < 1000; i++) {
        float j = i + 100.0;
        q_force_enqueue(&Qf, &(j));
    }

    size = q_getSize(&Qf);
    if (!(size == 1000))
        errorCode |= (1 << 9);
    for (int i = 0; i < 1000; i++) {
        float val = *(float*)(q_atN(&Qf, i));
        if (fabs(val - i - 100.0) >= eps)
            errorCode |= (1 << 10);
    }

    for (int i = 100; i < 1000; i++) {
        float j = (float)i - 1000.0;
        q_force_enqueue(&Qf, &j);
    }

    size = q_getSize(&Qf);
    if (!(size == 1000))
        errorCode |= (1 << 11);

    for (int i = 0; i < 100; i++) {
        float val = *(float*)(q_atN(&Qf, i));
        if (fabs(val - i - 1000) >= eps)
            errorCode |= (1 << 12);
    }

    for (int i = 100; i < 1000; i++) {
        float val = *(float*)(q_atN(&Qf, i));
        if (fabs(val - i + 1000) >= eps)
            errorCode |= (1 << 12);
    }

    // Testing push pop
    int container3[10];
    FIFOCircularQ Q3;
    q_init(&Q3, container3, 10, cb_write_int, cb_read_int);

    for (int i = 0; i < 10; i++) {
        if (q_enqueue(&Q3, &i)) {
            errorCode |= (1 << 16);
        }
    }
    for (int i = 10; i < 15; i++) {
        if (0 == q_enqueue(&Q3, &i)) {
            errorCode |= (1 << 17);
        }
    }

    if(*(int*)q_oldest(&Q3) != 0) errorCode |= (1 << 18);
    for (int i = 0; i < 10; i++) {
        int j = *(int*)q_atN(&Q3, i);
        if (j != i) errorCode |= (1 << 18);
    }

    q_del_oldest(&Q3); q_del_oldest(&Q3);
    if((*(int*)q_oldest(&Q3)) != 2) errorCode |= (1 << 19);
    for (int i = 0; i < 8; i++) {
        int j = *(int*)q_atN(&Q3, i);
        if (j - 2 != i) errorCode |= (1 << 19);
    }

    q_del_oldest(&Q3);
    for (int i = 0; i < 7; i++){
        int j = *(int*)q_oldest(&Q3);
        size = q_getSize(&Q3);
        if ((size != 7 - i) ||
            (j != i + 3))
            errorCode |= (1 << 20);
        q_del_oldest(&Q3);
    }

    size = q_getSize(&(Q3));
    if (size != 0) errorCode |= (1 << 21);

    int *p1, *p2;
    p1 = (int *)q_atN(&(Q3), 0);
    p2 = (int *)q_oldest(&Q3);
    if(p1 != NULL || p2 != NULL) errorCode |= (1 << 22);

    for (int i = 0; i < 19; i++)
        q_enqueue(&(Q3), &i);
    for (int i = 0; i < 10; i++) {
        int k = *(int*)q_oldest(&(Q3));
        if ((k != i)) errorCode |= (1 << 23);
        q_del_oldest(&(Q3));
    }
    if(q_del_oldest(&(Q3)) != -1) errorCode |= 23;

    // Testing enqueu batch
    float containerbq[10];
    FIFOCircularQ bq;
    q_init(&bq, containerbq, 10, cb_write_float, cb_read_float);
    float vals[] = {1, 2, 3, 4};
    q_enqueue_batch(&(bq), vals, sizeof(float), 4);
    size = q_getSize(&bq);
    if (size != 4) errorCode |= (1 << 24);
    if(*(float*)q_oldest(&bq) != 1)
        errorCode |= (1 << 24);

    vals[0] = 5; vals[1] = 6;
    q_enqueue_batch(&(bq), vals, sizeof(float), 2);
    size = q_getSize(&bq);
    if (size != 6) errorCode |= (1 << 25);
    vals[0] = 7; vals[1] = 8; vals[2] = 9; vals[3] = 10;
    q_enqueue_batch(&(bq), vals, sizeof(float), 4);
    size = q_getSize(&bq);
    if (size != 10) errorCode |= (1 << 25);
    for(int i = 0; i < 10; i++){
        if(fabs(*(float*)q_atN(&bq, i) - i - 1) > eps)
            errorCode |= 1 << 25;
    }
    for(int i = 0; i < 10; i++){
        float j = *(float*)q_oldest(&bq);
        if(fabs(j - i - 1) > eps)
            errorCode |= 1 << 25;
        q_del_oldest(&bq);
    }
    size = q_getSize(&bq);
    if (size != 0) errorCode |= (1 << 25);
    q_enqueue_batch(&(bq), vals, sizeof(float), 4);
    q_enqueue_batch(&(bq), vals, sizeof(float), 4);
    if(q_enqueue_batch(&(bq), vals, sizeof(float), 4) != -1)
        errorCode |= (1 << 25);
    return errorCode;
}
#endif //__TEST_CIRCULAR_Q__