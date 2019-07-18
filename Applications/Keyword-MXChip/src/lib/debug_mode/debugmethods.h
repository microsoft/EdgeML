#include <arm_math.h>

#ifdef __cplusplus
extern "C" {
#endif

void printFloatArrF32(float32_t *, int, float);
void printFloatArrQ31(q31_t *, int, float);
void printIntArr(int32_t *, int, int);
void printHexQ31(q31_t);
void printInt32(int32_t);
void printVoid(void *);
void printStr(char *);
void printFloatAddr(float *);

#ifdef __cplusplus
}
#endif