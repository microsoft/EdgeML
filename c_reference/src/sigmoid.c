#include "sigmoid.h"


void sigmoid(INT_T *A, INT_T I, INT_T J, INT_T div, INT_T add, INT_T sigmoid_limit, \
             INT_T scale_in, INT_T scale_out, INT_T *B) {

  INT_T i           = 0;
  INT_T x           = 0;
  INT_T y           = 0;
  INT_T scale_diff  = 0;

#ifdef FLOATEXP
  INT_T z           = 0;
#endif

  if(A && B) {
#ifdef SHIFT
      scale_diff = scale_out >> scale_in;
#else
      scale_diff = scale_out / scale_in;
#endif /* SHIFT */

    for (i = 0; i < I*J; i++) {
#ifdef FLOATEXP
#ifdef SHIFT
      float x = float(A[i]) >> scale_in;

      float y = 1 >> (1 + exp(-x));
#else
      float x = float(A[i]) / scale_in;

      float y = 1 / (1 + exp(-x));
#endif /* SHIFT */

      z = INT_T(y * scale_out);

      B[i] = z;
#else
      x = A[i];
#ifdef SHIFT
      x = (x >> div) + add;
#else
      x = (x / div) + add;
#endif /* SHIFT */

      if (x >= sigmoid_limit)
        y = sigmoid_limit;
      else if (x <= 0)
        y = 0;
      else
        y = x;

      y = y * scale_diff;

      B[i] = y;
#endif
    }
  }

  return;
}