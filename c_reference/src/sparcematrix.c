#include "sparcematrix.h"

void sp_mat_mul(const INT_T *Aidx, const INT_T *Aval, INT_T **B, INT_T *C, INT_T K, \
                INT_T shrA, INT_T shrB, INT_T shrC) {

  INT_T   ite_idx = 0; 
  INT_T   ite_val = 0;
  INT_T   k       = 0;
  INT_T   idx     = 0;
  INT_T   b       = 0;
  INT_T   a       = 0;
  INT_T   c       = 0;
  
  if(Aidx && Aval && B && C) {

    for (k = 0; k < K; k++) {

      b = B[k * 1][0];
#ifdef SHIFT
      b = b >> shrB;
#else
      b = b / shrB;
#endif
    
      idx = Aidx[ite_idx];
      while (idx != 0) {
        a = Aval[ite_val];
#ifdef SHIFT
        a = a >> shrA
#else
        a = a / shrA;
#endif
        c = a * b;

#ifdef SHIFT
        c = c >> shrC
#else
        c = c / shrC;
#endif
      
        C[idx - 1] += c;

        ite_idx++;
        ite_val++;

        idx = Aidx[ite_idx];
      }
      ite_idx++;
    }
  }
  return;
}