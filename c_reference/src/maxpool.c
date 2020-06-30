#include "maxpool.h"

void maxpool(INT_T *A, INT_T *B, INT_T N, INT_T H, INT_T W, INT_T C, INT_T FH, \
             INT_T FW, INT_T strideH, INT_T strideW, INT_T HPADL, INT_T HPADR, \
            INT_T WPADL, INT_T WPADR) {

  INT_T n   = 0;
  INT_T ho  = 0;
  INT_T wo  = 0;
  INT_T c   = 0;
  INT_T max = 0;
  INT_T a   = 0;
  INT_T hs  = 0;
  INT_T ws  = 0;
  INT_T HO  = 0;
  INT_T WO  = 0;


#ifdef SHIFT 
  HO = H >> strideH;
  WO = W >> strideW;
#else
  HO = H / strideH;
  WO = W / strideW;
#endif /* SHIFT */
  if(A && B) {
    for (n = 0; n < N; n++) {
      for (ho = 0; ho < HO; ho++) {
        for (wo = 0; wo < WO; wo++) {
          for (c = 0; c < C; c++) {

            max = A[n * H * W * C + (strideH * ho) * W * C + (strideW * wo) * C + c];
            for (hs = 0; hs < FH; hs++) {
              for (ws = 0; ws < FW; ws++) {
                a = A[n * H * W * C + ((strideH * ho) + hs) * W * C + ((strideW * wo) + ws) * C + c];
                if (a > max)
                  max = a;
              }
            }

            B[n * HO * WO * C + ho * WO * C + wo * C + c] = max;
          }
        }
      }
    }
  }
  return;
}