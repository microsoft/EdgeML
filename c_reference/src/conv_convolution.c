#include "conv_convolution.h"

void conv(INT_T *A, const INT_T *B, INT_T *C, INT_T *tmp,                   \
          INT_T N, INT_T H, INT_T W, INT_T CI, INT_T HF,                    \
          INT_T WF, INT_T CO, INT_T shrA, INT_T shrB, INT_T H1, INT_T H2) {

  INT_T     padH      =  0;
  INT_T     padW      =  0;
  INT_T     n         =  0;
  INT_T     h         =  0;
  INT_T     w         =  0;
  INT_T     co        =  0;
  INT_T     counter   =  0;
  INT_T     hf        =  0;
  INT_T     wf        =  0;
  INT_T     ci        =  0;
  INT_T     a         =  0;
  INT_T     b         =  0;
  INT_T     totalEle  =  0;
  INT_T     count     =  0;
  INT_T     depth     =  0;
  INT_T     p         =  0;
  INT_T     sum       =  0;
  uint8_t   shr       =  0;

  if(A && B && C && tmp) { 
    padH      = (HF - 1) / 2;
    padW      = (WF - 1) / 2;

    for (n = 0; n < N; n++) {
      for (h = 0; h < H; h++) {
        for (w = 0; w < W; w++) {
          for (co = 0; co < CO; co++) {

            counter = 0;
            for (hf = 0; hf < HF; hf++) {
              for (wf = 0; wf < WF; wf++) {
                for (ci = 0; ci < CI; ci++) {
                  a = (((((h + hf) < padH) || ((h + hf) >= (H + padH))) ||     \
                      (((w + wf) < padW) || ((w + wf) >= (W + padW)))) ? 0 :   \
                      A[n * H * W * CI + ((h + hf) - padH) * W * CI + ((w + wf) - padW) * CI + ci]);

                  b = B[hf * WF * CI * CO + wf * CI * CO + ci * CO + co];

#ifdef SHIFT
                  a = a >> shrA;
                  b = b >> shrB;
#else
                  a = a / shrA;
                  b = b / shrB;
#endif
                  
                  tmp[counter] = a * b;
                  counter++;
                }
              }
            }

            totalEle  = HF * WF * CI;
            count     = HF * WF * CI; 
            shr       = 1;

            while (depth < (H1 + H2)){

              if (depth >= H1)
                shr = 0;
              
#ifdef SHIFT
              for (p = 0; p < ( (totalEle >> 1 ) + 1); p++) {
#else
              for (p = 0; p < ( (totalEle / 2) + 1); p++) {
#endif
                if (p < (count >> 1)) {
                  if (shr)
#ifdef SHIFT
                    sum = ( tmp[2 * p] >> 1 ) + ( tmp[(2 * p) + 1] >> 1 );
#else
                    sum = tmp[2 * p] / 2 + tmp[(2 * p) + 1] / 2;
#endif
                  else
                    sum = tmp[2 * p] + tmp[(2 * p) + 1];
                }
                else if ((p == (count >> 1)) && ((count & 1) == 1)) {
                  if (shr)
#ifdef SHIFT
                    sum = tmp[2 * p] >> 1;
#else
                    sum = tmp[2 * p] / 2;
#endif
                  else
                    sum = tmp[2 * p];
                }
                else
                  sum = 0;

                tmp[p] = sum;
              }
#ifdef SHIFT
              count = (count + 1) >> 1;
#else
              count = (count + 1) / 2;
#endif

              depth++;
            }

            C[n * H * W * CO + h * W * CO + w * CO + co] = tmp[0];
          }
        }
      }
    }
  }

  return;
}
void convolution(INT_T *A, const INT_T *B, INT_T *C, INT_T *tmp, 
                 INT_T N, INT_T H, INT_T W, INT_T CIN, INT_T HF,        \
                 INT_T WF, INT_T CINF, INT_T COUTF, INT_T HOUT,         \
                 INT_T WOUT, INT_T HPADL, INT_T HPADR, INT_T WPADL,     \
                 INT_T WPADR, INT_T HSTR, INT_T WSTR, INT_T HDL,        \
                 INT_T WDL, INT_T G, INT_T shrA, INT_T shrB, INT_T H1,  \
                 INT_T H2) {
  
  INT_T     HOffsetL  = 0; 
  INT_T     WOffsetL  = 0;
  INT_T     HOffsetR  = 0;
  INT_T     WOffsetR  = 0;
  INT_T     n         = 0;
  INT_T     h         = 0;
  INT_T     w         = 0;
  INT_T     g         = 0;
  INT_T     co        = 0;
  INT_T     counter   = 0;
  INT_T     hf        = 0;
  INT_T     wf        = 0;
  INT_T     ci        = 0;
  INT_T     a         = 0;
  INT_T     b         = 0;
  INT_T     p         = 0;
  INT_T     totalEle  = 0;
  INT_T     count     = 0;
  INT_T     depth     = 0;
  INT_T     sum       = 0;
  INT_T     hout      = 0;
  INT_T     wout      = 0;
  uint8_t   shr       = 0;

  if( A && B && C && tmp) {
    HOffsetL = HDL*(HF/2) - HPADL;
    WOffsetL = WDL*(WF/2) - WPADL;
    HOffsetR = HDL*(HF/2) - HPADR;
    WOffsetR = WDL*(WF/2) - WPADR;

    for(n = 0; n < N; n++) {
      for(h = HOffsetL, hout = 0; h < H - HOffsetR; h += HSTR, hout++) {
        for(w = WOffsetL, wout = 0; w < W - WOffsetR; w += WSTR, wout++) {
          for(g = 0; g < G; g++) {
            for(co = 0; co < COUTF; co ++) {

              counter = 0;
              for(hf = -(HF/2); hf <= HF/2; hf++) {
                for(wf = -(WF/2); wf <= WF/2; wf++) {
                  for(ci = 0; ci < CINF; ci++) {

                    a = (((h + HDL * hf) < 0) || ((h + HDL * hf) >= H) ||   \
                         ((w + WDL * wf) < 0) || ((w + WDL * wf) >= W))     \
                          ? 0 : A[n * H * W * CIN + (h + HDL * hf) * W *    \
                            CIN + (w + WDL * wf) * CIN + (ci + g * CINF)];

                    b = B[g * HF * WF * CINF * COUTF + (hf + HF/2) * WF *   \
                        CINF * COUTF + (wf + WF/2) * CINF * COUTF + ci * COUTF + co];

#ifdef SHIFT
                    a = a >> shrA;
                    b = b >> shrB;
#else
                    a = a / shrA;
                    b = b / shrB;
#endif

                    tmp[counter] = a * b;
                    counter++;
                  }
                }
              }

              totalEle  = HF * WF * CINF;
              count     = HF * WF * CINF;
              shr       = 1;

              while (depth < (H1 + H2)) {
                if (depth >= H1)
                  shr = 0;

                for (p = 0; p < (totalEle / 2 + 1); p++) {
                  
                  if (p < (count >> 1)) {
                    if (shr)
#ifdef SHIFT
                      sum = tmp[2 * p] >> 1 + tmp[(2 * p) + 1] >> 1;
#else
                      sum = tmp[2 * p] / 2 + tmp[(2 * p) + 1] / 2;
#endif
                    else
                      sum = tmp[2 * p] + tmp[(2 * p) + 1];
                  }
                  else if ((p == (count >> 1)) && ((count & 1) == 1)) {
                    if (shr)
#ifdef SHIFT
                      sum = tmp[2 * p] >> 1;
#else
                      sum = tmp[2 * p] / 2;
#endif
                    else
                      sum = tmp[2 * p];
                  }
                  else
                    sum = 0;

                  tmp[p] = sum;
                }
#ifdef SHIFT
                count = (count + 1) >> 1;
#else
                count = (count + 1) / 2 ;
#endif
                depth++;
              }

              C[n * HOUT * WOUT * (COUTF * G) + hout * WOUT * (COUTF * G)   \
                + wout * (COUTF * G) + (co + g * COUTF)] = tmp[0];
            }
          }
        }
      }
    }
  }
}
