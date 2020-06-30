
#ifndef __CON_CONVOLUTION_H__
#define __CON_CONVOLUTION_H__


/**
 * @brief 
 * @param[in]       
 * @param[in]       
 * @param[in]       
 * @param[in]       
   @return          none
 * @example         
 *                            
 *                            
 *                  
*/
void conv(INT_T *A, const INT_T *B, INT_T *C, INT_T *tmp,                   \
          INT_T N, INT_T H, INT_T W, INT_T CI, INT_T HF,                    \
          INT_T WF, INT_T CO, INT_T shrA, INT_T shrB, INT_T H1, INT_T H2);


/**
 * @brief 
 * @param[in]       
 * @param[in]       
 * @param[in]       
 * @param[in]       
   @return          none
 * @example         
 *                            
 *                            
 *                  
*/
void convolution(INT_T *A, const INT_T *B, INT_T *C, INT_T *tmp, 
                 INT_T N, INT_T H, INT_T W, INT_T CIN, INT_T HF,        \
                 INT_T WF, INT_T CINF, INT_T COUTF, INT_T HOUT,         \
                 INT_T WOUT, INT_T HPADL, INT_T HPADR, INT_T WPADL,     \
                 INT_T WPADR, INT_T HSTR, INT_T WSTR, INT_T HDL,        \
                 INT_T WDL, INT_T G, INT_T shrA, INT_T shrB, INT_T H1,  \
                 INT_T H2);
#endif
