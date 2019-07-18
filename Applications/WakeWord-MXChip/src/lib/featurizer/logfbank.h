#pragma once
#include <arm_math.h>
#include <arm_const_structs.h>
#include <string.h>
#include "../utils/helpermath.h"

#ifdef DEBUG_MODE
    #include "../debug_mode/debugmethods.h"
#endif

#define DBL_EPS 2.2204460492503131e-16
// #define PI 3.14159265358979323846264338327
#define SAMPLING_RATE 16000
#define FRAME_LEN 400
#define STRIDE 160
#define NFILT 32
#define HAMMING 2
#define DO_BIT_REVERSE 1

// Various supported windowing schemes
#define WIN_RECTANGULAR 0
#define WIN_HANNING 1
#define WIN_HAMMING 2
#define PREEMPH 0.97

#ifdef __cplusplus
extern "C" {
#endif 

#if defined(NFFT_512)
    #define NFFT 512
    #define NFFT_SHIFT 9
#endif
#if defined(FFT_Q31)
    #define CFFT_INSTANCE arm_cfft_sR_q31_len512
    #define CFFT_FUNC arm_cfft_q31
#elif  defined(FFT_F32)
    #define CFFT_INSTANCE arm_cfft_sR_f32_len512
    #define CFFT_FUNC arm_cfft_f32
#endif
//
// Computes the MFCC of one frame (WINLEN) of data. Note that windowing
// is not performed as part of this function. It needs to be done outside this
// method before data is passed on. Also note that many MFCC parameters like
// number of filters, stride, sampling rate, fft length etc are fixed as
// as compile time constants. Hence, this method is not a generic 
// implementation - like the one found in the Pi3 or Pi0 implementations.
//
// mfcc_result 		- stores the output of the MFCC computation.
// data 			- Data of length WINLEN
// fbank            - The filter bank for this configuration. Can be obtained by
//                    get_filterbank_parameters_xx
// preemph_tail     - Tail element of the previous window for preemphasis.
//                    Set to 0 if this is the first window.

void logfbank(
    float32_t *mfcc_result,
    const int32_t *data,
    // The fbank filters. These are calculated
    // externally and passed in the interest of
    // speed. Use get_filterbank_parameters for this.
    const void *vfbank,
    const int32_t preemph_tail
);

void get_filterbank_parameters(
    float32_t *fbank,
    int nfilt,
    int samplingRate,
    int nfft
);

#ifdef __cplusplus
}
#endif // __cplusplus

