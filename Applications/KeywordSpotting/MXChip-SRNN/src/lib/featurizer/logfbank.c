#include "logfbank.h"

#ifdef __cplusplus
extern "C" {
#endif

float32_t meltohz(float32_t mel);
float32_t hztomel(float32_t hz);

#ifdef FFT_Q31
    static q31_t temp_in_q31[2 * NFFT];
    static q31_t powspectrum_q31[(NFFT/ 2 + 1)];
    static q31_t feat_q31[NFILT];
#elif FFT_F32
    static float32_t temp_in[2 * NFFT];
    static float32_t powspectrum[(NFFT/ 2 + 1)];
    static float32_t feat[NFILT];
#endif
    static void preemph_q31(q31_t *temp_in, q31_t preemph,
        int len, q31_t tail);
    static void preemph_f32(float32_t *temp_in, float32_t preemph,
        int len, float tail);
    static void next_pow_2_uint32(uint32_t n, uint32_t *pow2, uint8_t *index);
    static void int32_to_q31(const int32_t *srcVec,  q31_t *destVec,
        int32_t len, uint8_t max_shift); 

#ifdef __cplusplus
}
#endif


void logfbank(
    float32_t *mfcc_result,
    const int32_t *data,
    const void *vfbank,
    const int32_t preemph_tail
    ){
    int ifftFlag = 0, doBitReverse = DO_BIT_REVERSE;

#ifdef FFT_Q31
    uint32_t scaleVal; uint8_t scaleShift;
    int32_t max = 0, min = 0;
    for(int i = 0; i < NFFT; i++){
        int32_t val = data[i];
        if(val > max) max = val;
        if(val < min) min = val;
    }
    max = (max > -1 * min) ? max: -1 * min;
    max = (max > fabs(preemph_tail)) ? max: fabs(preemph_tail);
    next_pow_2_uint32((uint32_t) max, &scaleVal, &scaleShift);
    scaleVal *= 2; scaleShift += 1;
    int32_to_q31(data, temp_in_q31, NFFT, scaleShift);
    // windowing q31 
    q31_t tail, preemph;
    float pre_f = (float)PREEMPH;
    float tail_f = (float)preemph_tail / scaleVal;
    arm_float_to_q31(&tail_f, &tail, 1);
    arm_float_to_q31(&pre_f, &preemph, 1);
    preemph_q31(temp_in_q31, preemph, (int)FRAME_LEN, tail);
    for(int i = 0; i < NFFT; i++){
        q31_t val = (temp_in_q31[NFFT - 1 - i] >> (NFFT_SHIFT));
        temp_in_q31[2 * NFFT - 2 - 2 * i] = val;
        temp_in_q31[2 * NFFT - 1 - 2 * i] = 0;
    }
    arm_cfft_q31(&arm_cfft_sR_q31_len512, temp_in_q31, ifftFlag, doBitReverse);
    // Scale back otherwise we will underflow when computing magnitude
    // We scale by NFFT_SHIT only. Scaling up too much causes saturation.
    arm_scale_q31(temp_in_q31, 0x7fffffff, NFFT_SHIFT, temp_in_q31, 2 * NFFT);
    arm_cmplx_mag_squared_q31(temp_in_q31, powspectrum_q31, NFFT);
    // We are now in q29. Need to scale by 4 (shift by 2) to get back to q31
    arm_scale_q31(powspectrum_q31, 0x7fffffff, 2, powspectrum_q31, (NFFT/2+1));
    // Trying the matmul version. Scaling a little bit more so that
    // we dont loose too much precision in matmuls
    arm_scale_q31(powspectrum_q31, 0x7fffffff, NFFT_SHIFT,
        powspectrum_q31, (NFFT/2 + 1));
    arm_matrix_instance_q31 mfilters, mpowspectrum, mfeat;
    mfilters.numRows = NFILT; mfilters.numCols = NFFT/2 + 1;
    mfilters.pData = (q31_t *)vfbank;
    mpowspectrum.numRows = NFFT/2 + 1; mpowspectrum.numCols = 1;
    mpowspectrum.pData = powspectrum_q31;
    mfeat.numRows = NFILT; mfeat.numCols = 1;
    mfeat.pData = feat_q31;
    arm_status status;
    // We are sufficiently scaled down at this point;
    // don't want to rescale.
    status = arm_mat_mult_fast_q31(&mfilters, &mpowspectrum, &mfeat);
    // if (status != ARM_MATH_SUCCESS); do something ?
    arm_q31_to_float(mfeat.pData, mfcc_result, NFILT);
    arm_scale_f32(mfcc_result, scaleVal * scaleVal, mfcc_result, NFILT);
    for(int i = 0; i < NFILT; i++)
        if (mfcc_result[i] > 0.0f) mfcc_result[i] = log(mfcc_result[i]);
        else mfcc_result[i] = log(DBL_EPS);
    return;
#elif FFT_F32
    // Fbank is in float32_t
    const float32_t *fbank = (float32_t*) vfbank;
    // Pre-emphasis
    for(int i = 0; i < NFFT; i++){
        temp_in[i] = (float)data[i];
    }
    preemph_f32(temp_in, (float32_t)PREEMPH, (int)FRAME_LEN, preemph_tail);
    // Convert to complex notation
    // memset(temp_in, 0, 2 * NFFT * sizeof(float32_t)); is slower
    for(int i = 0; i < NFFT; i++){
        temp_in[2 * NFFT - 2 - 2 * i] = temp_in[NFFT - 1 - i];
        temp_in[2 * NFFT - 1 - 2 * i] = 0;
    }
    arm_cfft_f32(&arm_cfft_sR_f32_len512, temp_in, ifftFlag, doBitReverse);
    arm_cmplx_mag_squared_f32(temp_in, powspectrum, NFFT);
    scalarVectorMul(powspectrum, (NFFT / 2 + 1), (float32_t)(1.0 / NFFT));

    // The below is the non-mat-mul version of 
    // computing filter bank energies. This is faster than matmul
    // version without FPU
    // ---
    // const float32_t *fbank2 = (float32_t*) vfbank;
    // for (int l = 0; l < NFILT; l++) {
    //     feat[l] = 0.0f;
    //     int k = 0;
    //     for (k = 0; k < NFFT / 2 + 1; k++)
    //         feat[l] += powspectrum[k] * fbank2[l * (NFFT/ 2 + 1) + k];
    //     if (feat[l] > 0.0f) feat[l] = log(feat[l]);
    //     else feat[l] = log(DBL_EPS);
    // }
    // The below is the matmul-equivalent
    // This is slightly slower with soft float ABI. Keeping it here
    // since things out to be faster with SIMD and FPU.
    // ---
    arm_matrix_instance_f32 mfilters, mpowspectrum, mfeat;
    mfilters.numRows = NFILT; mfilters.numCols = NFFT/2 + 1;
    mfilters.pData = fbank;
    mpowspectrum.numRows = NFFT/2 + 1; mpowspectrum.numCols = 1;
    mpowspectrum.pData = powspectrum;
    mfeat.numRows = NFILT; mfeat.numCols = 1;
    mfeat.pData = feat;
    arm_status status;
    status = arm_mat_mult_f32(&mfilters, &mpowspectrum, &mfeat);
    for(int i = 0; i < NFILT; i++)
        if (feat[i] > 0.0f) feat[i] = log(feat[i]);
        else feat[i] = log(DBL_EPS);
    // --
    arm_copy_f32(feat, mfcc_result, NFILT);
    return;
#endif
}

void get_filterbank_parameters(float32_t *fbank, int nfilt,
    int samplingRate, int nfft){
    float32_t lowmel = hztomel(0.0f);
    float32_t highmel = hztomel(samplingRate / 2.0f);

    // Generate nfilt center frequencies linearly spaced in the mel scale
    float32_t bin[nfilt + 2];
    int i = 0;
    for (i = 0; i <= nfilt + 1; i++)
        bin[i] = floor(meltohz(i * (highmel - lowmel) /
            (nfilt + 1) + lowmel) * (nfft + 1) / samplingRate);

    memset(fbank, 0, (nfft / 2 + 1) * nfilt * sizeof(float32_t));
    for (i = 0; i < nfilt; i++) {
        int j = 0;
        for (j = (int)bin[i]; j < (int)bin[i + 1]; j++)
            fbank[i * (nfft / 2 + 1) + j] = (j - bin[i]) / (bin[i + 1] - bin[i]);
        for (j = (int)bin[i + 1]; j < (int)bin[i + 2]; j++)
            fbank[i * (nfft / 2 + 1) + j] = (bin[i + 2] - j) / (bin[i + 2] - bin[i + 1]);
    }
}

float32_t hztomel(float32_t hz) {
    return 2595 * log10(1 + hz / 700.0f);
}

float32_t meltohz(float32_t mel){
    return 700 * (pow(10, mel / 2595.0f) - 1);
}

void preemph_q31(q31_t *temp_in, q31_t preemph, int len, q31_t tail){
    int lim;
    int remaining = len;
    static q31_t subs[10], batch = 10;
    int start = 0;
    while (remaining > 0){
        lim = (remaining < batch ? remaining: batch);
        subs[0] = tail;
        tail = temp_in[start + lim - 1];
        for(int i = 1 ; i < lim; i++)
            subs[i] = temp_in[start + i - 1];
        arm_scale_q31(subs, preemph, 0, subs, lim);
        // printFloatArrQ31(&temp_in[start], 10, 512);
        // printFloatArrQ31(subs, 10, 512);
        arm_sub_q31(&temp_in[start], subs, &temp_in[start], lim);
        // printFloatArrQ31(&temp_in[start], 10, 512);
        // printInt32(0);
        start += lim;
        remaining -= lim;
    }
}

void preemph_f32(float32_t *temp_in, float32_t preemph,
    int len, float tail){
    for(int i = len-1; i > 0; i--){
        temp_in[i] = temp_in[i] - preemph * temp_in[i-1];
    }
    temp_in[0] = temp_in[0] - preemph * tail;
}

void next_pow_2_uint32(uint32_t n, uint32_t *pow2, uint8_t *index) { 
    // Convert N to power of 2
    n--; 
    n |= n >> 1; 
    n |= n >> 2; 
    n |= n >> 4; 
    n |= n >> 8; 
    n |= n >> 16; 
    n++;
    *pow2 = n;
    // Do a fast log to find which index this is (shift amount)
    // WARNING: 0 means no shift (if n==1 or n==0)
    if (n == 0) {*index = 0; return;}
    if ((n & 1) == 1) {*index = 0; return;}
    for(uint8_t i = 1; i <= 31; i++)
        if ((n >> i) == 1) {*index = i; return;}
} 

// Converting int32 to q31 requires reversing the order.
// For instance, the number 17 in int is 0x00000011 but this
// can't be represented in q31. We need atleast q8.24 for this.
// In q8.24, we the representation is 0x11.000000 .
//
// Hence we implement this conversion as follows - we copy 'max_shift'
// number of  bits from the left side of the int and into the
// first 'max_shift'  bits of a zero-fileld 32 sized bitvector (for
// lack of better terminology). Hence, the returned number in q31 is the
// original number scaled down by 2^(max_shift).
//
// max_shift should at least be 2 and atmost 31 for the behaviour
// to be defined.
void int32_to_q31(const int32_t *srcVec,  q31_t *destVec,
    int32_t len, uint8_t max_shift){
    // Create a mask of max_shift
    for(int i = 0; i < len; i++){
        destVec[i] = srcVec[i] << (31 - (max_shift));
    }
}