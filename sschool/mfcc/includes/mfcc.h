#pragma once

#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#include "fft.h"
#include <complex.h>

typedef float FPTYPE;

#define DBL_EPS 2.2204460492503131e-16
#define PI 3.14159265358979323846264338327
#define PREEMPH 0.97
#define SAMPLING_RATE 16000
#define NFILT 32
#define HAMMING 2
#define NFFT 512
#define WINLEN 400
#define STRIDE 160
#define NODCT 0
// #define FRAMESIZE 26032
// #define NUMFRAMES FRAMESIZE/STRIDE

void mfcc(FPTYPE* mfcc_result, FPTYPE *data, int samplingRate, int nfilt, int numcep, int nfft, int ceplifter, int appendEnergy, int window, int numentries, int doDCT);
void get_filterbank_parameters(FPTYPE *fbank, int nfilt, int samplingRate, int nfft);
FPTYPE hztomel(FPTYPE hz);
FPTYPE meltohz(FPTYPE mel);
void windowing(FPTYPE *temp_in, int numentries, int window);