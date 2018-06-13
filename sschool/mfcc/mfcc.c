// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "mfcc.h"

// mfcc_result 		- stores the output of the MFCC computation - array of length numcep
// data 			- input data of length numentries
// samplingRate 	- the rate at which audio was sampled in Hz
// nfilt 			- number of filters desired between 0 and samplingRate/2
// numcep 			- number of MFCC coefficients
// nfft 			- N in N-point FFT
// ceplifter 		- 0 indicates no lifter 
// window 			- 0/1/2, 0 - rectangular, 1 - Hanning, 2 - Hamming
// doDCT			- Whether you want MFCC or filter bank outputs
void mfcc(
    FPTYPE* mfcc_result, FPTYPE *data, int samplingRate, int nfilt, int numcep,
    int nfft, int ceplifter, int appendEnergy, int window, int numentries, int doDCT)
{
    // Get frame data into temp_in and pad with zeros if numentries < nfft
    FPTYPE temp_in[nfft];
    FPTYPE feat[nfilt];
    FPTYPE powspectrum[(nfft / 2 + 1)];
    FPTYPE fbank[nfilt * (nfft / 2 + 1)];

    complex float *temp_out = mufft_alloc(nfft * sizeof(complex float));

    memset(temp_in, 0, nfft * sizeof(FPTYPE));
    memcpy(temp_in, data, sizeof(FPTYPE) * numentries);
    // Windowing
    if (window != 0)
        windowing(temp_in, numentries, window);
    mufft_plan_1d *muplan = mufft_create_plan_1d_r2c(nfft, 0);
    mufft_execute_plan_1d(muplan, temp_out, temp_in);

    // Compute power spectra
    int i = 0;
    for (i = 0; i < nfft / 2 + 1; i++) {
        powspectrum[i] = (1.0f / nfft) * (pow(creal(temp_out[i]), 2) + pow(cimag(temp_out[i]), 2));
    }

    // Compute the filterbank parameters
    get_filterbank_parameters(fbank, nfilt, samplingRate, nfft);

    FPTYPE specenergy = 0.0f;
    for (i = 0; i < nfft / 2 + 1; i++)
        specenergy += powspectrum[i];
    if (specenergy <= 0.0f)
        specenergy = DBL_EPS;

    // Get filter bank output
    int l = 0;
    for (l = 0; l < nfilt; l++) {
        feat[l] = 0.0f;
        int k = 0;
        for (k = 0; k < nfft / 2 + 1; k++)
            feat[l] += powspectrum[k] * fbank[l * (nfft / 2 + 1) + k];
        if (feat[l] > 0.0f)
            feat[l] = log(feat[l]);
        else
            feat[l] = log(DBL_EPS);
    }

    if (doDCT == 1)
    {
        for (i = 0; i < numcep; i++)
        {
            // DCT - II of filter bank output
            mfcc_result[i] = 0.0f;
            int j = 0;
            for (j = 0; j < nfilt; j++)
                mfcc_result[i] += feat[j] * cos((i * PI / nfilt) * (j + 0.5f));
            // Orthogonalization of DCT output
            if (i == 0)
                mfcc_result[i] *= sqrt(1.0f / nfilt);
            else
                mfcc_result[i] *= sqrt(2.0f / nfilt);
            // Ceplifter
            if (ceplifter != 0)
                mfcc_result[i] *= 1.0f + (ceplifter / 2.0f) * sin(PI * i / ceplifter);
        }
        // Append Energy
        if (appendEnergy == 1)
            mfcc_result[0] = log(specenergy);
    }
    else {
        memcpy(mfcc_result, feat, sizeof(FPTYPE) * nfilt);
    }
    mufft_free(temp_out);
    mufft_free_plan_1d(muplan);
}

void get_filterbank_parameters(FPTYPE *fbank, int nfilt, int samplingRate, int nfft) {
    FPTYPE lowmel = hztomel(0.0f);
    FPTYPE highmel = hztomel(samplingRate / 2.0f);

    // Generate nfilt center frequencies linearly spaced in the mel scale
    FPTYPE bin[nfilt + 2];
    int i = 0;
    for (i = 0; i <= nfilt + 1; i++)
        bin[i] = floor(meltohz(i * (highmel - lowmel) / (nfilt + 1) + lowmel) * (nfft + 1) / samplingRate);

    memset(fbank, 0, (nfft / 2 + 1) * nfilt * sizeof(FPTYPE));
    // Triangular Filter Banks 
    for (i = 0; i < nfilt; i++) {
        int j = 0;
        for (j = (int)bin[i]; j < (int)bin[i + 1]; j++)
            fbank[i * (nfft / 2 + 1) + j] = (j - bin[i]) / (bin[i + 1] - bin[i]);
        for (j = (int)bin[i + 1]; j < (int)bin[i + 2]; j++)
            fbank[i * (nfft / 2 + 1) + j] = (bin[i + 2] - j) / (bin[i + 2] - bin[i + 1]);
    }
}

FPTYPE hztomel(FPTYPE hz)
{
    return 2595 * log10(1 + hz / 700.0f);
}

FPTYPE meltohz(FPTYPE mel)
{
    return 700 * (pow(10, mel / 2595.0f) - 1);
}

void windowing(FPTYPE *temp_in, int numentries, int window)
{
    // Apply respective window before FFT
    int i = 0;
    for (i = 0; i < numentries; i++)
    {
        if (window == 1)
            temp_in[i] *= (0.5f - 0.5f * cos(2 * PI * i / (numentries - 1)));
        else if (window == 2)
            temp_in[i] *= (0.54f - 0.46f * cos(2 * PI * i / (numentries - 1)));
    }
}