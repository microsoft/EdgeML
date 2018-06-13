// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#ifndef MUFFT_DEBUG
#define MUFFT_DEBUG
#endif

#include "mfcc.h"

void rolling_prediction(FPTYPE *input_data, FPTYPE *mfcc_result, int numframes, int N)
{
	// output has to be number of numframes * NFILT 
	// For first frame, the startIndex and end Index are as follows 

    int endIndex;
    int startIndex;
    int currentFrameID = 0;
    int i = 0;

    FPTYPE *local_data = (FPTYPE *)malloc(sizeof(FPTYPE) * WINLEN);

	while (currentFrameID < numframes)
	{
        // This is important, because last frame will have less than WINLEN
        memset(local_data, 0, sizeof(FPTYPE) * WINLEN);

        startIndex = currentFrameID * STRIDE;
        endIndex = (currentFrameID != (numframes - 1)) 
                    ? startIndex + WINLEN - 1 
                    : startIndex + N - currentFrameID * STRIDE - 1;

        for (i = endIndex; i >= startIndex + 1; i--)
            local_data[i - startIndex] = 1.0f * input_data[i] - input_data[i - 1] * PREEMPH;

        local_data[0] = (currentFrameID == 0) 
                    ? 1.0f * input_data[startIndex] 
                    : 1.0f * input_data[startIndex] - input_data[startIndex - 1] * PREEMPH;

        mfcc(&mfcc_result[currentFrameID * NFILT], local_data, SAMPLING_RATE, NFILT, 0, NFFT, 0, 0, HAMMING, WINLEN, NODCT);
        currentFrameID += 1;
	}
    free(local_data);
	
}

int main(void)
{   
    // Beware to change the winlen in python_speech_features when you change this
    unsigned N = 26032;

    // numframes is not calculated by simple division
    int numframes;
    if (N <= WINLEN)
        numframes = 1;
    else
        numframes = 1 + (int)ceil((1.0 * N - WINLEN) / STRIDE);

    // For input and output
    float *input = mufft_alloc(N * sizeof(float));
    FPTYPE *mfcc_result = (FPTYPE *)malloc(sizeof(FPTYPE) * numframes * NFILT);

    // Creating data
    unsigned i = 0;
    for (i = 0; i < N; i++)
    {
        float real = (float)i;
        input[i] = real;
    }

    rolling_prediction(input, mfcc_result, numframes, N);
    for(i = 0; i < numframes; i++) {
        int j =0;
        for(j = 0; j < NFILT; j++) {
            // printf("%d %d %f ", i, j, mfcc_result[i*NFILT+j]);
        }
        // printf("\n");
    }
}

