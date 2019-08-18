/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT license.
 * 
 * Declaration of Featurizer class for ML predictions
 */

#ifndef __Featurizer__
#define __Featurizer__

#include "config.h"
#include "utils.h"
#include <cstdlib>

class Featurizer {
    FIFOCircularQ<float, 400> *ax, *ay, *az, *gx, *gy, *gz;
    float sensorValue1Dflat[400];
    int bucketIndex;
    int bucketWidth=20; // Default value of number of buckets 
    int getBucket(
        FIFOCircularQ<float, 400>*, 
        int bucketDistribution[]);
    /*
     * Format of feature vector:[longestPosIndex,longestPosCount,
     * longestNegCount,longestNegIndex,ax[20buckets],
     * ay[20buckets],az[20buckets],
     * gx[20buckets],gy[20buckets],gz[20buckets]]
     */
public:
    Featurizer(
        int bucketWidth, 
        FIFOCircularQ<float, 400>*, 
        FIFOCircularQ<float, 400>*,
        FIFOCircularQ<float, 400>*, 
        FIFOCircularQ<float, 400>*,
        FIFOCircularQ<float, 400>*, 
        FIFOCircularQ<float, 400>*
        );
    int featurize(int bucketDistribution[]);
};

#endif //__Featurizer__