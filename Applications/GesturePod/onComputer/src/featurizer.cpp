/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT license.
 * 
 * Definition of featurizer class
 */

#include "featurizer.h"

Featurizer::Featurizer(
    int _bucketWidth, 
    FIFOCircularQ<float, 400> *_ax, 
    FIFOCircularQ<float, 400> *_ay,
    FIFOCircularQ<float, 400> *_az, 
    FIFOCircularQ<float, 400> *_gx,
    FIFOCircularQ<float, 400> *_gy, 
    FIFOCircularQ<float, 400> *_gz
    ){
    if(_bucketWidth != 20) 
        exit(-1);
    this->ax = _ax;
    this->ay = _ay;
    this->az = _az;
    this->gx = _gx;
    this->gy = _gy;
    this->gz = _gz;
}

int Featurizer::featurize(int bucketDistribution[]){
    //Ensure buckets are initialised to zero    
    bucketDistribution[0]=-1; 
    bucketDistribution[3]=-1;               
    /*
    * bucket[0]=longestPosIndex, init to -1
    * bucket[1]=longestPosCount, init to 0
    * bucket[2]=longestNegCount, init to 0
    * bucket[3]=longestNegIndex, init tp -1
    */
    // get features for acc and gyr across 3 axes
    bucketIndex=4;
    if((getBucket(this->ax, bucketDistribution))!=1) return 0;
    if((getBucket(this->ay, bucketDistribution))!=1) return 0;
    if((getBucket(this->az, bucketDistribution))!=1) return 0;
    if((getBucket(this->gx, bucketDistribution))!=1) return 0;
    if((getBucket(this->gy, bucketDistribution))!=1) return 0;
    if((getBucket(this->gz, bucketDistribution))!=1) return 0;
    return 1;
}

int Featurizer::getBucket(
    FIFOCircularQ<float, 400>*sensorValue1D,
    int bucketDistribution[]
    ){
    sensorValue1D->flatten(sensorValue1Dflat);
    int imax = 0, imin = 0, maxval = 0, minval = 0, 
        maxcount = 0, mincount = 0, postemp = 0, negtemp = 0, 
        bucketCount;
    bool flagPos = false, flagNeg = false;
    float j,k;
    for(int i =0; i < 400; i++){
        // This typecasting and all the further values are
        // scaled to optimize performance
        int val = (int)100 *sensorValue1Dflat[i];

        if(val < 0){
            bucketDistribution[bucketIndex]+=1;
        } else if(val > 100){
            bucketDistribution[bucketIndex+19]+=1;
        } else {
            int index=val/5;
            bucketDistribution[bucketIndex+index]+=1;
        } 

        // Pos/neg Index/Count
        if(sensorValue1D==this->gy){
            int thresholdCount=3;
            //PosEdgeValues
            if(val>62 && !flagPos){
                maxcount = 1;
                postemp = i;
                flagPos = true;
                flagNeg = false;
            } else if(val>62 && flagPos){
                flagNeg = 0;
                mincount = 0;
                maxcount++;
                if(maxcount>maxval){
                    maxval = maxcount;
                    if(maxval>thresholdCount) bucketDistribution[1] = maxval;
                    imax = postemp;
                    if(maxval>thresholdCount) bucketDistribution[0] = imax;
                }
            } else if(val<62 && flagPos && val >32){
                flagPos=false;
                maxcount = 0;
                mincount = 0;
            //NegEdgeBegins
            } else if(val<62 && flagPos && val <32){
                flagPos=false;
                maxcount=0;
                mincount = 1;
                negtemp = i;
                flagNeg = true;
            //NegEdgeValues         
            } else if(val<32 && !flagNeg){
                mincount =1;
                negtemp = i;
                flagNeg = true;
                flagPos = false;
            } else if(val<32 && flagNeg){
                mincount++;
                if(mincount>minval){
                    minval=mincount;
                    if(minval>thresholdCount) bucketDistribution[2] = minval;
                    imin = negtemp;
                    if(minval>thresholdCount) bucketDistribution[3] = imin;
                }
            } else if(val>32 && flagNeg && val<65){
                flagNeg=false;
                mincount = 0;
            } else if(val>32 && flagNeg && val>65){
                flagNeg=false;
                mincount = 0;
                maxcount = 1;
                postemp = i;
                flagPos = true;
            } else {
                flagPos = false;
                maxcount = 0;
                flagNeg = false;
                mincount = 0;
            }
        }//pos-neg edge count and index loop termination
    }//Iterating over windowwidth(400) elements loop termination

    bucketIndex+=bucketWidth;//Updating Bucket Index
    // Sanity checks
    if(bucketIndex>124 || bucketIndex<23) return 0;
    return 1;
}
