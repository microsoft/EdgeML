/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT license.
 */

#include "protoNN.h"

int8_t ProtoNNF::getInitErrorCode(){
    this->errorCode = 0;
    int d = this->featDim;
    int d_cap = this->ldDim;
    int m = this->numPrototypes;
    int L = this->numLabels;
    float gamma = this->gamma;
    if ((d != pgm_read_word_near(&(protoNNParam::featDim))))
        this->errorCode |= 1;

    if (d_cap != pgm_read_word_near(&(protoNNParam::ldDim)))
        this->errorCode |= 2;

    if (m != pgm_read_word_near(&(protoNNParam::numPrototypes)))
        this->errorCode |= 4;

    if(L != pgm_read_word_near(&(protoNNParam::numLabels)))
        this->errorCode |= 8;

    if ((gamma - pgm_read_float_near(&(protoNNParam::gamma))) >= 0.001 ||
        (gamma - pgm_read_float_near(&(protoNNParam::gamma))) <= -0.001){
        this->errorCode |= 16;
    }
    return this->errorCode;
}

/**
 * Constructor initializes the ProtoNNF predictor with values
 * from the data.h file. Use ProtoNNF(unsigned int d, unsigned int d_cap, unsigned int m, unsigned int L)
 * whenever possible.
 *
 * @see ProtoNNF(unsigned int d, unsigned int d_cap, unsigned int m, unsigned int L)
 * @see getErrorCode()
 */
ProtoNNF::ProtoNNF() {
    this->featDim = pgm_read_word_near(&(protoNNParam::featDim));
    this->ldDim = pgm_read_word_near(&(protoNNParam::ldDim));
    this->numPrototypes = pgm_read_word_near(&(protoNNParam::numPrototypes));
    this->numLabels = pgm_read_word_near(&(protoNNParam::numLabels));
    this->gamma = pgm_read_float_near(&(protoNNParam::gamma));
    this->errorCode = getInitErrorCode();
}

/**
 * This constructor initializes the ProtoNNF predictory
 * with the given matrix dimensions. This is the safer
 * method as here we can check if the dimensions provided
 * and that in the data.h file are the same.
 * @param d         The dimension of the feature vector.
 * @param d_cap     The dimension of the lower dimension matrix onto
                    which projection happens. The projection matrix
                    W is of dimensions d x d_cap.
 * @param m         The number of prototypes.
 * @param L         The number of output labels.
 * @param gamma     Gamma for gaussian kernel.
 * @see getErrorCode()
 */
ProtoNNF::ProtoNNF(unsigned d, unsigned d_cap,
    unsigned m, unsigned L, float gamma){
    this->featDim = d;
    this->ldDim = d_cap;
    this->numPrototypes = m;
    this->numLabels = L;
    this->gamma = gamma;
    this->errorCode = getInitErrorCode();
}

/**
 * Project the feature vector x of dimension dx1 (length x 1)
 * onto low dimensional subspace using the projection matrix W.
 * The low dimensional vector is stored in x_cap.
 * That is, `W.x` is calculated for feature vector x and ld-projection
 * matrix W.
 * @param   x   A pointer to the vector x.
 * @param   x_cap A pointer to the location where the projected vector is to be saved.
 *                  Ensure that correct amount of memory is allocated.
 * @param   length  The length of vector x
 * @returns 0 
 */
int8_t ProtoNNF::denseLDProjection(float* x, float* x_cap){
    unsigned int d = this->featDim;
    unsigned int d_cap = this->ldDim;
    for (int i = 0; i < d_cap; i++){
        float dotProd = 0.0;
        for(int j = 0; j < d; j++){
            float component = getProjectionComponent(i, j);
            dotProd += x[j] * component;
        }
        x_cap[i] = dotProd;
    }
    return 0;
}

/**
 * Gaussian kernel for vectors (x and y)
 *
 * @param x Pointer to vector x.
 * @param y Pointer to vector y
 * @param length The length of the vectors.
 * @param gamma The gamma used in Gaussian kernel.
 * @returns Gaussian distance between vectors x and y.
 */
float ProtoNNF::gaussian(const float *x, const float *y,
    unsigned length, float gamma = 1.0) {
    float sumSq = 0.0;
    for(unsigned i = 0; i < length; i++){
        sumSq += (x[i] - y[i])*(x[i] - y[i]);
    }
    sumSq = -1*gamma*gamma*sumSq;
    sumSq = exp(sumSq);
    return sumSq;
}

/**
 * Helper method for scalar vector multiplication.
 * @param vec Pointer to vector (float).
 * @param length The length of vector.
 * @param scalar The (float) scalar to multiply the vector by.
 * @returns 0
 */
int8_t ProtoNNF::scalarVectorMul(float *vec, unsigned length,
    float scalar) {

    for(unsigned i = 0; i < length; i++) {
        vec[i] = vec[i] * scalar;
    }
    return 0;
}

/**
 * Helper method for scalar vector Addition
 * @param srcVec Pointer to (float)vector.
 * @param dstVec Pointer to (float)vector.
 * @param length The length of vector.
 * @returns 0. dstVec += srcVec .
 */
int8_t ProtoNNF::vectorVectorAdd(float *dstVec, float *srcVec,
    unsigned length){

    for(unsigned i = 0; i < length; i++)
        dstVec[i] += srcVec[i];
    return 0;
}

/**
 * Reads the prototype at `index` from the data.h file.
 *
 * **NOTE:** Prototypes Matrix B and Prototype Label matrix Z
 * are assumed to be stored in column major format.
 *
 * @param index The index of the prototype.
 * @param prototype A pointer to the destination where the prototype
 * will be stored. Should have `d_cap` memory allocated.
 *
 * @returns 0
 */
int8_t ProtoNNF::getPrototype(unsigned index, float *prototype){
    unsigned int d_cap = this->ldDim;
    for(unsigned i = 0; i < d_cap; i++){
        float component = pgm_read_float_near(&(protoNNParam::prototypeMatrix[index * d_cap + i]));
        prototype[i] = component;
    }
    return 0;
}

/**
 * Reads the label vector for the prototype at `index` in the prototype matrix
 * stored in data.h.
 *
 * @param index The index of the prototype
 * @param prototypeLabel The destination for the label vector
 *
 * @returns 0;
 */
int8_t ProtoNNF::getPrototypeLabel(unsigned index, float *prototypeLabel){
    unsigned int L = this->numLabels;
    for(unsigned i = 0; i < L; i++){
        float component = pgm_read_float_near(&(protoNNParam::prototypeLabelMatrix[index * L + i]));
        prototypeLabel[i] = component;
    }
    return 0;
}

/**
 * Iterating through the projection matrix W. W is `d_cap x d` dimensions.
 *
 * @param i Row index
 * @param j Column index
 * @returns W[i][j]
 */
float ProtoNNF::getProjectionComponent(unsigned i, unsigned j){
    unsigned int d = this->featDim;
    return pgm_read_float_near(&(protoNNParam::ldProjectionMatrix[i * d + j]));
}

/**
 * Method that returns the final prediction class.
 *
 * @param labelScores Pointer to the scores of each of the labels
 * obtained from prediction.
 * @param length The number of labels.
 * @param getScore Returns the score instead of the label if true.
 * @returns Label with max score or the max score.
 */
float ProtoNNF::rho(float* labelScores, unsigned length) {
    float maxScore = -FLT_MAX;
    float maxIndex = 0;
    for(int i = 0; i < length; i++){
        if (labelScores[i] > maxScore){
            maxIndex = i;
            maxScore = labelScores[i];
        }
    }
    return maxIndex;
}

/**
 * The method used to perform prediction. This depends on the
 * data.h file and assumes that the data.h file is correct.
 * @param x The feature vector.
 * @param length The length of the feature vector.
 * @param getScore If true, returns the score of the max label rather
 * than the label itself.
 * @returns The label, the score or -1. -1 if the length and expected
 feature vector length mismatch.
 */
float ProtoNNF::predict(float *x, unsigned length,
    int *scores) {

    unsigned m = this->numPrototypes;
    unsigned int d = this->featDim;
    unsigned int d_cap = this->ldDim;
    unsigned int L = this->numLabels;
    float gamma = this->gamma;
    if (length != d)
        return -1.0;
    float x_cap[d_cap];
    float y_cap[L];
    float prototype[d_cap];
    float prototypeLabel[L];
    float weight = 0.0;

    for(unsigned i = 0; i < L; i++){
        y_cap[i] = 0.0;
    }
    // Project x onto the d_cap dimension
    denseLDProjection(x, x_cap);
    for(unsigned i = 0; i < m; i++){
        // at this stage, we are holding a feature vector
        // its LD projection and a prototype in memory
        getPrototype(i, prototype);
        weight = gaussian(x_cap, prototype, d_cap, gamma);
        getPrototypeLabel(i, prototypeLabel);
        scalarVectorMul(prototypeLabel, L, weight);
        vectorVectorAdd(y_cap, prototypeLabel, L);
    }
    if (scores != nullptr)
        for(int i = 0; i < L; i++)
            scores[i] = (int)(100000 * y_cap[i]);
    return rho(y_cap, L);
}

/**
 * Returns the error code that was registered
 * when ProtoNN predictor was initialized. The
 * 8 bit error code has the following bits turned
 * on depending on the mismatch.
 *
 * * **Error codes:**
 * 
 *        |    3    |    2    |    1    |    0    |
 *        |    L    |    m    |  d_cap  |    d    |
 *        
 *        |    7    |    6    |    5    |    4    |
 *        |         |         |         |  gamma  |
 *   
 * @returns int8_t errorCode
 *
 */
int8_t ProtoNNF::getErrorCode(){
    return this->errorCode;
}