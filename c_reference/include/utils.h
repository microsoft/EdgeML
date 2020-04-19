// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#ifndef __UTILS_H__
#define __UTILS_H__

#include <math.h>
#include <float.h>

inline float min(const float a, const float b) {
  return (a < b) ? a : b;
}

inline float max(const float a, const float b) {
  return (a > b) ? a : b;
}

inline float relu(const float x) {
  if (x < 0.0) return 0.0;
  else return x;
}

inline float sigmoid(const float x) {
  return 1.0f / (1.0f + expf(-1.0f * x));
}

inline float tanhyperbolic(const float x) {
  float ex = expf(x);
  float enx = expf(-1.0f * x);
  return (ex - enx) / (ex + enx);
}

inline float quantTanh(const float x) {
  return max(min(x, 1.0f), -1.0f);
}

inline float quantSigmoid(const float x) {
  return max(min((x + 1.0f) / 2.0f, 1.0f), 0.0f);
}

void v_relu(const float* const vec, const unsigned len, float* const ret) {
  for (unsigned i = 0; i < len; i++) ret[i] = relu(vec[i]);
}

void v_sigmoid(const float* const vec, const unsigned len, float* const ret) {
  for (unsigned i = 0; i < len; i++) ret[i] = sigmoid(vec[i]);
}

void v_tanh(const float* const vec, const unsigned len, float* const ret) {
  for (unsigned i = 0; i < len; i++) ret[i] = tanh(vec[i]);
}

void v_quantSigmoid(const float* const vec, const unsigned len, float* const ret) {
  for (unsigned i = 0; i < len; i++) ret[i] = quantSigmoid(vec[i]);
}

void v_quantTanh(const float* const vec, const unsigned len, float* const ret) {
  for (unsigned i = 0; i < len; i++) ret[i] = quantTanh(vec[i]);
}

/* Scaled matrix-vector multiplication:  ret = alpha * ret + beta * mat * vec
   alpha and beta are scalars
   ret is of size nrows, vec is of size ncols
   mat is of size nrows * ncols, stored in row major */
void matVec(const float* const mat, const float* const vec,
  const unsigned nrows, const unsigned ncols,
  const float alpha, const float beta,
  float* const ret) {

  for (unsigned row = 0; row < nrows; row++) {
    float sum = 0.0f;
    float* mat_offset = (float*)mat + row * ncols;
    for (unsigned col = 0; col < ncols; col++) {
      sum += *mat_offset++ * vec[col];
    }
    ret[row] = alpha * ret[row] + beta * sum;
  }
}

// scaled vector addition: ret = scalar1 * vec1 + scalar2 * vector2
void v_add(const float scalar1, const float* const vec1,
  const float scalar2, const float* const vec2,
  const unsigned len, float* const ret) {
  for (unsigned i = 0; i < len; i++)
    ret[i] = scalar1 * vec1[i] + scalar2 * vec2[i];
}

// point-wise vector division ret = vec2 / vec1 
void v_mult(const float* const vec1, const float* const vec2,
  const unsigned len, float* const ret) {
  for (unsigned i = 0; i < len; i++)
    ret[i] = vec1[i] * vec2[i];
}

// point-wise vector division ret = vec2 / vec1 
void v_div(const float* const vec1, const float* const vec2,
  const unsigned len, float* const ret) {
  for (unsigned i = 0; i < len; i++)
    ret[i] = vec2[i] / vec1[i];
}

// Return index with max value, if tied, return first tied index.
unsigned argmax(const float* const vec, const unsigned len) {
  unsigned maxId = 0;
  float maxScore = FLT_MIN;
  for (unsigned i = 0; i < len; i++) {
    if (vec[i] > maxScore) {
      maxScore = vec[i];
      maxId = i;
    }
  }
  return maxId;
}

// ret[i] = exp(input[i]) / \sum_i exp(input[i])
void softmax(const float* const input, const unsigned len, float* const ret) {
  float m = input[argmax(input, len)];
  float sum = 0.0f;
  for (unsigned i = 0; i < len; i++)
    sum += expf(input[i] - m);

  float offset = m + logf(sum);
  for (unsigned i = 0; i < len; i++)
    ret[i] = expf(input[i] - offset);
}

#endif
