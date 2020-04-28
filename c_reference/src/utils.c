// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <math.h>
#include <float.h>
#include "utils.h"

float min(float a, float b) {
  return (a < b) ? a : b;
}

float max(float a, float b) {
  return (a > b) ? a : b;
}

float relu(float x) {
  if (x < 0.0) return 0.0;
  else return x;
}

float sigmoid(float x) {
  return 1.0f / (1.0f + expf(-1.0f * x));
}

float tanhyperbolic(float x) {
  float ex = expf(x);
  float enx = expf(-1.0f * x);
  return (ex - enx) / (ex + enx);
}

float quantTanh(float x) {
  return max(min(x, 1.0f), -1.0f);
}

float quantSigmoid(float x) {
  return max(min((x + 1.0f) / 2.0f, 1.0f), 0.0f);
}

void v_relu(const float* const vec, unsigned len, float* const ret) {
  for (unsigned i = 0; i < len; i++) ret[i] = relu(vec[i]);
}

void v_sigmoid(const float* const vec, unsigned len, float* const ret) {
  for (unsigned i = 0; i < len; i++) ret[i] = sigmoid(vec[i]);
}

void v_tanh(const float* const vec, unsigned len, float* const ret) {
  for (unsigned i = 0; i < len; i++) ret[i] = tanhyperbolic(vec[i]);
}

void v_quantSigmoid(const float* const vec, unsigned len, float* const ret) {
  for (unsigned i = 0; i < len; i++) ret[i] = sigmoid(vec[i]);
}

void v_quantTanh(const float* const vec, unsigned len, float* const ret) {
  for (unsigned i = 0; i < len; i++) ret[i] = tanh(vec[i]);
}

void matVec(const float* const mat, const float* const vec,
  unsigned nrows, unsigned ncols,
  float alpha, float beta,
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

void v_add(float scalar1, const float* const vec1,
  float scalar2, const float* const vec2,
  unsigned len, float* const ret) {
  for (unsigned i = 0; i < len; i++)
    ret[i] = scalar1 * vec1[i] + scalar2 * vec2[i];
}

void v_mult(const float* const vec1, const float* const vec2,
  unsigned len, float* const ret) {
  for (unsigned i = 0; i < len; i++)
    ret[i] = vec1[i] * vec2[i];
}

void v_div(const float* const vec1, const float* const vec2,
  unsigned len, float* const ret) {
  for (unsigned i = 0; i < len; i++)
    ret[i] = vec2[i] / vec1[i];
}

float l2squared(const float* const vec1,
  const float* const vec2, unsigned dim) {
  float sum = 0.0f;
  for (unsigned i = 0; i < dim; i++)
    sum += (vec1[i] - vec2[i]) * (vec1[i] - vec2[i]);
  return sum;
}

unsigned argmax(const float* const vec, unsigned len) {
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

void softmax(const float* const input, unsigned len, float* const ret) {
  float m = input[argmax(input, len)];
  float sum = 0.0f;
  for (unsigned i = 0; i < len; i++)
    sum += expf(input[i] - m);

  float offset = m + logf(sum);
  for (unsigned i = 0; i < len; i++)
    ret[i] = expf(input[i] - offset);
}
