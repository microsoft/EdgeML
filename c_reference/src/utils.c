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

void offset_matVec_conv1d(const float* mat, const float* vec,
  unsigned nrows, unsigned ncols,
  unsigned row_stride, unsigned vec_stride,
  unsigned depthwise, float* ret) {

  while (nrows--) {
    // For depthwise, the vec(input) pointer is updated 
    // Since each row of the mat corresponds to a separate channel index
    float* vec_offset = depthwise ? (float*)vec++ : (float*)vec;
    float* mat_offset = (float*)mat;
    float sum = 0.0f;
    unsigned cols = ncols;
    
    #ifdef LOOP_UNROLL
      unsigned len_unroll = cols >> 2;
      cols %= 4; // ncols % 4
      while (len_unroll--) {
        sum += (*mat_offset++) * (*vec_offset);
        vec_offset += vec_stride;
        sum += (*mat_offset++) * (*vec_offset);
        vec_offset += vec_stride;
        sum += (*mat_offset++) * (*vec_offset);
        vec_offset += vec_stride;
        sum += (*mat_offset++) * (*vec_offset);
        vec_offset += vec_stride;
      }
    #endif
    
    while (cols--) {
      sum += (*mat_offset++) * (*vec_offset);
      vec_offset += vec_stride;
    }
    *ret++ = sum;
    mat += row_stride;
  }
}

void tiledMatMul_float(const float* const matA, const float* const matB,
  unsigned nrows, unsigned ncommon, unsigned ncols,
  unsigned total_comm_A, unsigned total_cols_B,
  float* const ret, unsigned block_size) {
  for (unsigned row = 0; row < nrows; row += block_size) {
    unsigned row_block_size = (row + block_size < nrows) ? block_size : nrows - row;
    for (unsigned col = 0; col < ncols; col += block_size) {
      unsigned col_block_size = (col + block_size < ncols) ? block_size : ncols - col;
      for (unsigned comm = 0; comm < ncommon; comm += block_size) {
        unsigned comm_block_size = (comm + block_size < ncommon) ? block_size : ncommon - comm;
        for (unsigned block_row = row; block_row < row + row_block_size; block_row++) {
          float *ret_offset = (float *)ret + block_row * ncols + col;
          for (unsigned block_col = col; block_col < col + col_block_size; block_col++) {
            float sum = 0;
            unsigned temp_block_size = comm_block_size;
            const float *matA_offset = (const float*)matA + block_row * total_comm_A + comm;
            const float *matB_offset = (const float*)matB + comm * total_cols_B + block_col;

            #ifdef LOOP_UNROLL
              unsigned len_unroll = temp_block_size >> 2;
              temp_block_size %= 4; // comm_block_size % 4
              while (len_unroll--) {
                sum += (*matA_offset++) * (*matB_offset);
                matB_offset += ncols;
                sum += (*matA_offset++) * (*matB_offset);
                matB_offset += ncols;
                sum += (*matA_offset++) * (*matB_offset);
                matB_offset += ncols;
                sum += (*matA_offset++) * (*matB_offset);
                matB_offset += ncols;
              }
            #endif

            while (temp_block_size--) {
              sum += (*matA_offset++) * (*matB_offset);
              matB_offset += ncols;
            }
            *ret_offset++ += sum;
          }
        } 
      }
    }
  }
}

void transposed_tiledMatMul(const float* const matA, const float* const matB,
  unsigned nrows, unsigned ncommon, unsigned ncols,
  unsigned total_comm_A, unsigned total_comm_B,
  float* const ret, unsigned block_size) {
  for (unsigned row = 0; row < nrows; row += block_size) {
    unsigned row_block_size = (row + block_size < nrows) ? block_size : nrows - row;
    for (unsigned col = 0; col < ncols; col += block_size) {
      unsigned col_block_size = (col + block_size < ncols) ? block_size : ncols - col;
      for (unsigned comm = 0; comm < ncommon; comm += block_size) {
        unsigned comm_block_size = (comm + block_size < ncommon) ? block_size : ncommon - comm;
        for (unsigned block_row = row; block_row < row + row_block_size; block_row++) {
          float *ret_offset = (float *)ret + block_row * ncols + col;
          for (unsigned block_col = col; block_col < col + col_block_size; block_col++) {
            float sum = 0;
            unsigned temp_block_size = comm_block_size;
            const float *matA_offset = (const float*)matA + block_row * total_comm_A + comm;
            const float *matB_offset = (const float*)matB + block_col * total_comm_B + comm;

            #ifdef LOOP_UNROLL
              unsigned len_unroll = temp_block_size >> 2;
              temp_block_size %= 4; // comm_block_size % 4
              while (len_unroll--) {
                sum += (*matA_offset++) * (*matB_offset++);
                sum += (*matA_offset++) * (*matB_offset++);
                sum += (*matA_offset++) * (*matB_offset++);
                sum += (*matA_offset++) * (*matB_offset++);
              }
            #endif

            while (temp_block_size--) {
              sum += (*matA_offset++) * (*matB_offset++);
            }
            *ret_offset++ += sum;
          }
        } 
      }
    }
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

void semi_sigmoid_tanh(float* output_signal, const float* const input_signal, 
  unsigned in_time, unsigned in_channels) {
  unsigned time_step = 0; // used to avoid index multiplication
  while (in_time--) {
    unsigned pivot = in_channels >> 1;
    float* input_sigmoid_offset = (float*)input_signal + time_step;
    float* input_tanh_offset = (float*)input_signal + time_step + pivot;
    
    #ifdef LOOP_UNROLL
      unsigned len_unroll = pivot >> 2;
      pivot %= 4;
      while (len_unroll--) {
        *output_signal++ = sigmoid(*input_sigmoid_offset++) *
                          tanh(*input_tanh_offset++);
        *output_signal++ = sigmoid(*input_sigmoid_offset++) *
                          tanh(*input_tanh_offset++);
        *output_signal++ = sigmoid(*input_sigmoid_offset++) *
                          tanh(*input_tanh_offset++);
        *output_signal++ = sigmoid(*input_sigmoid_offset++) *
                          tanh(*input_tanh_offset++);
      }
    #endif

    while (pivot--) {
      *output_signal++ = sigmoid(*input_sigmoid_offset++) *
                        tanh(*input_tanh_offset++);
    }
    time_step += in_channels;
  }
}
