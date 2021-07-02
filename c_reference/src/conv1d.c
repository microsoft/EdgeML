// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "conv1d.h"
#include "utils.h"

int conv1d_lr(float* output_signal, unsigned out_time, unsigned out_channels,
  const float* input_signal, unsigned in_time, unsigned in_channels,
  unsigned padding, unsigned kernel_size,
  const void* params, unsigned stride, unsigned activation) {

  const ConvLayers_LR_Params* tparams= (ConvLayers_LR_Params*) params;
  
  // Perform the convolution. Zero-pad is from 0 to padding and in_time + padding to in_time + 2 * padding
  unsigned rank = tparams->rank;
  // Buffer for W2 out
  float* temp_rank_out = (float*)malloc(rank * sizeof(float));
  // Buffer for W1 out
  float* temp_out = (float*)malloc(out_channels * sizeof(float));
  for (unsigned t_in_start = 0, t_in_end = kernel_size - 1, t_out = 0; 
        t_out < out_time; t_out++, t_in_start += stride, t_in_end += stride) {
    unsigned t_index = t_out * out_channels;

    if ((t_in_start >= padding) && (t_in_end < (in_time + padding))) {
      // Filter fully inside the input. Kept as the initial condition, since this is the most common one
      offset_matVec_conv1d(tparams->W2,
        input_signal + (t_in_start - padding) * in_channels,
        rank, kernel_size * in_channels,
        kernel_size * in_channels, 1, 0, temp_rank_out);
      // row_stride = ncols, vec_stride = 1, depthwise = 0. Hence the call is identical to a regular MatVec (without scaling)
      offset_matVec_conv1d(tparams->W1, temp_rank_out, out_channels,
        rank, rank, 1, 0, temp_out);
      memcpy(output_signal + t_index, temp_out, out_channels * sizeof(float));
    } 
    else if ((t_in_start < padding) && (t_in_end >= padding)) {
      // Filter partially entered the input
      // As a part of the filter is outside the input, we need less than "kernel_size" time-steps. 
      // We will only be using a part of the weight matrix(assuming shape = out_channels, kernel_size * in_channels)
      // Hence we provide a separate row_stride paramemter to discard/skip certain columns in the weight matrix
      offset_matVec_conv1d(tparams->W2 + (padding - t_in_start) * in_channels, 
        input_signal, rank,
        (t_in_end - padding + 1) * in_channels, 
        kernel_size * in_channels, 1, 0, temp_rank_out);
      // row_stride = ncols, vec_stride = 1, depthwise = 0. Hence the call is identical to a regular MatVec (without scaling)
      offset_matVec_conv1d(tparams->W1, temp_rank_out, out_channels,
        rank, rank, 1, 0, temp_out); 
      memcpy(output_signal + t_index, temp_out, out_channels * sizeof(float));
    }
    else if (t_in_start < (in_time + padding) && (t_in_end >= (in_time + padding))) {
      // Filter partially exited the input
      // As a part of the filter is outside the input, we need less than "kernel_size" time-steps. 
      // We will only be using a part of the weight matrix(assuming shape = out_channels, kernel_size * in_channels)
      // Hence we provide a separate row_stride paramemter to discard/skip certain columns in the weight matrix
      offset_matVec_conv1d(tparams->W2,
        input_signal  + (t_in_start - padding) * in_channels, 
        rank, (in_time + padding - t_in_start) * in_channels, 
        kernel_size * in_channels, 1, 0, temp_rank_out);
      // row_stride = ncols, vec_stride = 1, depthwise = 0. Hence the call is identical to a regular MatVec (without scaling)
      offset_matVec_conv1d(tparams->W1, temp_rank_out, out_channels,
        rank, rank, 1, 0, temp_out);
      memcpy(output_signal + t_index, temp_out, out_channels * sizeof(float));
    }
    else {
      // Filter completely in the padding region
      // The filter is either fully outside the input or has not yet entered the input
      memset(output_signal + t_index, 0, out_channels * sizeof(float));
    }
    for (unsigned co = 0; co < out_channels; co++) {
      // Post-Conv activation. More activation functions can be added should the necessity arise
      switch (activation) {
        case 1 :
          output_signal[t_index + co] = sigmoid(output_signal[t_index + co] +
                                          tparams->B[co]);
          break;

        case 2 :
          output_signal[t_index + co] = tanh(output_signal[t_index + co] +
                                          tparams->B[co]);
          break;

        case 3 :
          output_signal[t_index + co] = relu(output_signal[t_index + co] +
                                          tparams->B[co]);
          break;
        
        default :
          output_signal[t_index + co] += tparams->B[co];
          break;
      }
    }
  }
  free(temp_out);
  free(temp_rank_out);
  return 0;
}

int conv1d_lr_parallel(float* output_signal, unsigned out_time, unsigned out_channels,
  const float* input_signal, unsigned in_time, unsigned in_channels,
  unsigned padding, unsigned kernel_size,
  const void* params, unsigned stride, unsigned activation) {

  unsigned ncols = kernel_size * in_channels, num_iter = 0, num_steps_one_row = 0;
  // Calculate the number of time steps in one row for the first non-overlapping instance
  while (num_steps_one_row < kernel_size) {
    num_steps_one_row += stride;
    num_iter++;
  }
  unsigned total_in_cols = num_steps_one_row * in_channels;
  
  const ConvLayers_LR_Parallel_Params* tparams = (ConvLayers_LR_Parallel_Params*) params;
  // Perform the convolution. Zero-pad is from 0 to padding and in_time + padding to in_time + 2 * padding
  // Buffer to hold the output. For corner cases, this will be realtively big. 
  // But will be needed for the central condition (filter inside input).
  // If there are not enough time steps to linearise into one row, then allocate only 1 time step
  unsigned buffer_steps = ((in_time / num_steps_one_row) > 1) ? 
                            in_time / num_steps_one_row : 1;
  unsigned rank = tparams->rank;
  // Buffer for W2 out
  float* temp_rank_out = (float*)malloc(buffer_steps * rank * sizeof(float));
  // Buffer for W1 out
  float* temp_out = (float*)malloc(buffer_steps * out_channels * sizeof(float));

  unsigned t_in_start, t_in_end, t_out; // Values are needed outside the loops. Hence declared here
  for (t_in_start = 0, t_in_end = kernel_size - 1, t_out = 0;
        t_in_start < padding && t_out < out_time;
        t_out++, t_in_start += stride, t_in_end += stride) {
    if (t_in_end < padding) {
      // Filter outside the input region and in the padded region
      memset(output_signal + t_out * out_channels, 0, 
        out_channels * sizeof(float));
    } 
    else { //(t_in_end >= padding)
      // Filter partially entered the input
      // As a part of the filter is outside the input, we need less than "kernel_size" time-steps. 
      // We will only be using a part of the weight matrix(assuming shape = out_channels, kernel_size * in_channels)
      // Hence we provide a separate row_stride paramemter to discard/skip certain columns in the weight matrix
      offset_matVec_conv1d(tparams->W2 + (padding - t_in_start) * in_channels, 
        input_signal, rank, (t_in_end - padding + 1) * in_channels, 
        kernel_size * in_channels, 1, 0, temp_rank_out);
      // row_stride = ncols, vec_stride = 1, depthwise = 0. Hence the call is identical to a regular MatVec (without scaling)
      offset_matVec_conv1d(tparams->W1, temp_rank_out, out_channels,
        rank, rank, 1, 0, temp_out); 
      memcpy(output_signal + t_out * out_channels, 
        temp_out, out_channels * sizeof(float));
    }
  }
  // The main part => the filter is fully inside the input. We can think of the non-overlapping cases as parallel cases
  // Each of the iterations are for the kernel striding to the next point till the filter is out of the overlapping region 
  // Hence we use the num_steps_one_row for calculating the number of time steps to be linearized in one row
  // Using the above logic, we can convert the MatVec opeartion into a MatMul operation
  // Ideally both implementation would be the same. However for edge devices the matMul was found to be faster matVec (both tilied)
  // Skip if atleast 2 rows cannot be formed. The condition 2 * num_steps_one_row + stride is the worst case criteria
  // The MatVec will be used for the computation in-case the following block is skipped
  if (in_time > ((num_steps_one_row << 1) + stride)) {
    t_in_start -= padding; // remove the padding offset temporarily
    t_in_end -= padding; // Used to keep track of the final processed index
    for (unsigned iter = 0; (iter < num_iter) && (t_out < out_channels);
          iter++, t_in_start += stride, t_out++) {
      unsigned in_rows = (in_time - t_in_start) / num_steps_one_row;
      memset(temp_rank_out, 0, buffer_steps * rank * sizeof(float));
      memset(temp_out, 0, buffer_steps * out_channels * sizeof(float));
      if (t_in_end < (t_in_start + ((in_rows - 1) * num_steps_one_row))) {
        // t_in_end is used to find the furthest time step was used in the MatMul calculation
        // This value will be used for calculating the index for the final section of the processing
        t_in_end = ((in_rows - 1) * num_steps_one_row) + t_in_start + stride;
      }
      transposed_tiledMatMul(input_signal  + t_in_start * in_channels , tparams->W2,  
        in_rows, ncols, rank, total_in_cols, ncols,
        temp_rank_out, tparams->block_size_to_lr);
      transposed_tiledMatMul(temp_rank_out , tparams->W1,  
        in_rows, rank, out_channels, rank, rank,
        temp_out, tparams->block_size_from_lr);
      // Copy all the data into the output
      float* output_offset = (float*)output_signal + t_out * out_channels;
      float* temp_offset = (float*)temp_out;
      unsigned t_iter = in_rows, offset_factor_for_out =  num_iter * out_channels;
      while (t_iter--) {
        memcpy(output_offset, temp_offset, out_channels * sizeof(float));
        output_offset += offset_factor_for_out;
        temp_offset += out_channels;
      }
    }
    // Initialize the time iterators
    // Use the stored value in t_in_end to calculate the iterators
    t_in_start = t_in_end + padding; // Add the padding and stride offsets again
    t_in_end = t_in_start + kernel_size - 1;
    t_out = t_in_start / stride;
  }
  for (; t_out < out_time; t_out++, t_in_start += stride, t_in_end += stride) {
    if (t_in_start < (in_time + padding) && (t_in_end < (in_time + padding))) {
      // Filter fully in the input but very close to the edges. 
      // Due to the num_steps_one_row divisibility usage in the parallel step, some computations would be skipped
      // Incase the MatMul is skipped, this block will be used to compute the results
      offset_matVec_conv1d(tparams->W2,
        input_signal + (t_in_start - padding) * in_channels,
        rank, kernel_size * in_channels,
        kernel_size * in_channels, 1, 0, temp_rank_out);
      // row_stride = ncols, vec_stride = 1, depthwise = 0. Hence the call is identical to a regular MatVec (without scaling)
      offset_matVec_conv1d(tparams->W1, temp_rank_out, out_channels,
        rank, rank, 1, 0, temp_out);
      memcpy(output_signal + t_out * out_channels,
        temp_out, out_channels * sizeof(float));
    }
    else if (t_in_start < (in_time + padding) && (t_in_end >= (in_time + padding))) {
      // Filter partially exited the input
      // As a part of the filter is outside the input, we need less than "kernel_size" time-steps. 
      // We will only be using a part of the weight matrix(assuming shape = out_channels, kernel_size * in_channels)
      // Hence we provide a separate row_stride paramemter to discard/skip certain columns in the weight matrix
      offset_matVec_conv1d(tparams->W2,
        input_signal  + (t_in_start - padding) * in_channels, 
        rank, (in_time + padding - t_in_start) * in_channels, 
        kernel_size * in_channels, 1, 0, temp_rank_out);
      // row_stride = ncols, vec_stride = 1, depthwise = 0. Hence the call is identical to a regular MatVec (without scaling)
      offset_matVec_conv1d(tparams->W1, temp_rank_out, out_channels,
        rank, rank, 1, 0, temp_out);
      memcpy(output_signal + t_out * out_channels,
        temp_out, out_channels * sizeof(float));
    }
    else {
      // Filter completely outside the input and in the padding region
      memset(output_signal + t_out * out_channels,
        0, out_channels * sizeof(float));
    }
  }
  // Bias and activation
  for (t_out = 0; t_out < out_time; t_out++) {
    unsigned t_index = t_out * out_channels;
    for (unsigned co = 0; co < out_channels; co++) {
      // Post-Conv activation. More activation functions can be added should the necessity arise
      switch (activation) {
        case 1 :
          output_signal[t_index + co] = sigmoid(output_signal[t_index + co] +
                                          tparams->B[co]);
          break;

        case 2 :
          output_signal[t_index + co] = tanh(output_signal[t_index + co] +
                                          tparams->B[co]);
          break;

        case 3 :
          output_signal[t_index + co] = relu(output_signal[t_index + co] +
                                          tparams->B[co]);
          break;
        
        default :
          output_signal[t_index + co] += tparams->B[co];
          break;
      }
    }
  }
  free(temp_out);
  return 0;
}

int conv1d(float* output_signal, unsigned out_time, unsigned out_channels,
  const float* input_signal, unsigned in_time, unsigned in_channels,
  unsigned padding, unsigned kernel_size,
  const void* params, unsigned stride, unsigned activation) {

  const ConvLayers_Params* tparams= (ConvLayers_Params*) params;
  unsigned vec_stride = 1, cols_scale = in_channels;
  if (tparams->depthwise) {
    vec_stride = in_channels;
    out_channels = in_channels;
    cols_scale = 1;
  }

  // Perform the Convolution. Pad is from 0 to padding and in_time + padding to in_time + 2 * padding
  float* temp_out = (float*)malloc(out_channels * sizeof(float));
  for (unsigned t_in_start = 0, t_in_end = kernel_size - 1, t_out = 0; 
        t_out < out_time; t_out++, t_in_start += stride, t_in_end += stride) {
    unsigned t_index = t_out * out_channels;

    if ((t_in_start >= padding) && (t_in_end < (in_time + padding))) {
      // Filter fully inside the input. Kept as the initial condition, since this is the most common one
      offset_matVec_conv1d(tparams->W,
        input_signal + (t_in_start - padding) * in_channels,
          out_channels, kernel_size * cols_scale,
          kernel_size * cols_scale, vec_stride, tparams->depthwise, temp_out);
      memcpy(output_signal + t_index, temp_out, out_channels * sizeof(float));
    } 
    else if ((t_in_start < padding) && (t_in_end >= padding)) {
      // Filter partially entered the input
      // As a part of the filter is outside the input, we need less than "kernel_size" time-steps. 
      // We will only be using a part of the weight matrix(assuming shape = out_channels, kernel_size * in_channels)
      // Hence we provide a separate row_stride paramemter to discard/skip certain columns in the weight matrix
      offset_matVec_conv1d(tparams->W + (padding - t_in_start) * cols_scale, 
        input_signal, out_channels, (t_in_end - padding + 1) * cols_scale,
        kernel_size * cols_scale, vec_stride, tparams->depthwise, temp_out);
      memcpy(output_signal + t_index, temp_out, out_channels * sizeof(float));
    }
    else if (t_in_start < (in_time + padding) && (t_in_end >= (in_time + padding))) {
      // Filter partially exited the input
      // As a part of the filter is outside the input, we need less than "kernel_size" time-steps. 
      // We will only be using a part of the weight matrix(assuming shape = out_channels, kernel_size * in_channels)
      // Hence we provide a separate row_stride paramemter to discard/skip certain columns in the weight matrix
      offset_matVec_conv1d(tparams->W,
        input_signal  + (t_in_start - padding) * in_channels, 
        out_channels, (in_time + padding - t_in_start) * cols_scale,
        kernel_size * cols_scale, vec_stride, tparams->depthwise, temp_out);
      memcpy(output_signal + t_index, temp_out, out_channels * sizeof(float));
    }
    else {
      // Filter completely in the padding region
      // The filter is either fully outside the input or has not yet entered the input
      memset(output_signal + t_index, 0, out_channels * sizeof(float));
    }
    for (unsigned co = 0; co < out_channels; co++) {
      // Post-Conv activation. More activation functions can be added should the necessity arise
      switch (activation) {
        case 1 :
          output_signal[t_index + co] = sigmoid(output_signal[t_index + co] +
                                          tparams->B[co]);
          break;

        case 2 :
          output_signal[t_index + co] = tanh(output_signal[t_index + co] +
                                          tparams->B[co]);
          break;

        case 3 :
          output_signal[t_index + co] = relu(output_signal[t_index + co] +
                                          tparams->B[co]);
          break;
        
        default :
          output_signal[t_index + co] += tparams->B[co];
          break;
      }
    }
  }
  free(temp_out);
  return 0;
}

int conv1d_parallel(float* output_signal, unsigned out_time, unsigned out_channels,
  const float* input_signal, unsigned in_time, unsigned in_channels,
  unsigned padding, unsigned kernel_size,
  const void* params, unsigned stride, unsigned activation) {
  
  unsigned ncols = kernel_size * in_channels, num_iter = 0, num_steps_one_row = 0;
  // Calculate the number of time steps in one row for the first non-overlapping instance
  while (num_steps_one_row < kernel_size) {
    num_steps_one_row += stride;
    num_iter++;
  }
  unsigned total_in_cols = num_steps_one_row * in_channels;

  const ConvLayers_Parallel_Params* tparams = (ConvLayers_Parallel_Params*) params;
  // Perform the Convolution. Pad is from 0 to padding and in_time + padding to in_time + 2 * padding
  // Buffer to hold the output. For corner cases, this will be realtively big. 
  // But will be needed for the central condition (filter inside input).
  // If there are not enough time steps to linearise into one row, then allocate only 1 time step
  unsigned buffer_steps = ((in_time / num_steps_one_row) > 1) ? 
                            in_time / num_steps_one_row : 1;
  float* temp_out = (float*)malloc(buffer_steps * out_channels * sizeof(float));
  unsigned t_in_start, t_in_end, t_out; // Values are needed outside the loops. Hence declared here
  for (t_in_start = 0, t_in_end = kernel_size - 1, t_out = 0; 
        t_in_start < padding && t_out < out_time;
        t_out++, t_in_start += stride, t_in_end += stride) {
    if (t_in_end < padding) {
      // Filter outside the input region and in the padded region
      memset(output_signal + t_out * out_channels,
        0, out_channels * sizeof(float));
    } 
    else { //(t_in_end >= padding)
      // Filter partially entered the input
      // As a part of the filter is outside the input, we need less than "kernel_size" time-steps. 
      // We will only be using a part of the weight matrix(assuming shape = out_channels, kernel_size * in_channels)
      // Hence we provide a separate row_stride paramemter to discard/skip certain columns in the weight matrix
      offset_matVec_conv1d(tparams->W + (padding - t_in_start) * in_channels,
        input_signal, out_channels, (t_in_end - padding + 1) * in_channels,
        ncols, 1, 0, temp_out);
      memcpy(output_signal + t_out * out_channels,
        temp_out, out_channels * sizeof(float));
    }
  }
  // The main part => the filter is fully inside the input. We can think of the non-overlapping cases as parallel cases
  // Each of the iterations are for the kernel striding to the next point till the filter is out of the overlapping region 
  // Hence we use the num_steps_one_row for calculating the number of time steps to be linearized in one row
  // Using the above logic, we can convert the MatVec opeartion into a MatMul operation
  // Ideally both implementation would be the same. However for edge devices the matMul was found to be faster matVec (both tilied)
  // Skip if atleast 2 rows cannot be formed. The condition 2 * num_steps_one_row + stride is the worst case criteria
  // The MatVec will be used for the computation in-case the following block is skipped
  if (in_time > ((num_steps_one_row << 1) + stride)) {
    t_in_start -= padding; // remove the padding offset temporarily
    t_in_end -= padding; // Used to keep track of the final processed index
    for (unsigned iter = 0; (iter < num_iter) && (t_out < out_channels);
          iter++, t_in_start += stride, t_out++) {
      unsigned in_rows = (in_time - t_in_start) / num_steps_one_row;
      memset(temp_out, 0, buffer_steps * out_channels * sizeof(float));
      if (t_in_end < (t_in_start + ((in_rows - 1) * num_steps_one_row))) {
        // t_in_end is used to find the furthest time step was used in the MatMul calculation
        // This value will be used for calculating the index for the final section of the processing
        t_in_end = ((in_rows - 1) * num_steps_one_row) + t_in_start + stride;
      }
      transposed_tiledMatMul(input_signal  + t_in_start * in_channels , tparams->W,  
        in_rows, ncols, out_channels, total_in_cols, ncols,
        temp_out, tparams->block_size);
      // Copy all the data into the output
      float* output_offset = (float*)output_signal + t_out * out_channels;
      float* temp_offset = (float*)temp_out;
      unsigned t_iter = in_rows, offset_factor_for_out =  num_iter * out_channels;
      while (t_iter--) {
        memcpy(output_offset, temp_offset, out_channels * sizeof(float));
        output_offset += offset_factor_for_out;
        temp_offset += out_channels;
      }
    }
    // Initialize the time iterators
    // Use the stored value in t_in_end to calculate the iterators
    t_in_start = t_in_end + padding; // Add the padding and stride offsets again
    t_in_end = t_in_start + kernel_size - 1;
    t_out = t_in_start / stride;
  }
  for (; t_out < out_time; t_out++, t_in_start += stride, t_in_end += stride) {
    if (t_in_start < (in_time + padding) && (t_in_end < (in_time + padding))) {
      // Filter fully in the input but very close to the edges. 
      // Due to the num_steps_one_row divisibility usage in the parallel step, some computations would be skipped
      // Incase the MatMul is skipped, this block will be used to compute the results
      offset_matVec_conv1d(tparams->W,
        input_signal + (t_in_start - padding) * in_channels,
        out_channels, kernel_size * in_channels,
        kernel_size * in_channels, 1, 0, temp_out);
      memcpy(output_signal + t_out * out_channels,
        temp_out, out_channels * sizeof(float));
    }
    else if (t_in_start < (in_time + padding) && (t_in_end >= (in_time + padding))) {
      // Filter partially exited the input
      // As a part of the filter is outside the input, we need less than "kernel_size" time-steps. 
      // We will only be using a part of the weight matrix(assuming shape = out_channels, kernel_size * in_channels)
      // Hence we provide a separate row_stride paramemter to discard/skip certain columns in the weight matrix
      offset_matVec_conv1d(tparams->W,
        input_signal  + (t_in_start - padding) * in_channels, 
        out_channels, (in_time + padding - t_in_start) * in_channels, 
        ncols, 1, 0, temp_out);
      memcpy(output_signal + t_out * out_channels,
        temp_out, out_channels * sizeof(float));
    }
    else {
      // Filter completely outside the input and in the padding region
      memset(output_signal + t_out * out_channels,
        0, out_channels * sizeof(float));
    }
  }
  // Bias and activation
  for (t_out = 0; t_out < out_time; t_out++) {
    unsigned t_index = t_out * out_channels;
    for (unsigned co = 0; co < out_channels; co++) {
      // Post-Conv activation. More activation functions can be added should the necessity arise
      switch (activation) {
        case 1 :
          output_signal[t_index + co] = sigmoid(output_signal[t_index + co] +
                                          tparams->B[co]);
          break;

        case 2 :
          output_signal[t_index + co] = tanh(output_signal[t_index + co] +
                                          tparams->B[co]);
          break;

        case 3 :
          output_signal[t_index + co] = relu(output_signal[t_index + co] +
                                          tparams->B[co]);
          break;
        
        default :
          output_signal[t_index + co] += tparams->B[co];
          break;
      }
    }
  }
  free(temp_out);
  return 0;
}

int avgpool1d(float* output_signal, unsigned out_time, const float* input_signal,
  unsigned in_time, unsigned in_channels,
  unsigned padding, unsigned kernel_size, unsigned stride, unsigned activation) {

  // Iterate over the time steps and average them
  float scale = 1.0/(float)kernel_size; // To avoid divisions
  for (unsigned t_in = 0, t_out = 0; t_out < out_time; t_out++, t_in += stride) {
    for (unsigned ci = 0; ci < in_channels; ci++) {
      float sum = 0;
      for (unsigned tf = 0; tf < kernel_size; tf++) {
        if (((t_in + tf) < padding) || ((t_in + tf) >= (in_time + padding))) {
          continue;
        }
        else {
          sum += (input_signal[((tf + t_in) - padding) * in_channels + ci]);
        }
      }
      switch (activation) {
        case 1 :
          output_signal[t_out * in_channels + ci] = sigmoid(sum * scale);
          break;

        case 2 :
          output_signal[t_out * in_channels + ci] = tanh(sum * scale);
          break;

        case 3 :
          output_signal[t_out * in_channels + ci] = relu(sum * scale);
          break;

        default :
          output_signal[t_out * in_channels + ci] = sum * scale;
          break;
      }
    }
  }
  return 0;
}

int batchnorm1d(float* output_signal, float* input_signal,
  unsigned in_time, unsigned in_channels,
  const float* const mean, const float* const var,
  unsigned affine_config, const float* const gamma , const float* const beta,
  unsigned in_place, float eps) {
  float* ret = in_place ? (float*)input_signal : (float*)output_signal;
  
  // Check for affine_config
  // = 1 ; Use gamma, beta, mean and var
  // = 2 ; Use only gamma and beta
  // = 3 ; Use only mean and var
  if (affine_config == 1) {
    while (in_time--) {
      float* gamma_offset = (float*)gamma;
      float* beta_offset = (float*)beta;
      float* mean_offset = (float*)mean;
      float* var_offset = (float*)var;
      unsigned channels = in_channels;

      #ifdef LOOP_UNROLL
        unsigned len_unroll = channels >> 2;
        channels %= 4;
        while (len_unroll--) {
          *ret++ = (*gamma_offset++) * (((*input_signal++) - (*mean_offset++)) /
                      sqrt((*var_offset++) + eps)) + (*beta_offset++);
          *ret++ = (*gamma_offset++) * (((*input_signal++) - (*mean_offset++)) /
                      sqrt((*var_offset++) + eps)) + (*beta_offset++);
          *ret++ = (*gamma_offset++) * (((*input_signal++) - (*mean_offset++)) /
                      sqrt((*var_offset++) + eps)) + (*beta_offset++);
          *ret++ = (*gamma_offset++) * (((*input_signal++) - (*mean_offset++)) /
                      sqrt((*var_offset++) + eps)) + (*beta_offset++);
        }
      #endif

      while (channels--) {
        *ret++ = (*gamma_offset++) * (((*input_signal++) - (*mean_offset++)) /
                    sqrt((*var_offset++) + eps)) + (*beta_offset++);
      }
    }
  }
  else if (affine_config == 2) {
    while (in_time--) {
      float* gamma_offset = (float*)gamma;
      float* beta_offset = (float*)beta;
      unsigned channels = in_channels;

      #ifdef LOOP_UNROLL
        unsigned len_unroll = channels >> 2;
        channels %= 4;
        while (len_unroll--) {
          *ret++ = ((*gamma_offset++) * (*input_signal++)) + (*beta_offset++);
          *ret++ = ((*gamma_offset++) * (*input_signal++)) + (*beta_offset++);
          *ret++ = ((*gamma_offset++) * (*input_signal++)) + (*beta_offset++);
          *ret++ = ((*gamma_offset++) * (*input_signal++)) + (*beta_offset++);
        }
      #endif

      while (channels--) {
        *ret++ = ((*gamma_offset++) * (*input_signal++)) + (*beta_offset++);
      }
    }
  }
  else {
    while (in_time--) {
      float* mean_offset = (float*)mean;
      float* var_offset = (float*)var;
      unsigned channels = in_channels;

      #ifdef LOOP_UNROLL
        unsigned len_unroll = channels >> 2;
        channels %= 4;
        while (len_unroll--) {
          *ret++ = ((*input_signal++) - (*mean_offset++)) /
                    sqrt((*var_offset++) + eps);
          *ret++ = ((*input_signal++) - (*mean_offset++)) /
                    sqrt((*var_offset++) + eps);
          *ret++ = ((*input_signal++) - (*mean_offset++)) /
                    sqrt((*var_offset++) + eps);
          *ret++ = ((*input_signal++) - (*mean_offset++)) /
                    sqrt((*var_offset++) + eps);
        }
      #endif
      
      while (channels--) {
        *ret++ = ((*input_signal++) - (*mean_offset++)) /
                  sqrt((*var_offset++) + eps);
      }
    }
  }
  return 0;
}
