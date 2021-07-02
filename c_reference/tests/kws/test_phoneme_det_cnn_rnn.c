// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "conv1d.h"
#include "dscnn.h"
#include "utils.h"
#include "rnn_bricked.h"

#include "keyword_spotting_io_2.h"
#include "precnn_params.h"
#include "rnn_params.h"
#include "postcnn_params.h"

// Check number of output time-steps with the number of label time-steps
int checkTime(unsigned out_time) {
  if (out_time != KWS_OUT_TIME) {
    printf("Error, estimated and actual ouput time-steps mismatch");
    return 1;
  }
  return 0;
}
// Error Check
void checkError(float* pred, float* label) {
  float error = 0, denom = 0;
  for (unsigned t = 0; t < KWS_OUT_TIME; t++) {
    for (unsigned d = 0; d < POST_CNN_OUT_FEATURES; d++) {
      error += ((pred[t * POST_CNN_OUT_FEATURES + d] 
                - label[t * POST_CNN_OUT_FEATURES + d])
                * (pred[t * POST_CNN_OUT_FEATURES + d] 
                - label[t * POST_CNN_OUT_FEATURES + d]));
      denom += label[t * POST_CNN_OUT_FEATURES + d] 
                * label[t * POST_CNN_OUT_FEATURES + d];
    }
  }
  printf("Full Network\n");
  printf("Agg Squared Error : %f\n", error);
  printf("MSE : %f\n", error / (KWS_OUT_TIME*POST_CNN_OUT_FEATURES));
  printf("RMSE : %f\n", error / denom);
}

/* CNN-RNN based Phoneme Detection Model
 
  The phoneme detection model used consists of 6 blocks.
  1st block is a CNN, where kernel size is 5 and regular tanh activation
  2nd block is an RNN, which has a specified forward and a backward context running at a stride/hop of 3.
  Hence it reduces the sequence length by a factor of 3.
  Rest of the blocks(3rd, 4th, 5th and 6th) are a combination of CNNs
  Each of the final 4 blocks consist of a depth cnn (kernel size of 5) and a point cnn (kernel size of 1)

  Input to the architecture is of the form (seq_len, feature_dim) where feature dim refers to n_mels (number of mel features/number of features from the featurizer).
  Output is of the form (seq_len/3, 41) where 41 is the number of phonemes over which the classification is performed. 
  Phonemes are predicted for every 3rd time frame, operating under the assumption that they don't vary faster than that.

  NOTE: Before deployment for real-time streaming applications, we would need to make minor modification
  These changes are subject to the input specs i.e fixing input buffer time steps, number of features from the deployed featurizer, method of reading the input into a buffer
*/
void phoneme_prediction(float* mem_buf) {
  ConvLayers_LR_Parallel_Params conv_params = {
    .W1 = CNN1_W1,
    .W2 = CNN1_W2,
    .B = CNN1_BIAS,
    .rank = PRE_CNN_LOW_RANK,
    .block_size_to_lr = 100,
    .block_size_from_lr = 100,
  };

  ConvLayers_Params depth_param_2 = {
    .W = CNN2_DEPTH_W,
    .B = CNN2_DEPTH_BIAS,
    .depthwise = 1,
  };

  ConvLayers_LR_Parallel_Params point_param_2 = {
    .W1 = CNN2_POINT_W1,
    .W2 = CNN2_POINT_W2,
    .B = CNN2_POINT_BIAS,
    .rank = POST_CNN_LOW_RANK,
    .block_size_to_lr = 100,
    .block_size_from_lr = 100,
  };

  ConvLayers_Params depth_param_3 = {
    .W = CNN3_DEPTH_W,
    .B = CNN3_DEPTH_BIAS,
    .depthwise = 1,
  };

  ConvLayers_LR_Parallel_Params point_param_3 = {
    .W1 = CNN3_POINT_W1,
    .W2 = CNN3_POINT_W2,
    .B = CNN3_POINT_BIAS,
    .rank = POST_CNN_LOW_RANK,
    .block_size_to_lr = 100,
    .block_size_from_lr = 100,
  };

  ConvLayers_Params depth_param_4 = {
    .W = CNN4_DEPTH_W,
    .B = CNN4_DEPTH_BIAS,
    .depthwise = 1,
  };

  ConvLayers_LR_Parallel_Params point_param_4 = {
    .W1 = CNN4_POINT_W1,
    .W2 = CNN4_POINT_W2,
    .B = CNN4_POINT_BIAS,
    .rank = POST_CNN_LOW_RANK,
    .block_size_to_lr = 100,
    .block_size_from_lr = 100,
  };

  ConvLayers_Params depth_param_5 = {
    .W = CNN5_DEPTH_W,
    .B = CNN5_DEPTH_BIAS,
    .depthwise = 1,
  };

  ConvLayers_LR_Parallel_Params point_param_5 = {
    .W1 = CNN5_POINT_W1,
    .W2 = CNN5_POINT_W2,
    .B = CNN5_POINT_BIAS,
    .rank = POST_CNN_LOW_RANK,
    .block_size_to_lr = 100,
    .block_size_from_lr = 100,
  };

  BrickedFastGRNN_LR_Params bwd_RNN_params = {
    .W1     = B_W1,
    .W2     = B_W2,
    .wRank  = RNN_LOW_RANK,
    .U1     = B_U1,
    .U2     = B_U2,
    .uRank  = RNN_LOW_RANK,
    .Bg     = B_BIAS_GATE,
    .Bh     = B_BIAS_UPDATE,
    .sigmoid_zeta = sigmoid(B_ZETA),
    .sigmoid_nu   = sigmoid(B_NU),
    .block_size_u_from_lr = 100,
    .block_size_u_to_lr = 100,
    .block_size_w_from_lr = 100,
    .block_size_w_to_lr = 100,
  };

  BrickedFastGRNN_LR_Params fwd_RNN_params = {
    .W1     = F_W1,
    .W2     = F_W2,
    .wRank  = RNN_LOW_RANK,
    .U1     = F_U1,
    .U2     = F_U2,
    .uRank  = RNN_LOW_RANK,
    .Bg     = F_BIAS_GATE,
    .Bh     = F_BIAS_UPDATE,
    .sigmoid_zeta = sigmoid(F_ZETA),
    .sigmoid_nu   = sigmoid(F_NU),
    .block_size_u_from_lr = 100,
    .block_size_u_to_lr = 100,
    .block_size_w_from_lr = 100,
    .block_size_w_to_lr = 100,
  };

  unsigned in_time, out_time;

  /* Pre-CNN */
  in_time = KWS_IN_TIME;
  out_time = in_time - PRE_CNN_FILT + (PRE_CNN_FILT_PAD << 1) + 1;
  float* cnn1_out = (float*)malloc(out_time * PRE_CNN_OUT_FEATURES * sizeof(float));
  // Since batchnorm1d is the first layer and in-place will alter the input. 
  // Use the in-place computation only if the input can be discarded/altered. Else avoid in-place computation for this layer
  phon_pred_lr_cnn(cnn1_out, mem_buf,
    conv1d_lr_parallel, in_time, PRE_CNN_IN_FEATURES,
    0, 0, PRE_CNN_BNORM_AFFINE, CNN1_SCALE, CNN1_OFFSET, PRE_CNN_BNORM_INPLACE,
    PRE_CNN_OUT_FEATURES, PRE_CNN_FILT_PAD, PRE_CNN_FILT,
    &conv_params, PRE_CNN_STRIDE, PRE_CNN_FILT_ACT); // regular tanh activation

  batchnorm1d(0, cnn1_out, in_time, RNN_IN_FEATURES, 
    0, 0, RNN_BNORM_AFFINE, RNN_SCALE, RNN_OFFSET, 1, 0.00001);

  /* Bricked Bi-FastGRNN Block */
  out_time = in_time/RNN_HOP + 1;
  float* rnn_out = (float*)malloc(out_time * RNN_OUT_FEATURES * sizeof(float));
  forward_bricked_fastgrnn_lr(rnn_out, RNN_OUT_FEATURES >> 1, cnn1_out,
    in_time, RNN_IN_FEATURES, RNN_FWD_WINDOW, RNN_HOP,
    &fwd_RNN_params, RNN_BI_DIR, RNN_SAMPLE_FIRST_BRICK);

  backward_bricked_fastgrnn_lr(rnn_out + (RNN_OUT_FEATURES >> 1), 
    RNN_OUT_FEATURES >> 1, cnn1_out,
    in_time, RNN_IN_FEATURES, RNN_BWD_WINDOW, RNN_HOP,
    &bwd_RNN_params, RNN_BI_DIR, RNN_SAMPLE_LAST_BRICK);
  free(cnn1_out);

  /* Post-CNN */
  // Since all inputs to the subsequent layers are temporary, in-place batchnorm1d can be used without any input(initial buffer)/output(final layer) data alteration/corruption
  // CNN2
  in_time = out_time;
  out_time = in_time - POST_CNN_DEPTH_FILT + (POST_CNN_DEPTH_PAD << 1) + 1;
  out_time = out_time - POST_CNN_POOL + (POST_CNN_POOL_PAD << 1) + 1;
  float* cnn2_out = (float*)malloc(out_time * POST_CNN_INTER_FEATURES * sizeof(float));
  phon_pred_depth_point_lr_cnn(cnn2_out, rnn_out,
    conv1d_lr_parallel, in_time, POST_CNN_INTER_FEATURES,
    0, 0, POST_CNN_BNORM_AFFINE, CNN2_SCALE, CNN2_OFFSET, POST_CNN_BNORM_INPLACE,
    POST_CNN_DEPTH_PAD, POST_CNN_DEPTH_FILT,
    &depth_param_2, POST_CNN_DEPTH_STRIDE, POST_CNN_DEPTH_ACT,
    POST_CNN_INTER_FEATURES, POST_CNN_POINT_PAD, POST_CNN_POINT_FILT,
    &point_param_2, POST_CNN_POINT_STRIDE, POST_CNN_POINT_ACT,
    POST_CNN_POOL_PAD, POST_CNN_POOL, POST_CNN_POOL_STRIDE, POST_CNN_POOL_ACT);
  free(rnn_out);

  // CNN3
  in_time = out_time;
  out_time = in_time - POST_CNN_DEPTH_FILT + (POST_CNN_DEPTH_PAD << 1) + 1;
  out_time = out_time - POST_CNN_POOL + (POST_CNN_POOL_PAD << 1) + 1;
  float* cnn3_out = (float*)malloc(out_time * POST_CNN_INTER_FEATURES * sizeof(float));
  phon_pred_depth_point_lr_cnn(cnn3_out, cnn2_out,
    conv1d_lr_parallel, in_time, POST_CNN_INTER_FEATURES,
    0, 0, POST_CNN_BNORM_AFFINE, CNN3_SCALE, CNN3_OFFSET, POST_CNN_BNORM_INPLACE,
    POST_CNN_DEPTH_PAD, POST_CNN_DEPTH_FILT,
    &depth_param_3, POST_CNN_DEPTH_STRIDE, POST_CNN_DEPTH_ACT,
    POST_CNN_INTER_FEATURES, POST_CNN_POINT_PAD, POST_CNN_POINT_FILT,
    &point_param_3, POST_CNN_POINT_STRIDE, POST_CNN_POINT_ACT,
    POST_CNN_POOL_PAD, POST_CNN_POOL, POST_CNN_POOL_STRIDE, POST_CNN_POOL_ACT);
  free(cnn2_out);

  // CNN4
  in_time = out_time;
  out_time = in_time - POST_CNN_DEPTH_FILT + (POST_CNN_DEPTH_PAD << 1) + 1;
  out_time = out_time - POST_CNN_POOL + (POST_CNN_POOL_PAD << 1) + 1;
  float* cnn4_out = (float*)malloc(out_time * POST_CNN_INTER_FEATURES * sizeof(float));
  phon_pred_depth_point_lr_cnn(cnn4_out, cnn3_out,
    conv1d_lr_parallel, in_time, POST_CNN_INTER_FEATURES,
    0, 0, POST_CNN_BNORM_AFFINE, CNN4_SCALE, CNN4_OFFSET, POST_CNN_BNORM_INPLACE,
    POST_CNN_DEPTH_PAD, POST_CNN_DEPTH_FILT,
    &depth_param_4, POST_CNN_DEPTH_STRIDE, POST_CNN_DEPTH_ACT,
    POST_CNN_INTER_FEATURES, POST_CNN_POINT_PAD, POST_CNN_POINT_FILT,
    &point_param_4, POST_CNN_POINT_STRIDE, POST_CNN_POINT_ACT,
    POST_CNN_POOL_PAD, POST_CNN_POOL, POST_CNN_POOL_STRIDE, POST_CNN_POOL_ACT);
  free(cnn3_out);

  // CNN5
  in_time = out_time;
  out_time = in_time - POST_CNN_DEPTH_FILT + (POST_CNN_DEPTH_PAD << 1) + 1;
  out_time = out_time - POST_CNN_POOL + (POST_CNN_POOL_PAD << 1) + 1;
  float* pred = (float*)malloc(out_time * POST_CNN_OUT_FEATURES * sizeof(float));
  phon_pred_depth_point_lr_cnn(pred, cnn4_out,
    conv1d_lr_parallel, in_time, POST_CNN_INTER_FEATURES,
    0, 0, POST_CNN_BNORM_AFFINE, CNN5_SCALE, CNN5_OFFSET, POST_CNN_BNORM_INPLACE,
    POST_CNN_DEPTH_PAD, POST_CNN_DEPTH_FILT,
    &depth_param_5, POST_CNN_DEPTH_STRIDE, POST_CNN_DEPTH_ACT,
    POST_CNN_OUT_FEATURES, POST_CNN_POINT_PAD, POST_CNN_POINT_FILT,
    &point_param_5, POST_CNN_POINT_STRIDE, POST_CNN_POINT_ACT,
    POST_CNN_POOL_PAD, POST_CNN_POOL, POST_CNN_POOL_STRIDE, POST_CNN_POOL_ACT);
  free(cnn4_out);

  /* Output Time and Prediction Check. Created for Debugging */
  if (checkTime(out_time))
    return;
  else
    checkError(pred, OUTPUT);
  free(pred);
}

int main() {
  #ifdef LOOP_UNROLL
    printf("Loop Unrolling Active\n");
  #endif
  clock_t begin = clock();
  phoneme_prediction(INPUT);
  clock_t end = clock();
  double time_spent = (float)(end - begin) / CLOCKS_PER_SEC;
  printf("Time elapsed is %f seconds\n", time_spent);
  return 0;
}
