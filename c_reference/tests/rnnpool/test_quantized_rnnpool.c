// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <string.h>

#include "quantized_fastgrnn.h"
#include "quantized_rnnpool.h"

#include "q_wider_regression_model/rnn1.h"
#include "q_wider_regression_model/rnn2.h"

// Comparator function for sorting floats.
int compare_floats(const void *a, const void *b) {
  const float *da = (const float *) a;
  const float *db = (const float *) b;

  return (*da > *db) - (*da < *db);
}

// Function for computing the deviation from the expected floating point
// result and returning the largest such deviation found.
float compute_error(Q15_T pred[4 * HIDDEN_DIM2], float label[4 * HIDDEN_DIM2],
                   float* const errors, SCALE_T scl) {
  float epsilon = 0.01;
  float agg_diff = 0.0;

  for (unsigned i = 0; i < 4 * HIDDEN_DIM2; i++) {
    float f_pred = ((float)pred[i]) / pow(2, scl);
    float single_diff = 100.0 * fabs(f_pred - label[i]) / (fabs(label[i]) + epsilon);
    agg_diff = single_diff > agg_diff ? single_diff : agg_diff;
    errors[i] = single_diff;
  }

  return agg_diff;
}

// Function for computing the 95th percentile deviation among all the outputs.
float aggregate_error(float* errors, unsigned len) {
  qsort(errors, len, sizeof(float), compare_floats);
  unsigned index = (unsigned) round(fmax((0.95 * len - 1), 0));
  return errors[index];
}

/** Run this test using the following command:
 * $: ./test_quantized_rnnpool <num_patches> <input.npy> <output.npy>
 *    <expected_output.npy> <log.txt>
 *  By default, all tests run without using bit-shifting operations.
 */
int main(int argc, char **argv) {
  unsigned patches;
  SCALE_T XScale = 12, YScale = 14;

  FILE *xFile, *yFile, *floatResFile, *outputLog;

  if (argc != 6) {
    printf("Improper Number of Arguments Provided!\n");
    return -1;
  } else {
    patches = atoi(argv[1]);
    xFile = fopen(argv[2], "rb");
    yFile = fopen(argv[3], "wb");
    floatResFile = fopen(argv[4], "rb");
    outputLog = fopen(argv[5], "w");
  }

  if (xFile == NULL) {
    printf("An error occured while opening the input file.\n");
    return -1;
  }
  if (yFile == NULL) {
    printf("An error occured while opening the predicted output file.\n");
    return -1;
  }
  if (floatResFile == NULL) {
    printf("An error occured while opening the expected output file.\n");
    return -1;
  }
  if (outputLog == NULL) {
    printf("An error occured while opening the output log file.\n");
    return -1;
  }

  char line[9];
  fgets(line, 9, xFile);
  fgets(line, 9, floatResFile);

  int16_t headerSize;
  fread(&headerSize, sizeof(int16_t), 1, xFile);
  char* headerLine = malloc((headerSize + 1) * sizeof(*headerLine));
  fgets(headerLine, headerSize + 1, xFile);

  int16_t floatHeaderSize;
  fread(&floatHeaderSize, sizeof(int16_t), 1, floatResFile);
  char* floatHeaderLine = malloc((floatHeaderSize + 1) * sizeof(*floatHeaderLine));
  fgets(floatHeaderLine, floatHeaderSize + 1, floatResFile);
  free(floatHeaderLine);
  free(headerLine);

  char numpyHeader1[] = "{'descr': '<f4', 'fortran_order': False, 'shape': (";
  unsigned len = snprintf(NULL, 0, "%d", patches);
  char* numpyHeader2 = malloc((len + 1) * sizeof(char));
  snprintf(numpyHeader2, len + 1, "%d", patches);
  char numpyHeader3[] = ", 1, 32), }";

  size_t headerLength = strlen(numpyHeader1) + strlen(numpyHeader2) +
                        strlen(numpyHeader3);
  int count = 1;
  for (size_t i = headerLength + 10; i % 64 != 63; i++) {
    count++;
  }

  char numpyHeader4[count + 1];
  numpyHeader4[count] = '\0';
  numpyHeader4[count - 1] = (char)(10);
  for (int i = count - 2; i >= 0; i--) {
    numpyHeader4[i] = ' ';
  }

  headerLength += strlen(numpyHeader4);
  char a = headerLength / 256 , b = headerLength % 256;

  char numpyMagix = 147;
  char numpyVersionMajor = 1, numpyVersionMinor = 0;

  fputc(numpyMagix, yFile);
  fputs("NUMPY", yFile);
  fputc(numpyVersionMajor, yFile);
  fputc(numpyVersionMinor, yFile);
  fputc(b, yFile);
  fputc(a, yFile);
  fputs(numpyHeader1, yFile);
  fputs(numpyHeader2, yFile);
  fputs(numpyHeader3, yFile);
  fputs(numpyHeader4, yFile);

  Q15_FastGRNN_Params rnn1_params = {
    .mean = NULL,
    .stdDev = NULL,
    .W = W1,
    .U = U1,
    .Bg = Bg1,
    .Bh = Bh1,
    .sigmoid_zeta = sigmoid_zeta1,
    .sigmoid_nu = sigmoid_nu1
  };

  Q15_FastGRNN_Params rnn2_params = {
    .mean = NULL,
    .stdDev = NULL,
    .W = W2,
    .U = U2,
    .Bg = Bg2,
    .Bh = Bh2,
    .sigmoid_zeta = sigmoid_zeta2,
    .sigmoid_nu = sigmoid_nu2
  };

  Q15_T preComp11[HIDDEN_DIM1];
  Q15_T preComp12[HIDDEN_DIM1];
  Q15_T preComp13[HIDDEN_DIM1];
  Q15_T normFeatures1[INPUT_CHANNELS];
  memset(preComp11, 0, sizeof(Q15_T) * HIDDEN_DIM1);
  memset(preComp12, 0, sizeof(Q15_T) * HIDDEN_DIM1);
  memset(preComp13, 0, sizeof(Q15_T) * HIDDEN_DIM1);
  memset(normFeatures1, 0, sizeof(Q15_T) * INPUT_CHANNELS);
  Q15_FastGRNN_Buffers rnn1_buffers = {
  	.preComp1 = preComp11,
    .preComp2 = preComp12,
    .preComp3 = preComp13,
    .normFeatures = normFeatures1
  };

  Q15_T preComp21[HIDDEN_DIM2];
  Q15_T preComp22[HIDDEN_DIM2];
  Q15_T preComp23[HIDDEN_DIM2];
  Q15_T normFeatures2[HIDDEN_DIM1];
  memset(preComp21, 0, sizeof(Q15_T) * HIDDEN_DIM2);
  memset(preComp22, 0, sizeof(Q15_T) * HIDDEN_DIM2);
  memset(preComp23, 0, sizeof(Q15_T) * HIDDEN_DIM2);
  memset(normFeatures2, 0, sizeof(Q15_T) * HIDDEN_DIM1);
  Q15_FastGRNN_Buffers rnn2_buffers = {
    .preComp1 = preComp21,
    .preComp2 = preComp22,
    .preComp3 = preComp23,
    .normFeatures = normFeatures2
  };

  Q15_FastGRNN_Scales rnn1_scales = {
    .input = input1,
    .mean = meanScale1,
    .meanSub = meanSub1,
    .stdDev = stdDevScale1,
    .normFeaturesHDStdDev = normFeaturesHDStdDev1,
    .W = WScale1,
    .normFeaturesMVW = normFeaturesMVW1,
    .H1W = H1W1,
    .H2W = H2W1,
    .U = UScale1,
    .hiddenStateMVU = hiddenStateMVU1,
    .H1U = H1U1,
    .H2U = H2U1,
    .mV1AddMV2 = mV1AddMV21,
    .mV2AddMV1 = mV2AddMV11,
    .mV1AddMV2Out = mV1AddMV2Out1,
    .pC1AddBg = pC1AddBg1,
    .Bg = BgScale1,
    .pC1AddBgOut = pC1AddBgOut1,
    .sigmoidLimit = sigmoidLimit1,
    .sigmoidScaleIn = sigmoidScaleIn1,
    .sigmoidScaleOut = sigmoidScaleOut1,
    .pC1AddBh = pC1AddBh1,
    .Bh = BhScale1,
    .pC1AddBhOut = pC1AddBhOut1,
    .tanhScaleIn = tanhScaleIn1,
    .tanhScaleOut = tanhScaleOut1,
    .gateHDHiddenState = gateHDHiddenState1,
    .hiddenStateHDGate = hiddenStateHDGate1,
    .qOneScale = qOneScale1,
    .qOneSubGate = qOneSubGate1,
    .qOneSubGateOut = qOneSubGateOut1,
    .sigmoidZeta = sigmoidZetaScale1,
    .sigmoidZetaMulQOneSubGate = sigmoidZetaMulQOneSubGate1,
    .sigmoidNu = sigmoidNuScale1,
    .sigmoidNuAddQOneSubGate = sigmoidNuAddQOneSubGate1,
    .sigmoidNuAddQOneSubGateOut = sigmoidNuAddQOneSubGateOut1,
    .sigmoidNuAddQOneSubGateHDUpdate = sigmoidNuAddQOneSubGateHDUpdate1,
    .updateHDSigmoidNuAddQOneSubGate = updateHDSigmoidNuAddQOneSubGate1,
    .pC3AddPC1 = pC3AddPC11,
    .pC1AddPC3 = pC1AddPC31,
    .hiddenStateOut = hiddenStateOut1,
    .div = div1,
    .add = add1,
    .qOne = qOne1
  };

  Q15_FastGRNN_Scales rnn2_scales = {
    .input = input2,
    .mean = meanScale2,
    .meanSub = meanSub2,
    .stdDev = stdDevScale2,
    .normFeaturesHDStdDev = normFeaturesHDStdDev2,
    .W = WScale2,
    .normFeaturesMVW = normFeaturesMVW2,
    .H1W = H1W2,
    .H2W = H2W2,
    .U = UScale2,
    .hiddenStateMVU = hiddenStateMVU2,
    .H1U = H1U2,
    .H2U = H2U2,
    .mV1AddMV2 = mV1AddMV22,
    .mV2AddMV1 = mV2AddMV12,
    .mV1AddMV2Out = mV1AddMV2Out2,
    .pC1AddBg = pC1AddBg2,
    .Bg = BgScale2,
    .pC1AddBgOut = pC1AddBgOut2,
    .sigmoidLimit = sigmoidLimit2,
    .sigmoidScaleIn = sigmoidScaleIn2,
    .sigmoidScaleOut = sigmoidScaleOut2,
    .pC1AddBh = pC1AddBh2,
    .Bh = BhScale2,
    .pC1AddBhOut = pC1AddBhOut2,
    .tanhScaleIn = tanhScaleIn2,
    .tanhScaleOut = tanhScaleOut2,
    .gateHDHiddenState = gateHDHiddenState2,
    .hiddenStateHDGate = hiddenStateHDGate2,
    .qOneScale = qOneScale2,
    .qOneSubGate = qOneSubGate2,
    .qOneSubGateOut = qOneSubGateOut2,
    .sigmoidZeta = sigmoidZetaScale2,
    .sigmoidZetaMulQOneSubGate = sigmoidZetaMulQOneSubGate2,
    .sigmoidNu = sigmoidNuScale2,
    .sigmoidNuAddQOneSubGate = sigmoidNuAddQOneSubGate2,
    .sigmoidNuAddQOneSubGateOut = sigmoidNuAddQOneSubGateOut2,
    .sigmoidNuAddQOneSubGateHDUpdate = sigmoidNuAddQOneSubGateHDUpdate2,
    .updateHDSigmoidNuAddQOneSubGate = updateHDSigmoidNuAddQOneSubGate2,
    .pC3AddPC1 = pC3AddPC12,
    .pC1AddPC3 = pC1AddPC32,
    .hiddenStateOut = hiddenStateOut2,
    .div = div2,
    .add = add2,
    .qOne = qOne2
  };

  Q15_T output_test[4 * HIDDEN_DIM2];
  Q15_T buffer[HIDDEN_DIM1 * PATCH_DIM];
  float xLine[INPUT_CHANNELS * PATCH_DIM * PATCH_DIM];
  float yLine[4 * HIDDEN_DIM2];
  float* allErrors = malloc(patches * 4 * HIDDEN_DIM2 * (sizeof(float)));

  double time_spent = 0.0;
  for (unsigned i = 0; i < patches; i++) {
    fread(&xLine[0], sizeof(float), INPUT_CHANNELS * PATCH_DIM * PATCH_DIM, xFile);
    fread(&yLine[0], sizeof(float), 4 * HIDDEN_DIM2, floatResFile);
    Q15_T reshapedXLine[INPUT_CHANNELS * PATCH_DIM * PATCH_DIM];

    for (unsigned a = 0; a < INPUT_CHANNELS; a ++) {
      for (unsigned b = 0; b < PATCH_DIM; b++) {
        for (unsigned c = 0; c < PATCH_DIM; c++) {
          reshapedXLine[b * PATCH_DIM * INPUT_CHANNELS + c * INPUT_CHANNELS + a] =
          (Q15_T)((xLine[a * PATCH_DIM * PATCH_DIM + b * PATCH_DIM + c]) * pow(2, XScale));
        }
      }
    }

    fprintf(outputLog, "Running Quantized RNNPool on Patch %d\n", i + 1);
    clock_t begin = clock();
    q15_rnnpool_block(reshapedXLine, INPUT_CHANNELS, PATCH_DIM, PATCH_DIM,
                      q15_fastgrnn, HIDDEN_DIM1, (const void*)(&rnn1_params),
                      (void*)(&rnn1_buffers), (const void*)(&rnn1_scales),
                      q15_fastgrnn, HIDDEN_DIM2, (const void*)(&rnn2_params),
                      (void*)(&rnn2_buffers), (const void*)(&rnn2_scales),
                      output_test, buffer);
    clock_t end = clock();
    time_spent += (double)(end - begin) / CLOCKS_PER_SEC;
    fprintf(outputLog, "Time elapsed is %f seconds\n", time_spent);

    float max_diff = compute_error(output_test, yLine,
                                   allErrors + i * 4 * HIDDEN_DIM2, YScale);
    fprintf(outputLog, "Maximum Observed Deviation: %f percent\n", max_diff);

    for (unsigned j = 0; j < 4 * HIDDEN_DIM2; j++) {
      float val = ((float)output_test[j]) / pow(2, YScale);
      fwrite((char*)&val, sizeof(float), 1, yFile);
    }
  }

  fclose(xFile);
  fclose(yFile);
  fclose(floatResFile);

  float aggregate = aggregate_error(allErrors, patches * 4 * HIDDEN_DIM2);
  fprintf(outputLog, "Aggregated 95th Percentile Error: %f\n", aggregate);
  if (aggregate < 1.61) {
    fprintf(outputLog, "Quantized RNNPool Numerical Test Passed!\n");
  } else {
    fprintf(outputLog, "Quantized RNNPool Numerical Test Failed!\n");
    return -1;
  }

  free(allErrors);
  return 0;
}
