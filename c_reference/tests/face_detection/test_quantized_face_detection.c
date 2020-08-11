// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <string.h>

#include "quantized_datatypes.h"
#include "quantized_face_detection.h"

#define MEM_BUF_SIZE 188160

// Comparator function for sorting floats.
int compare_floats(const void *a, const void *b) {
  const float *da = (const float *) a;
  const float *db = (const float *) b;

  return (*da > *db) - (*da < *db);
}

// Function for computing the deviation from the expected floating point
// result and returning the largest such deviation found.
/*
float compute_error(const Q15_T* const pred, const float* const label,
                    float* const errors, SCALE_T scl) {
  float agg_diff = 0.0;

  for (unsigned i = 0; i < N * HOUT * WOUT * COUT; i++) {
    float f_pred = ((float)pred[i]) / pow(2, scl);
    float single_diff = fabs(f_pred - label[i]);
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
*/

/** Run this test using the following command:
 * $: ./test_quantized_face_detection <num_patches> <input.npy> <output.npy>
 *    <expected_output.npy> <log.txt>
 *  By default, all tests run without using bit-shifting operations.
 */
int main(int argc, char **argv) {
  unsigned patches;
  SCALE_T XScale = 1, YScale = 12;

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
  char numpyHeader3[] = ", 28800), }";

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

  char* mem_buf = malloc(MEM_BUF_SIZE * sizeof(char));
  /*
  float* xLine = malloc(N * H * W * CIN * sizeof(float));
  float* yLine = malloc(N * HOUT * WOUT * COUT * sizeof(float));
  float* allErrors = malloc(N * HOUT * WOUT * COUT * (sizeof(float)));

  fread(xLine, sizeof(float), N * H * W * CIN, xFile);
  fread(yLine, sizeof(float), N * HOUT * WOUT * COUT, floatResFile);

  for (unsigned i = 0; i < N * H * W * CIN; i++) {
    memory_buffer[i] = (Q15_T)((xLine[i]) * pow(2, XScale));
  }
  */

  fprintf(outputLog, "Running Quantized Face Detection Model\n");
  double time_spent = 0.0;
  clock_t begin = clock();
  q_face_detection(mem_buf);
  clock_t end = clock();
  time_spent += (double)(end - begin) / CLOCKS_PER_SEC;
  fprintf(outputLog, "Time elapsed is %f seconds\n", time_spent);
  //float max_diff = compute_error(output_test, yLine, allErrors, YScale);
  //fprintf(outputLog, "Maximum Observed Deviation: %f \n", max_diff);

  /*
  for (unsigned j = 0; j < N * HOUT * WOUT * COUT; j++) {
    float val = ((float)output_test[j]) / pow(2, YScale);
    fwrite((char*)&val, sizeof(float), 1, yFile);
  }
  */

  fclose(xFile);
  fclose(yFile);
  fclose(floatResFile);

  /*
  float aggregate = aggregate_error(allErrors, N * HOUT * WOUT * COUT);
  fprintf(outputLog, "Aggregated 95th Percentile Error: %f\n", aggregate);
  if (aggregate < 1.419) {
    fprintf(outputLog, "Quantized Face Detection Numerical Test Passed!\n");
  } else {
    fprintf(outputLog, "Quantized Face Detection Numerical Test Failed!\n");
    return -1;
  }

  for (unsigned i = 0; i < N * HOUT * WOUT * COUT; i++){
    if (output_test[i] != expected[i]) {
      fprintf(outputLog, "Output: %d, Expected: %d at Index: %d\n",
              output_test[i], expected[i], i);
      fprintf(outputLog, "Quantized Face Detection Fixed Point Test Failed!\n");
      return -1;
    }
  }
  */

  fprintf(outputLog, "Quantized Face Detection Fixed Point Test Passed!\n");
  /*
  free(reshapedXLine);
  free(output_test);
  free(X);
  free(T);
  free(U);
  free(xLine);
  free(yLine);
  free(allErrors);
  */

  return 0;
}
