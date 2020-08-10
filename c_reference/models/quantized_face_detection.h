// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#ifndef __QUANTIZED_FACE_DETECTION_H__
#define __QUANTIZED_FACE_DETECTION_H__

#include "quantized_datatypes.h"

/**
 * @brief Compute the element-wise subtraction between two vectors.
 * @param[in]     input       pointer to the input tensor
 * @param[out]    output      pointer to the output tensor
 * @return        none
 * @example
 */
void q_face_detection(Q7_T* input, Q15_T* output);

#endif
