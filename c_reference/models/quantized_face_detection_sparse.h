// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#ifndef __QUANTIZED_FACE_DETECTION_SPARSE_H__
#define __QUANTIZED_FACE_DETECTION_SPARSE_H__

/**
 * @brief Routine for running the entire face detection model pipeline.
 * @param[in, out]    mem_buf   pointer to the singleton memory buffer for input, intermediate computations and output, used for fragmentation invariance
 * @return            none
 * @example
 */
void q_face_detection_sparse(char* const mem_buf);

#endif
