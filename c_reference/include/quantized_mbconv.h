// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#ifndef __QUANTIZED_MBCONV_H__
#define __QUANTIZED_MBCONV_H__

#include "quantized_utils.h"

void MBConv(const INT_T* const A, const INT_T* const F1, const INT_T* const BN1W,
            const INT_T* const BN1B, const INT_T* const F2, const INT_T* const BN2W,
            const INT_T* const BN2B, const INT_T* const F3, const INT_T* const BN3W,
            const INT_T* const BN3B, INT_T* const C, INT_T* const X, INT_T* const T,
            INTM_T* const U, ITER_T N, ITER_T H, ITER_T W, ITER_T Cin, ITER_T Ct,
            ITER_T HF, ITER_T WF, ITER_T Cout, ITER_T Hout, ITER_T Wout,
            ITER_T HPADL, ITER_T HPADR, ITER_T WPADL, ITER_T WPADR, ITER_T HSTR,
            ITER_T WSTR, SCALE_T D1, SCALE_T D2, SCALE_T D3, INTM_T limit1,
            INTM_T limit2, INTM_T shr1, INTM_T shr2, INTM_T shr3, INTM_T shr4,
            INTM_T shr5, INTM_T shr6, INTM_T shr7, INTM_T shr8, INTM_T shr9,
            INTM_T shl1, INTM_T shl2, INTM_T shl3, INTM_T shl4, INTM_T shl5,
            INTM_T shl6, INTM_T shl7, INTM_T shl8, INTM_T shl9);

#endif
