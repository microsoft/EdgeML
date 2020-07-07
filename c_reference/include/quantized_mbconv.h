// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#ifndef __QUANTIZED_MBCONV_H__
#define __QUANTIZED_MBCONV_H__

#include "quantized_datatypes.h"

void MBConv(TypeA *A, TypeF1 *F1, TypeB1W *BN1W, TypeB1B *BN1B, TypeF2 *F2,
            TypeB2W *BN2W, TypeB2B *BN2B, TypeF3 *F3, TypeB3W *BN3W,
            TypeB3B *BN3B, TypeC *C, TypeX *X, TypeT *T, TypeU *U, MYITE N,
            MYITE H, MYITE W, MYITE Cin, MYITE Ct, MYITE HF, MYITE WF,
            MYITE Cout, MYITE Hout, MYITE Wout, MYITE HPADL, MYITE HPADR,
            MYITE WPADL, MYITE WPADR, MYITE HSTR, MYITE WSTR, MYITE D1,
            MYITE D2, MYITE D3, TypeUB1W SIX_1, TypeUB2W SIX_2, TypeUB1W shr1,
            TypeUB1W shr2, TypeUB1W shr3, TypeUB2W shr4, TypeUB2W shr5,
            TypeUB2W shr6, TypeUB3W shr7, TypeUB3W shr8, TypeUB3W shr9,
            TypeUB1W shl1, TypeUB1W shl2, TypeUB1W shl3, TypeUB2W shl4,
            TypeUB2W shl5, TypeUB2W shl6, TypeUB3W shl7, TypeUB3W shl8,
            TypeUB3W shl9);

#endif
