// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "quantized_mbconv.h"

<class TypeA, class TypeF1, class TypeB1W, class TypeB1B, class TypeF2, class TypeB2W,
class TypeB2B, class TypeF3, class TypeB3W, class TypeB3B, class TypeC, class TypeX,
class TypeT, class TypeU, class TypeUB1W, class TypeUB2W, class TypeUB3W>

<int16_t, int16_t, int16_t, int16_t, int16_t, int16_t,
int16_t, int16_t, int16_t, int16_t, int16_t, int16_t,
int16_t, int32_t, int32_t, int32_t, int32_t>

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
            TypeUB3W shl9) {
	MYITE HOffsetL = (HF/2) - HPADL;
	MYITE WOffsetL = (WF/2) - WPADL;
	MYITE HOffsetR = (HF/2) - HPADR;
	MYITE WOffsetR = (WF/2) - WPADR;

	for (MYITE n = 0; n < N; n++) {
		MYITE margin = HOffsetL + (HF / 2 + 1) - HSTR > 0 ? HOffsetL + (HF/2 + 1) - HSTR : 0;
		MYITE nstart = HOffsetL - (HF/2) < 0 ? 0 : HOffsetL - (HF/2);
		for (MYITE i = nstart; i < margin; i++) {
			for (MYITE j = 0; j < W; j++) {
				for (MYITE k = 0; k < Ct; k++) {
					for (MYITE l = 0; l < Cin; l++) {
						U[l] = ((TypeU) A[n * H * W * Cin + i * W * Cin + j * Cin + l]) * ((TypeU) F1[l * Ct + k]);
					}
					MYITE totalEle = Cin;
					MYITE count = Cin;
					MYITE depth = 0;

					while (depth < D1) {
						for (MYITE p = 0; p < (totalEle / 2 + 1); p++) {
							if (p < count / 2)
								U[p] = U[2 * p] / 2 + U[(2 * p) + 1] / 2;
							else if ((p == (count / 2)) && ((count % 2) == 1))
								U[p] = U[2 * p] / 2;
							else
								U[p] = 0;
						}
						count = (count + 1) / 2;
						depth++;
					}

					TypeUB1W x = (((TypeUB1W)((U[0] * shl1) / shr1 + (BN1B[k] * shl2) / shr2)) * ((TypeUB1W)BN1W[k]));
					x = x < 0 ? 0 : x;
					x = x > SIX_1 ? SIX_1 : x;
					X[i * W * Ct + j * Ct + k] =  (x * shl3) / shr3;
				}
			}
		}

		for (MYITE h = HOffsetL, hout = 0; h < H - HOffsetR; hout++, h += HSTR) {

			for (MYITE i = 0; i < HSTR; i++) {
				for (MYITE j = 0; j < W; j++) {
					for (MYITE k = 0; k < Ct; k++) {
						MYITE iRed = (i + margin + hout * HSTR) % HF, iFull = i + margin + hout * HSTR;
						X[iRed * W * Ct + j * Ct + k] = 0.0;
						for (MYITE l = 0; l < Cin; l++) {
							TypeA a = iFull < H ? A[n * H * W * Cin + iFull * W * Cin + j * Cin + l] : 0.0;
							U[l] = ((TypeU) a) * ((TypeU) F1[l * Ct + k]);
						}
						MYITE totalEle = Cin;
						MYITE count = Cin;
						MYITE depth = 0;

						while (depth < D1) {
							for (MYITE p = 0; p <(totalEle / 2 + 1); p++) {
								if (p < count / 2)
									U[p] = U[2 * p] / 2 + U[(2 * p) + 1] / 2;
								else if ((p == (count / 2)) && ((count % 2) == 1))
									U[p] = U[2 * p] / 2;
								else
									U[p] = 0;
							}
							count = (count + 1) / 2;
							depth++;
						}

						TypeUB1W x = (((TypeUB1W)((U[0] * shl1) / shr1 + (BN1B[k] * shl2) / shr2)) * ((TypeUB1W)BN1W[k]));
						x = x < 0 ? 0 : x;
						x = x > SIX_1 ? SIX_1 : x;
						X[iRed * W * Ct + j * Ct + k] =  (x * shl3) / shr3;
					}
				}
			}

			for (MYITE w = WOffsetL, wout = 0; w < W - WOffsetR; w += WSTR, wout++) {
				for (MYITE g = 0; g < Ct; g++) {
					MYITE counter = 0;
					for (MYITE hf = -(HF/2); hf <= (HF/2); hf++) {
						for (MYITE wf = -(WF/2); wf <= (WF/2); wf++) {
							TypeX x = (((h + hf) < 0) || ((h + hf) >= H) || ((w + wf) < 0) || ((w + wf) >= W)) ? 0.0 : X[((h + hf) % HF) * W * Ct + (w + wf) * Ct + g];
							TypeF2 b = F2[g * HF * WF + (hf + HF/2) * WF + (wf + WF/2)];
							U[counter] = ((TypeU) x) * ((TypeU) b);
							counter++;
						}
					}
					MYITE totalEle = HF * WF;
					MYITE count = HF * WF;
					MYITE depth = 0;

					while (depth < D2) {
						for (MYITE p = 0; p < (totalEle / 2 + 1); p++) {
							if (p < count / 2)
								U[p] = U[2 * p] / 2 + U[(2 * p) + 1] / 2;
							else if ((p == (count / 2)) && ((count % 2) == 1))
								U[p] = U[2 * p] / 2;
							else
								U[p] = 0;
						}
						count = (count + 1) / 2;
						depth++;
					}

					TypeUB2W x = (((TypeUB2W)((U[0] * shl4) / shr4 + (BN2B[g] * shl5) / shr5)) * ((TypeUB2W)BN2W[g]));
					x = x < 0 ? 0 : x;
					x = x > SIX_2 ? SIX_2 : x;
					T[g] =  (x * shl6) / shr6;
				}

				for (MYITE i = 0; i < Cout; i++) {
					for (MYITE g = 0; g < Ct; g++)
						U[g] = T[g] * F3[g * Cout + i];
					MYITE totalEle = Ct;
					MYITE count = Ct;
					MYITE depth = 0;

					while (depth < D3) {
						for (MYITE p = 0; p<(totalEle / 2 + 1); p++) {
							if (p < count / 2)
								U[p] = U[2 * p] / 2 + U[(2 * p) + 1] / 2;
							else if ((p == (count / 2 )) && ((count % 2) == 1))
								U[p] = U[2 * p] / 2;
							else
								U[p] = 0;
						}
						count = (count + 1) / 2;
						depth++;
					}
					C[n * Hout * Wout * Cout + hout * Wout * Cout + wout * Cout + i] = ((((TypeUB3W)((U[0] * shl7) / shr7 + (BN3B[i] * shl8) / shr8)) * ((TypeUB3W)BN3W[i])) * shl9) / shr9;
				}
			}
		}
	}
}
