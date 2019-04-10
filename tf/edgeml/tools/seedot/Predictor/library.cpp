// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <iostream>

#include "datatypes.h"
#include "library.h"

// This file contains implementations of the linear algebra operators supported by SeeDot.
// Each function takes the scaling factors as arguments along with the pointers to the operands.

// C = A + B
void MatAdd(MYINT *A, MYINT *B, MYINT *C, MYINT I, MYINT J, MYINT shrA, MYINT shrB, MYINT shrC) {
	for (MYINT i = 0; i < I; i++) {
		for (MYINT j = 0; j < J; j++) {
			MYINT a = A[i * J + j];
			MYINT b = B[i * J + j];

			a = a / shrA;
			b = b / shrB;

			MYINT c = a + b;
			c = c / shrC;

			C[i * J + j] = c;
		}
	}
	return;
}

// C = A - B
void MatSub(MYINT *A, const MYINT *B, MYINT *C, MYINT I, MYINT J, MYINT shrA, MYINT shrB, MYINT shrC) {
	for (MYINT i = 0; i < I; i++) {
		for (MYINT j = 0; j < J; j++) {
			MYINT a = A[i * J + j];
			MYINT b = B[i * J + j];

			a = a / shrA;
			b = b / shrB;

			MYINT c = a - b;
			c = c / shrC;

			C[i * J + j] = c;
		}
	}
	return;
}

// C = A * B
void MatMulNN(MYINT *A, MYINT *B, MYINT *C, MYINT *tmp, MYINT I, MYINT K, MYINT J, MYINT shrA, MYINT shrB, MYINT H1, MYINT H2) {

	for (MYINT i = 0; i < I; i++) {
		for (MYINT j = 0; j < J; j++) {
			for (MYINT k = 0; k < K; k++) {
				MYINT a = A[i * K + k];
				MYINT b = B[k * J + j];

				a = a / shrA;
				b = b / shrB;

				tmp[k] = a * b;
			}

			MYINT count = K, depth = 0;
			bool shr = true;

			while (depth < (H1 + H2)) {
				if (depth >= H1)
					shr = false;

				for (MYINT p = 0; p < (K / 2 + 1); p++) {
					MYINT sum;
					if (p < (count >> 1))
						sum = tmp[2 * p] + tmp[(2 * p) + 1];
					else if ((p == (count >> 1)) && ((count & 1) == 1))
						sum = tmp[2 * p];
					else
						sum = 0;

					if (shr)
						tmp[p] = sum / 2;
					else
						tmp[p] = sum;
				}
				count = (count + 1) >> 1;

				depth++;
			}

			C[i * J + j] = tmp[0];
		}
	}
	return;
}

// C = A * B
void MatMulCN(const MYINT *A, MYINT *B, MYINT *C, MYINT *tmp, MYINT I, MYINT K, MYINT J, MYINT shrA, MYINT shrB, MYINT H1, MYINT H2) {

	for (MYINT i = 0; i < I; i++) {
		for (MYINT j = 0; j < J; j++) {
			for (MYINT k = 0; k < K; k++) {
				MYINT a = A[i * K + k];
				MYINT b = B[k * J + j];

				a = a / shrA;
				b = b / shrB;

				tmp[k] = a * b;
			}

			MYINT count = K, depth = 0;
			bool shr = true;

			while (depth < (H1 + H2)) {
				if (depth >= H1)
					shr = false;

				for (MYINT p = 0; p < (K / 2 + 1); p++) {
					MYINT sum;
					if (p < (count >> 1))
						sum = tmp[2 * p] + tmp[(2 * p) + 1];
					else if ((p == (count >> 1)) && ((count & 1) == 1))
						sum = tmp[2 * p];
					else
						sum = 0;

					if (shr)
						tmp[p] = sum / 2;
					else
						tmp[p] = sum;
				}
				count = (count + 1) >> 1;

				depth++;
			}

			C[i * J + j] = tmp[0];
		}
	}
	return;
}

// C = A * B
void MatMulNC(MYINT *A, const MYINT *B, MYINT *C, MYINT *tmp, MYINT I, MYINT K, MYINT J, MYINT shrA, MYINT shrB, MYINT H1, MYINT H2) {

	for (MYINT i = 0; i < I; i++) {
		for (MYINT j = 0; j < J; j++) {
			for (MYINT k = 0; k < K; k++) {
				MYINT a = A[i * K + k];
				MYINT b = B[k * J + j];

				a = a / shrA;
				b = b / shrB;

				tmp[k] = a * b;
			}

			MYINT count = K, depth = 0;
			bool shr = true;

			while (depth < (H1 + H2)) {
				if (depth >= H1)
					shr = false;

				for (MYINT p = 0; p < (K / 2 + 1); p++) {
					MYINT sum;
					if (p < (count >> 1))
						sum = tmp[2 * p] + tmp[(2 * p) + 1];
					else if ((p == (count >> 1)) && ((count & 1) == 1))
						sum = tmp[2 * p];
					else
						sum = 0;

					if (shr)
						tmp[p] = sum / 2;
					else
						tmp[p] = sum;
				}
				count = (count + 1) >> 1;

				depth++;
			}

			C[i * J + j] = tmp[0];
		}
	}
	return;
}

// C = A * B
void MatMulCC(const MYINT *A, const MYINT *B, MYINT *C, MYINT *tmp, MYINT I, MYINT K, MYINT J, MYINT shrA, MYINT shrB, MYINT H1, MYINT H2) {

	for (MYINT i = 0; i < I; i++) {
		for (MYINT j = 0; j < J; j++) {
			for (MYINT k = 0; k < K; k++) {
				MYINT a = A[i * K + k];
				MYINT b = B[k * J + j];

				a = a / shrA;
				b = b / shrB;

				tmp[k] = a * b;
			}

			MYINT count = K, depth = 0;
			bool shr = true;

			while (depth < (H1 + H2)) {
				if (depth >= H1)
					shr = false;

				for (MYINT p = 0; p < (K / 2 + 1); p++) {
					MYINT sum;
					if (p < (count >> 1))
						sum = tmp[2 * p] + tmp[(2 * p) + 1];
					else if ((p == (count >> 1)) && ((count & 1) == 1))
						sum = tmp[2 * p];
					else
						sum = 0;

					if (shr)
						tmp[p] = sum / 2;
					else
						tmp[p] = sum;
				}
				count = (count + 1) >> 1;

				depth++;
			}

			C[i * J + j] = tmp[0];
		}
	}
	return;
}

// C = A |*| B
void SparseMatMul(const MYINT *Aidx, const MYINT *Aval, MYINT **B, MYINT *C, MYINT K, MYINT shrA, MYINT shrB, MYINT shrC) {

	MYINT ite_idx = 0, ite_val = 0;
	for (MYINT k = 0; k < K; k++) {
		// MYINT b = getIntFeature(k);
		MYINT b = B[k * 1][0];
		b = b / shrB;

		MYINT idx = Aidx[ite_idx];
		while (idx != 0) {
			MYINT a = Aval[ite_val];
			a = a / shrA;

			MYINT c = a * b;
			c = c / shrC;

			C[idx - 1] += c;

			ite_idx++;
			ite_val++;

			idx = Aidx[ite_idx];
		}
		ite_idx++;
	}

	return;
}

// C = A <*> B
void MulCir(MYINT *A, MYINT *B, MYINT *C, MYINT I, MYINT J, MYINT shrA, MYINT shrB) {
	for (MYINT i = 0; i < I; i++) {
		for (MYINT j = 0; j < J; j++) {
			MYINT a = A[i * J + j];
			MYINT b = B[i * J + j];

			a = a / shrA;
			b = b / shrB;

			C[i * J + j] = a * b;
		}
	}
	return;
}

// A = tanh(A)
void TanH(MYINT *A, MYINT I, MYINT J, MYINT tanh_limit) {
	for (MYINT i = 0; i < I; i++) {
		for (MYINT j = 0; j < J; j++) {
			MYINT x = A[i * J + j], y;

			if (x >= tanh_limit)
				y = tanh_limit;
			else if (x <= -tanh_limit)
				y = -tanh_limit;
			else
				y = x;

			A[i * J + j] = y;
		}
	}
	return;
}

// index = argmax(A)
void ArgMax(MYINT *A, MYINT I, MYINT J, MYINT *index) {

	MYINT max = A[0], maxIndex = 0;
	MYINT counter = 0;
	for (MYINT i = 0; i < I; i++) {
		for (MYINT j = 0; j < J; j++) {
			MYINT x = A[i * J + j];

			if (max < x) {
				maxIndex = counter;
				max = x;
			}

			counter++;
		}
	}

	*index = maxIndex;

	return;
}

// A = A^T
void Transpose(MYINT *A, MYINT *B, MYINT I, MYINT J) {
	for (MYINT i = 0; i < I; i++) {
		for (MYINT j = 0; j < J; j++) {
			B[i * J + j] = A[j * I + i];
		}
	}
	return;
}

// C = a * B
void ScalarMul(MYINT *A, MYINT *B, MYINT *C, MYINT I, MYINT J, MYINT shrA, MYINT shrB) {

	MYINT a = *A;
	a = a / shrA;

	for (MYINT i = 0; i < I; i++) {
		for (MYINT j = 0; j < J; j++) {
			MYINT b = B[i * J + j];
			b = b / shrB;

			C[i * J + j] = a * b;
		}
	}

	return;
}

// C = A # B
// A[N][H][W][CI], B[HF][WF][CI][CO], C[N][H][W][CO]
void Conv(MYINT *A, const MYINT *B, MYINT *C, MYINT *tmp, MYINT N, MYINT H, MYINT W, MYINT CI, MYINT HF, MYINT WF, MYINT CO, MYINT shrA, MYINT shrB, MYINT H1, MYINT H2) {
	MYINT padH = (HF - 1) / 2;
	MYINT padW = (WF - 1) / 2;

	for (MYINT n = 0; n < N; n++) {
		for (MYINT h = 0; h < H; h++) {
			for (MYINT w = 0; w < W; w++) {
				for (MYINT co = 0; co < CO; co++) {

					MYINT counter = 0;
					for (MYINT hf = 0; hf < HF; hf++) {
						for (MYINT wf = 0; wf < WF; wf++) {
							for (MYINT ci = 0; ci < CI; ci++) {
								MYINT a = (((((h + hf) < padH) || ((h + hf) >= (H + padH))) || (((w + wf) < padW) || ((w + wf) >= (W + padW)))) ? 0 : A[n * H * W * CI + ((h + hf) - padH) * W * CI + ((w + wf) - padW) * CI + ci]);
								a = a / shrA;

								MYINT b = B[hf * WF * CI * CO + wf * CI * CO + ci * CO + co];
								b = b / shrB;

								tmp[counter] = a * b;
								counter++;
							}
						}
					}

					MYINT totalEle = HF * WF * CI;
					MYINT count = HF * WF * CI, depth = 0;
					bool shr = true;

					while (depth < (H1 + H2)) {
						if (depth >= H1)
							shr = false;

						for (MYINT p = 0; p < (totalEle / 2 + 1); p++) {
							MYINT sum;
							if (p < (count >> 1))
								sum = tmp[2 * p] + tmp[(2 * p) + 1];
							else if ((p == (count >> 1)) && ((count & 1) == 1))
								sum = tmp[2 * p];
							else
								sum = 0;

							if (shr)
								tmp[p] = sum / 2;
							else
								tmp[p] = sum;
						}
						count = (count + 1) >> 1;

						depth++;
					}

					C[n * H * W * CO + h * W * CO + w * CO + co] = tmp[0];
				}
			}
		}
	}

	return;
}

// A = A <+> B
// A[N][H][W][C], B[C]
void AddOrSubCir4D(MYINT *A, const MYINT *B, MYINT N, MYINT H, MYINT W, MYINT C, MYINT shrA, MYINT shrB, MYINT shrC, bool add) {

	for (MYINT n = 0; n < N; n++) {
		for (MYINT h = 0; h < H; h++) {
			for (MYINT w = 0; w < W; w++) {
				for (MYINT c = 0; c < C; c++) {
					MYINT a = A[n * H * W * C + h * W * C + w * C + c];
					a = a / shrA;

					MYINT b = B[c];
					b = b / shrB;

					MYINT res;
					if (add)
						res = a + b;
					else
						res = a - b;

					res = res / shrC;

					A[n * H * W * C + h * W * C + w * C + c] = res;
				}
			}
		}
	}

	return;
}

// A = A <+> B
// A[N][H][W][C], B[C]
void AddOrSubCir2D(MYINT *A, const MYINT *B, MYINT H, MYINT W, MYINT shrA, MYINT shrB, MYINT shrC, bool add) {

	for (MYINT h = 0; h < H; h++) {
		for (MYINT w = 0; w < W; w++) {
			MYINT a = A[h * W + w];
			a = a / shrA;

			MYINT b = B[w];
			b = b / shrB;

			MYINT res;
			if (add)
				res = a + b;
			else
				res = a - b;

			res = res / shrC;

			A[h * W + w] = res;
		}
	}

	return;
}

// A = relu(A)
// A[N][H][W][C]
void Relu4D(MYINT *A, MYINT N, MYINT H, MYINT W, MYINT C) {

	for (MYINT n = 0; n < N; n++) {
		for (MYINT h = 0; h < H; h++) {
			for (MYINT w = 0; w < W; w++) {
				for (MYINT c = 0; c < C; c++) {
					MYINT a = A[n * H * W * C + h * W * C + w * C + c];
					if (a < 0)
						a = 0;

					A[n * H * W * C + h * W * C + w * C + c] = a;
				}
			}
		}
	}

	return;
}

// A = relu(A)
// A[N][H][W][C]
void Relu2D(MYINT *A, MYINT H, MYINT W) {

	for (MYINT h = 0; h < H; h++) {
		for (MYINT w = 0; w < W; w++) {
			MYINT a = A[h * W + w];
			if (a < 0)
				a = 0;

			A[h * W + w] = a;
		}
	}

	return;
}

// B = maxpool(A)
// A[N][H][W][C], B[N][H][W][C]
void Maxpool(MYINT *A, MYINT *B, MYINT N, MYINT H, MYINT W, MYINT C, MYINT stride) {
	MYINT HO = H / stride;
	MYINT WO = W / stride;

	for (MYINT n = 0; n < N; n++) {
		for (MYINT ho = 0; ho < HO; ho++) {
			for (MYINT wo = 0; wo < WO; wo++) {
				for (MYINT c = 0; c < C; c++) {

					MYINT max = A[n * H * W * C + (stride * ho) * W * C + (stride * wo) * C + c];
					for (MYINT hs = 0; hs < stride; hs++) {
						for (MYINT ws = 0; ws < stride; ws++) {
							MYINT a = A[n * H * W * C + ((stride * ho) + hs) * W * C + ((stride * wo) + ws) * C + c];
							if (a > max)
								max = a;
						}
					}

					B[n * HO * WO * C + ho * WO * C + wo * C + c] = max;
				}
			}
		}
	}

	return;
}
