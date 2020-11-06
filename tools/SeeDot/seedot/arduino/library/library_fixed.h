// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

//#define SATURATE
//#define FASTAPPROX
#define FLOATEXP

#include <Arduino.h>

#include "config.h"
#include "predict.h"

template<class TypeA>
inline TypeA Saturate(int32_t inp) {
	return (TypeA)inp;
}

template<>
inline int16_t Saturate(int32_t inp) {
#ifdef SATURATE
	return (int16_t)((inp > 32767 ? 32767 : inp) < -32768 ? -32768 : inp);
#else
	return (int16_t)inp;
#endif
}

template<>
inline int8_t Saturate(int32_t inp) {
#ifdef SATURATE
	return (int8_t)((inp > 127 ? 127 : inp) < -128 ? -128 : inp);
#else
	return (int8_t)inp;
#endif
}

template<class T, class U>
bool isSame() {
	return false;
}

template<>
bool isSame<int8_t, int8_t>() {
	return true;
}

template<>
bool isSame<int16_t, int16_t>() {
	return true;
}

template<>
bool isSame<int32_t, int32_t>() {
	return true;
}


// C = A + B
inline __attribute__((always_inline)) void MatAddNN(MYINT* A, MYINT* B, MYINT* C, MYINT I, MYINT J, MYINT shrA, MYINT shrB, MYINT shrC) {
	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
			MYINT a = A[i * J + j];
			MYINT b = B[i * J + j];

			a = a / shrA;
			b = b / shrB;

			MYINT c = Saturate<MYINT>(a / shrC + b / shrC);

			C[i * J + j] = c;
		}
	}
	return;
}
// C = A + B
template<class TypeA, class TypeB, class TypeTemp, class TypeC>
inline __attribute__((always_inline)) void MatAddNN(TypeA* A, TypeB* B, TypeC* C, MYINT I, MYINT J, MYINT shrA, MYINT shrB, MYINT shrC, MYINT demote) {
	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
			TypeTemp a = (TypeTemp)A[i * J + j];
			TypeTemp b = (TypeTemp)B[i * J + j];

			a = a / shrA;
			b = b / shrB;

			TypeTemp c = a / shrC + b / shrC;

			C[i * J + j] = Saturate<TypeC>(c / demote);
		}
	}
	return;
}

// C = A + B
inline __attribute__((always_inline)) void MatAddCN(const MYINT* A, MYINT* B, MYINT* C, MYINT I, MYINT J, MYINT shrA, MYINT shrB, MYINT shrC) {
	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
			#if defined(INT8)
			MYINT a = ((MYINT) pgm_read_byte_near(&A[i * J + j]));
			#elif defined(INT16)
			MYINT a = ((MYINT) pgm_read_word_near(&A[i * J + j]));
			#else
			MYINT a = ((MYINT) pgm_read_dword_near(&A[i * J + j]));
			#endif

			MYINT b = B[i * J + j];

			a = a / shrA;
			b = b / shrB;

			MYINT c = Saturate<MYINT>(a / shrC + b / shrC);

			C[i * J + j] = c;
		}
	}
	return;
}
// C = A + B
template<class TypeA, class TypeB, class TypeTemp, class TypeC>
inline __attribute__((always_inline)) void MatAddCN(const TypeA* A, TypeB* B, TypeC* C, MYINT I, MYINT J, MYINT shrA, MYINT shrB, MYINT shrC, MYINT demote) {
	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
			TypeTemp a;
			if (isSame<TypeA, int8_t>()) {
				a = (TypeA)pgm_read_byte_near(&A[i * J + j]);
			}
			else if (isSame<TypeA, int16_t>()) {
				a = (TypeA)pgm_read_word_near(&A[i * J + j]);
			}
			else if (isSame<TypeA, int32_t>()) {
				a = (TypeA)pgm_read_dword_near(&A[i * J + j]);
			}
			
			TypeTemp b = (TypeTemp)B[i * J + j];

			a = a / shrA;
			b = b / shrB;

			TypeTemp c = a / shrC + b / shrC;

			C[i * J + j] = Saturate<TypeC>(c / demote);
		}
	}
	return;
}

// C = A + B
inline __attribute__((always_inline)) void MatAddNC(MYINT* A, const MYINT* B, MYINT* C, MYINT I, MYINT J, MYINT shrA, MYINT shrB, MYINT shrC) {
	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
			MYINT a = A[i * J + j];

			#if defined(INT8)
			MYINT b = ((MYINT) pgm_read_byte_near(&B[i * J + j]));
			#elif defined(INT16)
			MYINT b = ((MYINT) pgm_read_word_near(&B[i * J + j]));
			#else
			MYINT b = ((MYINT) pgm_read_dword_near(&B[i * J + j]));
			#endif

			a = a / shrA;
			b = b / shrB;

			MYINT c = Saturate<MYINT>(a / shrC + b / shrC);

			C[i * J + j] = c;
		}
	}
	return;
}
// C = A + B
template<class TypeA, class TypeB, class TypeTemp, class TypeC>
inline __attribute__((always_inline)) void MatAddNC(TypeA* A, const TypeB* B, TypeC* C, MYINT I, MYINT J, MYINT shrA, MYINT shrB, MYINT shrC, MYINT demote) {
	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {

			TypeTemp a = (TypeTemp)A[i * J + j];

			TypeTemp b;
			if (isSame<TypeB, int8_t>()) {
				b = (TypeB)pgm_read_byte_near(&B[i * J + j]);
			}
			else if (isSame<TypeB, int16_t>()) {
				b = (TypeB)pgm_read_word_near(&B[i * J + j]);
			}
			else if (isSame<TypeB, int32_t>()) {
				b = (TypeB)pgm_read_dword_near(&B[i * J + j]);
			}

			a = a / shrA;
			b = b / shrB;

			TypeTemp c = a / shrC + b / shrC;

			C[i * J + j] = Saturate<TypeC>(c / demote);
		}
	}
	return;
}

// C = A + B
inline __attribute__((always_inline)) void MatAddCC(const MYINT* A, const MYINT* B, MYINT* C, MYINT I, MYINT J, MYINT shrA, MYINT shrB, MYINT shrC) {
	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
			#if defined(INT8)
			MYINT a = ((MYINT) pgm_read_byte_near(&A[i * J + j]));
			#elif defined(INT16)
			MYINT a = ((MYINT) pgm_read_word_near(&A[i * J + j]));
			#else
			MYINT a = ((MYINT) pgm_read_dword_near(&A[i * J + j]));
			#endif
			
			#if defined(INT8)
			MYINT b = ((MYINT) pgm_read_byte_near(&B[i * J + j]));
			#elif defined(INT16)
			MYINT b = ((MYINT) pgm_read_word_near(&B[i * J + j]));
			#else
			MYINT b = ((MYINT) pgm_read_dword_near(&B[i * J + j]));
			#endif

			a = a / shrA;
			b = b / shrB;

			MYINT c = Saturate<MYINT>(a / shrC + b / shrC);

			C[i * J + j] = c;
		}
	}
	return;
}
// C = A + B
template<class TypeA, class TypeB, class TypeTemp, class TypeC>
inline __attribute__((always_inline)) void MatAddCC(const TypeA* A, const TypeB* B, TypeC* C, MYINT I, MYINT J, MYINT shrA, MYINT shrB, MYINT shrC, MYINT demote) {
	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {

			TypeTemp a;
			if (isSame<TypeA, int8_t>()) {
				a = (TypeA)pgm_read_byte_near(&A[i * J + j]);
			}
			else if (isSame<TypeA, int16_t>()) {
				a = (TypeA)pgm_read_word_near(&A[i * J + j]);
			}
			else if (isSame<TypeA, int32_t>()) {
				a = (TypeA)pgm_read_dword_near(&A[i * J + j]);
			}

			TypeTemp b;
			if (isSame<TypeB, int8_t>()) {
				b = (TypeB)pgm_read_byte_near(&B[i * J + j]);
			}
			else if (isSame<TypeB, int16_t>()) {
				b = (TypeB)pgm_read_word_near(&B[i * J + j]);
			}
			else if (isSame<TypeB, int32_t>()) {
				b = (TypeB)pgm_read_dword_near(&B[i * J + j]);
			}

			a = a / shrA;
			b = b / shrB;

			TypeTemp c = a / shrC + b / shrC;

			C[i * J + j] = Saturate<TypeC>(c / demote);
	}
}
	return;
}

// C = a + B
inline __attribute__((always_inline)) void MatAddBroadCastA(MYINT* A, MYINT* B, MYINT* C, MYINT I, MYINT J, MYINT shrA, MYINT shrB, MYINT shrC) {
	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
			MYINT a = *A;
			MYINT b = B[i * J + j];

			a = a / shrA;
			b = b / shrB;

			MYINT c = Saturate<MYINT>(a / shrC + b / shrC);

			C[i * J + j] = c;
		}
	}
	return;
}
// C = a + B
template<class TypeA, class TypeB, class TypeTemp, class TypeC>
inline __attribute__((always_inline)) void MatAddBroadCastA(TypeA* A, TypeB* B, TypeC* C, MYINT I, MYINT J, MYINT shrA, MYINT shrB, MYINT shrC, MYINT demote) {
	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
			TypeTemp a = (TypeTemp)*A;
			TypeTemp b = (TypeTemp)B[i * J + j];

			a = a / shrA;
			b = b / shrB;

			TypeTemp c = a / shrC + b / shrC;

			C[i * J + j] = Saturate<TypeC>(c / demote);
		}
	}
	return;
}

// C = A + b
inline __attribute__((always_inline)) void MatAddBroadCastB(MYINT* A, MYINT* B, MYINT* C, MYINT I, MYINT J, MYINT shrA, MYINT shrB, MYINT shrC) {
	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
			MYINT a = A[i * J + j];
			MYINT b = *B;

			a = a / shrA;
			b = b / shrB;

			MYINT c = Saturate<MYINT>(a / shrC + b / shrC);

			C[i * J + j] = c;
		}
	}
	return;
}
// C = A + b
template<class TypeA, class TypeB, class TypeTemp, class TypeC>
inline __attribute__((always_inline)) void MatAddBroadCastB(TypeA* A, TypeB* B, TypeC* C, MYINT I, MYINT J, MYINT shrA, MYINT shrB, MYINT shrC, MYINT demote) {
	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
			TypeTemp a = (TypeTemp)A[i * J + j];
			TypeTemp b = (TypeTemp)*B;

			a = a / shrA;
			b = b / shrB;

			TypeTemp c = a / shrC + b / shrC;

			C[i * J + j] = Saturate<TypeC>(c / demote);
		}
	}
	return;
}

// C = A - B
// TODO: shrB is int32_t because in 8-bit/16-bit code, shrB is usually very high and int8_t/int16_t will overflow.
inline __attribute__((always_inline)) void MatSub(MYINT* A, const MYINT* B, MYINT* C, MYINT I, MYINT J, MYINT shrA, int32_t shrB, MYINT shrC) {
	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
			MYINT a = A[i * J + j];
			
			#if defined(INT8)
			MYINT b = ((MYINT) pgm_read_byte_near(&B[i * J + j]));
			#elif defined(INT16)
			MYINT b = ((MYINT) pgm_read_word_near(&B[i * J + j]));
			#else
			MYINT b = ((MYINT) pgm_read_dword_near(&B[i * J + j]));
			#endif

			a = a / shrA;
			b = b / shrB;

			MYINT c = Saturate<MYINT>(a / shrC - b / shrC);

			C[i * J + j] = c;
		}
	}
	return;
}
// C = A - B
template<class TypeA, class TypeB, class TypeTemp, class TypeC>
// TODO: shrB is int32_t because in 8-bit/16-bit code, shrB is usually very high and int8_t/int16_t will overflow.
inline __attribute__((always_inline)) void MatSub(TypeA* A, const TypeB* B, TypeC* C, MYINT I, MYINT J, MYINT shrA, int32_t shrB, MYINT shrC, MYINT demote) {
	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
			TypeTemp a = (TypeTemp)A[i * J + j];
			
			TypeTemp b;
			if (isSame<TypeB, int8_t>()) {
				b = (TypeB)pgm_read_byte_near(&B[i * J + j]);
			}
			else if (isSame<TypeB, int16_t>()) {
				b = (TypeB)pgm_read_word_near(&B[i * J + j]);
			}
			else if (isSame<TypeB, int32_t>()) {
				b = (TypeB)pgm_read_dword_near(&B[i * J + j]);
			}

			a = a / shrA;
			b = b / shrB;

			TypeTemp c = a / shrC - b / shrC;

			C[i * J + j] = Saturate<TypeC>(c / demote);
		}
	}
	return;
}

// C = a - B
// TODO: shrB is int32_t because in 8-bit/16-bit code, shrB is usually very high and int8_t/int16_t will overflow.
inline __attribute__((always_inline)) void MatSubBroadCastA(MYINT* A, MYINT* B, MYINT* C, MYINT I, MYINT J, MYINT shrA, int32_t shrB, MYINT shrC) {
	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
			MYINT a = *A;
			MYINT b = B[i * J + j];

			a = a / shrA;
			b = b / shrB;

			MYINT c = Saturate<MYINT>(a / shrC - b / shrC);

			C[i * J + j] = c;
		}
	}
	return;
}
// C = a - B
// TODO: shrB is int32_t because in 8-bit/16-bit code, shrB is usually very high and int8_t/int16_t will overflow.
template<class TypeA, class TypeB, class TypeTemp, class TypeC>
inline __attribute__((always_inline)) void MatSubBroadCastA(TypeA* A, TypeB* B, TypeC* C, MYINT I, MYINT J, MYINT shrA, MYINT shrB, MYINT shrC, MYINT demote) {
	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
			TypeTemp a = (TypeTemp)*A;
			TypeTemp b = (TypeTemp)B[i * J + j];

			a = a / shrA;
			b = b / shrB;

			TypeTemp c = a / shrC - b / shrC;

			C[i * J + j] = Saturate<TypeC>(c / demote);
		}
	}
	return;
}

// C = A - b
// TODO: shrB is int32_t because in 8-bit/16-bit code, shrB is usually very high and int8_t/int16_t will overflow.
inline __attribute__((always_inline)) void MatSubBroadCastB(MYINT* A, MYINT* B, MYINT* C, MYINT I, MYINT J, MYINT shrA, int32_t shrB, MYINT shrC) {
	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
			MYINT a = A[i * J + j];
			MYINT b = *B;

			a = a / shrA;
			b = b / shrB;

			MYINT c = Saturate<MYINT>(a / shrC - b / shrC);

			C[i * J + j] = c;
		}
	}
	return;
}
// C = A - b
// TODO: shrB is int32_t because in 8-bit/16-bit code, shrB is usually very high and int8_t/int16_t will overflow.
template<class TypeA, class TypeB, class TypeTemp, class TypeC>
inline __attribute__((always_inline)) void MatSubBroadCastB(TypeA* A, TypeB* B, TypeC* C, MYINT I, MYINT J, MYINT shrA, MYINT shrB, MYINT shrC, MYINT demote) {
	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
			TypeTemp a = (TypeTemp)A[i * J + j];
			TypeTemp b = (TypeTemp)*B;

			a = a / shrA;
			b = b / shrB;

			TypeTemp c = a / shrC - b / shrC;

			C[i * J + j] = Saturate<TypeC>(c / demote);
		}
	}
	return;
}

// C = A * B
inline __attribute__((always_inline)) void MatMulNN(MYINT *A, MYINT *B, MYINT *C, MYINT *tmp, MYINT I, MYINT K, MYINT J, MYINT shrA, MYINT shrB, MYINT H1, MYINT H2) {

	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
			for (MYITE k = 0; k < K; k++) {
				MYINT a = A[i * K + k];
				MYINT b = B[k * J + j];
#ifdef FASTAPPROX
				a = a / shrA;
				b = b / shrB;

				tmp[k] = a * b;
#else
				int64_t prod = ((int64_t)a * (int64_t)b);
				tmp[k] = Saturate<MYINT>((prod / ((int64_t)shrB * (int64_t)shrA)));
#endif
			}

			MYITE count = K, depth = 0;
			bool shr = true;

			while (depth < (H1 + H2)) {
				if (depth >= H1)
					shr = false;

				for (MYITE p = 0; p < (K / 2 + 1); p++) {
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
template<class TypeA, class TypeB, class TypeTemp, class TypeC>
inline __attribute__((always_inline)) void MatMulNN(TypeA* A, TypeB* B, TypeC* C, TypeTemp* tmp, MYINT I, MYINT K, MYINT J, MYINT shrA, MYINT shrB, MYINT H1, MYINT H2, MYINT demote) {
	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
			for (MYITE k = 0; k < K; k++) {
				TypeTemp a = (TypeTemp)A[i * K + k];
				TypeTemp b = (TypeTemp)B[k * J + j];

				TypeTemp prod = a * b;

				tmp[k] = prod;
			}

			MYITE count = K, depth = 0;
			bool shr = true;

			while (depth < (H1 + H2)) {
				if (depth >= H1)
					shr = false;

				for (MYITE p = 0; p < (K / 2 + 1); p++) {
					TypeTemp sum;
					if (p < (count >> 1)) {
						if (shr)
							sum = tmp[2 * p] / 2 + tmp[(2 * p) + 1] / 2;
						else
							sum = tmp[2 * p] + tmp[(2 * p) + 1];
					}
					else if ((p == (count >> 1)) && ((count & 1) == 1)) {
						if (shr)
							sum = tmp[2 * p] / 2;
						else
							sum = tmp[2 * p];
					}
					else
						sum = 0;

					tmp[p] = sum;
				}
				count = (count + 1) >> 1;

				depth++;
			}

			C[i * J + j] = Saturate<TypeC>(((tmp[0] / shrA) / shrB) / demote);
		}
	}
	return;
}

// C = A * B
inline __attribute__((always_inline)) void MatMulCN(const MYINT *A, MYINT *B, MYINT *C, MYINT *tmp, MYINT I, MYINT K, MYINT J, MYINT shrA, MYINT shrB, MYINT H1, MYINT H2) {

	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
			for (MYITE k = 0; k < K; k++) {
				#if defined(INT8)
				MYINT a = ((MYINT) pgm_read_byte_near(&A[i * K + k]));
				#elif defined(INT16)
				MYINT a = ((MYINT) pgm_read_word_near(&A[i * K + k]));
				#else
				MYINT a = ((MYINT) pgm_read_dword_near(&A[i * K + k]));
				#endif

				MYINT b = B[k * J + j];

#ifdef FASTAPPROX
				a = a / shrA;
				b = b / shrB;

				tmp[k] = a * b;
#else
				int64_t prod = ((int64_t)a * (int64_t)b);
				tmp[k] = Saturate<MYINT>((prod / ((int64_t)shrB * (int64_t)shrA)));
#endif
			}

			MYITE count = K, depth = 0;
			bool shr = true;

			while (depth < (H1 + H2)) {
				if (depth >= H1)
					shr = false;

				for (MYITE p = 0; p < (K / 2 + 1); p++) {
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
template<class TypeA, class TypeB, class TypeTemp, class TypeC>
inline __attribute__((always_inline)) void MatMulCN(const TypeA* A, TypeB* B, TypeC* C, TypeTemp* tmp, MYINT I, MYINT K, MYINT J, MYINT shrA, MYINT shrB, MYINT H1, MYINT H2, MYINT demote) {
	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
			for (MYITE k = 0; k < K; k++) {
				TypeTemp a;
				if (isSame<TypeA, int8_t>()) {
					a = (TypeA)pgm_read_byte_near(&A[i * K + k]);
				}
				else if (isSame<TypeA, int16_t>()) {
					a = (TypeA)pgm_read_word_near(&A[i * K + k]);
				}
				else if (isSame<TypeA, int32_t>()) {
					a = (TypeA)pgm_read_dword_near(&A[i * K + k]);
				}

				TypeTemp b = (TypeTemp)B[k * J + j];

				TypeTemp prod = a * b;

				tmp[k] = prod;
		}

			MYITE count = K, depth = 0;
			bool shr = true;

			while (depth < (H1 + H2)) {
				if (depth >= H1)
					shr = false;

				for (MYITE p = 0; p < (K / 2 + 1); p++) {
					TypeTemp sum;
					if (p < (count >> 1)) {
						if (shr)
							sum = tmp[2 * p] / 2 + tmp[(2 * p) + 1] / 2;
						else
							sum = tmp[2 * p] + tmp[(2 * p) + 1];
					}
					else if ((p == (count >> 1)) && ((count & 1) == 1)) {
						if (shr)
							sum = tmp[2 * p] / 2;
						else
							sum = tmp[2 * p];
					}
					else
						sum = 0;

					tmp[p] = sum;
				}
				count = (count + 1) >> 1;

				depth++;
			}

			C[i * J + j] = Saturate<TypeC>(((tmp[0] / shrA) / shrB) / demote);
	}
}
	return;
}

// C = A * B
inline __attribute__((always_inline)) void MatMulNC(MYINT *A, const MYINT *B, MYINT *C, MYINT *tmp, MYINT I, MYINT K, MYINT J, MYINT shrA, MYINT shrB, MYINT H1, MYINT H2) {

	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
			for (MYITE k = 0; k < K; k++) {
				MYINT a = A[i * K + k];

				#if defined(INT8)
				MYINT b = ((MYINT) pgm_read_byte_near(&B[k * J + j]));
				#elif defined(INT16)
				MYINT b = ((MYINT) pgm_read_word_near(&B[k * J + j]));
				#else
				MYINT b = ((MYINT) pgm_read_dword_near(&B[k * J + j]));
				#endif

#ifdef FASTAPPROX
				a = a / shrA;
				b = b / shrB;

				tmp[k] = a * b;
#else
				int64_t prod = ((int64_t)a * (int64_t)b);
				tmp[k] = Saturate<MYINT>((prod / ((int64_t)shrB * (int64_t)shrA)));
#endif
			}

			MYITE count = K, depth = 0;
			bool shr = true;

			while (depth < (H1 + H2)) {
				if (depth >= H1)
					shr = false;

				for (MYITE p = 0; p < (K / 2 + 1); p++) {
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
template<class TypeA, class TypeB, class TypeTemp, class TypeC>
inline __attribute__((always_inline)) void MatMulNC(TypeA* A, const TypeB* B, TypeC* C, TypeTemp* tmp, MYINT I, MYINT K, MYINT J, MYINT shrA, MYINT shrB, MYINT H1, MYINT H2, MYINT demote) {
	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
			for (MYITE k = 0; k < K; k++) {
				TypeTemp a = (TypeTemp)A[i * K + k];
				
				TypeTemp b;
				if (isSame<TypeB, int8_t>()) {
					b = (TypeB)pgm_read_byte_near(&B[k * J + j]);
				}
				else if (isSame<TypeB, int16_t>()) {
					b = (TypeB)pgm_read_word_near(&B[k * J + j]);
				}
				else if (isSame<TypeB, int32_t>()) {
					b = (TypeB)pgm_read_dword_near(&B[k * J + j]);
				}

				TypeTemp prod = a * b;

				tmp[k] = prod;
		}

			MYITE count = K, depth = 0;
			bool shr = true;

			while (depth < (H1 + H2)) {
				if (depth >= H1)
					shr = false;

				for (MYITE p = 0; p < (K / 2 + 1); p++) {
					TypeTemp sum;
					if (p < (count >> 1)) {
						if (shr)
							sum = tmp[2 * p] / 2 + tmp[(2 * p) + 1] / 2;
						else
							sum = tmp[2 * p] + tmp[(2 * p) + 1];
					}
					else if ((p == (count >> 1)) && ((count & 1) == 1)) {
						if (shr)
							sum = tmp[2 * p] / 2;
						else
							sum = tmp[2 * p];
					}
					else
						sum = 0;

					tmp[p] = sum;
				}
				count = (count + 1) >> 1;

				depth++;
			}

			C[i * J + j] = Saturate<TypeC>(((tmp[0] / shrA) / shrB) / demote);
	}
}
	return;
}

// C = A * B
inline __attribute__((always_inline)) void MatMulCC(const MYINT *A, const MYINT *B, MYINT *C, MYINT *tmp, MYINT I, MYINT K, MYINT J, MYINT shrA, MYINT shrB, MYINT H1, MYINT H2) {

	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
			for (MYITE k = 0; k < K; k++) {
				#if defined(INT8)
				MYINT a = ((MYINT) pgm_read_byte_near(&A[i * K + k]));
				#elif defined(INT16)
				MYINT a = ((MYINT) pgm_read_word_near(&A[i * K + k]));
				#else
				MYINT a = ((MYINT) pgm_read_dword_near(&A[i * K + k]));
				#endif

				#if defined(INT8)
				MYINT b = ((MYINT) pgm_read_byte_near(&B[k * J + j]));
				#elif defined(INT16)
				MYINT b = ((MYINT) pgm_read_word_near(&B[k * J + j]));
				#else
				MYINT b = ((MYINT) pgm_read_dword_near(&B[k * J + j]));
				#endif

#ifdef FASTAPPROX
				a = a / shrA;
				b = b / shrB;

				tmp[k] = a * b;
#else
				int64_t prod = ((int64_t)a * (int64_t)b);
				tmp[k] = Saturate<MYINT>((prod / ((int64_t)shrB * (int64_t)shrA)));
#endif
			}

			MYITE count = K, depth = 0;
			bool shr = true;

			while (depth < (H1 + H2)) {
				if (depth >= H1)
					shr = false;

				for (MYITE p = 0; p < (K / 2 + 1); p++) {
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
template<class TypeA, class TypeB, class TypeTemp, class TypeC>
inline __attribute__((always_inline)) void MatMulCC(const TypeA* A, const TypeB* B, TypeC* C, TypeTemp* tmp, MYINT I, MYINT K, MYINT J, MYINT shrA, MYINT shrB, MYINT H1, MYINT H2, MYINT demote) {
	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
			for (MYITE k = 0; k < K; k++) {
				TypeTemp a;
				if (isSame<TypeA, int8_t>()) {
					a = (TypeA)pgm_read_byte_near(&A[i * K + k]);
			}
				else if (isSame<TypeA, int16_t>()) {
					a = (TypeA)pgm_read_word_near(&A[i * K + k]);
				}
				else if (isSame<TypeA, int32_t>()) {
					a = (TypeA)pgm_read_dword_near(&A[i * K + k]);
				}

				TypeTemp b;
				if (isSame<TypeB, int8_t>()) {
					b = (TypeB)pgm_read_byte_near(&B[k * J + j]);
				}
				else if (isSame<TypeB, int16_t>()) {
					b = (TypeB)pgm_read_word_near(&B[k * J + j]);
				}
				else if (isSame<TypeB, int32_t>()) {
					b = (TypeB)pgm_read_dword_near(&B[k * J + j]);
				}

				TypeTemp prod = a * b;

				tmp[k] = prod;
		}

			MYITE count = K, depth = 0;
			bool shr = true;

			while (depth < (H1 + H2)) {
				if (depth >= H1)
					shr = false;

				for (MYITE p = 0; p < (K / 2 + 1); p++) {
					TypeTemp sum;
					if (p < (count >> 1)) {
						if (shr)
							sum = tmp[2 * p] / 2 + tmp[(2 * p) + 1] / 2;
						else
							sum = tmp[2 * p] + tmp[(2 * p) + 1];
					}
					else if ((p == (count >> 1)) && ((count & 1) == 1)) {
						if (shr)
							sum = tmp[2 * p] / 2;
						else
							sum = tmp[2 * p];
					}
					else
						sum = 0;

					tmp[p] = sum;
				}
				count = (count + 1) >> 1;

				depth++;
			}

			C[i * J + j] = Saturate<TypeC>(((tmp[0] / shrA) / shrB) / demote);
	}
}
	return;
}

// C = A |*| B
// TODO: K is int16_t because K is usually very high and int8_t will overflow in 8-bit code.
inline __attribute__((always_inline)) void SparseMatMulX(const MYINT *Aidx, const MYINT *Aval, MYINT *C, int16_t K, MYINT shrA, MYINT shrB, MYINT shrC) {

	MYITE ite_idx = 0, ite_val = 0;
	for (MYITE k = 0; k < K; k++) {
		MYINT b = getIntFeature(k);
		//MYINT b = B[k * 1][0];
#ifdef FASTAPPROX
		b = b / shrB;
#endif

		#if defined(INT8)
		MYINT idx = ((MYINT) pgm_read_byte_near(&Aidx[ite_idx]));
		#elif defined(INT16)
		MYINT idx = ((MYINT) pgm_read_word_near(&Aidx[ite_idx]));
		#else
		MYINT idx = ((MYINT) pgm_read_dword_near(&Aidx[ite_idx]));
		#endif

		while (idx != 0) {
			#if defined(INT8)
			MYINT a = ((MYINT) pgm_read_byte_near(&Aval[ite_val]));
			#elif defined(INT16)
			MYINT a = ((MYINT) pgm_read_word_near(&Aval[ite_val]));
			#else
			MYINT a = ((MYINT) pgm_read_dword_near(&Aval[ite_val]));
			#endif
#ifdef FASTAPPROX
			a = a / shrA;

			MYINT c = a * b;
			c = c / shrC;
#else
			MYINT c = Saturate<MYINT>(((int64_t)a * (int64_t)b) / ((int64_t)shrC * (int64_t)shrA * (int64_t)shrB));
#endif

			C[idx - 1] += c;

			ite_idx++;
			ite_val++;

			#if defined(INT8)
			idx = ((MYINT) pgm_read_byte_near(&Aidx[ite_idx]));
			#elif defined(INT16)
			idx = ((MYINT) pgm_read_word_near(&Aidx[ite_idx]));
			#else
			idx = ((MYINT) pgm_read_dword_near(&Aidx[ite_idx]));
			#endif
		}
		ite_idx++;
	}

	return;
}
// C = A |*| B
// TODO: K is int16_t because K is usually very high and int8_t will overflow in 8-bit code.
inline __attribute__((always_inline)) void SparseMatMul(const MYINT *Aidx, const MYINT *Aval, MYINT *B, MYINT *C, int16_t K, MYINT shrA, MYINT shrB, MYINT shrC) {

	MYITE ite_idx = 0, ite_val = 0;
	for (MYITE k = 0; k < K; k++) {
		MYINT b = B[k];
#ifdef FASTAPPROX
		b = b / shrB;
#endif

		#if defined(INT8)
		MYINT idx = ((MYINT) pgm_read_byte_near(&Aidx[ite_idx]));
		#elif defined(INT16)
		MYINT idx = ((MYINT) pgm_read_word_near(&Aidx[ite_idx]));
		#else
		MYINT idx = ((MYINT) pgm_read_dword_near(&Aidx[ite_idx]));
		#endif

		while (idx != 0) {
			#if defined(INT8)
			MYINT a = ((MYINT) pgm_read_byte_near(&Aval[ite_val]));
			#elif defined(INT16)
			MYINT a = ((MYINT) pgm_read_word_near(&Aval[ite_val]));
			#else
			MYINT a = ((MYINT) pgm_read_dword_near(&Aval[ite_val]));
			#endif
#ifdef FASTAPPROX
			a = a / shrA;

			MYINT c = a * b;
			c = c / shrC;
#else
			MYINT c = Saturate<MYINT>(((int64_t)a * (int64_t)b) / ((int64_t)shrC * (int64_t)shrA * (int64_t)shrB));
#endif

			C[idx - 1] += c;

			ite_idx++;
			ite_val++;

			#if defined(INT8)
			idx = ((MYINT) pgm_read_byte_near(&Aidx[ite_idx]));
			#elif defined(INT16)
			idx = ((MYINT) pgm_read_word_near(&Aidx[ite_idx]));
			#else
			idx = ((MYINT) pgm_read_dword_near(&Aidx[ite_idx]));
			#endif
		}
		ite_idx++;
	}

	return;
}
// C = A |*| B
// TODO: K is int16_t because K is usually very high and int8_t will overflow in 8-bit code.
template<class TypeA, class TypeAidx, class TypeB, class TypeTemp, class TypeC>
inline __attribute__((always_inline)) void SparseMatMulX(const TypeAidx* Aidx, const TypeA* Aval, TypeC* C, int16_t K, MYINT shrA, MYINT shrB, MYINT shrC, MYINT demote) {

	MYITE ite_idx = 0, ite_val = 0;
	for (MYITE k = 0; k < K; k++) {
		TypeTemp b = (TypeTemp)getIntFeature(k);
		
		//b = b / shrB;

		TypeAidx idx;
		if (isSame<TypeAidx, int8_t>()) {
			idx = (TypeAidx)pgm_read_byte_near(&Aidx[ite_idx]);
		}
		else if (isSame<TypeAidx, int16_t>()) {
			idx = (TypeAidx)pgm_read_word_near(&Aidx[ite_idx]);
		}
		else if (isSame<TypeAidx, int32_t>()) {
			idx = (TypeAidx)pgm_read_dword_near(&Aidx[ite_idx]);
		}
		while (idx != 0) {
			TypeTemp a;
			if (isSame<TypeA, int8_t>()) {
				a = (TypeA)pgm_read_byte_near(&Aval[ite_val]);
			}
			else if (isSame<TypeTemp, int16_t>()) {
				a = (TypeA)pgm_read_word_near(&Aval[ite_val]);
			}
			else if (isSame<TypeTemp, int32_t>()) {
				a = (TypeA)pgm_read_dword_near(&Aval[ite_val]);
			}
			//a = a / shrA;
			TypeTemp c = (TypeTemp)(a * b);
			//c = c / shrC;

			C[idx - 1] += Saturate<TypeC>((((c / shrA) / shrB) / shrC) / demote);

			ite_idx++;
			ite_val++;

			if (isSame<TypeAidx, int8_t>()) {
				idx = (TypeAidx)pgm_read_byte_near(&Aidx[ite_idx]);
			}
			else if (isSame<TypeAidx, int16_t>()) {
				idx = (TypeAidx)pgm_read_word_near(&Aidx[ite_idx]);
			}
			else if (isSame<TypeAidx, int32_t>()) {
				idx = (TypeAidx)pgm_read_dword_near(&Aidx[ite_idx]);
			}
		}
		ite_idx++;
	}

	return;
}
// C = A |*| B
// TODO: K is int16_t because K is usually very high and int8_t will overflow in 8-bit code.
template<class TypeA, class TypeAidx, class TypeB, class TypeTemp, class TypeC>
inline __attribute__((always_inline)) void SparseMatMul(const TypeAidx* Aidx, const TypeA* Aval, TypeB* B, TypeC* C, int16_t K, MYINT shrA, MYINT shrB, MYINT shrC, MYINT demote) {

	MYITE ite_idx = 0, ite_val = 0;
	for (MYITE k = 0; k < K; k++) {
		TypeTemp b = (TypeTemp)B[k];
		
		//b = b / shrB;

		TypeAidx idx;
		if (isSame<TypeAidx, int8_t>()) {
			idx = (TypeAidx)pgm_read_byte_near(&Aidx[ite_idx]);
		}
		else if (isSame<TypeAidx, int16_t>()) {
			idx = (TypeAidx)pgm_read_word_near(&Aidx[ite_idx]);
		}
		else if (isSame<TypeAidx, int32_t>()) {
			idx = (TypeAidx)pgm_read_dword_near(&Aidx[ite_idx]);
		}
		while (idx != 0) {
			TypeTemp a;
			if (isSame<TypeA, int8_t>()) {
				a = (TypeA)pgm_read_byte_near(&Aval[ite_val]);
			}
			else if (isSame<TypeTemp, int16_t>()) {
				a = (TypeA)pgm_read_word_near(&Aval[ite_val]);
			}
			else if (isSame<TypeTemp, int32_t>()) {
				a = (TypeA)pgm_read_dword_near(&Aval[ite_val]);
			}
			//a = a / shrA;
			TypeTemp c = (TypeTemp)(a * b);
			//c = c / shrC;

			C[idx - 1] += Saturate<TypeC>((((c / shrA) / shrB) / shrC) / demote);

			ite_idx++;
			ite_val++;

			if (isSame<TypeAidx, int8_t>()) {
				idx = (TypeAidx)pgm_read_byte_near(&Aidx[ite_idx]);
			}
			else if (isSame<TypeAidx, int16_t>()) {
				idx = (TypeAidx)pgm_read_word_near(&Aidx[ite_idx]);
			}
			else if (isSame<TypeAidx, int32_t>()) {
				idx = (TypeAidx)pgm_read_dword_near(&Aidx[ite_idx]);
			}
		}
		ite_idx++;
	}

	return;
}


// C = A <*> B
inline __attribute__((always_inline)) void MulCir(MYINT *A, MYINT *B, MYINT *C, MYINT I, MYINT J, MYINT shrA, MYINT shrB) {
	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
			MYINT a = A[i * J + j];
			MYINT b = B[i * J + j];

#ifdef FASTAPPROX
			a = a / shrA;
			b = b / shrB;

			C[i * J + j] = a * b;
#else
			int64_t prod = ((int64_t)a * (int64_t)b);
			C[i * J + j] = Saturate<MYINT>(prod / ((int64_t)shrB * (int64_t)shrA));
#endif
		}
	}
	return;
}
// C = A <*> B
template<class TypeA, class TypeB, class TypeTemp, class TypeC>
inline __attribute__((always_inline)) void MulCir(TypeA* A, TypeB* B, TypeC* C, MYINT I, MYINT J, MYINT shrA, MYINT shrB, MYINT demote) {
	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
			TypeTemp a = (TypeTemp)A[i * J + j];
			TypeTemp b = (TypeTemp)B[i * J + j];

			TypeTemp prod = a * b;
			C[i * J + j] = Saturate<TypeC>(((prod / shrA) / shrB) / demote);
		}
	}
	return;
}

// index = argmax(A)
inline __attribute__((always_inline)) void ArgMax(MYINT *A, MYINT I, MYINT J, MYINT *index) {

	MYINT max = A[0];
	MYITE maxIndex = 0, counter = 0;
	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
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
// index = argmax(A)
template<class TypeA>
inline __attribute__((always_inline)) void ArgMax(TypeA* A, MYINT I, MYINT J, MYITE* index) {
	TypeA max = A[0];
	MYITE maxIndex = 0, counter = 0;
	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
			TypeA x = A[i * J + j];

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
inline __attribute__((always_inline)) void Transpose(MYINT *A, MYINT *B, MYINT I, MYINT J) {
	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
			B[i * J + j] = A[j * I + i];
		}
	}
	return;
}
// A = A^T
template<class TypeA>
inline __attribute__((always_inline)) void Transpose(TypeA* A, TypeA* B, MYINT I, MYINT J) {
	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
			B[i * J + j] = A[j * I + i];
		}
	}
	return;
}

// C = a * B
inline __attribute__((always_inline)) void ScalarMul(MYINT *A, MYINT *B, MYINT *C, MYINT I, MYINT J, MYINT shrA, MYINT shrB) {
	MYINT a = *A;
#ifdef FASTAPPROX
	a = a / shrA;
#endif

	for (MYITE i = 0; i < I; i++)
	{
		for (MYITE j = 0; j < J; j++)
		{
			MYINT b = B[i * J + j];

#ifdef FASTAPPROX
			b = b / shrB;

			C[i * J + j] = a * b;
#else
			int64_t prod = ((int64_t)a * (int64_t)b);
			C[i * J + j] = Saturate<MYINT>(prod / ((int64_t)shrA * (int64_t)shrB));
#endif
		}
	}

	return;
}
// C = a * B
template<class TypeA, class TypeB, class TypeTemp, class TypeC>
inline __attribute__((always_inline)) void ScalarMul(TypeA* A, TypeB* B, TypeC* C, MYINT I, MYINT J, MYINT shrA, MYINT shrB, int demote) {
	TypeTemp a = (TypeTemp)*A;
	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
			TypeTemp b = (TypeTemp)B[i * J + j];

			TypeTemp prod = a * b;
			C[i * J + j] = Saturate<TypeC>(((prod / shrA) / shrB) / demote);
		}
	}

	return;
}

// C = A # B
// A[N][H][W][CI], B[HF][WF][CI][CO], C[N][H][W][CO]
inline __attribute__((always_inline)) void Conv(MYINT *A, const MYINT *B, MYINT *C, MYINT *tmp, MYINT N, MYINT H, MYINT W, MYINT CI, MYINT HF, MYINT WF, MYINT CO, MYINT shrA, MYINT shrB, MYINT H1, MYINT H2) {
	MYITE padH = (HF - 1) / 2;
	MYITE padW = (WF - 1) / 2;

	for (MYITE n = 0; n < N; n++) {
		for (MYITE h = 0; h < H; h++) {
			for (MYITE w = 0; w < W; w++) {
				for (MYITE co = 0; co < CO; co++) {

					MYITE counter = 0;
					for (MYITE hf = 0; hf < HF; hf++) {
						for (MYITE wf = 0; wf < WF; wf++) {
							for (MYITE ci = 0; ci < CI; ci++) {
								MYINT a = (((((h + hf) < padH) || ((h + hf) >= (H + padH))) || (((w + wf) < padW) || ((w + wf) >= (W + padW)))) ? 0 : A[n * H * W * CI + ((h + hf) - padH) * W * CI + ((w + wf) - padW) * CI + ci]);
								a = a / shrA;

								#if defined(INT8)
								MYINT b = ((MYINT) pgm_read_byte_near(&B[hf * WF * CI * CO + wf * CI * CO + ci * CO + co]));
								#elif defined(INT16)
								MYINT b = ((MYINT) pgm_read_word_near(&B[hf * WF * CI * CO + wf * CI * CO + ci * CO + co]));
								#else
								MYINT b = ((MYINT) pgm_read_dword_near(&B[hf * WF * CI * CO + wf * CI * CO + ci * CO + co]));
								#endif
								b = b / shrB;

								tmp[counter] = a * b;
								counter++;
							}
						}
					}

					MYITE totalEle = HF * WF * CI;
					MYITE count = HF * WF * CI, depth = 0;
					bool shr = true;

					while (depth < (H1 + H2)) {
						if (depth >= H1)
							shr = false;

						for (MYITE p = 0; p < (totalEle / 2 + 1); p++) {
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
// C = A # B
// A[N][H][W][CI], B[HF][WF][CI][CO], C[N][H][W][CO]
template<class TypeA, class TypeB, class TypeTemp, class TypeC>
inline __attribute__((always_inline)) void Conv(TypeA* A, const TypeB* B, TypeC* C, TypeTemp* tmp, MYINT N, MYINT H, MYINT W, MYINT CI, MYINT HF, MYINT WF, MYINT CO, MYINT shrA, MYINT shrB, MYINT H1, MYINT H2, MYINT demote) {
	MYITE padH = (HF - 1) / 2;
	MYITE padW = (WF - 1) / 2;

	for (MYITE n = 0; n < N; n++) {
		for (MYITE h = 0; h < H; h++) {
			for (MYITE w = 0; w < W; w++) {
				for (MYITE co = 0; co < CO; co++) {

					MYITE counter = 0;
					for (MYITE hf = 0; hf < HF; hf++) {
						for (MYITE wf = 0; wf < WF; wf++) {
							for (MYITE ci = 0; ci < CI; ci++) {
								TypeTemp a = (TypeTemp)(((((h + hf) < padH) || ((h + hf) >= (H + padH))) || (((w + wf) < padW) || ((w + wf) >= (W + padW)))) ? 0 : A[n * H * W * CI + ((h + hf) - padH) * W * CI + ((w + wf) - padW) * CI + ci]);
								TypeTemp b;
								if (isSame<TypeB, int8_t>()) {
									b = (TypeB)pgm_read_byte_near(&B[hf * WF * CI * CO + wf * CI * CO + ci * CO + co]);
								}
								else if (isSame<TypeB, int16_t>()) {
									b = (TypeB)pgm_read_word_near(&B[hf * WF * CI * CO + wf * CI * CO + ci * CO + co]);
								}
								else if (isSame<TypeB, int32_t>()) {
									b = (TypeB)pgm_read_dword_near(&B[hf * WF * CI * CO + wf * CI * CO + ci * CO + co]);
								}
								tmp[counter] = a * b;
								counter++;
							}
						}
					}

					MYITE totalEle = HF * WF * CI;
					MYITE count = HF * WF * CI, depth = 0;
					bool shr = true;

					while (depth < (H1 + H2)) {
						if (depth >= H1)
							shr = false;

						for (MYITE p = 0; p < (totalEle / 2 + 1); p++) {
							TypeTemp sum;
							if (p < (count >> 1)) {
								if (shr)
									sum = tmp[2 * p] / 2 + tmp[(2 * p) + 1] / 2;
								else
									sum = tmp[2 * p] + tmp[(2 * p) + 1];
							}
							else if ((p == (count >> 1)) && ((count & 1) == 1)) {
								if (shr)
									sum = tmp[2 * p] / 2;
								else
									sum = tmp[2 * p];
							}
							else
								sum = 0;

							tmp[p] = sum;
						}
						count = (count + 1) >> 1;

						depth++;
					}

					C[n * H * W * CO + h * W * CO + w * CO + co] = Saturate<TypeC>(((tmp[0] / shrA) / shrB) / demote);
				}
			}
		}
	}
	return;
}


// A = A <+> B
// A[N][H][W][C], B[C]
inline __attribute__((always_inline)) void AddOrSubCir4D(MYINT *A, const MYINT *B, MYINT N, MYINT H, MYINT W, MYINT C, MYINT shrA, MYINT shrB, MYINT shrC, bool add) {

	for (MYITE n = 0; n < N; n++) {
		for (MYITE h = 0; h < H; h++) {
			for (MYITE w = 0; w < W; w++) {
				for (MYITE c = 0; c < C; c++) {
					MYINT a = A[n * H * W * C + h * W * C + w * C + c];
					a = a / shrA;

					#if defined(INT8)
					MYINT b = ((MYINT) pgm_read_byte_near(&B[c]));
					#elif defined(INT16)
					MYINT b = ((MYINT) pgm_read_word_near(&B[c]));
					#else
					MYINT b = ((MYINT) pgm_read_dword_near(&B[c]));
					#endif

					b = b / shrB;

					MYINT res;
					if (add)
						res = Saturate<MYINT>(a / shrC + b / shrC);
					else
						res = Saturate<MYINT>(a / shrC - b / shrC);

					A[n * H * W * C + h * W * C + w * C + c] = res;
				}
			}
		}
	}

	return;
}
// A = A <+> B
// A[N][H][W][C], B[C]
template<class TypeA, class TypeB, class TypeTemp>
inline __attribute__((always_inline)) void AddOrSubCir4D(TypeA* A, const TypeB* B, MYINT N, MYINT H, MYINT W, MYINT C, MYINT shrA, MYINT shrB, MYINT shrC, bool add) {
	for (MYITE n = 0; n < N; n++) {
		for (MYITE h = 0; h < H; h++) {
			for (MYITE w = 0; w < W; w++) {
				for (MYITE c = 0; c < C; c++) {
					TypeTemp a = (TypeTemp)A[n * H * W * C + h * W * C + w * C + c];
					a = a / shrA;

					TypeTemp b;
					if (isSame<TypeB, int8_t>()) {
						b = (TypeB)pgm_read_byte_near(&B[c]);
					}
					else if (isSame<TypeB, int16_t>()) {
						b = (TypeB)pgm_read_word_near(&B[c]);
					}
					else if (isSame<TypeB, int32_t>()) {
						b = (TypeB)pgm_read_dword_near(&B[c]);
					}

					b = b / shrB;

					TypeTemp res;
					if (add)
						res = a / shrC + b / shrC;
					else
						res = a / shrC - b / shrC;

					A[n * H * W * C + h * W * C + w * C + c] = Saturate<TypeA>(res);
				}
			}
		}
	}
	return;
}


// A = A <+> B
// A[N][H][W][C], B[C]
inline __attribute__((always_inline)) void AddOrSubCir2D(MYINT *A, const MYINT *B, MYINT H, MYINT W, MYINT shrA, MYINT shrB, MYINT shrC, bool add) {

	for (MYITE h = 0; h < H; h++) {
		for (MYITE w = 0; w < W; w++) {
			MYINT a = A[h * W + w];
			a = a / shrA;

			#if defined(INT8)
			MYINT b = ((MYINT) pgm_read_byte_near(&B[w]));
			#elif defined(INT16)
			MYINT b = ((MYINT) pgm_read_word_near(&B[w]));
			#else
			MYINT b = ((MYINT) pgm_read_dword_near(&B[w]));
			#endif

			b = b / shrB;

			MYINT res;
			if (add)
				res = Saturate<MYINT>(a / shrC + b / shrC);
			else
				res = Saturate<MYINT>(a / shrC - b / shrC);

			A[h * W + w] = res;
		}
	}

	return;
}
// A = A <+> B
// A[N][H][W][C], B[C]
template<class TypeA, class TypeB, class TypeTemp>
inline __attribute__((always_inline)) void AddOrSubCir2D(TypeA* A, const TypeB* B, MYINT H, MYINT W, MYINT shrA, MYINT shrB, MYINT shrC, bool add) {
	for (MYITE h = 0; h < H; h++) {
		for (MYITE w = 0; w < W; w++) {
			TypeTemp a = (TypeTemp)A[h * W + w];
			a = a / shrA;

			TypeTemp b;
			if (isSame<TypeB, int8_t>()) {
				b = (TypeB)pgm_read_byte_near(&B[w]);
			}
			else if (isSame<TypeB, int16_t>()) {
				b = (TypeB)pgm_read_word_near(&B[w]);
			}
			else if (isSame<TypeB, int32_t>()) {
				b = (TypeB)pgm_read_dword_near(&B[w]);
			}

			b = b / shrB;

			TypeTemp res;
			if (add)
				res = a / shrC + b / shrC;
			else
				res = a / shrC - b / shrC;

			A[h * W + w] = Saturate<TypeA>(res);
		}
	}

	return;
}


// A = relu(A)
// A[N][H][W][C]
inline __attribute__((always_inline)) void Relu4D(MYINT *A, MYINT N, MYINT H, MYINT W, MYINT C) {

	for (MYITE n = 0; n < N; n++) {
		for (MYITE h = 0; h < H; h++) {
			for (MYITE w = 0; w < W; w++) {
				for (MYITE c = 0; c < C; c++) {
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
template<class TypeA>
inline __attribute__((always_inline)) void Relu4D(TypeA* A, MYINT N, MYINT H, MYINT W, MYINT C) {
	for (MYITE n = 0; n < N; n++) {
		for (MYITE h = 0; h < H; h++) {
			for (MYITE w = 0; w < W; w++) {
				for (MYITE c = 0; c < C; c++) {
					TypeA a = A[n * H * W * C + h * W * C + w * C + c];
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
inline __attribute__((always_inline)) void Relu2D(MYINT *A, MYINT H, MYINT W) {

	for (MYITE h = 0; h < H; h++) {
		for (MYITE w = 0; w < W; w++) {
			MYINT a = A[h * W + w];
			if (a < 0)
				a = 0;

			A[h * W + w] = a;
		}
	}

	return;
}
// A = relu(A)
// A[N][H][W][C]
template<class TypeA>
inline __attribute__((always_inline)) void Relu2D(TypeA* A, MYINT H, MYINT W) {
	for (MYITE h = 0; h < H; h++) {
		for (MYITE w = 0; w < W; w++) {
			TypeA a = A[h * W + w];
			if (a < 0)
				a = 0;
			A[h * W + w] = a;
		}
	}
	return;
}

// B = maxpool(A)
// A[N][H][W][C], B[N][H][W][C]
inline __attribute__((always_inline)) void Maxpool(MYINT *A, MYINT *B, MYINT N, MYINT H, MYINT W, MYINT C, MYINT stride) {
	MYITE HO = H / stride;
	MYITE WO = W / stride;

	for (MYITE n = 0; n < N; n++) {
		for (MYITE ho = 0; ho < HO; ho++) {
			for (MYITE wo = 0; wo < WO; wo++) {
				for (MYITE c = 0; c < C; c++) {

					MYINT max = A[n * H * W * C + (stride * ho) * W * C + (stride * wo) * C + c];
					for (MYITE hs = 0; hs < stride; hs++) {
						for (MYITE ws = 0; ws < stride; ws++) {
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
// B = maxpool(A)
// A[N][H][W][C], B[N][H][W][C]
template<class TypeA, class TypeB>
inline __attribute__((always_inline)) void Maxpool(TypeA* A, TypeB* B, MYINT N, MYINT H, MYINT W, MYINT C, MYINT stride, MYINT demote) {
	MYITE HO = H / stride;
	MYITE WO = W / stride;

	for (MYITE n = 0; n < N; n++) {
		for (MYITE ho = 0; ho < HO; ho++) {
			for (MYITE wo = 0; wo < WO; wo++) {
				for (MYITE c = 0; c < C; c++) {

					TypeA max = A[n * H * W * C + (stride * ho) * W * C + (stride * wo) * C + c];
					for (MYITE hs = 0; hs < stride; hs++) {
						for (MYITE ws = 0; ws < stride; ws++) {
							TypeA a = A[n * H * W * C + ((stride * ho) + hs) * W * C + ((stride * wo) + ws) * C + c];
							if (a > max)
								max = a;
						}
					}

					B[n * HO * WO * C + ho * WO * C + wo * C + c] = (TypeB)(max / demote);
				}
			}
		}
	}
	return;
}

// B = exp(A)
inline __attribute__((always_inline)) void Exp(MYINT *A, MYINT I, MYINT J, MYINT shrA, MYINT shrB, MYINT *B) {

	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
			B[i * J + j] = ((MYINT)(exp(((float)A[i * J + j]) / shrA) * shrB));
		}
	}

	return;
}
// B = exp(A)
//shrB overflows int16_t
template<class TypeA, class TypeB>
inline __attribute__((always_inline)) void Exp(TypeA* A, MYINT I, MYINT J, MYINT shrA, int32_t shrB, TypeB* B, MYINT demote) {
	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
			B[i * J + j] = (TypeB)((exp(((float)A[i * J + j]) / shrA) * shrB) / demote);
		}
	}
	return;
}

// Integer Exponentiation
// B = exp(A)
const PROGMEM int8_t expTable8[128] = {64, 60, 56, 53, 50, 47, 44, 41, 39, 36, 34, 32, 30, 28, 27, 25, 24, 22, 21, 20, 18, 17, 16, 15, 14, 13, 13, 12, 11, 10, 10, 9, 9, 8, 8, 7, 7, 6, 6, 6, 5, 5, 5, 4, 4, 4, 4, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
template<class TypeB>
inline __attribute__((always_inline)) TypeB expBase8(int8_t A, MYINT adjust) {
	int8_t val = (A == -128) ? 127 : -A;
	if(val < 0) {
		val = 127;
	}
	return (TypeB) (pgm_read_byte_near(&expTable8[val]) * adjust);
}
const PROGMEM int16_t expTable16A[256] = {16384, 15391, 14459, 13583, 12760, 11987, 11261, 10578, 9937, 9335, 8770, 8238, 7739, 7270, 6830, 6416, 6027, 5662, 5319, 4997, 4694, 4410, 4143, 3892, 3656, 3434, 3226, 3031, 2847, 2675, 2513, 2360, 2217, 2083, 1957, 1838, 1727, 1622, 1524, 1432, 1345, 1263, 1187, 1115, 1047, 984, 924, 868, 816, 766, 720, 676, 635, 597, 561, 527, 495, 465, 437, 410, 385, 362, 340, 319, 300, 282, 265, 249, 234, 220, 206, 194, 182, 171, 161, 151, 142, 133, 125, 118, 110, 104, 97, 92, 86, 81, 76, 71, 67, 63, 59, 56, 52, 49, 46, 43, 41, 38, 36, 34, 32, 30, 28, 26, 25, 23, 22, 20, 19, 18, 17, 16, 15, 14, 13, 12, 12, 11, 10, 10, 9, 9, 8, 8, 7, 7, 6, 6, 5, 5, 5, 5, 4, 4, 4, 4, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
const PROGMEM int16_t expTable16B[128] = {16384, 16376, 16368, 16360, 16352, 16344, 16336, 16328, 16320, 16312, 16304, 16296, 16288, 16280, 16272, 16264, 16256, 16249, 16241, 16233, 16225, 16217, 16209, 16201, 16193, 16185, 16177, 16169, 16162, 16154, 16146, 16138, 16130, 16122, 16114, 16106, 16099, 16091, 16083, 16075, 16067, 16059, 16051, 16044, 16036, 16028, 16020, 16012, 16004, 15997, 15989, 15981, 15973, 15965, 15958, 15950, 15942, 15934, 15927, 15919, 15911, 15903, 15895, 15888, 15880, 15872, 15864, 15857, 15849, 15841, 15833, 15826, 15818, 15810, 15803, 15795, 15787, 15779, 15772, 15764, 15756, 15749, 15741, 15733, 15726, 15718, 15710, 15703, 15695, 15687, 15680, 15672, 15664, 15657, 15649, 15641, 15634, 15626, 15618, 15611, 15603, 15596, 15588, 15580, 15573, 15565, 15558, 15550, 15542, 15535, 15527, 15520, 15512, 15504, 15497, 15489, 15482, 15474, 15467, 15459, 15452, 15444, 15437, 15429, 15421, 15414, 15406, 15399};
template<class TypeB>
inline __attribute__((always_inline)) TypeB expBase16(int16_t A, MYINT adjust) {
	int16_t val = (A == -32768) ? 32767 : -A;
	int16_t val1 = val % 128;
	int16_t val2 = val / 128;
	int32_t res = ((int32_t)pgm_read_word_near(&expTable16A[val2])) * ((int32_t)pgm_read_word_near(&expTable16B[val1]));
	return (TypeB) ((res / 16384) / adjust);
}
template<class TypeB>
inline __attribute__((always_inline)) void ExpNew8(int8_t *A, MYINT I, MYINT J, MYINT adjust, TypeB *B) {
	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
			B[i * J + j] = expBase8<TypeB>(A[i * J + j], adjust);
		}
	}
	return;
}
template<class TypeB>
inline __attribute__((always_inline)) void ExpNew16(int16_t *A, MYINT I, MYINT J, MYINT adjust, TypeB *B) {
	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
			B[i * J + j] = expBase16<TypeB>(A[i * J + j], adjust);
		}
	}
	return;
}

// A = Sigmoid(A)
inline __attribute__((always_inline)) void Sigmoid(MYINT *A, MYINT I, MYINT J, MYINT div, MYINT add, MYINT sigmoid_limit, MYINT scale_in, MYINT scale_out) {

	MYINT scale_diff = scale_out / scale_in;

	for (MYITE i = 0; i < I; i++)
	{
		for (MYITE j = 0; j < J; j++)
		{
#ifdef FLOATEXP
			float x = float(A[i * J + j]) / scale_in;

			float y = 1 / (1 + exp(-x));

			MYINT z = MYINT(y * scale_out);

			A[i * J + j] = z;
#else
			MYINT x = A[i * J + j];

			x = (x / div) + add;

			MYINT y;
			if (x >= sigmoid_limit)
				y = sigmoid_limit;
			else if (x <= 0)
				y = 0;
			else
				y = x;

			y = y * scale_diff;

			A[i * J + j] = y;
#endif
		}
	}

	return;
}
// A = Sigmoid(A)
template<class TypeA>
inline __attribute__((always_inline)) void Sigmoid(TypeA* A, MYINT I, MYINT J, MYINT div, MYINT add, MYINT sigmoid_limit, MYINT scale_in, MYINT scale_out) {
	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
			float x = float(A[i * J + j]) / scale_in;

			float y = 1 / (1 + exp(-x));

			TypeA z = (TypeA)(y * scale_out);

			A[i * J + j] = z;
		}
	}

	return;
}
// Integer sigmoid using new table exponentiation
template<int dummy>
inline __attribute__((always_inline)) void SigmoidNew8(int8_t* A, MYINT I, MYINT J, int8_t* B) {
	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
			int8_t a = A[i * J + j];
			if (a <= 0) {
				int8_t b = expBase8<int8_t>(a, 1);
				B[i * J + j] = (int8_t)((64 * (int16_t)b) / ((int16_t)b + (int16_t)64));
			} else {
				B[i * J + j] = (int8_t)(((int16_t)4096) / ((int16_t)64 + (int16_t)expBase8<int8_t>(-a, 1)));
			}
			
		}
	}
	return;
}
template<int dummy>
inline __attribute__((always_inline)) void SigmoidNew16(int16_t* A, MYINT I, MYINT J, int16_t* B) {
	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
			int16_t a = A[i * J + j];
			if (a <= 0) {
				int16_t b = expBase16<int16_t>(a, 1);
				B[i * J + j] = (int16_t)((16384 * (int32_t)b) / ((int32_t)b + (int32_t)16384));
			} else {
				B[i * J + j] = (int16_t)(((int32_t)267943936L) / ((int32_t)16384 + (int32_t)expBase16<int16_t>(-a, 1)));
			}
			
		}
	}
	return;
}


// A = tanh(A)
inline __attribute__((always_inline)) void TanH(MYINT *A, MYINT I, MYINT J, MYINT scale_in, MYINT scale_out, MYINT *B) {
	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
#ifdef FLOATEXP
			float x = float(A[i * J + j]) / scale_in;

			float y = tanh(x);

			MYINT z = MYINT(y * scale_out);

			B[i * J + j] = z;
#else
			MYINT x = A[i * J + j], y;

			if (x >= scale_in)
				y = scale_in;
			else if (x <= -scale_in)
				y = -scale_in;
			else
				y = x;

			MYINT scale_diff = scale_out / scale_in;

			y *= scale_diff;

			B[i * J + j] = y;
#endif
		}
	}
	return;
}
// A = tanh(A)
template<class TypeA>
inline __attribute__((always_inline)) void TanH(TypeA* A, MYINT I, MYINT J, TypeA scale_in, TypeA scale_out, TypeA* B) {
	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
			float x = float(A[i * J + j]) / scale_in;

			float y = tanh(x);

			MYINT z = (TypeA)(y * scale_out);

			B[i * J + j] = z;
		}
	}
	return;
}
// Integer TanH using new table exponentiation
template<int dummy>
inline __attribute__((always_inline)) void TanHNew8(int8_t* A, MYINT I, MYINT J, int8_t* B) {
	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
			int8_t a = A[i * J + j];
			if (a <= 0) {
				int16_t b = expBase8<int8_t>(2*a, 1);
				B[i * J + j] = (int8_t)( (((int16_t)64)*(b - 64)) / (b + 64));
			} else {
				int16_t b = expBase8<int8_t>(-2*a, 1);
				B[i * J + j] = (int8_t)( (((int16_t)64)*(64 - b)) / (b + 64));
			}
			
		}
	}
	return;
}
template<int dummy>
inline __attribute__((always_inline)) void TanHNew16(int16_t* A, MYINT I, MYINT J, int16_t* B) {
	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
			int16_t a = A[i * J + j];
			if (a <= 0) {
				int32_t b = expBase16<int16_t>(2*a, 1);
				B[i * J + j] = (int16_t)( (((int32_t)16384)*(b - 16384)) / (b + 16384));
			} else {
				int32_t b = expBase16<int16_t>(-2*a, 1);
				B[i * J + j] = (int16_t)( (((int32_t)16384)*(16384 - b)) / (b + 16384));
			}
			
		}
	}
	return;
}

// A = AdjustScaleShr(A)
inline __attribute__((always_inline)) void AdjustScaleShr(MYINT* A, MYINT I, MYINT J, MYINT scale) {

	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
			MYINT a = A[i * J + j];
			A[i * J + j] = a / scale;
		}
	}

	return;
}
// A = AdjustScaleShr(A)
template<class TypeA>
inline __attribute__((always_inline)) void AdjustScaleShr(TypeA* A, MYINT I, MYINT J, MYINT scale) {
	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
			TypeA a = A[i * J + j];
			A[i * J + j] = a / scale;
		}
	}
	return;
}

// A = AdjustScaleShl(A)
inline __attribute__((always_inline)) void AdjustScaleShl(MYINT* A, MYINT I, MYINT J, MYINT scale) {

	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
			MYINT a = A[i * J + j];
			A[i * J + j] = a * scale;
		}
	}

	return;
}
// A = AdjustScaleShl(A)
template<class TypeA>
inline __attribute__((always_inline)) void AdjustScaleShl(TypeA* A, MYINT I, MYINT J, MYINT scale) {
	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
			TypeA a = A[i * J + j];
			A[i * J + j] = a * scale;
		}
	}
	return;
}
// A = AdjustScaleShlSaturate(A)
template<class TypeA>
inline __attribute__((always_inline)) void AdjustScaleShlSaturate(TypeA* A, MYINT I, MYINT J, MYINT scale, MYINT saturate) {
	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
			TypeA a = A[i * J + j];
			a = (a < saturate && a > -saturate) ? a : (a > 0 ? saturate : -saturate);
			A[i * J + j] = a * scale;
		}
	}
	return;
}
