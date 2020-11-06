// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

#include "datatypes.h"

/**
 * Notation used:
 * 		By default, 'matrix' is to be interpreted as a matrix in fixed point representation
 * 		dim(X) = dimension of matrix X
 * 		bw(X) = number of bits each value of X uses
 * 		sc(X) = scale of matrix X
 * 		scale of a fixed point matrix X is an integer S such that
 * 			Xq (floating point matrix) = (2 ^ -S) * X where 
 * 				a ^ b (a and b are integers) is a raised to the power b, and 
 * 				a * b (a is integer and b is a matrix) is a multiplied to each element of b
 **/

/**
 * Dimensions: 	A, B, C are matrices, dim(A) = dim(B) = dim(C) = [I][J]; I, J, shrA, shrB, shrC are integers
 * 
 * Matrix Addition
 * Compute A + B and store it in C
 * shrA, shrB, shrC are scaling constants which are computed in irBuilder.py::getScaleForAddAndSub(sc(A), sc(B), sc(C))
 * 		shrA, shrB are used to bring matrices A and B to the same scale for addition 
 * 		shrC adjusts the output matrix if required to prevent overflows
 * The last two letters, which can be either C or N, denote the following:
 * 		If the last letter is N, it means the matrix B is an intermediate variable in RAM
 * 		If the last letter is C, it means the matrix B is a read only parameter which must be extracted from flash 
 * 		Similarly, the second last letter controls the input of matrix A
 * 		On Arduino-like devices with Harvard architecture, the reading of RAM and flash variables is different, hence the different functions
 **/
void MatAddNN(MYINT *A, MYINT *B, MYINT *C, MYINT I, MYINT J, MYINT shrA, MYINT shrB, MYINT shrC);
void MatAddCN(const MYINT *A, MYINT *B, MYINT *C, MYINT I, MYINT J, MYINT shrA, MYINT shrB, MYINT shrC);
void MatAddNC(MYINT *A, const MYINT *B, MYINT *C, MYINT I, MYINT J, MYINT shrA, MYINT shrB, MYINT shrC);
void MatAddCC(const MYINT *A, const MYINT *B, MYINT *C, MYINT I, MYINT J, MYINT shrA, MYINT shrB, MYINT shrC);

/**
 * Dimensions: 	I, J, shrA, shrB, shrC are integers
 * 				C is a matrix, dim(C) = [I][J]
 * 				For MatAddBroadCastA, B is a matrix, dim(B) = [I][J], A represents a scalar
 * 				For MatAddBroadCastB, A is a matrix, dim(A) = [I][J], B represents a scalar
 *  
 * Broadcasted Matrix Addition
 * 		For MatAddBroadCastA, add scalar A to all elements of B and store result in C
 * 		For MatAddBroadCastB, add scalar B to all elements of A and store result in C
 * shrA, shrB, shrC are scaling constants which are computed in irBuilder.py::getScaleForAddAndSub(sc(A), sc(B), sc(C))
 * 		shrA, shrB are used to bring matrices A and B to the same scale for addition 
 * 		shrC adjusts the output matrix if required to prevent overflows
 **/
void MatAddBroadCastA(MYINT *A, MYINT *B, MYINT *C, MYINT I, MYINT J, MYINT shrA, MYINT shrB, MYINT shrC);
void MatAddBroadCastB(MYINT *A, MYINT *B, MYINT *C, MYINT I, MYINT J, MYINT shrA, MYINT shrB, MYINT shrC);

/**
 * Dimensions: 	A, B, C are matrices, dim(A) = dim(B) = dim(C) = [I][J]; I, J, shrA, shrB, shrC are integers
 * 
 * Matrix Subtraction
 * Compute A - B and store it in C
 * shrA, shrB, shrC are scaling constants which are computed in irBuilder.py::getScaleForAddAndSub(sc(A), sc(B), sc(C))
 * 		shrA, shrB are used to bring matrices A and B to the same scale for addition 
 * 		shrC adjusts the output matrix if required to prevent overflows
 * Mostly this operation is used for mean normalisation where the mean (matrix B) is known beforehand and hence stored on read only memory.
 **/
void MatSub(MYINT *A, const MYINT *B, MYINT *C, MYINT I, MYINT J, MYINT shrA, int32_t shrB, MYINT shrC);

/**
 * Dimensions: 	I, J, shrA, shrB, shrC are integers
 * 				C is a matrix, dim(C) = [I][J]
 * 				For MatSubBroadCastA, B is a matrix, dim(B) = [I][J], A represents a scalar
 * 				For MatSubBroadCastB, A is a matrix, dim(A) = [I][J], B represents a scalar
 *  
 * Broadcasted Matrix Subtraction
 * 		For MatSubBroadCastA, add scalar A to all elements of B and store result in C
 * 		For MatSubBroadCastB, add scalar B to all elements of A and store result in C
 * shrA, shrB, shrC are scaling constants which are computed in irBuilder.py::getScaleForAddAndSub(sc(A), sc(B), sc(C))
 * 		shrA, shrB are used to bring matrices A and B to the same scale for addition 
 * 		shrC adjusts the output matrix if required to prevent overflows
 **/
void MatSubBroadCastA(MYINT *A, MYINT *B, MYINT *C, MYINT I, MYINT J, MYINT shrA, int32_t shrB, MYINT shrC);
void MatSubBroadCastB(MYINT *A, MYINT *B, MYINT *C, MYINT I, MYINT J, MYINT shrA, int32_t shrB, MYINT shrC);

/**
 * Dimensions: 	A, B, C are matrices, dim(A) = [I][J], dim(B) = [J][K], dim(C) = [I][K]; tmp is a vector, dim(tmp) = [J] I, K, J, shrA, shrB, H1, H2 are integers
 * 
 * Matrix Multiplication
 * Compute A * B and store it in C, using tmp as a buffer.
 * 		To compute C[i][k], we have to compute summation_j[0:J](A[i][j]*B[j][k]). We store the J values in the vector tmp, 
 * 		and carry out Tree Sum (described below) on the vector to ensure minimum loss of bits 
 * shrA, shrB, H1, H2 are scaling constants which are computed in irBuilder.py::getShrTreeSumAndDemoteParamsForMul(bw(A), sc(A), bw(B), sc(B), bw(tmp), sc(tmp), bw(C), sc(C), J)
 * 		shrA, shrB are used to alter the scales of matrices A and B so that the multiplication avoids overflows but maintains as many bits as possible
 * 		H1, H2 are used for Tree Sum. Usage is described below
 * The last two letters, which can be either C or N, denote the following:
 * 		If the last letter is N, it means the matrix B is an intermediate variable in RAM
 * 		If the last letter is C, it means the matrix B is a read only parameter which must be extracted from flash 
 * 		Similarly, the second last letter controls the input of matrix A
 * 		On Arduino-like devices with Harvard architecture, the reading of RAM and flash variables is different, hence the different functions
 * 
 * Tree Sum
 * This is a technique used to sum up a long vector. To sum up a vector [a0, a1, a2, a3, a4, a5, a6...],
 * in the first stage we first store a0 + a1 at index 0, a2 + a3 at index 2, a4 + a5 at index 4 and so on.
 * Next stage we store index 0 + index 2 at index 0, index 4 + index 6 at index 4, and so on.
 * We continue this till all elements are summed up at index 0.
 * For fixed point arithmetic, in the first H1 (parameter) stages, we divide the addition result by 2 to avoid overflows,
 * and in the next H2 (parameter) stages (assuming no overflows), we do not do the division to conserve prevision
 **/
void MatMulNN(MYINT *A, MYINT *B, MYINT *C, MYINT *tmp, MYINT I, MYINT K, MYINT J, MYINT shrA, MYINT shrB, MYINT H1, MYINT H2);
void MatMulCN(const MYINT *A, MYINT *B, MYINT *C, MYINT *tmp, MYINT I, MYINT K, MYINT J, MYINT shrA, MYINT shrB, MYINT H1, MYINT H2);
void MatMulNC(MYINT *A, const MYINT *B, MYINT *C, MYINT *tmp, MYINT I, MYINT K, MYINT J, MYINT shrA, MYINT shrB, MYINT H1, MYINT H2);
void MatMulCC(const MYINT *A, const MYINT *B, MYINT *C, MYINT *tmp, MYINT I, MYINT K, MYINT J, MYINT shrA, MYINT shrB, MYINT H1, MYINT H2);

/**
 * Dimensions: 	A, B, C are matrices. dim(A) = [I][J], dim(B) = [J][1], dim(C)  [I][1]
 * 				Aval, Aidx combined is a sparse representation of A. dim(Aval) = [K], dim(Aidx) = [K+J]
 * 
 * Representation:	Aval[i] is the i^th non-zero value of A, and Aidx[i] encodes the location of Aval[i].
 * 					Number of zeroes before Aidx[i] : row of Aval[i]
 * 					Aidx[i] + ... + Aidx[l] where l is the largest value less than i such that A[idx] = 0 : column of Aval[i]
 * 
 * Sparse Matrix Multiplication
 * Compute A * B and store it in C.
 * shrA, shrB, shrC are constants used to scale down the result of individual multiplications to not cause overflows. 
 * 		Computed at irBuilder.py::getShrTreeSumAndDemoteParamsForMul(bw(A), sc(A), bw(B), sc(B), bw(C), sc(C), bw(C), sc(C), J)
 */
void SparseMatMulX(const MYINT *Aidx, const MYINT *Aval, MYINT **B, MYINT *C, int16_t K, MYINT shrA, MYINT shrB, MYINT shrC);
void SparseMatMul(const MYINT *Aidx, const MYINT *Aval, MYINT *B, MYINT *C, int16_t K, MYINT shrA, MYINT shrB, MYINT shrC);

/**
 * Dimensions: 	A, B, C are matrices, dim(A) = dim(B) = dim(C) = [I][J]; I, J, shrA, shrB, shrC are integers
 * 
 * Hadamard Matrix Product
 * Compute A * B element-wise and store it in C
 * shrA, shrB are scaling constants which are computed in irBuilder.py::getShrTreeSumAndDemoteParamsForMul(bw(A), sc(A), bw(B), sc(B), bw(C), sc(C), bw(C), sc(C), 1)
 * 		shrA, shrB are used to alter the scales of matrices A and B so that the multiplication avoids overflows but maintains as many bits as possible
 **/
void MulCir(MYINT *A, MYINT *B, MYINT *C, MYINT I, MYINT J, MYINT shrA, MYINT shrB);

/**
 * Dimensions:	A, B are matrices, dim(A) = dim(B) = [I][J]. I, J, scale_in, scale_out are integers
 * 
 * TanH
 * Computes tanH(A) element-wise and stores the result in B
 * scale_in is the scale of the input matrix A, and scale_out is the scale of the output matrix B
 */
void TanH(MYINT *A, MYINT I, MYINT J, MYINT scale_in, MYINT scale_out, MYINT *B);

/**
 * Dimensions:	A is a matrix, dim(A) = [I][J]. I, J are integers, index points to an integer.
 * 				Currently assumes either I or J = 1
 * 
 * ArgMax
 * Computes argmax(A) and stores the result in index
 */
void ArgMax(MYINT *A, MYINT I, MYINT J, MYINT *index);

/**
 * Dimensions:	A, B are matrices. dim(A) = [J][I], dim(B) = [I][J]
 * 
 * Transpose
 * Computes transpose(A) and stores the result in B
 */
void Transpose(MYINT *A, MYINT *B, MYINT I, MYINT J);

/**
 * Dimensions: 	I, J, shrA, shrB are integers
 * 				B, C is are matrices, dim(B) = dim(C) = [I][J]
 * 				A represents a scalar
 *  
 * Scalar Matrix Addition
 * 		Multiply scalar A to all elements of B and store result in C
 * shrA, shrB are scaling constants which are computed in irBuilder.py::getShrTreeSumAndDemoteParamsForMul(bw(A), sc(A), bw(B), sc(B), bw(C), sc(C), bw(C), sc(C), 1)
 * 		shrA, shrB are used to alter the scales of matrices A and B so that the multiplication avoids overflows but maintains as many bits as possible
 */
void ScalarMul(MYINT *A, MYINT *B, MYINT *C, MYINT I, MYINT J, MYINT shrA, MYINT shrB);

/**
 * (only second signature is described as it encompasses the first method)
 * 
 * Dimensions:	A, B, C are matrices, dim(A) = [N][H][W][CI], dim(B) = [G][HF][WF][CINF][COUTF], dim(C) = [N][HOUT][WOUT][COUTF*G]
 * 				computation of HOUT and WOUT is in type.py::visitConvolution()
 * 				tmp is a vector, dim(tmp) = [HF*WF*CINF]; all other parameters are integers
 * 
 * Convolution
 * Computes the convolution of batched and multi-channeled 2D image A with filter B, and stores the result in C, using tmp as a buffer
 * Precomputed parameters: (computed using irBuilder.py::getShrTreeSumAndDemoteParamsForMul(bw(A), sc(A), bw(B), sc(B), bw(tmp), sc(tmp), bw(C), sc(C), HF*WF*CINF))
 * 		shrA, shrB: dividing input matrices' elements to prevent overflows
 * 		H1, H2 : parameters for Tree Sum, described below
 * Raw parameters (directly passed from input code to function):
 * 		HPADL, HPADR : Thickness of padding on top, bottom of the image
 * 		WPADL, WPADR : Thickness of padding on left, right of the image
 * 		HSTR, WSTR : Convolution horizontal, vertical stride
 * 		HDL, WDL : Convolution horizontal, vertical dilations
 * 		G : Number of groups
 * 
 * Tree Sum
 * This is a technique used to sum up a long vector. To sum up a vector [a0, a1, a2, a3, a4, a5, a6...],
 * in the first stage we first store a0 + a1 at index 0, a2 + a3 at index 2, a4 + a5 at index 4 and so on.
 * Next stage we store index 0 + index 2 at index 0, index 4 + index 6 at index 4, and so on.
 * We continue this till all elements are summed up at index 0.
 * For fixed point arithmetic, in the first H1 (parameter) stages, we divide the addition result by 2 to avoid overflows,
 * and in the next H2 (parameter) stages (assuming no overflows), we do not do the division to conserve prevision
 */
void Conv(MYINT *A, const MYINT *B, MYINT *C, MYINT *tmp, MYINT N, MYINT H, MYINT W, MYINT CI, MYINT HF, MYINT WF, MYINT CO, MYINT shrA, MYINT shrB, MYINT H1, MYINT H2);
void Convolution(MYINT *A, const MYINT *B, MYINT *C, MYINT *tmp, MYINT N, MYINT H, MYINT W, MYINT CIN, MYINT HF, MYINT WF, MYINT CINF, MYINT COUTF, MYINT HOUT, MYINT WOUT, MYINT HPADL, MYINT HPADR, MYINT WPADL, MYINT WPADR, MYINT HSTR, MYINT WSTR, MYINT HDL, MYINT WDL, MYINT G, MYINT shrA, MYINT shrB, MYINT H1, MYINT H2);

/**
 * (only describing first signature. second signature is the same, just without N and C dimensions)
 * 
 * Dimensions: 	A, B, X are matrices, dim(A) = dim(X) = [N][H][W][C], dim(B) = [C]
 * 				N, H, W, C, shrA, shrB are integers
 * 				add is a boolean. If true, A + B is computed. If false, A - B is computed
 * 
 * Channel-wise addition/subtraction
 * For c over all channel (C) dimensions, add/subtract scalar B[c] to all values of A[:][:][:][c] and store in X. 
 * shrA, shrB, shrC are scaling constants which are computed in irBuilder.py::getScaleForAddAndSub(sc(A), sc(B), sc(X))
 *		shrA, shrB are used to bring matrices A and B to the same scale for addition 
 *		shrC adjusts the output matrix if required to prevent overflows
 */
void AddOrSubCir4D(MYINT *A, const MYINT *B, MYINT *X, MYINT N, MYINT H, MYINT W, MYINT C, MYINT shrA, MYINT shrB, MYINT shrC, bool add);
void AddOrSubCir2D(MYINT *A, const MYINT *B, MYINT *X, MYINT H, MYINT W, MYINT shrA, MYINT shrB, MYINT shrC, bool add);

/**
 * (describing first signature. second signature is the same, just without the N and C dimensions)
 * 
 * Dimensions: A is a matrix, dim(A) = [N][H][W][C]; N, H, W, C are integers
 * 
 * Relu
 * Computes relu(A) for all elements and stores the result back in A
 */
void Relu4D(MYINT *A, MYINT N, MYINT H, MYINT W, MYINT C);
void Relu2D(MYINT *A, MYINT H, MYINT W);
void Relu6(MYINT *A, MYINT *B, MYINT N, MYINT H, MYINT W, MYINT C, MYINT six, MYINT div);

/**
 * Dimensions:	A, B are matrices, dim(A) = dim(B) = [N][H][W][C]; N, H, W, C are integers
 * 				FH, FW, strideH, strideW, HPADL, HPADR, WPADL, WPADR
 * 
 * Maxpool
 * Computes the maxpool of A and stores the result in B
 * Raw parameters (directly passed from input code to function):
 * 		FH, FW : Size of filter amongst which max is taken
 * 		HPADL, HPADR : Thickness of padding on top, bottom of the image
 * 		WPADL, WPADR : Thickness of padding on left, right of the image
 * 		strideH, strideW : Convolution horizontal, vertical stride
 */
void Maxpool(MYINT *A, MYINT *B, MYINT N, MYINT H, MYINT W, MYINT C, MYINT FH, MYINT FW, MYINT strideH, MYINT strideW, MYINT HPADL, MYINT HPADR, MYINT WPADL, MYINT WPADR);


/**
 * Dimensions:	A is a tensor. dim(A) = [N][H][W][C]
 * 				scaleA, shrA are integers
 * 
 * Exponentiation
 * For each channel computes the L2 norm of all its elements. And divides each number in that channel by the norm.
 */

void NormaliseL2(MYINT* A, MYINT* B, MYINT N, MYINT H, MYINT W, MYINT C, MYINT scaleA, MYINT shrA); 


/**
 * Dimensions:	A, B are matrices. dim(A) = dim(B) = [I][J]
 * 				shrA, shrB are integers
 * 
 * Exponentiation
 * Computes exponentiation of all elements in A (interpreted as a floating point value) to the base e and stores the result in B
 * shrA, shrB are integers which satisfy the following:
 * 		Dividing (float division) each element of matrix A by shrA gives the floating point matrix of A
 * 		Dividing (float division) each element of matrix B by shrB gives the floating point matrix of B
 */
void Exp(MYINT *A, MYINT I, MYINT J, MYINT shrA, MYINT shrB, MYINT *B);

/**
 * Dimensions:	A, B are matrices, dim(A) = dim(B) = [I][J]; div, add, sigmoid_limit, scale_in, scale_out are integers
 * 
 * Sigmoid activation
 * Computes the sigmoid activation for all elements of A and stores the result in B
 * scale_in, scale_out are integers which satisfy the following:
 * 		Dividing (float division) each element of matrix A by scale_in gives the floating point matrix of A
 * 		Dividing (float division) each element of matrix B by scale_out gives the floating point matrix of B
 * 
 * Ifn some cases, a piecewise linear approximation is used for sigmoid: min(max((X+2.0)/4.0, 0.0), 1.0) in floating point version
 * In this case, 
 * 		div represents the fixed point version of 4.0 in the expression 
 * 		add represents the fixed point version of 2.0 in the expression
 * 		sigmoid_limit represents the fixed point version of 1.0 in the expression
 * If flag FLOATEXP is disabled, and if new table exponentiation (Util.py::class Config) is not used, this piecewise approximation is used. Else, the above 3 parameters are not used
 */
void Sigmoid(MYINT *A, MYINT I, MYINT J, MYINT div, MYINT add, MYINT sigmoid_limit, MYINT scale_in, MYINT scale_out, MYINT *B);

/**
 * Dimensions:	A is a matrix, dim(A) = [I][J][K][L] or [I][J]; scale, I, J, (K, L) are integers
 * 
 * Scale adjustment methods
 * AdjustScaleShr divides all elements of A by scale and stores the result in A
 * AdjustScaleShl multiplies all elements of A by scale and stores the result in A
 */
void AdjustScaleShr(MYINT *A, MYINT I, MYINT J, MYINT scale);
void AdjustScaleShl(MYINT *A, MYINT I, MYINT J, MYINT scale);
void AdjustScaleShr(MYINT *A, MYINT I, MYINT J, MYINT K, MYINT L, MYINT scale);
void AdjustScaleShl(MYINT *A, MYINT I, MYINT J, MYINT K, MYINT L, MYINT scale);

/**
 * Dimensions:	A, B is a matrix, dim(A) = dim(B) = [I][J], axis, I, J are integers
 * 
 * Reverse a Matrix
 * Reverses the matrix A along axis (can be 0 for axis I and 1 for axis J) and stores the result in B
 */
void Reverse2(MYINT *A, MYINT axis, MYINT I, MYINT J, MYINT *B);

//Templated Operations: For cases when Variable BitWidth is enabled

template<class TypeA>
inline TypeA Saturate(int32_t inp) {
	return (TypeA)inp;
}

template<>
inline int16_t Saturate(int32_t inp) {
#ifdef SATURATE
	inp = inp > 32767 ? 32767 : inp;
	return (int16_t)(inp < -32768 ? -32768 : inp);
#else
	return (int16_t)inp;
#endif
}

template<>
inline int8_t Saturate(int32_t inp) {
#ifdef SATURATE
	inp = inp > 127 ? 127 : inp;
	return (int8_t)(inp < -128 ? -128 : inp);
#else
	return (int8_t)inp;
#endif
}

template<class TypeA, class TypeB, class TypeTemp, class TypeC>
void MatAddNN(TypeA* A, TypeB* B, TypeC* C, MYINT I, MYINT J, MYINT shrA, MYINT shrB, MYINT shrC, MYINT demote) {
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
template<class TypeA, class TypeB, class TypeTemp, class TypeC>
void MatAddCN(const TypeA* A, TypeB* B, TypeC* C, MYINT I, MYINT J, MYINT shrA, MYINT shrB, MYINT shrC, MYINT demote) {
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
template<class TypeA, class TypeB, class TypeTemp, class TypeC>
void MatAddNC(TypeA* A, const TypeB* B, TypeC* C, MYINT I, MYINT J, MYINT shrA, MYINT shrB, MYINT shrC, MYINT demote) {
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
template<class TypeA, class TypeB, class TypeTemp, class TypeC>
void MatAddCC(const TypeA* A, const TypeB* B, TypeC* C, MYINT I, MYINT J, MYINT shrA, MYINT shrB, MYINT shrC, MYINT demote) {
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

template<class TypeA, class TypeB, class TypeTemp, class TypeC>
void MatAddBroadCastA(TypeA* A, TypeB* B, TypeC* C, MYINT I, MYINT J, MYINT shrA, MYINT shrB, MYINT shrC, MYINT demote) {
	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
			TypeTemp a = (TypeTemp)* A;
			TypeTemp b = (TypeTemp)B[i * J + j];

			a = a / shrA;
			b = b / shrB;

			TypeTemp c = a / shrC + b / shrC;

			C[i * J + j] = Saturate<TypeC>(c / demote);
		}
	}
	return;
}
template<class TypeA, class TypeB, class TypeTemp, class TypeC>
void MatAddBroadCastB(TypeA* A, TypeB* B, TypeC* C, MYINT I, MYINT J, MYINT shrA, MYINT shrB, MYINT shrC, MYINT demote) {
	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
			TypeTemp a = (TypeTemp)A[i * J + j];
			TypeTemp b = (TypeTemp)* B;

			a = a / shrA;
			b = b / shrB;

			TypeTemp c = a / shrC + b / shrC;

			C[i * J + j] = Saturate<TypeC>(c / demote);
		}
	}
	return;
}

template<class TypeA, class TypeB, class TypeTemp, class TypeX>
void MatAdd4(TypeA *A, TypeB *B, TypeX *X, MYINT N, MYINT H, MYINT W, MYINT C, MYINT shrA, MYINT shrB, MYINT shrC, MYINT demote) {
	for (MYITE n = 0; n < N; n++) {
		for (MYITE h = 0; h < H; h++) {
			for (MYITE w = 0; w < W; w++) {
				for (MYITE c = 0; c < C; c++) {
					TypeTemp a = (TypeTemp)A[n * H * W * C + h * W * C + w * C + c];
					TypeTemp b = (TypeTemp)B[n * H * W * C + h * W * C + w * C + c];

					a = a / shrA;
					b = b / shrB;

					TypeTemp x = a / shrC + b / shrC;

					X[n * H * W * C + h * W * C + w * C + c] = Saturate<TypeX>(x / demote);
				}
			}
			
		}
	}
	return;
}

template<class TypeA, class TypeB, class TypeTemp, class TypeC>
// TODO: shrB is int32_t because in 8-bit/16-bit code, shrB is usually very high and int8_t/int16_t will overflow.
void MatSub(TypeA* A, const TypeB* B, TypeC* C, MYINT I, MYINT J, MYINT shrA, int32_t shrB, MYINT shrC, MYINT demote) {
	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
			TypeTemp a = (TypeTemp)A[i * J + j];
			TypeTemp b = (TypeTemp)B[i * J + j];

			a = a / shrA;
			b = b / shrB;

			TypeTemp c = a / shrC - b / shrC;

			C[i * J + j] = Saturate<TypeC>(c / demote);
		}
	}
	return;
}
template<class TypeA, class TypeB, class TypeTemp, class TypeC>
void MatSubBroadCastA(TypeA* A, TypeB* B, TypeC* C, MYINT I, MYINT J, MYINT shrA, MYINT shrB, MYINT shrC, MYINT demote) {
	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
			TypeTemp a = (TypeTemp)* A;
			TypeTemp b = (TypeTemp)B[i * J + j];

			a = a / shrA;
			b = b / shrB;

			TypeTemp c = a / shrC - b / shrC;

			C[i * J + j] = Saturate<TypeC>(c / demote);
		}
	}
	return;
}
template<class TypeA, class TypeB, class TypeTemp, class TypeC>
void MatSubBroadCastB(TypeA* A, TypeB* B, TypeC* C, MYINT I, MYINT J, MYINT shrA, MYINT shrB, MYINT shrC, MYINT demote) {
	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
			TypeTemp a = (TypeTemp)A[i * J + j];
			TypeTemp b = (TypeTemp)* B;

			a = a / shrA;
			b = b / shrB;

			TypeTemp c = a / shrC - b / shrC;

			C[i * J + j] = Saturate<TypeC>(c / demote);
		}
	}
	return;
}

template<class TypeA, class TypeB, class TypeTemp, class TypeC>
void MatMulNN(TypeA* A, TypeB* B, TypeC* C, TypeTemp* tmp, MYINT I, MYINT K, MYINT J, MYINT shrA, MYINT shrB, MYINT H1, MYINT H2, MYINT demote) {
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
template<class TypeA, class TypeB, class TypeTemp, class TypeC>
void MatMulCN(const TypeA* A, TypeB* B, TypeC* C, TypeTemp* tmp, MYINT I, MYINT K, MYINT J, MYINT shrA, MYINT shrB, MYINT H1, MYINT H2, MYINT demote) {
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
template<class TypeA, class TypeB, class TypeTemp, class TypeC>
void MatMulNC(TypeA* A, const TypeB* B, TypeC* C, TypeTemp* tmp, MYINT I, MYINT K, MYINT J, MYINT shrA, MYINT shrB, MYINT H1, MYINT H2, MYINT demote) {
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
template<class TypeA, class TypeB, class TypeTemp, class TypeC>
void MatMulCC(const TypeA* A, const TypeB* B, TypeC* C, TypeTemp* tmp, MYINT I, MYINT K, MYINT J, MYINT shrA, MYINT shrB, MYINT H1, MYINT H2, MYINT demote) {
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

template<class TypeA, class TypeAidx, class TypeB, class TypeTemp, class TypeC>
void SparseMatMulX(const TypeAidx* Aidx, const TypeA* Aval, TypeB** B, TypeC* C, int16_t K, MYINT shrA, MYINT shrB, MYINT shrC, MYINT demote) {

	MYITE ite_idx = 0, ite_val = 0;
	for (MYITE k = 0; k < K; k++) {
		// MYINT b = getIntFeature(k);
		TypeTemp b = (TypeTemp)B[k * 1][0];
		//b = b / shrB;

		MYITE idx = Aidx[ite_idx];
		while (idx != 0) {
			TypeTemp a = (TypeTemp)Aval[ite_val];
			//a = a / shrA;
			TypeTemp c = (TypeTemp)(a * b);
			//c = c / shrC;

			C[idx - 1] += Saturate<TypeC>((((c / shrA) / shrB) / shrC) / demote);

			ite_idx++;
			ite_val++;

			idx = Aidx[ite_idx];
		}
		ite_idx++;
	}

	return;
}
template<class TypeA, class TypeAidx, class TypeB, class TypeTemp, class TypeC>
void SparseMatMul(const TypeAidx* Aidx, const TypeA* Aval, TypeB* B, TypeC* C, int16_t K, MYINT shrA, MYINT shrB, MYINT shrC, MYINT demote) {

	MYITE ite_idx = 0, ite_val = 0;
	for (MYITE k = 0; k < K; k++) {
		TypeTemp b = (TypeTemp)B[k];
		//b = b / shrB;

		MYITE idx = Aidx[ite_idx];
		while (idx != 0) {
			TypeTemp a = (TypeTemp)Aval[ite_val];
			//a = a / shrA;
			TypeTemp c = (TypeTemp)(a * b);
			//c = c / shrC;

			C[idx - 1] += Saturate<TypeC>((((c / shrA) / shrB) / shrC) / demote);

			ite_idx++;
			ite_val++;

			idx = Aidx[ite_idx];
		}
		ite_idx++;
	}

	return;
}

template<class TypeA, class TypeB, class TypeTemp, class TypeC>
void MulCir(TypeA* A, TypeB* B, TypeC* C, MYINT I, MYINT J, MYINT shrA, MYINT shrB, MYINT demote) {
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

template<class TypeA>
void Confidence(TypeA* A, float* confidence) {
	*confidence = *A;
	if (*confidence < 0)
		* confidence = -(*confidence);
}

template<class TypeA>
void Confidence(TypeA* A, MYINT I, MYINT J, MYITE* index, float* confidence) {
	TypeA max = A[0];
	TypeA min = A[0];
	MYITE maxIndex = 0, counter = 0;
	float sum = 0;
	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
			TypeA x = A[i * J + j];
			//sum += x;
			if (max < x) {
				maxIndex = counter;
				max = x;
			}
			if (min > x) {
				min = x;
			}
			counter++;
		}
	}

	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
			sum += (A[i * J + j] - min);
		}
	}

	*index = maxIndex;
	if (sum < 0.0001 && sum > -0.0001)
		* confidence = ((float)1) / (I * J); //Maybe could penalise more as this is a underflow
	else
		*confidence = (float)(A[*index]-min) / (sum);
	return;
}


template<class TypeA>
void ArgMax(TypeA* A, MYINT I, MYINT J, int* index) {
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

template<class TypeA>
void Transpose(TypeA* A, TypeA* B, MYINT I, MYINT J) {
	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
			B[i * J + j] = A[j * I + i];
		}
	}
	return;
}

template<class TypeA, class TypeB, class TypeTemp, class TypeC>
void ScalarMul(TypeA* A, TypeB* B, TypeC* C, MYINT I, MYINT J, MYINT shrA, MYINT shrB, int demote) {
	TypeTemp a = (TypeTemp)* A;
	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
			TypeTemp b = (TypeTemp)B[i * J + j];

			TypeTemp prod = a * b;
			C[i * J + j] = Saturate<TypeC>(((prod / shrA) / shrB) / demote);
		}
	}

	return;
}

template<class TypeA, class TypeF1, class TypeB1W, class TypeB1B, class TypeF2, class TypeB2W, class TypeB2B, class TypeF3, class TypeB3W, class TypeB3B, class TypeC, class TypeX, class TypeT, class TypeU, class TypeUB1W, class TypeUB2W, class TypeUB3W> 
void MBConv(TypeA *A, TypeF1 *F1, TypeB1W *BN1W, TypeB1B *BN1B, TypeF2 *F2, TypeB2W *BN2W, TypeB2B *BN2B, TypeF3 *F3, TypeB3W *BN3W, TypeB3B *BN3B, TypeC *C, TypeX *X, TypeT *T, TypeU *U, MYITE N, MYITE H, MYITE W, MYITE Cin, MYITE Ct, MYITE HF, MYITE WF, MYITE Cout, MYITE Hout, MYITE Wout, MYITE HPADL, MYITE HPADR, MYITE WPADL, MYITE WPADR, MYITE HSTR, MYITE WSTR, MYITE D1, MYITE D2, MYITE D3, TypeUB1W SIX_1, TypeUB2W SIX_2, TypeUB1W shr1, TypeUB1W shr2, TypeUB1W shr3, TypeUB2W shr4, TypeUB2W shr5, TypeUB2W shr6, TypeUB3W shr7, TypeUB3W shr8, TypeUB3W shr9, TypeUB1W shl1, TypeUB1W shl2, TypeUB1W shl3, TypeUB2W shl4, TypeUB2W shl5, TypeUB2W shl6, TypeUB3W shl7, TypeUB3W shl8, TypeUB3W shl9) {
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
					X[i * W * Ct + j * Ct + k] =  Saturate<TypeX>((x * shl3) / shr3);
				}
			}
		}

		for (MYITE h = HOffsetL, hout = 0; h < H - HOffsetR; hout++, h += HSTR) {

			for (MYITE i = 0; i < HSTR; i++) {
				for (MYITE j = 0; j < W; j++) {
					for (MYITE k = 0; k < Ct; k++) {
						MYITE iRed = (i + margin + hout * HSTR) % HF, iFull = i + margin + hout * HSTR;
						X[iRed * W * Ct + j * Ct + k] = 0;
						for (MYITE l = 0; l < Cin; l++) {
							TypeA a = iFull < H ? A[n * H * W * Cin + iFull * W * Cin + j * Cin + l] : 0;
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
						X[iRed * W * Ct + j * Ct + k] =  Saturate<TypeX>((x * shl3) / shr3);
					}
				}
			}

			for (MYITE w = WOffsetL, wout = 0; w < W - WOffsetR; w += WSTR, wout++) {
				for (MYITE g = 0; g < Ct; g++) {
					MYITE counter = 0;
					for (MYITE hf = -(HF/2); hf <= (HF/2); hf++) {
						for (MYITE wf = -(WF/2); wf <= (WF/2); wf++) {
							TypeX x = (((h + hf) < 0) || ((h + hf) >= H) || ((w + wf) < 0) || ((w + wf) >= W)) ? 0 : X[((h + hf) % HF) * W * Ct + (w + wf) * Ct + g];
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
					T[g] =  Saturate<TypeT>((x * shl6) / shr6);
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
					C[n * Hout * Wout * Cout + hout * Wout * Cout + wout * Cout + i] = Saturate<TypeC>(((((TypeUB3W)((U[0] * shl7) / shr7 + (BN3B[i] * shl8) / shr8)) * ((TypeUB3W)BN3W[i])) * shl9) / shr9);
				}
			}
		}
	}
}

template<class TypeA, class TypeB, class TypeTemp, class TypeC>
void Conv(TypeA* A, const TypeB* B, TypeC* C, TypeTemp* tmp, MYINT N, MYINT H, MYINT W, MYINT CI, MYINT HF, MYINT WF, MYINT CO, MYINT shrA, MYINT shrB, MYINT H1, MYINT H2, MYINT demote) {
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

								TypeTemp b = (TypeTemp)B[hf * WF * CI * CO + wf * CI * CO + ci * CO + co];

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

// C = conv(A, B, <params>)
// A[N][H][W][CIN], B[G][HF][WF][CINF][COUTF], C[N][HOUT][WOUT][COUTF*G]
template<class TypeA, class TypeB, class TypeTemp, class TypeC>
void Convolution(TypeA *A, const TypeB *B, TypeC *C, TypeTemp *tmp, MYINT N, MYINT H, MYINT W, MYINT CIN, MYINT HF, MYINT WF, MYINT CINF, MYINT COUTF, MYINT HOUT, MYINT WOUT, MYINT HPADL, MYINT HPADR, MYINT WPADL, MYINT WPADR, MYINT HSTR, MYINT WSTR, MYINT HDL, MYINT WDL, MYINT G, MYINT shrA, MYINT shrB, MYINT H1, MYINT H2, MYINT demote) {
	MYITE HOffsetL = HDL*(HF/2) - HPADL;
	MYITE WOffsetL = WDL*(WF/2) - WPADL;
	MYITE HOffsetR = HDL*(HF/2) - HPADR;
	MYITE WOffsetR = WDL*(WF/2) - WPADR;

	for(MYITE n = 0; n < N; n++) {
		for(MYITE h = HOffsetL, hout = 0; h < H - HOffsetR; h += HSTR, hout++) {
			for(MYITE w = WOffsetL, wout = 0; w < W - WOffsetR; w += WSTR, wout++) {
				for(MYITE g = 0; g < G; g++) {
					for(MYITE co = 0; co < COUTF; co ++) {

						MYITE counter = 0;
						for(MYITE hf = -(HF/2); hf <= HF/2; hf++) {
							for(MYITE wf = -(WF/2); wf <= WF/2; wf++) {
								for(MYITE ci = 0; ci < CINF; ci++) {

									TypeTemp a = (TypeTemp) (((h + HDL * hf) < 0) || ((h + HDL * hf) >= H) || ((w + WDL * wf) < 0) || ((w + WDL * wf) >= W)) ? 0 : A[n * H * W * CIN + (h + HDL * hf) * W * CIN + (w + WDL * wf) * CIN + (ci + g * CINF)];

									TypeTemp b = (TypeTemp) B[g * HF * WF * CINF * COUTF + (hf + HF/2) * WF * CINF * COUTF + (wf + WF/2) * CINF * COUTF + ci * COUTF + co];

									tmp[counter] = a * b;
									counter++;
								}
							}
						}

						MYITE totalEle = HF * WF * CINF;
						MYITE count = HF * WF * CINF, depth = 0;
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

						C[n * HOUT * WOUT * (COUTF * G) + hout * WOUT * (COUTF * G) + wout * (COUTF * G) + (co + g * COUTF)] = Saturate<TypeC>(((tmp[0] / shrA) / shrB) / demote);
					}
				}
			}
		}
	}
}


template<class TypeA, class TypeB, class TypeTemp, class TypeC>
void AddOrSubCir4D(TypeA* A, const TypeB* B, TypeC* X, MYINT N, MYINT H, MYINT W, MYINT C, MYINT shrA, MYINT shrB, MYINT shrC, bool add, MYINT demote) {
	for (MYITE n = 0; n < N; n++) {
		for (MYITE h = 0; h < H; h++) {
			for (MYITE w = 0; w < W; w++) {
				for (MYITE c = 0; c < C; c++) {
					TypeTemp a = (TypeTemp)A[n * H * W * C + h * W * C + w * C + c];
					a = a / shrA;

					TypeTemp b = (TypeTemp)B[c];
					b = b / shrB;

					TypeTemp res;
					if (add)
						res = a / shrC + b / shrC;
					else
						res = a / shrC - b / shrC;

					X[n * H * W * C + h * W * C + w * C + c] = Saturate<TypeC>(res / demote);
				}
			}
		}
	}
	return;
}
template<class TypeA, class TypeB, class TypeTemp, class TypeC>
void AddOrSubCir2D(TypeA* A, const TypeB* B, TypeC* X, MYINT H, MYINT W, MYINT shrA, MYINT shrB, MYINT shrC, bool add, MYINT demote) {
	for (MYITE h = 0; h < H; h++) {
		for (MYITE w = 0; w < W; w++) {
			TypeTemp a = (TypeTemp)A[h * W + w];
			a = a / shrA;

			TypeTemp b = (TypeTemp)B[w];
			b = b / shrB;

			TypeTemp res;
			if (add)
				res = a / shrC + b / shrC;
			else
				res = a / shrC - b / shrC;

			X[h * W + w] = Saturate<TypeC>(res / demote);
		}
	}

	return;
}

template<class TypeA>
void Relu4D(TypeA* A, MYINT N, MYINT H, MYINT W, MYINT C) {
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

template<class TypeA, class TypeB>
void Relu6(TypeA* A, TypeB* B, MYINT N, MYINT H, MYINT W, MYINT C, TypeA six, TypeA div) {
	for (MYITE n = 0; n < N; n++) {
		for (MYITE h = 0; h < H; h++) {
			for (MYITE w = 0; w < W; w++) {
				for (MYITE c = 0; c < C; c++) {
					TypeA a = A[n * H * W * C + h * W * C + w * C + c];
					if (a < 0)
						a = 0;
					if (a > six)
						a = six;
					B[n * H * W * C + h * W * C + w * C + c] = (TypeB) (a / div);
				}
			}
		}
	}
	return;
}

template<class TypeA>
void Relu2D(TypeA* A, MYINT H, MYINT W) {
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

template<class TypeA, class TypeB>
void Maxpool(TypeA* A, TypeB* B, MYINT N, MYINT H, MYINT W, MYINT C, MYINT FH, MYINT FW, MYINT strideH, MYINT strideW, MYINT HPADL, MYINT HPADR, MYINT WPADL, MYINT WPADR, MYINT demote) {
	MYITE HO = H / strideH;
	MYITE WO = W / strideW;

	for (MYITE n = 0; n < N; n++) {
		for (MYITE ho = 0; ho < HO; ho++) {
			for (MYITE wo = 0; wo < WO; wo++) {
				for (MYITE c = 0; c < C; c++) {

					TypeA max = A[n * H * W * C + (strideH * ho) * W * C + (strideW * wo) * C + c];
					for (MYITE hs = 0; hs < FH; hs++) {
						for (MYITE ws = 0; ws < FW; ws++) {
							TypeA a = A[n * H * W * C + ((strideH * ho) + hs) * W * C + ((strideW * wo) + ws) * C + c];
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

template<class TypeA>
void NormaliseL2(TypeA* A, TypeA* B, MYINT N, MYINT H, MYINT W, MYINT C, MYINT scaleA, MYINT shrA) {
	for (MYITE n = 0; n < N; n++) {
		for (MYITE h = 0; h < H; h++) {
			for (MYITE w = 0; w < W; w++) {

				// calculate the sum square
				int32_t sumSquare = 0;
				MYINT shrAdiv = (1<<shrA);

				for (MYITE c = 0; c < C; c++) {
#ifdef FASTAPPROX
				MYINT tmp = (A[n * H * W * C + h * W * C + w * C + c] / shrAdiv);
				sumSquare += tmp*tmp;
#else           
				int32_t tmp = A[n * H * W * C + h * W * C + w * C + c];
				sumSquare += (((tmp*tmp)/shrAdiv)/shrAdiv);						
#endif
				}
				

				// calculate the inverse square root of sumSquare
				MYINT yLow = 1;

				// yHigh: A number of length shrA with all 1s in binary representation e.g. for shrA=8 --> y_high = 0b11111111
				MYINT yHigh = (1<<shrA - 1);   

				// one: value of 1 with same scale as y*y*sumSquare
				// scale of sumSquare = 2*scale_in + 2*shrA
				// since we assume scale of y = 1 - shrA
				// scale of y*y*sumSquare =  2*scale_in + 2*shrA + 2(1-shrA) = 2*scale_in + 2
				int32_t one = ( 1<< (-(2*scaleA + 2)) ); 

				//binary search for the inverse square root 
				while( yLow+1 < yHigh ){

					//using int32_t sotherwise (y*y*sumSquare) will overflow
					MYINT yMid = ((yHigh + yLow)>>1);

					int64_t cmpValue = (int64_t)sumSquare*yMid*yMid;

					if(cmpValue > one){
						yHigh = yMid;	
					}	
					else {
						yLow = yMid;
					}
				}
				MYINT inverseNorm = yLow;


				// multiply all elements by the 1/sqrt(sumSquare)
				for (MYITE c = 0; c < C; c++) {
						B[n * H * W * C + h * W * C + w * C + c]  = (A[n * H * W * C + h * W * C + w * C + c]  / shrAdiv)*inverseNorm;  
				}
			}				
		}
	}
	return;
}

//shrB overflows int16_t
template<class TypeA, class TypeB>
void Exp(TypeA* A, MYINT I, MYINT J, MYINT shrA, int32_t shrB, TypeB* B, MYINT demote) {
	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
			B[i * J + j] = (TypeB)((exp(((float)A[i * J + j]) / shrA) * shrB) / demote);
		}
	}
	return;
}

const int8_t expTable8[128] = {64, 60, 56, 53, 50, 47, 44, 41, 39, 36, 34, 32, 30, 28, 27, 25, 24, 22, 21, 20, 18, 17, 16, 15, 14, 13, 13, 12, 11, 10, 10, 9, 9, 8, 8, 7, 7, 6, 6, 6, 5, 5, 5, 4, 4, 4, 4, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
template<class TypeB>
inline TypeB expBase8(int8_t A, MYINT adjust) {
	int8_t val = (A == -128) ? 127 : -A;
	if(val < 0) {
		val = 127;
	}
	return (TypeB) (expTable8[val] * adjust);
}

const int16_t expTable16A[256] = {16384, 15391, 14459, 13583, 12760, 11987, 11261, 10578, 9937, 9335, 8770, 8238, 7739, 7270, 6830, 6416, 6027, 5662, 5319, 4997, 4694, 4410, 4143, 3892, 3656, 3434, 3226, 3031, 2847, 2675, 2513, 2360, 2217, 2083, 1957, 1838, 1727, 1622, 1524, 1432, 1345, 1263, 1187, 1115, 1047, 984, 924, 868, 816, 766, 720, 676, 635, 597, 561, 527, 495, 465, 437, 410, 385, 362, 340, 319, 300, 282, 265, 249, 234, 220, 206, 194, 182, 171, 161, 151, 142, 133, 125, 118, 110, 104, 97, 92, 86, 81, 76, 71, 67, 63, 59, 56, 52, 49, 46, 43, 41, 38, 36, 34, 32, 30, 28, 26, 25, 23, 22, 20, 19, 18, 17, 16, 15, 14, 13, 12, 12, 11, 10, 10, 9, 9, 8, 8, 7, 7, 6, 6, 5, 5, 5, 5, 4, 4, 4, 4, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
const int16_t expTable16B[128] = {16384, 16376, 16368, 16360, 16352, 16344, 16336, 16328, 16320, 16312, 16304, 16296, 16288, 16280, 16272, 16264, 16256, 16249, 16241, 16233, 16225, 16217, 16209, 16201, 16193, 16185, 16177, 16169, 16162, 16154, 16146, 16138, 16130, 16122, 16114, 16106, 16099, 16091, 16083, 16075, 16067, 16059, 16051, 16044, 16036, 16028, 16020, 16012, 16004, 15997, 15989, 15981, 15973, 15965, 15958, 15950, 15942, 15934, 15927, 15919, 15911, 15903, 15895, 15888, 15880, 15872, 15864, 15857, 15849, 15841, 15833, 15826, 15818, 15810, 15803, 15795, 15787, 15779, 15772, 15764, 15756, 15749, 15741, 15733, 15726, 15718, 15710, 15703, 15695, 15687, 15680, 15672, 15664, 15657, 15649, 15641, 15634, 15626, 15618, 15611, 15603, 15596, 15588, 15580, 15573, 15565, 15558, 15550, 15542, 15535, 15527, 15520, 15512, 15504, 15497, 15489, 15482, 15474, 15467, 15459, 15452, 15444, 15437, 15429, 15421, 15414, 15406, 15399};
template<class TypeB>
inline TypeB expBase16(int16_t A, MYINT adjust) {
	int16_t val = (A == -32768) ? 32767 : -A;
	int16_t val1 = val % 128;
	int16_t val2 = val / 128;
	int32_t res = expTable16A[val2] * expTable16B[val1];
	return (TypeB) (res / (16384 * adjust));
}

template<class TypeB>
void ExpNew8(int8_t *A, MYINT I, MYINT J, MYINT adjust, TypeB *B) {
	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
			B[i * J + j] = expBase8<TypeB>(A[i * J + j], adjust);
		}
	}
	return;
}

template<class TypeB>
void ExpNew16(int16_t *A, MYINT I, MYINT J, MYINT adjust, TypeB *B) {
	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
			B[i * J + j] = expBase16<TypeB>(A[i * J + j], adjust);
		}
	}
	return;
}


template<class TypeA>
void Sigmoid(TypeA* A, MYINT I, MYINT J, MYINT div, MYINT add, MYINT sigmoid_limit, MYINT scale_in, MYINT scale_out, TypeA* B) {
	TypeA scale_diff = scale_out / scale_in;
	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
		#ifdef FLOATEXP
			float x = float(A[i * J + j]) / scale_in;

			float y = 1 / (1 + exp(-x));

			TypeA z = (TypeA)(y * scale_out);

			B[i * J + j] = z;
		#else
			TypeA x = A[i * J + j];

			x = (x / div) + add;

			TypeA y;
			if (x >= sigmoid_limit)
				y = sigmoid_limit;
			else if (x <= 0)
				y = 0;
			else
				y = x;

			y = y * scale_diff;

			B[i * J + j] = y;
		#endif
		}
	}

	return;
}
// Integer sigmoid using new table exponentiation
template<int dummy>
void SigmoidNew8(int8_t* A, MYINT I, MYINT J, int8_t* B) {
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
void SigmoidNew16(int16_t* A, MYINT I, MYINT J, int16_t* B) {
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

template<class TypeA>
void TanH(TypeA* A, MYINT I, MYINT J, TypeA scale_in, TypeA scale_out, TypeA* B) {
	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
		#ifdef FLOATEXP
			float x = float(A[i * J + j]) / scale_in;

			float y = tanh(x);

			MYINT z = (TypeA)(y * scale_out);

			B[i * J + j] = z;
		#else
			TypeA x = A[i * J + j], y;

			if (x >= scale_in)
				y = scale_in;
			else if (x <= -scale_in)
				y = -scale_in;
			else
				y = x;

			TypeA scale_diff = scale_out / scale_in;

			y *= scale_diff;

			B[i * J + j] = y;
		#endif
		}
	}
	return;
}
// Integer TanH using new table exponentiation
template<int dummy>
void TanHNew8(int8_t* A, MYINT I, MYINT J, int8_t* B) {
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
void TanHNew16(int16_t* A, MYINT I, MYINT J, int16_t* B) {
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


template<class TypeA>
void AdjustScaleShr(TypeA* A, MYINT I, MYINT J, MYINT scale) {
	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
			TypeA a = A[i * J + j];
			A[i * J + j] = a / scale;
		}
	}
	return;
}
template<class TypeA>
void AdjustScaleShr(TypeA* A, MYINT I, MYINT J, MYINT K, MYINT L, MYINT scale) {
	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
			for(MYITE k = 0; k < K; k++) {
				for(MYITE l = 0; l < L; l++) {
					TypeA a = A[i * J * K * L + j * K * L + k * L + l];
					A[i * J * K * L + j * K * L + k * L + l] = a / scale;
				}
			}
		}
	}
	return;
}
template<class TypeA>
void AdjustScaleShl(TypeA* A, MYINT I, MYINT J, MYINT scale) {
	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
			TypeA a = A[i * J + j];
			A[i * J + j] = a * scale;
		}
	}
	return;
}
template<class TypeA>
void AdjustScaleShl(TypeA* A, MYINT I, MYINT J, MYINT K, MYINT L, MYINT scale) {
	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
			for(MYITE k = 0; k < K; k++) {
				for(MYITE l = 0; l < L; l++) {
					TypeA a = A[i * J * K * L + j * K * L + k * L + l];
					A[i * J * K * L + j * K * L + k * L + l] = a * scale;
				}
			}
		}
	}
	return;
}
template<class TypeA>
void AdjustScaleShlSaturate(TypeA* A, MYINT I, MYINT J, MYINT scale, MYINT saturate) {
	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
			TypeA a = A[i * J + j];
			a = (a < saturate && a > -saturate) ? a : (a > 0 ? saturate : -saturate);
			A[i * J + j] = a * scale;
		}
	}
	return;
}
template<class TypeA>
void AdjustScaleShlSaturate(TypeA* A, MYINT I, MYINT J, MYINT K, MYINT L, MYINT scale, MYINT saturate) {
	for (MYITE i = 0; i < I; i++) {
		for (MYITE j = 0; j < J; j++) {
			for(MYITE k = 0; k < K; k++) {
				for(MYITE l = 0; l < L; l++) {
					TypeA a = A[i * J * K * L + j * K * L + k * L + l];
					a = (a < saturate && a > -saturate) ? a : (a > 0 ? saturate : -saturate);
					A[i * J * K * L + j * K * L + k * L + l] = a * scale;
				}
			}
		}
	}
	return;
}

// B = reverse(A, axis)
template<class TypeA>
void Reverse2(TypeA *A, MYINT axis, MYINT I, MYINT J, TypeA *B)
{
	for (MYITE i = 0; i < I; i++)
	{
		for (MYITE j = 0; j < J; j++)
		{	
			MYINT i_prime = (axis == 0 ? (I-1-i) : i);
			MYINT j_prime = (axis == 1 ? (J-1-j) : j); 

			B[i * J + j] = A[i_prime*J + j_prime];
		}
	}
	return;
}