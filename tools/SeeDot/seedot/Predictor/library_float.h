// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

// This file contains declarations for floating point versions of all operators supported by SeeDot.
// Please refer to library_fixed.h for a description of each operator.

void MatAddNN(float* A, float* B, float* C, MYITE I, MYITE J, MYINT shrA, MYINT shrB, MYINT shrC);
void MatAddCN(const float* A, float* B, float* C, MYITE I, MYITE J, MYINT shrA, MYINT shrB, MYINT shrC);
void MatAddNC(float* A, const float* B, float* C, MYITE I, MYITE J, MYINT shrA, MYINT shrB, MYINT shrC);
void MatAddCC(const float* A, const float* B, float* C, MYITE I, MYITE J, MYINT shrA, MYINT shrB, MYINT shrC);

void MatAddBroadCastA(float* A, float* B, float* C, MYITE I, MYITE J, MYINT shrA, MYINT shrB, MYINT shrC);
void MatAddBroadCastB(float* A, float* B, float* C, MYITE I, MYITE J, MYINT shrA, MYINT shrB, MYINT shrC);

void MatAdd4(float* A, float* B, float* X, MYITE N, MYITE H, MYITE W, MYITE C, MYINT shrA, MYINT shrB, MYINT shrC);

void MatSub(float* A, const float* B, float* C, MYITE I, MYITE J, MYINT shrA, MYINT shrB, MYINT shrC);
void MatSubBroadCastA(float* A, float* B, float* C, MYITE I, MYITE J, MYINT shrA, MYINT shrB, MYINT shrC);
void MatSubBroadCastB(float* A, float* B, float* C, MYITE I, MYITE J, MYINT shrA, MYINT shrB, MYINT shrC);

void MatMulNN(float* A, float* B, float* C, float* tmp, MYITE I, MYITE K, MYITE J, MYINT shrA, MYINT shrB, MYITE H1, MYITE H2);
void MatMulCN(const float* A, float* B, float* C, float* tmp, MYITE I, MYITE K, MYITE J, MYINT shrA, MYINT shrB, MYITE H1, MYITE H2);
void MatMulNC(float* A, const float* B, float* C, float* tmp, MYITE I, MYITE K, MYITE J, MYINT shrA, MYINT shrB, MYITE H1, MYITE H2);
void MatMulCC(const float* A, const float* B, float* C, float* tmp, MYITE I, MYITE K, MYITE J, MYINT shrA, MYINT shrB, MYITE H1, MYITE H2);

void SparseMatMulX(const MYITE* Aidx, const float* Aval, float** B, float* C, int16_t K, MYINT shrA, MYINT shrB, MYINT shrC);
void SparseMatMul(const MYITE* Aidx, const float* Aval, float* B, float* C, int16_t K, MYINT shrA, MYINT shrB, MYINT shrC);

void MulCir(float* A, float* B, float* C, MYITE I, MYITE J, MYINT shrA, MYINT shrB);

void TanH(float* A, MYITE I, MYITE J, float scale_in, float scale_out, float* B);

void ArgMax(float* A, MYITE I, MYITE J, int* index);

void Transpose(float* A, float* B, MYITE I, MYITE J);

void ScalarMul(float* A, float* B, float* C, MYITE I, MYITE J, MYINT shrA, MYINT shrB);

void MBConv(float* A, const float* F1, const float* BN1W, const float* BN1B, const float* F2, const float* BN2W, const float* BN2B, const float* F3, const float* BN3W, const float* BN3B, float* C, float* X, float* T, float* U, MYITE N, MYITE H, MYITE W, MYITE Cin, MYITE Ct, MYITE HF, MYITE WF, MYITE Cout, MYITE Hout, MYITE Wout, MYITE HPADL, MYITE HPADR, MYITE WPADL, MYITE WPADR, MYITE HSTR, MYITE WSTR, MYITE D1, MYITE D2, MYITE D3, MYINT SIX_1, MYINT SIX_2, MYINT shr1, MYINT shr2, MYINT shr3, MYINT shr4, MYINT shr5, MYINT shr6, MYINT shr7, MYINT shr8, MYINT shr9, MYINT shl1, MYINT shl2, MYINT shl3, MYINT shl4, MYINT shl5, MYINT shl6, MYINT shl7, MYINT shl8, MYINT shl9, std::string name);

void Conv(float* A, const float* B, float* C, float* tmp, MYITE N, MYITE H, MYITE W, MYITE CI, MYITE HF, MYITE WF, MYITE CO, MYINT shrA, MYINT shrB, MYITE H1, MYITE H2);

void Convolution(float* A, const float* B, float* C, float* tmp, MYITE N, MYITE H, MYITE W, MYITE CIN, MYITE HF, MYITE WF, MYITE CINF, MYITE COUTF, MYITE HOUT, MYITE WOUT, MYITE HPADL, MYITE HPADR, MYITE WPADL, MYITE WPADR, MYITE HSTR, MYITE WSTR, MYITE HDL, MYITE WDL, MYITE G, MYINT shrA, MYINT shrB, MYITE H1, MYITE H2);

void AddOrSubCir4D(float* A, const float* B, float* X, MYITE N, MYITE H, MYITE W, MYITE C, MYINT shrA, MYINT shrB, MYINT shrC, bool add);

void AddOrSubCir2D(float* A, const float* B, float* X, MYITE H, MYITE W, MYINT shrA, MYINT shrB, MYINT shrC, bool add);

void Relu4D(float* A, MYITE N, MYITE H, MYITE W, MYITE C);

void Relu2D(float* A, MYITE H, MYITE W);

void Relu6(float* A, float* B, MYITE N, MYITE H, MYITE W, MYITE C, MYINT six, MYINT div);

void Maxpool(float* A, float* B, MYITE N, MYITE H, MYITE W, MYITE C, MYITE FH, MYITE FW, MYITE strideH, MYITE strideW, MYITE HPADL, MYITE HPADR, MYITE WPADL, MYITE WPADR);

void Exp(float* A, MYITE I, MYITE J, MYINT shrA, MYINT shrB, float* B);

void Sigmoid(float* A, MYITE I, MYITE J, float div, float add, float sigmoid_limit, MYINT scale_in, MYINT scale_out, float* B);

void AdjustScaleShr(float* A, MYITE I, MYITE J, MYINT scale);
void AdjustScaleShl(float* A, MYITE I, MYITE J, MYINT scale);

void Reverse2(float* A, MYITE axis, MYITE I, MYITE J, float* B);

void NormaliseL2(float* A, float* B, MYITE N, MYITE H, MYITE W, MYITE C, MYINT scaleA, MYINT shrA);
