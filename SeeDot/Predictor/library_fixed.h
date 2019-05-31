// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

void MatAddNN(MYINT *A, MYINT *B, MYINT *C, MYINT I, MYINT J, MYINT shrA, MYINT shrB, MYINT shrC);
void MatAddCN(const MYINT* A, MYINT* B, MYINT* C, MYINT I, MYINT J, MYINT shrA, MYINT shrB, MYINT shrC);
void MatAddNC(MYINT* A, const MYINT* B, MYINT* C, MYINT I, MYINT J, MYINT shrA, MYINT shrB, MYINT shrC);
void MatAddCC(const MYINT* A, const MYINT* B, MYINT* C, MYINT I, MYINT J, MYINT shrA, MYINT shrB, MYINT shrC);

void MatAddBroadCastA(MYINT* A, MYINT* B, MYINT* C, MYINT I, MYINT J, MYINT shrA, MYINT shrB, MYINT shrC);
void MatAddBroadCastB(MYINT* A, MYINT* B, MYINT* C, MYINT I, MYINT J, MYINT shrA, MYINT shrB, MYINT shrC);

void MatSub(MYINT *A, const MYINT *B, MYINT *C, MYINT I, MYINT J, MYINT shrA, int32_t shrB, MYINT shrC);
void MatSubBroadCastA(MYINT* A, MYINT* B, MYINT* C, MYINT I, MYINT J, MYINT shrA, int32_t shrB, MYINT shrC);
void MatSubBroadCastB(MYINT* A, MYINT* B, MYINT* C, MYINT I, MYINT J, MYINT shrA, int32_t shrB, MYINT shrC);

void MatMulNN(MYINT *A, MYINT *B, MYINT *C, MYINT *tmp, MYINT I, MYINT K, MYINT J, MYINT shrA, MYINT shrB, MYINT H1, MYINT H2);

void MatMulCN(const MYINT *A, MYINT *B, MYINT *C, MYINT *tmp, MYINT I, MYINT K, MYINT J, MYINT shrA, MYINT shrB, MYINT H1, MYINT H2);

void MatMulNC(MYINT *A, const MYINT *B, MYINT *C, MYINT *tmp, MYINT I, MYINT K, MYINT J, MYINT shrA, MYINT shrB, MYINT H1, MYINT H2);

void MatMulCC(const MYINT *A, const MYINT *B, MYINT *C, MYINT *tmp, MYINT I, MYINT K, MYINT J, MYINT shrA, MYINT shrB, MYINT H1, MYINT H2);

void SparseMatMul(const MYINT *Aidx, const MYINT *Aval, MYINT **B, MYINT *C, int16_t K, MYINT shrA, MYINT shrB, MYINT shrC);

void MulCir(MYINT *A, MYINT *B, MYINT *C, MYINT I, MYINT J, MYINT shrA, MYINT shrB);

void TanH(MYINT *A, MYINT I, MYINT J, MYINT tanh_limit);

void ArgMax(MYINT *A, MYINT I, MYINT J, MYINT *index);

void Transpose(MYINT *A, MYINT *B, MYINT I, MYINT J);

void ScalarMul(MYINT *A, MYINT *B, MYINT *C, MYINT I, MYINT J, MYINT shrA, MYINT shrB);

void Conv(MYINT *A, const MYINT *B, MYINT *C, MYINT *tmp, MYINT N, MYINT H, MYINT W, MYINT CI, MYINT HF, MYINT WF, MYINT CO, MYINT shrA, MYINT shrB, MYINT H1, MYINT H2);

void AddOrSubCir4D(MYINT *A, const MYINT *B, MYINT N, MYINT H, MYINT W, MYINT C, MYINT shrA, MYINT shrB, MYINT shrC, bool add);

void AddOrSubCir2D(MYINT *A, const MYINT *B, MYINT H, MYINT W, MYINT shrA, MYINT shrB, MYINT shrC, bool add);

void Relu4D(MYINT *A, MYINT N, MYINT H, MYINT W, MYINT C);

void Relu2D(MYINT *A, MYINT H, MYINT W);

void Maxpool(MYINT *A, MYINT *B, MYINT N, MYINT H, MYINT W, MYINT C, MYINT stride);

void Exp(MYINT *A, MYINT I, MYINT J, MYINT shrA, MYINT shrB, MYINT *B);

void Sigmoid(MYINT* A, MYINT I, MYINT J, MYINT div, MYINT add, MYINT sigmoid_limit, MYINT scale_in, MYINT scale_out);

void AdjustScaleShr(MYINT* A, MYINT I, MYINT J, MYINT scale);
void AdjustScaleShl(MYINT* A, MYINT I, MYINT J, MYINT scale);
