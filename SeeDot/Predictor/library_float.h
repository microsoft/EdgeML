// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

void MatAdd(float *A, float *B, float *C, MYINT I, MYINT J, MYINT shrA, MYINT shrB, MYINT shrC);

void MatSub(float *A, const float *B, float *C, MYINT I, MYINT J, MYINT shrA, MYINT shrB, MYINT shrC);

void MatMulNN(float *A, float *B, float *C, float *tmp, MYINT I, MYINT K, MYINT J, MYINT shrA, MYINT shrB, MYINT H1, MYINT H2);

void MatMulCN(const float *A, float *B, float *C, float *tmp, MYINT I, MYINT K, MYINT J, MYINT shrA, MYINT shrB, MYINT H1, MYINT H2);

void MatMulNC(float *A, const float *B, float *C, float *tmp, MYINT I, MYINT K, MYINT J, MYINT shrA, MYINT shrB, MYINT H1, MYINT H2);

void MatMulCC(const float *A, const float *B, float *C, float *tmp, MYINT I, MYINT K, MYINT J, MYINT shrA, MYINT shrB, MYINT H1, MYINT H2);

void SparseMatMul(const MYINT *Aidx, const float *Aval, float **B, float *C, MYINT K, MYINT shrA, MYINT shrB, MYINT shrC);

void MulCir(float *A, float *B, float *C, MYINT I, MYINT J, MYINT shrA, MYINT shrB);

void TanH(float *A, MYINT I, MYINT J, float tanh_limit);

void ArgMax(float *A, MYINT I, MYINT J, MYINT *index);

void Transpose(float *A, float *B, MYINT I, MYINT J);

void ScalarMul(float *A, float *B, float *C, MYINT I, MYINT J, MYINT shrA, MYINT shrB);

void Conv(float *A, const float *B, float *C, float *tmp, MYINT N, MYINT H, MYINT W, MYINT CI, MYINT HF, MYINT WF, MYINT CO, MYINT shrA, MYINT shrB, MYINT H1, MYINT H2);

void AddOrSubCir4D(float *A, const float *B, MYINT N, MYINT H, MYINT W, MYINT C, MYINT shrA, MYINT shrB, MYINT shrC, bool add);

void AddOrSubCir2D(float *A, const float *B, MYINT H, MYINT W, MYINT shrA, MYINT shrB, MYINT shrC, bool add);

void Relu4D(float *A, MYINT N, MYINT H, MYINT W, MYINT C);

void Relu2D(float *A, MYINT H, MYINT W);

void Maxpool(float *A, float *B, MYINT N, MYINT H, MYINT W, MYINT C, MYINT stride);

void Exp(float *A, MYINT I, MYINT J, MYINT shrA, MYINT shrB, float *B);
