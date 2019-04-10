// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <iostream>
#include <cstring>
#include <cmath>

#include "datatypes.h"
#include "predictors.h"
#include "library.h"
#include "seedot_fixed_model.h"

using namespace std;
using namespace bonsai_fixed;

int seedotFixed(MYINT **X) {
	MYINT tmp6[30][1];
	MYINT tmp7[30][1];
	MYINT node0;
	MYINT tmp9[1][1];
	MYINT tmp8[30];
	MYINT tmp11[1][1];
	MYINT tmp10[30];
	MYINT tmp12[1][1];
	MYINT tmp14[1][1];
	MYINT tmp13[30];
	MYINT node1;
	MYINT tmp16[1][1];
	MYINT tmp15[30];
	MYINT tmp18[1][1];
	MYINT tmp17[30];
	MYINT tmp19[1][1];
	MYINT tmp20[1][1];
	MYINT tmp22[1][1];
	MYINT tmp21[30];
	MYINT node2;
	MYINT tmp24[1][1];
	MYINT tmp23[30];
	MYINT tmp26[1][1];
	MYINT tmp25[30];
	MYINT tmp27[1][1];
	MYINT tmp28[1][1];
	MYINT tmp30[1][1];
	MYINT tmp29[30];
	MYINT node3;
	MYINT tmp32[1][1];
	MYINT tmp31[30];
	MYINT tmp34[1][1];
	MYINT tmp33[30];
	MYINT tmp35[1][1];
	MYINT tmp36[1][1];
	MYINT tmp37;



	// Z |*| X
	memset(tmp6, 0, sizeof(MYINT) * 30);
	SparseMatMul(&Zidx[0], &Zval[0], X, &tmp6[0][0], 257, 128, 128, 2);


	// tmp6 - mean
	MatSub(&tmp6[0][0], &mean[0][0], &tmp7[0][0], 30, 1, 1, 4, 1);

	node0 = 0;

	// W * ZX
	MatMulCN(&W[node0][0][0], &tmp7[0][0], &tmp9[0][0], &tmp8[0], 1, 30, 1, 128, 128, 0, 5);


	// V * ZX
	MatMulCN(&V[node0][0][0], &tmp7[0][0], &tmp11[0][0], &tmp10[0], 1, 30, 1, 128, 64, 0, 5);


	// tanh(V0)
	TanH(&tmp11[0][0], 1, 1, 2048);


	// W0 <*> V0_tanh
	MulCir(&tmp9[0][0], &tmp11[0][0], &tmp12[0][0], 1, 1, 64, 32);


	// T * ZX
	MatMulCN(&T[node0][0][0], &tmp7[0][0], &tmp14[0][0], &tmp13[0], 1, 30, 1, 128, 128, 1, 4);

	node1 = ((tmp14[0][0] > 0) ? ((2 * node0) + 1) : ((2 * node0) + 2));

	// W * ZX
	MatMulCN(&W[node1][0][0], &tmp7[0][0], &tmp16[0][0], &tmp15[0], 1, 30, 1, 128, 128, 0, 5);


	// V * ZX
	MatMulCN(&V[node1][0][0], &tmp7[0][0], &tmp18[0][0], &tmp17[0], 1, 30, 1, 128, 64, 0, 5);


	// tanh(V1)
	TanH(&tmp18[0][0], 1, 1, 2048);


	// W1 <*> V1_tanh
	MulCir(&tmp16[0][0], &tmp18[0][0], &tmp19[0][0], 1, 1, 64, 32);


	// score0 + tmp19
	MatAdd(&tmp12[0][0], &tmp19[0][0], &tmp20[0][0], 1, 1, 1, 1, 1);


	// T * ZX
	MatMulCN(&T[node1][0][0], &tmp7[0][0], &tmp22[0][0], &tmp21[0], 1, 30, 1, 128, 128, 1, 4);

	node2 = ((tmp22[0][0] > 0) ? ((2 * node1) + 1) : ((2 * node1) + 2));

	// W * ZX
	MatMulCN(&W[node2][0][0], &tmp7[0][0], &tmp24[0][0], &tmp23[0], 1, 30, 1, 128, 128, 0, 5);


	// V * ZX
	MatMulCN(&V[node2][0][0], &tmp7[0][0], &tmp26[0][0], &tmp25[0], 1, 30, 1, 128, 64, 0, 5);


	// tanh(V2)
	TanH(&tmp26[0][0], 1, 1, 2048);


	// W2 <*> V2_tanh
	MulCir(&tmp24[0][0], &tmp26[0][0], &tmp27[0][0], 1, 1, 64, 32);


	// score1 + tmp27
	MatAdd(&tmp20[0][0], &tmp27[0][0], &tmp28[0][0], 1, 1, 1, 1, 1);


	// T * ZX
	MatMulCN(&T[node2][0][0], &tmp7[0][0], &tmp30[0][0], &tmp29[0], 1, 30, 1, 128, 128, 1, 4);

	node3 = ((tmp30[0][0] > 0) ? ((2 * node2) + 1) : ((2 * node2) + 2));

	// W * ZX
	MatMulCN(&W[node3][0][0], &tmp7[0][0], &tmp32[0][0], &tmp31[0], 1, 30, 1, 128, 128, 0, 5);


	// V * ZX
	MatMulCN(&V[node3][0][0], &tmp7[0][0], &tmp34[0][0], &tmp33[0], 1, 30, 1, 128, 64, 0, 5);


	// tanh(V3)
	TanH(&tmp34[0][0], 1, 1, 2048);


	// W3 <*> V3_tanh
	MulCir(&tmp32[0][0], &tmp34[0][0], &tmp35[0][0], 1, 1, 64, 32);


	// score2 + tmp35
	MatAdd(&tmp28[0][0], &tmp35[0][0], &tmp36[0][0], 1, 1, 1, 1, 1);


	// sgn(score3)
	tmp37 = ((tmp36[0][0] > 0) ? 1 : 0);

	return tmp37;
}
