#include <iostream>

#include "datatypes.h"
#include "predictors.h"
#include "library_float.h"
#include "seedot_float\testing\model.h"

using namespace std;
using namespace bonsai_float;

int seedotFloat(float **X) {
	float tmp6[10][1];
	float tmp7[10][1];
	MYINT node0;
	float tmp9[1][1];
	float tmp8[10];
	float tmp11[1][1];
	float tmp10[10];
	float tmp12[1][1];
	float tmp14[1][1];
	float tmp13[10];
	MYINT node1;
	float tmp16[1][1];
	float tmp15[10];
	float tmp18[1][1];
	float tmp17[10];
	float tmp19[1][1];
	float tmp20[1][1];
	float tmp22[1][1];
	float tmp21[10];
	MYINT node2;
	float tmp24[1][1];
	float tmp23[10];
	float tmp26[1][1];
	float tmp25[10];
	float tmp27[1][1];
	float tmp28[1][1];
	MYINT tmp29;



	// Z |*| X
	memset(tmp6, 0, sizeof(float) * 10);
	SparseMatMul(&Zidx[0], &Zval[0], X, &tmp6[0][0], 1001, 128, 128, 8);


	// tmp6 - mean
	MatSub(&tmp6[0][0], &mean[0][0], &tmp7[0][0], 10, 1, 1, 512, 1);

	node0 = 0;

	// W * ZX
	MatMulCN(&W[node0][0][0], &tmp7[0][0], &tmp9[0][0], &tmp8[0], 1, 10, 1, 128, 64, 0, 4);


	// V * ZX
	MatMulCN(&V[node0][0][0], &tmp7[0][0], &tmp11[0][0], &tmp10[0], 1, 10, 1, 128, 128, 0, 4);


	// tanh(V0)
	TanH(&tmp11[0][0], 1, 1, 1.000000f);


	// W0 <*> V0_tanh
	MulCir(&tmp9[0][0], &tmp11[0][0], &tmp12[0][0], 1, 1, 16, 16);


	// T * ZX
	MatMulCN(&T[node0][0][0], &tmp7[0][0], &tmp14[0][0], &tmp13[0], 1, 10, 1, 64, 64, 0, 4);

	node1 = ((tmp14[0][0] > 0) ? ((2 * node0) + 1) : ((2 * node0) + 2));

	// W * ZX
	MatMulCN(&W[node1][0][0], &tmp7[0][0], &tmp16[0][0], &tmp15[0], 1, 10, 1, 128, 64, 0, 4);


	// V * ZX
	MatMulCN(&V[node1][0][0], &tmp7[0][0], &tmp18[0][0], &tmp17[0], 1, 10, 1, 128, 128, 0, 4);


	// tanh(V1)
	TanH(&tmp18[0][0], 1, 1, 1.000000f);


	// W1 <*> V1_tanh
	MulCir(&tmp16[0][0], &tmp18[0][0], &tmp19[0][0], 1, 1, 16, 16);


	// score0 + tmp19
	MatAdd(&tmp12[0][0], &tmp19[0][0], &tmp20[0][0], 1, 1, 1, 1, 1);


	// T * ZX
	MatMulCN(&T[node1][0][0], &tmp7[0][0], &tmp22[0][0], &tmp21[0], 1, 10, 1, 64, 64, 0, 4);

	node2 = ((tmp22[0][0] > 0) ? ((2 * node1) + 1) : ((2 * node1) + 2));

	// W * ZX
	MatMulCN(&W[node2][0][0], &tmp7[0][0], &tmp24[0][0], &tmp23[0], 1, 10, 1, 128, 64, 0, 4);


	// V * ZX
	MatMulCN(&V[node2][0][0], &tmp7[0][0], &tmp26[0][0], &tmp25[0], 1, 10, 1, 128, 128, 0, 4);


	// tanh(V2)
	TanH(&tmp26[0][0], 1, 1, 1.000000f);


	// W2 <*> V2_tanh
	MulCir(&tmp24[0][0], &tmp26[0][0], &tmp27[0][0], 1, 1, 16, 16);


	// score1 + tmp27
	MatAdd(&tmp20[0][0], &tmp27[0][0], &tmp28[0][0], 1, 1, 1, 1, 1);


	// sgn(score2)
	tmp29 = ((tmp28[0][0] > 0) ? 1 : 0);

	return tmp29;
}
