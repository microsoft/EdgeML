#include <iostream>

#include "datatypes.h"
#include "predictors.h"
#include "library_float.h"
#include "model_float.h"

using namespace std;
using namespace bonsai_float;

int seedotFloat(float **X) {
	float tmp6[20][1];
	float tmp7[20][1];
	MYINT node0;
	float tmp9[1][1];
	float tmp8[20];
	float tmp11[1][1];
	float tmp10[20];
	float tmp12[1][1];
	float tmp14[1][1];
	float tmp13[20];
	MYINT node1;
	float tmp16[1][1];
	float tmp15[20];
	float tmp18[1][1];
	float tmp17[20];
	float tmp19[1][1];
	float tmp20[1][1];
	float tmp22[1][1];
	float tmp21[20];
	MYINT node2;
	float tmp24[1][1];
	float tmp23[20];
	float tmp26[1][1];
	float tmp25[20];
	float tmp27[1][1];
	float tmp28[1][1];
	float tmp30[1][1];
	float tmp29[20];
	MYINT node3;
	float tmp32[1][1];
	float tmp31[20];
	float tmp34[1][1];
	float tmp33[20];
	float tmp35[1][1];
	float tmp36[1][1];
	MYINT tmp37;



	// Z |*| X
	memset(tmp6, 0, sizeof(float) * 20);
	SparseMatMul(&Zidx[0], &Zval[0], X, &tmp6[0][0], 401, 128, 128, 4);


	// tmp6 - mean
	MatSub(&tmp6[0][0], &mean[0][0], &tmp7[0][0], 20, 1, 1, 256, 1);

	node0 = 0;

	// W * ZX
	MatMulCN(&W[node0][0][0], &tmp7[0][0], &tmp9[0][0], &tmp8[0], 1, 20, 1, 128, 64, 0, 5);


	// V * ZX
	MatMulCN(&V[node0][0][0], &tmp7[0][0], &tmp11[0][0], &tmp10[0], 1, 20, 1, 128, 64, 0, 5);


	// tanh(V0)
	TanH(&tmp11[0][0], 1, 1, 1.000000f);


	// W0 <*> V0_tanh
	MulCir(&tmp9[0][0], &tmp11[0][0], &tmp12[0][0], 1, 1, 64, 32);


	// T * ZX
	MatMulCN(&T[node0][0][0], &tmp7[0][0], &tmp14[0][0], &tmp13[0], 1, 20, 1, 128, 64, 0, 5);

	node1 = ((tmp14[0][0] > 0) ? ((2 * node0) + 1) : ((2 * node0) + 2));

	// W * ZX
	MatMulCN(&W[node1][0][0], &tmp7[0][0], &tmp16[0][0], &tmp15[0], 1, 20, 1, 128, 64, 0, 5);


	// V * ZX
	MatMulCN(&V[node1][0][0], &tmp7[0][0], &tmp18[0][0], &tmp17[0], 1, 20, 1, 128, 64, 0, 5);


	// tanh(V1)
	TanH(&tmp18[0][0], 1, 1, 1.000000f);


	// W1 <*> V1_tanh
	MulCir(&tmp16[0][0], &tmp18[0][0], &tmp19[0][0], 1, 1, 64, 32);


	// score0 + tmp19
	MatAdd(&tmp12[0][0], &tmp19[0][0], &tmp20[0][0], 1, 1, 1, 1, 1);


	// T * ZX
	MatMulCN(&T[node1][0][0], &tmp7[0][0], &tmp22[0][0], &tmp21[0], 1, 20, 1, 128, 64, 0, 5);

	node2 = ((tmp22[0][0] > 0) ? ((2 * node1) + 1) : ((2 * node1) + 2));

	// W * ZX
	MatMulCN(&W[node2][0][0], &tmp7[0][0], &tmp24[0][0], &tmp23[0], 1, 20, 1, 128, 64, 0, 5);


	// V * ZX
	MatMulCN(&V[node2][0][0], &tmp7[0][0], &tmp26[0][0], &tmp25[0], 1, 20, 1, 128, 64, 0, 5);


	// tanh(V2)
	TanH(&tmp26[0][0], 1, 1, 1.000000f);


	// W2 <*> V2_tanh
	MulCir(&tmp24[0][0], &tmp26[0][0], &tmp27[0][0], 1, 1, 64, 32);


	// score1 + tmp27
	MatAdd(&tmp20[0][0], &tmp27[0][0], &tmp28[0][0], 1, 1, 1, 1, 1);


	// T * ZX
	MatMulCN(&T[node2][0][0], &tmp7[0][0], &tmp30[0][0], &tmp29[0], 1, 20, 1, 128, 64, 0, 5);

	node3 = ((tmp30[0][0] > 0) ? ((2 * node2) + 1) : ((2 * node2) + 2));

	// W * ZX
	MatMulCN(&W[node3][0][0], &tmp7[0][0], &tmp32[0][0], &tmp31[0], 1, 20, 1, 128, 64, 0, 5);


	// V * ZX
	MatMulCN(&V[node3][0][0], &tmp7[0][0], &tmp34[0][0], &tmp33[0], 1, 20, 1, 128, 64, 0, 5);


	// tanh(V3)
	TanH(&tmp34[0][0], 1, 1, 1.000000f);


	// W3 <*> V3_tanh
	MulCir(&tmp32[0][0], &tmp34[0][0], &tmp35[0][0], 1, 1, 64, 32);


	// score2 + tmp35
	MatAdd(&tmp28[0][0], &tmp35[0][0], &tmp36[0][0], 1, 1, 1, 1, 1);


	// sgn(score3)
	tmp37 = ((tmp36[0][0] > 0) ? 1 : 0);

	return tmp37;
}
