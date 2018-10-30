#include <iostream>

#include "datatypes.h"
#include "bonsai.h"
#include "Arduino.h"
#include "model.h"

using namespace std;
using namespace bonsai_fixed;

int bonsaiFixed(MYINT **X) {
	MYINT tmp6[12][1];
	MYINT tmp7[12][1];
	MYINT node0;
	MYINT tmp8[61][1];
	MYINT tmp9[12];
	MYINT tmp10[61][1];
	MYINT tmp11[12];
	MYINT tmp12[61][1];
	MYINT tmp13[1][1];
	MYINT tmp14[12];
	MYINT node1;
	MYINT tmp15[61][1];
	MYINT tmp16[12];
	MYINT tmp17[61][1];
	MYINT tmp18[12];
	MYINT tmp19[61][1];
	MYINT tmp20[61][1];
	MYINT tmp21;



	// Z |*| X
	memset(tmp6, 0, sizeof(MYINT) * 12);
	SparseMatMul(&Zidx[0], &Zval[0], X, &tmp6[0][0], 611, 128, 128, 4);


	// tmp6 - mean
	MatSub(&tmp6[0][0], &mean[0][0], &tmp7[0][0], 12, 1, 1, 512);

	node0 = 0;

	// W * ZX
	MatMulCN(&W[node0][0][0], &tmp7[0][0], &tmp8[0][0], &tmp9[0], 61, 12, 1, 64, 64, 0, 4);


	// V * ZX
	MatMulCN(&V[node0][0][0], &tmp7[0][0], &tmp10[0][0], &tmp11[0], 61, 12, 1, 64, 64, 0, 4);


	// tanh(V0)
	TanH(&tmp10[0][0], 61, 1, 2048);


	// W0 <*> V0_tanh
	MulCir(&tmp8[0][0], &tmp10[0][0], &tmp12[0][0], 61, 1, 64, 32);


	// T * ZX
	MatMulCN(&T[node0][0][0], &tmp7[0][0], &tmp13[0][0], &tmp14[0], 1, 12, 1, 128, 128, 0, 4);

	node1 = ((tmp13[0][0] > 0) ? ((2 * node0) + 1) : ((2 * node0) + 2));

	// W * ZX
	MatMulCN(&W[node1][0][0], &tmp7[0][0], &tmp15[0][0], &tmp16[0], 61, 12, 1, 64, 64, 0, 4);


	// V * ZX
	MatMulCN(&V[node1][0][0], &tmp7[0][0], &tmp17[0][0], &tmp18[0], 61, 12, 1, 64, 64, 0, 4);


	// tanh(V1)
	TanH(&tmp17[0][0], 61, 1, 2048);


	// W1 <*> V1_tanh
	MulCir(&tmp15[0][0], &tmp17[0][0], &tmp19[0][0], 61, 1, 64, 32);


	// score0 + tmp19
	MatAdd(&tmp12[0][0], &tmp19[0][0], &tmp20[0][0], 61, 1, 1, 1);


	// argmax(score1)
	ArgMax(&tmp20[0][0], 61, 1, &tmp21);


	return tmp21;
}
