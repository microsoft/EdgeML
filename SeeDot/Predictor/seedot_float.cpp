#include <iostream>

#include "datatypes.h"
#include "predictors.h"
#include "library_float.h"
#include "model_float.h"

using namespace std;
using namespace protonn_float;

int seedotFloat(float **X) {
	float tmp4;
	float tmp5[20][1];
	MYINT i;
	float tmp6[20][1];
	float tmp7;
	float tmp8[1][20];
	float tmp10[1][1];
	float tmp9[20];
	float tmp11[1][1];
	float tmp12[1][1];
	float tmp14[2][1];
	float tmp13[1];
	float tmp15[2][1];
	MYINT tmp16;

	tmp4 = 0.000515;


	// W |*| X
	memset(tmp5, 0, sizeof(float) * 20);
	SparseMatMul(&Widx[0], &Wval[0], X, &tmp5[0][0], 400, 128, 128, 512);

	memset(tmp15, 0, sizeof(float) * 2);
	i = 0;
	for (MYINT i0 = 0; (i0 < 40); i0++) {

		// WX - B
		MatSub(&tmp5[0][0], &B[i][0][0], &tmp6[0][0], 20, 1, 1, 128, 1);

		tmp7 = (-tmp4);

		// del^T
		Transpose(&tmp6[0][0], &tmp8[0][0], 1, 20);


		// tmp8 * del
		MatMulNN(&tmp8[0][0], &tmp6[0][0], &tmp10[0][0], &tmp9[0], 1, 20, 1, 1, 1, 0, 5);


		// tmp7 * tmp10
		ScalarMul(&tmp7, &tmp10[0][0], &tmp11[0][0], 1, 1, 128, 128);


		// exp(tmp11)
		Exp(&tmp11[0][0], 1, 1, 32767, 32767, &tmp12[0][0]);


		// Z * tmp12
		MatMulCN(&Z[i][0][0], &tmp12[0][0], &tmp14[0][0], &tmp13[0], 2, 1, 1, 128, 128, 0, 0);

		for (MYINT i1 = 0; (i1 < 2); i1++) {
			for (MYINT i2 = 0; (i2 < 1); i2++) {
				tmp15[i1][i2] = (tmp15[i1][i2] + (tmp14[i1][i2] / 64));
			}
		}
		i = (i + 1);
	}

	// argmax(res)
	ArgMax(&tmp15[0][0], 2, 1, &tmp16);


	return tmp16;
}
