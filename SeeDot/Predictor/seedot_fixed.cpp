// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <iostream>

#include "datatypes.h"
#include "predictors.h"
#include "library.h"
#include "model.h"

using namespace std;
using namespace protonn_fixed;

const MYINT EXP8A[64] = {
	8192, 3013, 1108, 407, 150, 55, 20, 7, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
};
const MYINT EXP8B[64] = {
	12962, 12761, 12563, 12368, 12177, 11988, 11802, 11619, 11439, 11261, 11087, 10915, 10746, 10579, 10415, 10254, 10095, 9938, 9784, 9632, 9483, 9336, 9191, 9049, 8908, 8770, 8634, 8500, 8369, 8239, 8111, 7985, 7862, 7740, 7620, 7502, 7385, 7271, 7158, 7047, 6938, 6830, 6724, 6620, 6517, 6416, 6317, 6219, 6123, 6028, 5934, 5842, 5752, 5662, 5575, 5488, 5403, 5319, 5237, 5156, 5076, 4997, 4919, 4843, 
};

int seedotFixed(MYINT **X) {
	MYINT tmp4;
	MYINT tmp5[20][1];
	MYINT i;
	MYINT tmp6[20][1];
	MYINT tmp7;
	MYINT tmp8[1][20];
	MYINT tmp9[1][1];
	MYINT tmp10[20];
	MYINT tmp11[1][1];
	MYINT tmp15[1][1];
	MYINT tmp12;
	MYINT tmp13;
	MYINT tmp14;
	MYINT tmp16[2][1];
	MYINT tmp17[1];
	MYINT tmp18[2][1];
	MYINT tmp19;

	tmp4 = 8640;


	// W |*| X
	memset(tmp5, 0, sizeof(MYINT) * 20);
	SparseMatMul(&Widx[0], &Wval[0], X, &tmp5[0][0], 400, 128, 128, 128);

	memset(tmp18, 0, sizeof(MYINT) * 2);
	i = 0;
	for (MYINT i0 = 0; (i0 < 40); i0++) {

		// WX - B
		MatSub(&tmp5[0][0], &B[i][0][0], &tmp6[0][0], 20, 1, 1, 32, 1);

		tmp7 = (-tmp4);

		// del^T
		Transpose(&tmp6[0][0], &tmp8[0][0], 1, 20);


		// tmp8 * del
		MatMulNN(&tmp8[0][0], &tmp6[0][0], &tmp9[0][0], &tmp10[0], 1, 20, 1, 8, 8, 0, 5);


		// tmp7 * tmp9
		ScalarMul(&tmp7, &tmp9[0][0], &tmp11[0][0], 1, 1, 128, 128);


		// exp(tmp11)
		if (((-tmp11[0][0]) < 59)) {
			tmp13 = 0;
			tmp14 = 0;
		} else {
			tmp12 = (((-tmp11[0][0]) - 59) << 2);
			tmp13 = ((tmp12 >> 10) & 63);
			tmp14 = ((tmp12 >> 4) & 63);
		}
		tmp15[0][0] = ((EXP8A[tmp13] >> 7) * (EXP8B[tmp14] >> 7));

		// Z * tmp15
		MatMulCN(&Z[i][0][0], &tmp15[0][0], &tmp16[0][0], &tmp17[0], 2, 1, 1, 128, 128, 0, 0);

		for (MYINT i1 = 0; (i1 < 2); i1++) {
			for (MYINT i2 = 0; (i2 < 1); i2++) {
				tmp18[i1][i2] = (tmp18[i1][i2] + (tmp16[i1][i2] / 64));
			}
		}
		i = (i + 1);
	}

	// argmax(res)
	ArgMax(&tmp18[0][0], 2, 1, &tmp19);


	return tmp19;
}
