#include <iostream>

#include "datatypes.h"
#include "protonn.h"
#include "Arduino.h"
#include "model.h"

using namespace std;
using namespace protonn_fixed;

const MYINT EXP7A[64] = {
	8192, 1108, 150, 20, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
};
const MYINT EXP7B[64] = {
	14466, 14021, 13590, 13171, 12766, 12373, 11993, 11624, 11266, 10919, 10583, 10258, 9942, 9636, 9340, 9052, 8774, 8504, 8242, 7989, 7743, 7505, 7274, 7050, 6833, 6623, 6419, 6221, 6030, 5844, 5665, 5490, 5321, 5158, 4999, 4845, 4696, 4552, 4412, 4276, 4144, 4017, 3893, 3773, 3657, 3545, 3436, 3330, 3227, 3128, 3032, 2939, 2848, 2760, 2676, 2593, 2513, 2436, 2361, 2288, 2218, 2150, 2084, 2019, 
};

int protonnFixed(MYINT **X) {
	MYINT tmp4;
	MYINT tmp5[10][1];
	MYINT i;
	MYINT tmp6[10][1];
	MYINT tmp7;
	MYINT tmp8[1][10];
	MYINT tmp9[1][1];
	MYINT tmp10[10];
	MYINT tmp11[1][1];
	MYINT tmp15[1][1];
	MYINT tmp12;
	MYINT tmp13;
	MYINT tmp14;
	MYINT tmp16[61][1];
	MYINT tmp17[1];
	MYINT tmp18[61][1];
	MYINT tmp19;

	tmp4 = 8321;


	// W |*| X
	memset(tmp5, 0, sizeof(MYINT) * 10);
	SparseMatMul(&Widx[0], &Wval[0], X, &tmp5[0][0], 610, 128, 128, 64);

	memset(tmp18, 0, sizeof(MYINT) * 61);
	i = 0;
	for (MYINT i0 = 0; (i0 < 61); i0++) {

		// WX - B
		MatSub(&tmp5[0][0], &B[i][0][0], &tmp6[0][0], 10, 1, 1, 32);

		tmp7 = (-tmp4);

		// del^T
		Transpose(&tmp6[0][0], &tmp8[0][0], 1, 10);


		// tmp8 * del
		MatMulNN(&tmp8[0][0], &tmp6[0][0], &tmp9[0][0], &tmp10[0], 1, 10, 1, 8, 8, 0, 4);


		// tmp7 * tmp9
		ScalarMul(&tmp7, &tmp9[0][0], &tmp11[0][0], 1, 1, 128, 128);


		// exp(tmp11)
		if (((-tmp11[0][0]) < 15)) {
			tmp13 = 0;
			tmp14 = 0;
		} else {
			tmp12 = (((-tmp11[0][0]) - 15) << 2);
			tmp13 = ((tmp12 >> 10) & 63);
			tmp14 = ((tmp12 >> 4) & 63);
		}
		tmp15[0][0] = ((EXP7A[tmp13] >> 7) * (EXP7B[tmp14] >> 7));

		// Z * tmp15
		MatMulCN(&Z[i][0][0], &tmp15[0][0], &tmp16[0][0], &tmp17[0], 61, 1, 1, 128, 128, 0, 0);

		for (MYINT i1 = 0; (i1 < 61); i1++) {
			for (MYINT i2 = 0; (i2 < 1); i2++) {
				tmp18[i1][i2] = (tmp18[i1][i2] + (tmp16[i1][i2] / 64));
			}
		}
		i = (i + 1);
	}

	// argmax(res)
	ArgMax(&tmp18[0][0], 61, 1, &tmp19);


	return tmp19;
}
