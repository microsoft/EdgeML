#include <Arduino.h>

#include "config.h"
#include "predict.h"
#include "library.h"
#include "model.h"

using namespace model;

const PROGMEM MYINT EXP13A[64] = {
	8192, 7695, 7229, 6791, 6379, 5993, 5630, 5289, 4968, 4667, 4384, 4119, 3869, 3635, 3414, 3208, 3013, 2831, 2659, 2498, 2347, 2204, 2071, 1945, 1827, 1717, 1613, 1515, 1423, 1337, 1256, 1180, 1108, 1041, 978, 919, 863, 811, 761, 715, 672, 631, 593, 557, 523, 491, 462, 434, 407, 383, 359, 338, 317, 298, 280, 263, 247, 232, 218, 205, 192, 180, 170, 159, 
};
const PROGMEM MYINT EXP13B[64] = {
	15967, 15952, 15936, 15921, 15905, 15890, 15874, 15859, 15843, 15828, 15812, 15797, 15781, 15766, 15751, 15735, 15720, 15704, 15689, 15674, 15659, 15643, 15628, 15613, 15598, 15582, 15567, 15552, 15537, 15522, 15506, 15491, 15476, 15461, 15446, 15431, 15416, 15401, 15386, 15371, 15356, 15341, 15326, 15311, 15296, 15281, 15266, 15251, 15236, 15221, 15206, 15192, 15177, 15162, 15147, 15132, 15118, 15103, 15088, 15073, 15059, 15044, 15029, 15015, 
};

int predict() {
	MYINT tmp4;
	MYINT tmp5[25][1];
	MYINT i;
	MYINT tmp6[25][1];
	MYINT tmp7;
	MYINT tmp8[1][25];
	MYINT tmp10[1][1];
	MYINT tmp9[25];
	MYINT tmp11[1][1];
	MYINT tmp15[1][1];
	MYINT tmp12;
	MYINT tmp13;
	MYINT tmp14;
	MYINT tmp17[10][1];
	MYINT tmp16[1];
	MYINT tmp18[10][1];
	MYINT tmp19;

	tmp4 = 16106;


	// W |*| X
	memset(tmp5, 0, sizeof(MYINT) * 25);
	SparseMatMul(&Widx[0], &Wval[0], &tmp5[0][0], 256, 128, 128, 8);

	memset(tmp18, 0, sizeof(MYINT) * 10);
	i = 0;
	for (MYINT i0 = 0; (i0 < 55); i0++) {

		// WX - B
		MatSub(&tmp5[0][0], &B[i][0][0], &tmp6[0][0], 25, 1, 1, 4, 1);

		tmp7 = (-tmp4);

		// del^T
		Transpose(&tmp6[0][0], &tmp8[0][0], 1, 25);


		// tmp8 * del
		MatMulNN(&tmp8[0][0], &tmp6[0][0], &tmp10[0][0], &tmp9[0], 1, 25, 1, 128, 64, 0, 5);


		// tmp7 * tmp10
		ScalarMul(&tmp7, &tmp10[0][0], &tmp11[0][0], 1, 1, 128, 128);


		// exp(tmp11)
		if (((-tmp11[0][0]) < 210)) {
			tmp13 = 0;
			tmp14 = 0;
		} else {
			tmp12 = (((-tmp11[0][0]) - 210) << 1);
			tmp13 = ((tmp12 >> 10) & 63);
			tmp14 = ((tmp12 >> 4) & 63);
		}
		tmp15[0][0] = ((((MYINT) pgm_read_word_near(&EXP13A[tmp13])) >> 7) * (((MYINT) pgm_read_word_near(&EXP13B[tmp14])) >> 7));

		// Z * tmp15
		MatMulCN(&Z[i][0][0], &tmp15[0][0], &tmp17[0][0], &tmp16[0], 10, 1, 1, 128, 128, 0, 0);

		for (MYINT i1 = 0; (i1 < 10); i1++) {
			for (MYINT i2 = 0; (i2 < 1); i2++) {
				tmp18[i1][i2] = (tmp18[i1][i2] + (tmp17[i1][i2] / 32));
			}
		}
		i = (i + 1);
	}

	// argmax(res)
	ArgMax(&tmp18[0][0], 10, 1, &tmp19);


	return tmp19;
}
