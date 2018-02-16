#include <iostream>

typedef int16_t MYINT;

#include "protonn.h"
#include "model.h"

using namespace std;
using namespace protonn_fixed;

const MYINT EXP9A[5] = {
	8192, 150, 2, 0, 0, 
};
const MYINT EXP9B[8] = {
	15040, 9122, 5533, 3356, 2035, 1234, 748, 454, 
};

int protonnFixed(MYINT **X) {
	MYINT tmp0;
	MYINT tmp1[10][1];
	MYINT tmp2;
	MYINT tmp3;
	MYINT tmp4;
	MYINT tmp5;
	MYINT tmp6;
	MYINT tmp7;
	MYINT tmp8[10][1];
	MYINT tmp9;
	MYINT tmp10;
	MYINT i;
	MYINT tmp11[10][1];
	MYINT tmp12;
	MYINT tmp13;
	MYINT tmp14;
	MYINT tmp15[1][10];
	MYINT tmp16[1][1];
	MYINT tmp17;
	MYINT tmp18;
	MYINT tmp19[10];
	MYINT tmp20;
	MYINT tmp21;
	MYINT tmp22[1][1];
	MYINT tmp23;
	MYINT tmp24[1][1];
	MYINT tmp28[1][1];
	MYINT tmp25;
	MYINT tmp26;
	MYINT tmp27;
	MYINT tmp29[2][1];
	MYINT tmp30;
	MYINT tmp31;
	MYINT tmp32[1];
	MYINT tmp33;
	MYINT tmp34;
	MYINT tmp35[2][1];
	MYINT tmp38;
	MYINT tmp36;
	MYINT tmp37;

	tmp0 = 9728;


	// WW |*| XX
	memset(tmp1, 0, sizeof(MYINT) * 10);
	tmp2 = 0;
	tmp3 = 0;
	for (MYINT i0 = 0; (i0 < 400); i0++) {
		tmp6 = X[i0][0];
		tmp6 = (tmp6 / 128);
		tmp4 = Widx[tmp2];
		while ((tmp4 != 0)) {
			tmp5 = Wval[tmp3];
			tmp5 = (tmp5 / 128);
			tmp7 = (tmp5 * tmp6);
			tmp7 = (tmp7 / 32);
			tmp1[(tmp4 - 1)][0] = (tmp1[(tmp4 - 1)][0] + tmp7);
			tmp2 = (tmp2 + 1);
			tmp3 = (tmp3 + 1);
			tmp4 = Widx[tmp2];
		}
		tmp2 = (tmp2 + 1);
	}

	// tmp1 - Min
	for (MYINT i1 = 0; (i1 < 10); i1++) {
		for (MYINT i2 = 0; (i2 < 1); i2++) {
			tmp9 = tmp1[i1][i2];
			tmp10 = (min[i1][i2] / 8);
			tmp8[i1][i2] = (tmp9 - tmp10);
			tmp8[i1][i2] = tmp8[i1][i2];
		}
	}
	memset(tmp35, 0, sizeof(MYINT) * 2);
	i = 0;
	for (MYINT i19 = 0; (i19 < 25); i19++) {

		// WX - BB
		for (MYINT i3 = 0; (i3 < 10); i3++) {
			for (MYINT i4 = 0; (i4 < 1); i4++) {
				tmp12 = tmp8[i3][i4];
				tmp13 = (B[i][i3][i4] / 8);
				tmp11[i3][i4] = (tmp12 - tmp13);
				tmp11[i3][i4] = tmp11[i3][i4];
			}
		}
		tmp14 = (-tmp0);

		// del^T
		for (MYINT i5 = 0; (i5 < 1); i5++) {
			for (MYINT i6 = 0; (i6 < 10); i6++) {
				tmp15[i5][i6] = tmp11[i6][i5];
			}
		}

		// tmp15 * del
		for (MYINT i7 = 0; (i7 < 1); i7++) {
			for (MYINT i9 = 0; (i9 < 1); i9++) {
				for (MYINT i8 = 0; (i8 < 10); i8++) {
					tmp17 = (tmp15[i7][i8] / 8);
					tmp18 = (tmp11[i8][i9] / 8);
					tmp19[i8] = (tmp17 * tmp18);
				}
				tmp20 = 10;
				for (MYINT i10 = 0; (i10 < 0); i10++) {
					tmp21 = 0;
					for (MYINT i11 = 0; (i11 < 6); i11++) {
						tmp19[i11] = ((tmp21 < (tmp20 >> 1)) ? ((tmp19[(2 * i11)] + tmp19[((2 * i11) + 1)]) / 2) : (((tmp21 == (tmp20 >> 1)) && ((tmp20 & 1) == 1)) ? (tmp19[(2 * i11)] / 2) : 0));
						tmp21 = (tmp21 + 1);
					}
					tmp20 = ((tmp20 + 1) >> 1);
				}
				for (MYINT i10 = 0; (i10 < 4); i10++) {
					tmp21 = 0;
					for (MYINT i11 = 0; (i11 < 6); i11++) {
						tmp19[i11] = ((tmp21 < (tmp20 >> 1)) ? (tmp19[(2 * i11)] + tmp19[((2 * i11) + 1)]) : (((tmp21 == (tmp20 >> 1)) && ((tmp20 & 1) == 1)) ? tmp19[(2 * i11)] : 0));
						tmp21 = (tmp21 + 1);
					}
					tmp20 = ((tmp20 + 1) >> 1);
				}
				tmp16[i7][i9] = tmp19[0];
			}
		}
		tmp23 = (tmp14 / 128);
		for (MYINT i12 = 0; (i12 < 1); i12++) {
			for (MYINT i13 = 0; (i13 < 1); i13++) {
				tmp24[i12][i13] = (tmp16[i12][i13] / 128);
			}
		}
		for (MYINT i12 = 0; (i12 < 1); i12++) {
			for (MYINT i13 = 0; (i13 < 1); i13++) {
				tmp22[i12][i13] = (tmp23 * tmp24[i12][i13]);
			}
		}

		// exp(tmp22)
		if (((-tmp22[0][0]) < 43)) {
			tmp26 = 0;
			tmp27 = 0;
		} else {
			tmp25 = (((-tmp22[0][0]) - 43) << 2);
			tmp26 = ((tmp25 >> 13) & 7);
			tmp27 = ((tmp25 >> 10) & 7);
		}
		tmp28[0][0] = ((EXP9A[tmp26] >> 7) * (EXP9B[tmp27] >> 7));

		// ZZ * tmp28
		for (MYINT i14 = 0; (i14 < 2); i14++) {
			for (MYINT i16 = 0; (i16 < 1); i16++) {
				for (MYINT i15 = 0; (i15 < 1); i15++) {
					tmp30 = (Z[i][i14][i15] / 128);
					tmp31 = (tmp28[i15][i16] / 128);
					tmp32[i15] = (tmp30 * tmp31);
				}
				tmp33 = 1;
				for (MYINT i17 = 0; (i17 < 0); i17++) {
					tmp34 = 0;
					for (MYINT i18 = 0; (i18 < 1); i18++) {
						tmp32[i18] = ((tmp34 < (tmp33 >> 1)) ? ((tmp32[(2 * i18)] + tmp32[((2 * i18) + 1)]) / 2) : (((tmp34 == (tmp33 >> 1)) && ((tmp33 & 1) == 1)) ? (tmp32[(2 * i18)] / 2) : 0));
						tmp34 = (tmp34 + 1);
					}
					tmp33 = ((tmp33 + 1) >> 1);
				}
				for (MYINT i17 = 0; (i17 < 0); i17++) {
					tmp34 = 0;
					for (MYINT i18 = 0; (i18 < 1); i18++) {
						tmp32[i18] = ((tmp34 < (tmp33 >> 1)) ? (tmp32[(2 * i18)] + tmp32[((2 * i18) + 1)]) : (((tmp34 == (tmp33 >> 1)) && ((tmp33 & 1) == 1)) ? tmp32[(2 * i18)] : 0));
						tmp34 = (tmp34 + 1);
					}
					tmp33 = ((tmp33 + 1) >> 1);
				}
				tmp29[i14][i16] = tmp32[0];
			}
		}
		for (MYINT i20 = 0; (i20 < 2); i20++) {
			for (MYINT i21 = 0; (i21 < 1); i21++) {
				tmp35[i20][i21] = (tmp35[i20][i21] + (tmp29[i20][i21] / 32));
			}
		}
		i = (i + 1);
	}
	tmp38 = 0;
	tmp36 = 0;
	tmp37 = tmp35[0][0];
	for (MYINT i22 = 0; (i22 < 2); i22++) {
		for (MYINT i23 = 0; (i23 < 1); i23++) {
			if ((tmp37 < tmp35[i22][i23])) {
				tmp36 = tmp38;
				tmp37 = tmp35[i22][i23];
			}
			tmp38 = (tmp38 + 1);
		}
	}

	return tmp36;
}
