#include <iostream>

typedef int16_t MYINT;

#include "bonsai.h"
#include "model.h"

using namespace std;
using namespace bonsai_fixed;

int bonsaiFixed(MYINT **X) {
	MYINT tmp0[10][1];
	MYINT tmp1;
	MYINT tmp2;
	MYINT tmp3;
	MYINT tmp4;
	MYINT tmp5;
	MYINT tmp6;
	MYINT tmp7[10][1];
	MYINT tmp8;
	MYINT tmp9;
	MYINT node0;
	MYINT tmp10[1][1];
	MYINT tmp11;
	MYINT tmp12;
	MYINT tmp13[10];
	MYINT tmp14;
	MYINT tmp15;
	MYINT tmp16[1][1];
	MYINT tmp17;
	MYINT tmp18;
	MYINT tmp19[10];
	MYINT tmp20;
	MYINT tmp21;
	MYINT tmp22[1][1];
	MYINT tmp23[1][1];
	MYINT tmp24;
	MYINT tmp25;
	MYINT tmp26[10];
	MYINT tmp27;
	MYINT tmp28;
	MYINT node1;
	MYINT tmp29[1][1];
	MYINT tmp30;
	MYINT tmp31;
	MYINT tmp32[10];
	MYINT tmp33;
	MYINT tmp34;
	MYINT tmp35[1][1];
	MYINT tmp36;
	MYINT tmp37;
	MYINT tmp38[10];
	MYINT tmp39;
	MYINT tmp40;
	MYINT tmp41[1][1];
	MYINT tmp42[1][1];
	MYINT tmp43;
	MYINT tmp44;
	MYINT tmp45;



	// ZZ |*| XX
	memset(tmp0, 0, sizeof(MYINT) * 10);
	tmp1 = 0;
	tmp2 = 0;
	for (MYINT i0 = 0; (i0 < 1001); i0++) {
		tmp5 = X[i0][0];
		tmp5 = (tmp5 / 128);
		tmp3 = Zidx[tmp1];
		while ((tmp3 != 0)) {
			tmp4 = Zval[tmp2];
			tmp4 = (tmp4 / 128);
			tmp6 = (tmp4 * tmp5);
			tmp6 = (tmp6 / 8);
			tmp0[(tmp3 - 1)][0] = (tmp0[(tmp3 - 1)][0] + tmp6);
			tmp1 = (tmp1 + 1);
			tmp2 = (tmp2 + 1);
			tmp3 = Zidx[tmp1];
		}
		tmp1 = (tmp1 + 1);
	}

	// tmp0 - Mean
	for (MYINT i1 = 0; (i1 < 10); i1++) {
		for (MYINT i2 = 0; (i2 < 1); i2++) {
			tmp8 = tmp0[i1][i2];
			tmp9 = (mean[i1][i2] / 1024);
			tmp7[i1][i2] = (tmp8 - tmp9);
			tmp7[i1][i2] = tmp7[i1][i2];
		}
	}
	node0 = 0;

	// WW * ZX
	for (MYINT i3 = 0; (i3 < 1); i3++) {
		for (MYINT i5 = 0; (i5 < 1); i5++) {
			for (MYINT i4 = 0; (i4 < 10); i4++) {
				tmp11 = (W[node0][i3][i4] / 128);
				tmp12 = (tmp7[i4][i5] / 64);
				tmp13[i4] = (tmp11 * tmp12);
			}
			tmp14 = 10;
			for (MYINT i6 = 0; (i6 < 0); i6++) {
				tmp15 = 0;
				for (MYINT i7 = 0; (i7 < 6); i7++) {
					tmp13[i7] = ((tmp15 < (tmp14 >> 1)) ? ((tmp13[(2 * i7)] + tmp13[((2 * i7) + 1)]) / 2) : (((tmp15 == (tmp14 >> 1)) && ((tmp14 & 1) == 1)) ? (tmp13[(2 * i7)] / 2) : 0));
					tmp15 = (tmp15 + 1);
				}
				tmp14 = ((tmp14 + 1) >> 1);
			}
			for (MYINT i6 = 0; (i6 < 4); i6++) {
				tmp15 = 0;
				for (MYINT i7 = 0; (i7 < 6); i7++) {
					tmp13[i7] = ((tmp15 < (tmp14 >> 1)) ? (tmp13[(2 * i7)] + tmp13[((2 * i7) + 1)]) : (((tmp15 == (tmp14 >> 1)) && ((tmp14 & 1) == 1)) ? tmp13[(2 * i7)] : 0));
					tmp15 = (tmp15 + 1);
				}
				tmp14 = ((tmp14 + 1) >> 1);
			}
			tmp10[i3][i5] = tmp13[0];
		}
	}

	// VV * ZX
	for (MYINT i8 = 0; (i8 < 1); i8++) {
		for (MYINT i10 = 0; (i10 < 1); i10++) {
			for (MYINT i9 = 0; (i9 < 10); i9++) {
				tmp17 = (V[node0][i8][i9] / 128);
				tmp18 = (tmp7[i9][i10] / 64);
				tmp19[i9] = (tmp17 * tmp18);
			}
			tmp20 = 10;
			for (MYINT i11 = 0; (i11 < 0); i11++) {
				tmp21 = 0;
				for (MYINT i12 = 0; (i12 < 6); i12++) {
					tmp19[i12] = ((tmp21 < (tmp20 >> 1)) ? ((tmp19[(2 * i12)] + tmp19[((2 * i12) + 1)]) / 2) : (((tmp21 == (tmp20 >> 1)) && ((tmp20 & 1) == 1)) ? (tmp19[(2 * i12)] / 2) : 0));
					tmp21 = (tmp21 + 1);
				}
				tmp20 = ((tmp20 + 1) >> 1);
			}
			for (MYINT i11 = 0; (i11 < 4); i11++) {
				tmp21 = 0;
				for (MYINT i12 = 0; (i12 < 6); i12++) {
					tmp19[i12] = ((tmp21 < (tmp20 >> 1)) ? (tmp19[(2 * i12)] + tmp19[((2 * i12) + 1)]) : (((tmp21 == (tmp20 >> 1)) && ((tmp20 & 1) == 1)) ? tmp19[(2 * i12)] : 0));
					tmp21 = (tmp21 + 1);
				}
				tmp20 = ((tmp20 + 1) >> 1);
			}
			tmp16[i8][i10] = tmp19[0];
		}
	}

	// tanh(V0)
	for (MYINT i13 = 0; (i13 < 1); i13++) {
		for (MYINT i14 = 0; (i14 < 1); i14++) {
			tmp16[i13][i14] = ((tmp16[i13][i14] >= 256) ? 256 : ((tmp16[i13][i14] <= -256) ? -256 : tmp16[i13][i14]));
		}
	}
	for (MYINT i15 = 0; (i15 < 1); i15++) {
		for (MYINT i16 = 0; (i16 < 1); i16++) {
			tmp22[i15][i16] = ((tmp10[i15][i16] / 16) * (tmp16[i15][i16] / 16));
		}
	}

	// TT * ZX
	for (MYINT i17 = 0; (i17 < 1); i17++) {
		for (MYINT i19 = 0; (i19 < 1); i19++) {
			for (MYINT i18 = 0; (i18 < 10); i18++) {
				tmp24 = (T[node0][i17][i18] / 64);
				tmp25 = (tmp7[i18][i19] / 64);
				tmp26[i18] = (tmp24 * tmp25);
			}
			tmp27 = 10;
			for (MYINT i20 = 0; (i20 < 0); i20++) {
				tmp28 = 0;
				for (MYINT i21 = 0; (i21 < 6); i21++) {
					tmp26[i21] = ((tmp28 < (tmp27 >> 1)) ? ((tmp26[(2 * i21)] + tmp26[((2 * i21) + 1)]) / 2) : (((tmp28 == (tmp27 >> 1)) && ((tmp27 & 1) == 1)) ? (tmp26[(2 * i21)] / 2) : 0));
					tmp28 = (tmp28 + 1);
				}
				tmp27 = ((tmp27 + 1) >> 1);
			}
			for (MYINT i20 = 0; (i20 < 4); i20++) {
				tmp28 = 0;
				for (MYINT i21 = 0; (i21 < 6); i21++) {
					tmp26[i21] = ((tmp28 < (tmp27 >> 1)) ? (tmp26[(2 * i21)] + tmp26[((2 * i21) + 1)]) : (((tmp28 == (tmp27 >> 1)) && ((tmp27 & 1) == 1)) ? tmp26[(2 * i21)] : 0));
					tmp28 = (tmp28 + 1);
				}
				tmp27 = ((tmp27 + 1) >> 1);
			}
			tmp23[i17][i19] = tmp26[0];
		}
	}
	node1 = ((tmp23[0][0] > 0) ? ((2 * node0) + 1) : ((2 * node0) + 2));

	// WW * ZX
	for (MYINT i22 = 0; (i22 < 1); i22++) {
		for (MYINT i24 = 0; (i24 < 1); i24++) {
			for (MYINT i23 = 0; (i23 < 10); i23++) {
				tmp30 = (W[node1][i22][i23] / 128);
				tmp31 = (tmp7[i23][i24] / 64);
				tmp32[i23] = (tmp30 * tmp31);
			}
			tmp33 = 10;
			for (MYINT i25 = 0; (i25 < 0); i25++) {
				tmp34 = 0;
				for (MYINT i26 = 0; (i26 < 6); i26++) {
					tmp32[i26] = ((tmp34 < (tmp33 >> 1)) ? ((tmp32[(2 * i26)] + tmp32[((2 * i26) + 1)]) / 2) : (((tmp34 == (tmp33 >> 1)) && ((tmp33 & 1) == 1)) ? (tmp32[(2 * i26)] / 2) : 0));
					tmp34 = (tmp34 + 1);
				}
				tmp33 = ((tmp33 + 1) >> 1);
			}
			for (MYINT i25 = 0; (i25 < 4); i25++) {
				tmp34 = 0;
				for (MYINT i26 = 0; (i26 < 6); i26++) {
					tmp32[i26] = ((tmp34 < (tmp33 >> 1)) ? (tmp32[(2 * i26)] + tmp32[((2 * i26) + 1)]) : (((tmp34 == (tmp33 >> 1)) && ((tmp33 & 1) == 1)) ? tmp32[(2 * i26)] : 0));
					tmp34 = (tmp34 + 1);
				}
				tmp33 = ((tmp33 + 1) >> 1);
			}
			tmp29[i22][i24] = tmp32[0];
		}
	}

	// VV * ZX
	for (MYINT i27 = 0; (i27 < 1); i27++) {
		for (MYINT i29 = 0; (i29 < 1); i29++) {
			for (MYINT i28 = 0; (i28 < 10); i28++) {
				tmp36 = (V[node1][i27][i28] / 128);
				tmp37 = (tmp7[i28][i29] / 64);
				tmp38[i28] = (tmp36 * tmp37);
			}
			tmp39 = 10;
			for (MYINT i30 = 0; (i30 < 0); i30++) {
				tmp40 = 0;
				for (MYINT i31 = 0; (i31 < 6); i31++) {
					tmp38[i31] = ((tmp40 < (tmp39 >> 1)) ? ((tmp38[(2 * i31)] + tmp38[((2 * i31) + 1)]) / 2) : (((tmp40 == (tmp39 >> 1)) && ((tmp39 & 1) == 1)) ? (tmp38[(2 * i31)] / 2) : 0));
					tmp40 = (tmp40 + 1);
				}
				tmp39 = ((tmp39 + 1) >> 1);
			}
			for (MYINT i30 = 0; (i30 < 4); i30++) {
				tmp40 = 0;
				for (MYINT i31 = 0; (i31 < 6); i31++) {
					tmp38[i31] = ((tmp40 < (tmp39 >> 1)) ? (tmp38[(2 * i31)] + tmp38[((2 * i31) + 1)]) : (((tmp40 == (tmp39 >> 1)) && ((tmp39 & 1) == 1)) ? tmp38[(2 * i31)] : 0));
					tmp40 = (tmp40 + 1);
				}
				tmp39 = ((tmp39 + 1) >> 1);
			}
			tmp35[i27][i29] = tmp38[0];
		}
	}

	// tanh(V1)
	for (MYINT i32 = 0; (i32 < 1); i32++) {
		for (MYINT i33 = 0; (i33 < 1); i33++) {
			tmp35[i32][i33] = ((tmp35[i32][i33] >= 256) ? 256 : ((tmp35[i32][i33] <= -256) ? -256 : tmp35[i32][i33]));
		}
	}
	for (MYINT i34 = 0; (i34 < 1); i34++) {
		for (MYINT i35 = 0; (i35 < 1); i35++) {
			tmp41[i34][i35] = ((tmp29[i34][i35] / 16) * (tmp35[i34][i35] / 16));
		}
	}

	// score0 + tmp41
	for (MYINT i36 = 0; (i36 < 1); i36++) {
		for (MYINT i37 = 0; (i37 < 1); i37++) {
			tmp43 = tmp22[i36][i37];
			tmp44 = tmp41[i36][i37];
			tmp42[i36][i37] = (tmp43 + tmp44);
			tmp42[i36][i37] = tmp42[i36][i37];
		}
	}
	tmp45 = ((tmp42[0][0] > 0) ? 1 : 0);

	return tmp45;
}
