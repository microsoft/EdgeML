#include <iostream>

#include "datatypes.h"
#include "lenet.h"
#include "model.h"

using namespace std;
using namespace lenet_fixed;

int lenetFixed(MYINT **X) {
	MYINT tmp0[1][32][32][3];
	MYINT tmp1;
	MYINT tmp2;
	MYINT tmp3;
	MYINT tmp4;
	MYINT tmp5[1][32][32][6];
	MYINT tmp6[75];
	MYINT tmp7;
	MYINT tmp8;
	MYINT tmp9;
	MYINT tmp10;
	MYINT tmp11;
	MYINT tmp12[1][32][32][6];
	MYINT tmp13;
	MYINT tmp14;
	MYINT tmp15[1][16][16][6];
	MYINT tmp16;
	MYINT tmp17;
	MYINT tmp18[1][16][16][16];
	MYINT tmp19[150];
	MYINT tmp20;
	MYINT tmp21;
	MYINT tmp22;
	MYINT tmp23;
	MYINT tmp24;
	MYINT tmp25[1][16][16][16];
	MYINT tmp26;
	MYINT tmp27;
	MYINT tmp28[1][8][8][16];
	MYINT tmp29;
	MYINT tmp30;
	MYINT tmp31[1][1024];
	MYINT tmp32;
	MYINT tmp33;
	MYINT tmp34[1][120];
	MYINT tmp35;
	MYINT tmp36;
	MYINT tmp37[1024];
	MYINT tmp38;
	MYINT tmp39;
	MYINT tmp40;
	MYINT tmp41;
	MYINT tmp42[1][120];
	MYINT tmp43;
	MYINT tmp44;
	MYINT tmp45[1][84];
	MYINT tmp46;
	MYINT tmp47;
	MYINT tmp48[120];
	MYINT tmp49;
	MYINT tmp50;
	MYINT tmp51;
	MYINT tmp52;
	MYINT tmp53[1][84];
	MYINT tmp54;
	MYINT tmp55;
	MYINT tmp56[1][10];
	MYINT tmp57;
	MYINT tmp58;
	MYINT tmp59[84];
	MYINT tmp60;
	MYINT tmp61;
	MYINT tmp62;
	MYINT tmp63;
	MYINT tmp64[1][10];
	MYINT tmp65;
	MYINT tmp66;
	MYINT tmp69;
	MYINT tmp67;
	MYINT tmp68;



	// reshape(XX, 1, 32, 32, 3)
	tmp1 = 0;
	tmp2 = 0;
	tmp3 = 0;
	tmp4 = 0;
	for (MYINT i0 = 0; (i0 < 3072); i0++) {
		for (MYINT i1 = 0; (i1 < 1); i1++) {
			tmp0[tmp1][tmp2][tmp3][tmp4] = X[i0][i1];
			tmp4 = (tmp4 + 1);
			if ((tmp4 == 3)) {
				tmp4 = 0;
				tmp3 = (tmp3 + 1);
				if ((tmp3 == 32)) {
					tmp3 = 0;
					tmp2 = (tmp2 + 1);
					if ((tmp2 == 32)) {
						tmp2 = 0;
						tmp1 = (tmp1 + 1);
					}
				}
			}
		}
	}

	// X2 # WWc1
	for (MYINT i2 = 0; (i2 < 1); i2++) {
		for (MYINT i3 = 0; (i3 < 32); i3++) {
			for (MYINT i4 = 0; (i4 < 32); i4++) {
				for (MYINT i8 = 0; (i8 < 6); i8++) {
					tmp9 = 0;
					for (MYINT i6 = 0; (i6 < 5); i6++) {
						for (MYINT i7 = 0; (i7 < 5); i7++) {
							for (MYINT i5 = 0; (i5 < 3); i5++) {
								tmp7 = (((((i3 + i6) < 2) || ((i3 + i6) >= 34)) || (((i4 + i7) < 2) || ((i4 + i7) >= 34))) ? 0 : tmp0[i2][((i3 + i6) - 2)][((i4 + i7) - 2)][i5]);
								tmp7 = (tmp7 / 128);
								tmp8 = Wc1[i6][i7][i5][i8];
								tmp8 = (tmp8 / 128);
								tmp6[tmp9] = (tmp7 * tmp8);
								tmp9 = (tmp9 + 1);
							}
						}
					}
					tmp10 = 75;
					for (MYINT i9 = 0; (i9 < 2); i9++) {
						tmp11 = 0;
						for (MYINT i10 = 0; (i10 < 38); i10++) {
							tmp6[i10] = ((tmp11 < (tmp10 >> 1)) ? ((tmp6[(2 * i10)] + tmp6[((2 * i10) + 1)]) / 2) : (((tmp11 == (tmp10 >> 1)) && ((tmp10 & 1) == 1)) ? (tmp6[(2 * i10)] / 2) : 0));
							tmp11 = (tmp11 + 1);
						}
						tmp10 = ((tmp10 + 1) >> 1);
					}
					for (MYINT i9 = 0; (i9 < 5); i9++) {
						tmp11 = 0;
						for (MYINT i10 = 0; (i10 < 38); i10++) {
							tmp6[i10] = ((tmp11 < (tmp10 >> 1)) ? (tmp6[(2 * i10)] + tmp6[((2 * i10) + 1)]) : (((tmp11 == (tmp10 >> 1)) && ((tmp10 & 1) == 1)) ? tmp6[(2 * i10)] : 0));
							tmp11 = (tmp11 + 1);
						}
						tmp10 = ((tmp10 + 1) >> 1);
					}
					tmp5[i2][i3][i4][i8] = tmp6[0];
				}
			}
		}
	}

	// tmp5 <+> BBc1
	for (MYINT i11 = 0; (i11 < 1); i11++) {
		for (MYINT i12 = 0; (i12 < 32); i12++) {
			for (MYINT i13 = 0; (i13 < 32); i13++) {
				for (MYINT i14 = 0; (i14 < 6); i14++) {
					tmp13 = tmp5[i11][i12][i13][i14];
					tmp14 = (Bc1[i14] / 32);
					tmp12[i11][i12][i13][i14] = (tmp13 + tmp14);
					tmp12[i11][i12][i13][i14] = tmp12[i11][i12][i13][i14];
				}
			}
		}
	}

	// relu(tmp12)
	for (MYINT i15 = 0; (i15 < 1); i15++) {
		for (MYINT i16 = 0; (i16 < 32); i16++) {
			for (MYINT i17 = 0; (i17 < 32); i17++) {
				for (MYINT i18 = 0; (i18 < 6); i18++) {
					tmp12[i15][i16][i17][i18] = ((tmp12[i15][i16][i17][i18] > 0) ? tmp12[i15][i16][i17][i18] : 0);
				}
			}
		}
	}

	// maxpool(Hc1, 2)
	for (MYINT i19 = 0; (i19 < 1); i19++) {
		for (MYINT i20 = 0; (i20 < 16); i20++) {
			for (MYINT i21 = 0; (i21 < 16); i21++) {
				for (MYINT i22 = 0; (i22 < 6); i22++) {
					tmp16 = tmp12[i19][(2 * i20)][(2 * i21)][i22];
					for (MYINT i23 = 0; (i23 < 2); i23++) {
						for (MYINT i24 = 0; (i24 < 2); i24++) {
							tmp17 = tmp12[i19][((2 * i20) + i23)][((2 * i21) + i24)][i22];
							tmp16 = ((tmp17 > tmp16) ? tmp17 : tmp16);
						}
					}
					tmp15[i19][i20][i21][i22] = tmp16;
				}
			}
		}
	}

	// Hc1P # WWc2
	for (MYINT i25 = 0; (i25 < 1); i25++) {
		for (MYINT i26 = 0; (i26 < 16); i26++) {
			for (MYINT i27 = 0; (i27 < 16); i27++) {
				for (MYINT i31 = 0; (i31 < 16); i31++) {
					tmp22 = 0;
					for (MYINT i29 = 0; (i29 < 5); i29++) {
						for (MYINT i30 = 0; (i30 < 5); i30++) {
							for (MYINT i28 = 0; (i28 < 6); i28++) {
								tmp20 = (((((i26 + i29) < 2) || ((i26 + i29) >= 18)) || (((i27 + i30) < 2) || ((i27 + i30) >= 18))) ? 0 : tmp15[i25][((i26 + i29) - 2)][((i27 + i30) - 2)][i28]);
								tmp20 = (tmp20 / 128);
								tmp21 = Wc2[i29][i30][i28][i31];
								tmp21 = (tmp21 / 64);
								tmp19[tmp22] = (tmp20 * tmp21);
								tmp22 = (tmp22 + 1);
							}
						}
					}
					tmp23 = 150;
					for (MYINT i32 = 0; (i32 < 0); i32++) {
						tmp24 = 0;
						for (MYINT i33 = 0; (i33 < 76); i33++) {
							tmp19[i33] = ((tmp24 < (tmp23 >> 1)) ? ((tmp19[(2 * i33)] + tmp19[((2 * i33) + 1)]) / 2) : (((tmp24 == (tmp23 >> 1)) && ((tmp23 & 1) == 1)) ? (tmp19[(2 * i33)] / 2) : 0));
							tmp24 = (tmp24 + 1);
						}
						tmp23 = ((tmp23 + 1) >> 1);
					}
					for (MYINT i32 = 0; (i32 < 8); i32++) {
						tmp24 = 0;
						for (MYINT i33 = 0; (i33 < 76); i33++) {
							tmp19[i33] = ((tmp24 < (tmp23 >> 1)) ? (tmp19[(2 * i33)] + tmp19[((2 * i33) + 1)]) : (((tmp24 == (tmp23 >> 1)) && ((tmp23 & 1) == 1)) ? tmp19[(2 * i33)] : 0));
							tmp24 = (tmp24 + 1);
						}
						tmp23 = ((tmp23 + 1) >> 1);
					}
					tmp18[i25][i26][i27][i31] = tmp19[0];
				}
			}
		}
	}

	// tmp18 <+> BBc2
	for (MYINT i34 = 0; (i34 < 1); i34++) {
		for (MYINT i35 = 0; (i35 < 16); i35++) {
			for (MYINT i36 = 0; (i36 < 16); i36++) {
				for (MYINT i37 = 0; (i37 < 16); i37++) {
					tmp26 = tmp18[i34][i35][i36][i37];
					tmp27 = (Bc2[i37] / 8);
					tmp25[i34][i35][i36][i37] = (tmp26 + tmp27);
					tmp25[i34][i35][i36][i37] = tmp25[i34][i35][i36][i37];
				}
			}
		}
	}

	// relu(tmp25)
	for (MYINT i38 = 0; (i38 < 1); i38++) {
		for (MYINT i39 = 0; (i39 < 16); i39++) {
			for (MYINT i40 = 0; (i40 < 16); i40++) {
				for (MYINT i41 = 0; (i41 < 16); i41++) {
					tmp25[i38][i39][i40][i41] = ((tmp25[i38][i39][i40][i41] > 0) ? tmp25[i38][i39][i40][i41] : 0);
				}
			}
		}
	}

	// maxpool(Hc2, 2)
	for (MYINT i42 = 0; (i42 < 1); i42++) {
		for (MYINT i43 = 0; (i43 < 8); i43++) {
			for (MYINT i44 = 0; (i44 < 8); i44++) {
				for (MYINT i45 = 0; (i45 < 16); i45++) {
					tmp29 = tmp25[i42][(2 * i43)][(2 * i44)][i45];
					for (MYINT i46 = 0; (i46 < 2); i46++) {
						for (MYINT i47 = 0; (i47 < 2); i47++) {
							tmp30 = tmp25[i42][((2 * i43) + i46)][((2 * i44) + i47)][i45];
							tmp29 = ((tmp30 > tmp29) ? tmp30 : tmp29);
						}
					}
					tmp28[i42][i43][i44][i45] = tmp29;
				}
			}
		}
	}

	// reshape(Hc2P, 1, 1024)
	tmp32 = 0;
	tmp33 = 0;
	for (MYINT i48 = 0; (i48 < 1); i48++) {
		for (MYINT i51 = 0; (i51 < 16); i51++) {
			for (MYINT i49 = 0; (i49 < 8); i49++) {
				for (MYINT i50 = 0; (i50 < 8); i50++) {
					tmp31[tmp32][tmp33] = tmp28[i48][i49][i50][i51];
					tmp33 = (tmp33 + 1);
					if ((tmp33 == 1024)) {
						tmp33 = 0;
						tmp32 = (tmp32 + 1);
					}
				}
			}
		}
	}

	// Hc2PP * WWf1
	for (MYINT i52 = 0; (i52 < 1); i52++) {
		for (MYINT i54 = 0; (i54 < 120); i54++) {
			for (MYINT i53 = 0; (i53 < 1024); i53++) {
				tmp38 = tmp31[i52][i53];
				tmp39 = Wf1[i53][i54];
				tmp35 = (tmp38 / 128);
				tmp36 = (tmp39 / 128);
				tmp37[i53] = (tmp35 * tmp36);
			}
			tmp40 = 1024;
			for (MYINT i55 = 0; (i55 < 1); i55++) {
				tmp41 = 0;
				for (MYINT i56 = 0; (i56 < 513); i56++) {
					tmp37[i56] = ((tmp41 < (tmp40 >> 1)) ? ((tmp37[(2 * i56)] + tmp37[((2 * i56) + 1)]) / 2) : (((tmp41 == (tmp40 >> 1)) && ((tmp40 & 1) == 1)) ? (tmp37[(2 * i56)] / 2) : 0));
					tmp41 = (tmp41 + 1);
				}
				tmp40 = ((tmp40 + 1) >> 1);
			}
			for (MYINT i55 = 0; (i55 < 9); i55++) {
				tmp41 = 0;
				for (MYINT i56 = 0; (i56 < 513); i56++) {
					tmp37[i56] = ((tmp41 < (tmp40 >> 1)) ? (tmp37[(2 * i56)] + tmp37[((2 * i56) + 1)]) : (((tmp41 == (tmp40 >> 1)) && ((tmp40 & 1) == 1)) ? tmp37[(2 * i56)] : 0));
					tmp41 = (tmp41 + 1);
				}
				tmp40 = ((tmp40 + 1) >> 1);
			}
			tmp34[i52][i54] = tmp37[0];
		}
	}

	// tmp34 <+> BBf1
	for (MYINT i57 = 0; (i57 < 1); i57++) {
		for (MYINT i58 = 0; (i58 < 120); i58++) {
			tmp43 = tmp34[i57][i58];
			tmp44 = (Bf1[i58] / 64);
			tmp42[i57][i58] = (tmp43 + tmp44);
			tmp42[i57][i58] = tmp42[i57][i58];
		}
	}

	// relu(tmp42)
	for (MYINT i59 = 0; (i59 < 1); i59++) {
		for (MYINT i60 = 0; (i60 < 120); i60++) {
			tmp42[i59][i60] = ((tmp42[i59][i60] > 0) ? tmp42[i59][i60] : 0);
		}
	}

	// Hf1 * WWf2
	for (MYINT i61 = 0; (i61 < 1); i61++) {
		for (MYINT i63 = 0; (i63 < 84); i63++) {
			for (MYINT i62 = 0; (i62 < 120); i62++) {
				tmp49 = tmp42[i61][i62];
				tmp50 = Wf2[i62][i63];
				tmp46 = (tmp49 / 128);
				tmp47 = (tmp50 / 128);
				tmp48[i62] = (tmp46 * tmp47);
			}
			tmp51 = 120;
			for (MYINT i64 = 0; (i64 < 1); i64++) {
				tmp52 = 0;
				for (MYINT i65 = 0; (i65 < 61); i65++) {
					tmp48[i65] = ((tmp52 < (tmp51 >> 1)) ? ((tmp48[(2 * i65)] + tmp48[((2 * i65) + 1)]) / 2) : (((tmp52 == (tmp51 >> 1)) && ((tmp51 & 1) == 1)) ? (tmp48[(2 * i65)] / 2) : 0));
					tmp52 = (tmp52 + 1);
				}
				tmp51 = ((tmp51 + 1) >> 1);
			}
			for (MYINT i64 = 0; (i64 < 6); i64++) {
				tmp52 = 0;
				for (MYINT i65 = 0; (i65 < 61); i65++) {
					tmp48[i65] = ((tmp52 < (tmp51 >> 1)) ? (tmp48[(2 * i65)] + tmp48[((2 * i65) + 1)]) : (((tmp52 == (tmp51 >> 1)) && ((tmp51 & 1) == 1)) ? tmp48[(2 * i65)] : 0));
					tmp52 = (tmp52 + 1);
				}
				tmp51 = ((tmp51 + 1) >> 1);
			}
			tmp45[i61][i63] = tmp48[0];
		}
	}

	// tmp45 <+> BBf2
	for (MYINT i66 = 0; (i66 < 1); i66++) {
		for (MYINT i67 = 0; (i67 < 84); i67++) {
			tmp54 = tmp45[i66][i67];
			tmp55 = (Bf2[i67] / 64);
			tmp53[i66][i67] = (tmp54 + tmp55);
			tmp53[i66][i67] = tmp53[i66][i67];
		}
	}

	// relu(tmp53)
	for (MYINT i68 = 0; (i68 < 1); i68++) {
		for (MYINT i69 = 0; (i69 < 84); i69++) {
			tmp53[i68][i69] = ((tmp53[i68][i69] > 0) ? tmp53[i68][i69] : 0);
		}
	}

	// Hf2 * WWf3
	for (MYINT i70 = 0; (i70 < 1); i70++) {
		for (MYINT i72 = 0; (i72 < 10); i72++) {
			for (MYINT i71 = 0; (i71 < 84); i71++) {
				tmp60 = tmp53[i70][i71];
				tmp61 = Wf3[i71][i72];
				tmp57 = (tmp60 / 128);
				tmp58 = (tmp61 / 128);
				tmp59[i71] = (tmp57 * tmp58);
			}
			tmp62 = 84;
			for (MYINT i73 = 0; (i73 < 1); i73++) {
				tmp63 = 0;
				for (MYINT i74 = 0; (i74 < 43); i74++) {
					tmp59[i74] = ((tmp63 < (tmp62 >> 1)) ? ((tmp59[(2 * i74)] + tmp59[((2 * i74) + 1)]) / 2) : (((tmp63 == (tmp62 >> 1)) && ((tmp62 & 1) == 1)) ? (tmp59[(2 * i74)] / 2) : 0));
					tmp63 = (tmp63 + 1);
				}
				tmp62 = ((tmp62 + 1) >> 1);
			}
			for (MYINT i73 = 0; (i73 < 6); i73++) {
				tmp63 = 0;
				for (MYINT i74 = 0; (i74 < 43); i74++) {
					tmp59[i74] = ((tmp63 < (tmp62 >> 1)) ? (tmp59[(2 * i74)] + tmp59[((2 * i74) + 1)]) : (((tmp63 == (tmp62 >> 1)) && ((tmp62 & 1) == 1)) ? tmp59[(2 * i74)] : 0));
					tmp63 = (tmp63 + 1);
				}
				tmp62 = ((tmp62 + 1) >> 1);
			}
			tmp56[i70][i72] = tmp59[0];
		}
	}

	// tmp56 <+> BBf3
	for (MYINT i75 = 0; (i75 < 1); i75++) {
		for (MYINT i76 = 0; (i76 < 10); i76++) {
			tmp65 = tmp56[i75][i76];
			tmp66 = (Bf3[i76] / 32);
			tmp64[i75][i76] = (tmp65 + tmp66);
			tmp64[i75][i76] = tmp64[i75][i76];
		}
	}
	tmp69 = 0;
	tmp67 = 0;
	tmp68 = tmp64[0][0];
	for (MYINT i77 = 0; (i77 < 1); i77++) {
		for (MYINT i78 = 0; (i78 < 10); i78++) {
			if ((tmp68 < tmp64[i77][i78])) {
				tmp67 = tmp69;
				tmp68 = tmp64[i77][i78];
			}
			tmp69 = (tmp69 + 1);
		}
	}

	return tmp67;
}
