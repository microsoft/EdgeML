#include <iostream>
#include <cstring>
#include <cmath>

#include "datatypes.h"
#include "predictors.h"
#include "profile.h"
#include "library_fixed.h"
#include "model_fixed.h"
#include "vars_fixed.h"

using namespace std;
using namespace seedot_fixed;
using namespace vars_fixed;

MYINT vars_fixed::tmp6[20][1];
MYINT vars_fixed::tmp7[20][1];
MYINT vars_fixed::node0;
MYINT vars_fixed::tmp9[1][1];
MYINT vars_fixed::tmp8[20];
MYINT vars_fixed::tmp11[1][1];
MYINT vars_fixed::tmp10[20];
MYINT vars_fixed::tmp12[1][1];
MYINT vars_fixed::tmp14[1][1];
MYINT vars_fixed::tmp13[20];
MYINT vars_fixed::node1;
MYINT vars_fixed::tmp16[1][1];
MYINT vars_fixed::tmp15[20];
MYINT vars_fixed::tmp18[1][1];
MYINT vars_fixed::tmp17[20];
MYINT vars_fixed::tmp19[1][1];
MYINT vars_fixed::tmp20[1][1];
MYINT vars_fixed::tmp22[1][1];
MYINT vars_fixed::tmp21[20];
MYINT vars_fixed::node2;
MYINT vars_fixed::tmp24[1][1];
MYINT vars_fixed::tmp23[20];
MYINT vars_fixed::tmp26[1][1];
MYINT vars_fixed::tmp25[20];
MYINT vars_fixed::tmp27[1][1];
MYINT vars_fixed::tmp28[1][1];
MYINT vars_fixed::tmp30[1][1];
MYINT vars_fixed::tmp29[20];
MYINT vars_fixed::node3;
MYINT vars_fixed::tmp32[1][1];
MYINT vars_fixed::tmp31[20];
MYINT vars_fixed::tmp34[1][1];
MYINT vars_fixed::tmp33[20];
MYINT vars_fixed::tmp35[1][1];
MYINT vars_fixed::tmp36[1][1];
MYINT vars_fixed::tmp37;

void seedotFixed(MYINT **X, int32_t* res)
{
	res[0] = -1;
}
const int switches = 0;
void seedotFixedSwitch(int i, MYINT **X_temp, int32_t* res) {
	switch(i) {
		default: res[0] = -1; return;
	}
}
