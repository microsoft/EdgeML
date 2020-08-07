// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#define CONV2D_CF 1
#define CONV2D_HF 3
#define CONV2D_WF 3
#define CONV2D_COUT 4

// The convention followed for filter weights is F[groups][filter_height][filter_width][in_channels][out_channels]
static Q15_T CBR1F[1 * CONV2D_HF * CONV2D_WF * CONV2D_CF * CONV2D_COUT] = {6436, 3724, -2339, -3459, 30083, 12475, 10569, -6687, 13151, -15532, -3044, -7236, -4457, 5650, -10706, 10279, -15064, 14987, 10476, 24315, -16423, -19336, 462, 16770, 2261, 481, -5310, -5713, -8107, 2591, 6205, -17504, -6909, -2618, 1814, -10145};
static Q15_T CBR1W[CONV2D_COUT] = {14221, 15093, 10072, 20547};
static Q15_T CBR1B[CONV2D_COUT] = {10604, 15056, 22502, -769};
