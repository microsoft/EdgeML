// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#define CONV2D_N 1
#define CBR1F_H 240
#define CBR1F_W 320
#define CBR1F_CIN 1
#define CBR1F_HF 3
#define CBR1F_WF 3
#define CBR1F_CF 1
#define CONV2D_COUT 4
#define CONV2D_HOUT 120
#define CONV2D_WOUT 160
#define CBR1F_HPADL 1
#define CBR1F_HPADR 1
#define CBR1F_WPADL 1
#define CBR1F_WPADR 1
#define CBR1F_HSTRIDE 2
#define CBR1F_WSTRIDE 2
#define CBR1F_HDILATION 1
#define CBR1F_WDILATION 1
#define CBR1F_G 1

#define CBR1W_HF 1
#define CBR1W_WF 1
#define CBR1W_CF 1
#define CBR1W_COUT 1
#define CBR1W_HPADL 0
#define CBR1W_HPADR 0
#define CBR1W_WPADL 0
#define CBR1W_WPADR 0
#define CBR1W_HSTRIDE 1
#define CBR1W_WSTRIDE 1
#define CBR1W_HDILATION 1
#define CBR1W_WDILATION 1
#define CBR1W_G 4

// The convention followed for filter weights is F[groups][filter_height][filter_width][in_channels][out_channels]
static Q15_T CBR1F[CBR1F_G * CBR1F_HF * CBR1F_WF * CBR1F_CF * CONV2D_COUT] = {6436, 3724, -2339, -3459, 30083, 12475, 10569, -6687, 13151, -15532, -3044, -7236, -4457, 5650, -10706, 10279, -15064, 14987, 10476, 24315, -16423, -19336, 462, 16770, 2261, 481, -5310, -5713, -8107, 2591, 6205, -17504, -6909, -2618, 1814, -10145};
static Q15_T CBR1W[CONV2D_COUT] = {14221, 15093, 10072, 20547};
static Q15_T CBR1B[CONV2D_COUT] = {10604, 15056, 22502, -769};

static SCALE_T CBR1F_H1 = 4;
static SCALE_T CBR1F_H2 = 0;
static SCALE_T CBR1W_H1 = 0;
static SCALE_T CBR1W_H2 = 0;
static Q7_T CONV2D_Limit = 96;
static Q7_T CONV2D_Div = 1;

#ifdef SHIFT
  static SCALE_T CBR1F_Scinput = 1;  //2
  static SCALE_T CBR1F_Scoutput = 2; //4
  static SCALE_T CBR1F_Demote = 8;   //256
  static SCALE_T CBR1B_Scten = 0;    //1
  static SCALE_T CBR1B_Scvec = 10;   //1024
  static SCALE_T CBR1B_Scret = 0;    //1
  static SCALE_T CBR1W_Scinput = 3;  //8
  static SCALE_T CBR1W_Scoutput = 3; //8
  static SCALE_T CBR1W_Demote = 8;   //256
#else
  static SCALE_T CBR1F_Scinput = 2;
  static SCALE_T CBR1F_Scoutput = 4;
  static SCALE_T CBR1F_Demote = 256;
  static SCALE_T CBR1B_Scten = 1;
  static SCALE_T CBR1B_Scvec = 1024;
  static SCALE_T CBR1B_Scret = 1;
  static SCALE_T CBR1W_Scinput = 8;
  static SCALE_T CBR1W_Scoutput = 8;
  static SCALE_T CBR1W_Demote = 256;
#endif
