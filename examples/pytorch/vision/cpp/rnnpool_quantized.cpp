// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <fstream>
#include <iostream>
#include <unordered_map>
#include <string>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <vector>

using namespace std;

#define MYINT int16_t
#define MYITE int16_t

#include "data.h"

#define SHIFT

#ifdef SHIFT
#define MYSCL int16_t
unordered_map<string, MYSCL> scale = {
    {"X", 12},

    {"one", 14},

    {"W1",14},
    {"H1",14},
    {"U1",15},
    {"Bg1",14},
    {"Bh1",15},
    {"zeta1",15},
    {"nu1",15},

    {"a1",11},
    {"b1",13},
    {"c1",11},
    {"cBg1",11},
    {"cBh1",11},
    {"g1",14},
    {"h1",14},
    {"z1",14},
    {"y1",14},
    {"w1",14},
    {"v1",14},
    {"u1",14},

    {"intermediate", 14},

    {"W2",14},
    {"H2",14},
    {"U2",14},
    {"Bg2",14},
    {"Bh2",14},
    {"zeta2",15},
    {"nu2",15},

    {"a2",14},
    {"b2",13},
    {"c2",13},
    {"cBg2",11},
    {"cBh2",11},
    {"g2",14},
    {"h2",14},
    {"z2",15},
    {"y2",14},
    {"w2",14},
    {"v2",14},
    {"u2",14},

    {"Y",14},
};
#else
#define MYSCL int32_t
unordered_map<string, MYSCL> scale = {
    {"X", 4096},

    {"one", 16384},

    {"W1",16384},
    {"H1",16384},
    {"U1",32768},
    {"Bg1",16384},
    {"Bh1",32768},
    {"zeta1",32768},
    {"nu1",32768},

    {"a1",2048},
    {"b1",8192},
    {"c1",2048},
    {"cBg1",2048},
    {"cBh1",2048},
    {"g1",16384},
    {"h1",16384},
    {"z1",16384},
    {"y1",16384},
    {"w1",16384},
    {"v1",16384},
    {"u1",16384},

    {"intermediate", 16384},

    {"W2",16384},
    {"H2",16384},
    {"U2",16384},
    {"Bg2",16384},
    {"Bh2",16384},
    {"zeta2",32768},
    {"nu2",32768},

    {"a2",16384},
    {"b2",8192},
    {"c2",8192},
    {"cBg2",2048},
    {"cBh2",2048},
    {"g2",16384},
    {"h2",16384},
    {"z2",32768},
    {"y2",16384},
    {"w2",16384},
    {"v2",16384},
    {"u2",16384},

    {"Y",16384},
};
#endif



void MatMul(int16_t* A, int16_t* B, int16_t* C, MYINT I, MYINT J, MYINT K, MYSCL scA, MYSCL scB, MYSCL scC) {

#ifdef SHIFT
  MYSCL addshrP = 1, addshr = 0;
  while (addshrP < J) {
    addshrP *= 2;
    addshr += 1;
  }
#else
  MYSCL addshr = 1;
  while (addshr < J)
    addshr *= 2;
#endif

#ifdef SHIFT
  MYSCL shr = scA + scB - scC - addshr;
#else
  MYSCL shr = (scA * scB) / (scC * addshr);
#endif

  for (int i = 0; i < I; i++) {
    for (int k = 0; k < K; k++) {
      int32_t s = 0;
      for (int j = 0; j < J; j++) {
#ifdef SHIFT
        s += ((int32_t)A[i * J + j] * (int32_t)B[j * K + k]) >> addshr;
#else
        s += ((int32_t)A[i * J + j] * (int32_t)B[j * K + k]) / addshr;
#endif
      }
#ifdef SHIFT
      C[i * K + k] = s >> shr;
#else
      C[i * K + k] = s / shr;
#endif
    }
  }
}

inline MYINT min(MYINT a, MYINT b) {
  return a < b ? a : b;
}

inline MYINT max(MYINT a, MYINT b) {
  return a > b ? a : b;
}

void MatAdd(int16_t* A, int16_t* B, int16_t* C, MYINT I, MYINT J, MYSCL scA, MYSCL scB, MYSCL scC) {

  MYSCL shrmin = min(scA, scB);
#ifdef SHIFT
  MYSCL shra = scA - shrmin;
  MYSCL shrb = scB - shrmin;
  MYSCL shrc = shrmin - scC;
#else
  MYSCL shra = scA / shrmin;
  MYSCL shrb = scB / shrmin;
  MYSCL shrc = shrmin / scC;
#endif

  for (int i = 0; i < I; i++) {
    for (int j = 0; j < J; j++) {
#ifdef SHIFT
      C[i * J + j] = ((A[i * J + j] >> (shra + shrc)) + (B[i * J + j] >> (shrb + shrc)));
#else
      C[i * J + j] = ((A[i * J + j] / (shra * shrc)) + (B[i * J + j] / (shrb * shrc)));
#endif
    }
  }
}

void ScalarMatSub(int16_t A, int16_t* B, int16_t* C, MYINT I, MYINT J, MYSCL scA, MYSCL scB, MYSCL scC) {

  MYSCL shrmin = min(scA, scB);
#ifdef SHIFT
  MYSCL shra = scA - shrmin;
  MYSCL shrb = scB - shrmin;
  MYSCL shrc = shrmin - scC;
#else
  MYSCL shra = scA / shrmin;
  MYSCL shrb = scB / shrmin;
  MYSCL shrc = shrmin / scC;
#endif

  for (int i = 0; i < I; i++) {
    for (int j = 0; j < J; j++) {
#ifdef SHIFT
      C[i * J + j] = ((A >> (shra + shrc)) - (B[i * J + j] >> (shrb + shrc)));
#else
      C[i * J + j] = ((A / (shra * shrc)) - (B[i * J + j] / (shrb * shrc)));
#endif
    }
  }
}

void ScalarMatAdd(int16_t A, int16_t* B, int16_t* C, MYINT I, MYINT J, MYSCL scA, MYSCL scB, MYSCL scC) {

  MYSCL shrmin = min(scA, scB);
#ifdef SHIFT
  MYSCL shra = scA - shrmin;
  MYSCL shrb = scB - shrmin;
  MYSCL shrc = shrmin - scC;
#else
  MYSCL shra = scA / shrmin;
  MYSCL shrb = scB / shrmin;
  MYSCL shrc = shrmin / scC;
#endif

  for (int i = 0; i < I; i++) {
    for (int j = 0; j < J; j++) {
#ifdef SHIFT
      C[i * J + j] = ((A >> (shra + shrc)) + (B[i * J + j] >> (shrb + shrc)));
#else
      C[i * J + j] = ((A / (shra * shrc)) + (B[i * J + j] / (shrb * shrc)));
#endif
    }
  }
}

void HadMul(int16_t* A, int16_t* B, int16_t* C, MYINT I, MYINT J, MYSCL scA, MYSCL scB, MYSCL scC) {

#ifdef SHIFT
  MYSCL shr = (scA + scB) - scC;
#else
  MYSCL shr = (scA * scB) / scC;
#endif

  for (int i = 0; i < I; i++) {
    for (int j = 0; j < J; j++) {
#ifdef SHIFT
      C[i * J + j] = (((int32_t)A[i * J + j]) * ((int32_t)B[i * J + j])) >> shr;
#else
      C[i * J + j] = (((int32_t)A[i * J + j]) * ((int32_t)B[i * J + j])) / shr;
#endif
    }
  }
}

void ScalarMul(int16_t A, int16_t* B, int16_t* C, MYINT I, MYINT J, MYSCL scA, MYSCL scB, MYSCL scC) {

#ifdef SHIFT
  MYSCL shr = (scA + scB) - scC;
#else
  MYSCL shr = (scA * scB) / scC;
#endif

  for (int i = 0; i < I; i++) {
    for (int j = 0; j < J; j++) {
#ifdef SHIFT
      C[i * J + j] = ((int32_t)(A) * (int32_t)(B[i * J + j])) >> shr;
#else
      C[i * J + j] = ((int32_t)(A) * (int32_t)(B[i * J + j])) / shr;
#endif
    }
  }
}

void SigmoidNew16(int16_t* A, MYINT I, MYINT J, int16_t* B) {
  for (MYITE i = 0; i < I; i++) {
    for (MYITE j = 0; j < J; j++) {
      int16_t a = A[i * J + j];
      B[i * J + j] = 8 * max(min((a + 2048) / 2, 2048), 0);
    }
  }
  return;
}

void TanHNew16(int16_t* A, MYINT I, MYINT J, int16_t* B) {
  for (MYITE i = 0; i < I; i++) {
    for (MYITE j = 0; j < J; j++) {
      int16_t a = A[i * J + j];
      B[i * J + j] = 8 * max(min(a, 2048), -2048);
    }
  }
  return;
}

void reverse(int16_t* A, int16_t* B, int I, int J) {
  for (int i = 0; i < I; i++) {
    for (int j = 0; j < J; j++) {
      B[i * J + j] = A[(I - i - 1) * J + j];
    }
  }
}


void print(int16_t* var, int I, int J, MYSCL scale) {
  for (int i = 0; i < I; i++) {
    for (int j = 0; j < J; j++) {
      cout << ((float)var[i * J + j]) / scale << " ";
    }
    cout << endl;
  }
  //exit(1);
}

void FastGRNN1(int16_t X[8][4], int16_t* H, int timestep) {
  memset(&H[0], 0, 8 * 2);

  for (int i = 0; i < timestep; i++) {
    int16_t a[1][8];
    MatMul(&X[i][0], &W1[0][0], &a[0][0], 1, 4, 8, scale["X"], scale["W1"], scale["a1"]);
    int16_t b[1][8];
    MatMul(&H[0], &U1[0][0], &b[0][0], 1, 8, 8, scale["H1"], scale["U1"], scale["b1"]);
    int16_t c[1][8];
    MatAdd(&a[0][0], &b[0][0], &c[0][0], 1, 8, scale["a1"], scale["b1"], scale["c1"]);
    int16_t cBg[1][8];
    MatAdd(&c[0][0], &Bg1[0][0], &cBg[0][0], 1, 8, scale["c1"], scale["Bg1"], scale["cBg1"]);
    int16_t g[1][8];
    SigmoidNew16(&cBg[0][0], 1, 8, &g[0][0]);
    int16_t cBh[1][8];
    MatAdd(&c[0][0], &Bh1[0][0], &cBh[0][0], 1, 8, scale["c1"], scale["Bh1"], scale["cBh1"]);
    int16_t h[1][8];
    TanHNew16(&cBh[0][0], 1, 8, &h[0][0]);
    int16_t z[1][8];
    HadMul(&g[0][0], &H[0], &z[0][0], 1, 8, scale["g1"], scale["H1"], scale["z1"]);
    int16_t y[1][8];
    ScalarMatSub(16384, &g[0][0], &y[0][0], 1, 8, scale["one"], scale["g1"], scale["y1"]);
    int16_t w[1][8];
    ScalarMul(zeta1, &y[0][0], &w[0][0], 1, 8, scale["zeta1"], scale["y1"], scale["w1"]);
    int16_t v[1][8];
    ScalarMatAdd(nu1, &w[0][0], &v[0][0], 1, 8, scale["nu1"], scale["w1"], scale["v1"]);
    int16_t u[1][8];
    HadMul(&w[0][0], &h[0][0], &u[0][0], 1, 8, scale["w1"], scale["h1"], scale["u1"]);

    MatAdd(&z[0][0], &u[0][0], &H[0], 1, 8, scale["z1"], scale["u1"], scale["H1"]);
  }
}

void FastGRNN2(int16_t X[8][8], int16_t* H, int timestep) {
  memset(&H[0], 0, 8 * 2);

  for (int i = 0; i < timestep; i++) {
    int16_t a[1][8];
    MatMul(&X[i][0], &W2[0][0], &a[0][0], 1, 8, 8, scale["intermediate"], scale["W2"], scale["a2"]);

    int16_t b[1][8];
    MatMul(&H[0], &U2[0][0], &b[0][0], 1, 8, 8, scale["H2"], scale["U2"], scale["b2"]);
    int16_t c[1][8];
    MatAdd(&a[0][0], &b[0][0], &c[0][0], 1, 8, scale["a2"], scale["b2"], scale["c2"]);
    int16_t cBg[1][8];
    MatAdd(&c[0][0], &Bg2[0][0], &cBg[0][0], 1, 8, scale["c2"], scale["Bg2"], scale["cBg2"]);
    int16_t g[1][8];
    SigmoidNew16(&cBg[0][0], 1, 8, &g[0][0]);
    int16_t cBh[1][8];
    MatAdd(&c[0][0], &Bh2[0][0], &cBh[0][0], 1, 8, scale["c2"], scale["Bh2"], scale["cBh2"]);
    int16_t h[1][8];
    TanHNew16(&cBh[0][0], 1, 8, &h[0][0]);
    int16_t z[1][8];
    HadMul(&g[0][0], &H[0], &z[0][0], 1, 8, scale["g2"], scale["H2"], scale["z2"]);
    int16_t y[1][8];
    ScalarMatSub(16384, &g[0][0], &y[0][0], 1, 8, scale["one"], scale["g2"], scale["y2"]);
    int16_t w[1][8];
    ScalarMul(zeta2, &y[0][0], &w[0][0], 1, 8, scale["zeta2"], scale["y2"], scale["w2"]);
    int16_t v[1][8];
    ScalarMatAdd(nu2, &w[0][0], &v[0][0], 1, 8, scale["nu2"], scale["w2"], scale["v2"]);
    int16_t u[1][8];
    HadMul(&w[0][0], &h[0][0], &u[0][0], 1, 8, scale["w2"], scale["h2"], scale["u2"]);

    MatAdd(&z[0][0], &u[0][0], &H[0], 1, 8, scale["z2"], scale["u2"], scale["H2"]);
  }
}

void RNNPool(int16_t X[8][8][4], int16_t pred[1][32]) {

  int16_t biinput1[8][8], biinput1r[8][8];
  for (int i = 0; i < 8; i++) {
    int16_t subX[8][4];
    for (int j = 0; j < 8; j++) {
      for (int k = 0; k < 4; k++) {
        subX[j][k] = X[i][j][k];
      }
    }
    int16_t H[1][8];
    FastGRNN1(subX, &H[0][0], 8);

    for (int j = 0; j < 8; j++) {
      biinput1[i][j] = H[0][j];
    }
  }

  int16_t res1[1][8], res2[1][8];
  FastGRNN2(biinput1, &res1[0][0], 8);
  reverse(&biinput1[0][0], &biinput1r[0][0], 8, 8);
  FastGRNN2(biinput1r, &res2[0][0], 8);

  int16_t biinput2[8][8], biinput2r[8][8];
  for (int i = 0; i < 8; i++) {
    int16_t subX[8][4];
    for (int j = 0; j < 8; j++) {
      for (int k = 0; k < 4; k++) {
        subX[j][k] = X[j][i][k];
      }
    }
    int16_t H[1][8];
    FastGRNN1(subX, &H[0][0], 8);

    for (int j = 0; j < 8; j++) {
      biinput2[i][j] = H[0][j];
    }
  }


  int16_t res3[1][8], res4[1][8];
  FastGRNN2(biinput2, &res3[0][0], 8);
  reverse(&biinput2[0][0], &biinput2r[0][0], 8, 8);
  FastGRNN2(biinput2r, &res4[0][0], 8);

  for (int i = 0; i < 8; i++)
    pred[0][i] = res1[0][i];
  for (int i = 0; i < 8; i++)
    pred[0][i + 8] = res2[0][i];
  for (int i = 0; i < 8; i++)
    pred[0][i + 16] = res3[0][i];
  for (int i = 0; i < 8; i++)
    pred[0][i + 24] = res4[0][i];
}

int main(int argc, char* argv[]) {
  string inputfile, outputfile;
  int patches;
  if (argc != 4) {
    cerr << "Improper number of arguments" << endl;
    return -1;
  }
  else {
    patches = atoi(argv[1]);
    inputfile = string(argv[2]);
    outputfile = string(argv[3]);
  }

  fstream Xfile, Yfile;

  Xfile.open(inputfile, ios::in | ios::binary);
  Yfile.open(outputfile, ios::out | ios::binary);


  char line[8];
  Xfile.read(line, 8);
  int headerSize;
  Xfile.read((char*)&headerSize, 1 * 2);

  char* headerLine = new char[headerSize]; //Ignored
  Xfile.read(headerLine, headerSize);
  delete[] headerLine;

  char numpyMagix = 147;
  char numpyVersionMajor = 1, numpyVersionMinor = 0;
  string numpyMetaHeader = "";
  numpyMetaHeader += numpyMagix;
  numpyMetaHeader += "NUMPY";
  numpyMetaHeader += numpyVersionMajor;
  numpyMetaHeader += numpyVersionMinor;

  string numpyHeader = "{'descr': '<f4', 'fortran_order': False, 'shape': (" + to_string(patches) + ", 1, 32), }";

  for (int i = numpyHeader.size() + numpyMetaHeader.size() + 2; i % 64 != 64 - 1; i++) {
    numpyHeader += ' ';
  }
  numpyHeader += (char)(10);

  char a = numpyHeader.size() / 256, b = numpyHeader.size() % 256;
  Yfile << numpyMetaHeader;
  Yfile << b << a;
  Yfile << numpyHeader;

  int total = 0;
  int correct = 0;

  for (int i = 0; i < 6241; i++) {

    float Xline[256];
    Xfile.read((char*)&Xline[0], 256 * 4);


    int16_t y;
    int16_t reshapedX[8][8][4];

    for (int a = 0; a < 4; a++) {
      for (int b = 0; b < 8; b++) {
        for (int c = 0; c < 8; c++) {
#ifdef SHIFT
          reshapedX[b][c][a] = (int16_t)((Xline[a * 64 + b * 8 + c * 1]) * pow(2, scale["X"]));
#else
          reshapedX[b][c][a] = (int16_t)((Xline[a * 64 + b * 8 + c * 1]) * scale["X"]);
#endif
        }
      }
    }

    int16_t pred[1][32];
    RNNPool(reshapedX, pred);

    for (int j = 0; j < 32; j++) {
      float val = ((float)pred[0][j]) / pow(2, scale["Y"]);
      Yfile.write((char*)&val, sizeof(float));
    }
  }
  Xfile.close();
  Yfile.close();

  return 0;
}