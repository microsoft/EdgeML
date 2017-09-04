// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#ifndef __PAR_UTILS_H
#define __PAR_UTILS_H
#define SEQ_BASE 32

#include "pre_processor.h"

namespace EdgeML
{
  template<class INT_T, class UnaryFunction>
  void map_n(const INT_T& begin, const INT_T& end, UnaryFunction f);

  template<class INT_T, class UnaryFunction>
  void parmap_n(const INT_T& begin, const INT_T& end, UnaryFunction f, const INT_T seq_base = SEQ_BASE);

  void parallelExp(MatrixXuf& D);

  template<class INT_T>
  struct rowExponentiate
  {
    MatrixXuf* _D;
    rowExponentiate(MatrixXuf* D) : _D(D) {}
    void operator() (const INT_T& r) { _D->middleRows(r, 1) = _D->middleRows(r, 1).array().exp(); }
  };

  template<class INT_T>
  struct colExponentiate
  {
    MatrixXuf* _D;
    colExponentiate(MatrixXuf* D) : _D(D) {}
    void operator() (const INT_T& c) { _D->middleCols(c, 1) = _D->middleCols(c, 1).array().exp(); }
  };
  /*
  template<class INT_T>
  struct row_threshold {
    float* _D;
    row_threshold(MatrixXuf* D) : _D(D) {}
    void operator() (const INT_T& r, const int thresh) { if (_D->middleRows(r,1) = _D->middleRows(r,1).array().exp();  }
  };

  template<class INT_T>
  struct colExponentiate {
    MatrixXuf* _D;
    colExponentiate(MatrixXuf* D) : _D(D) {}
    void operator() (const INT_T& c) { _D->middleCols(c,1) = _D->middleCols(c,1).array().exp();  }
    };*/


    // In grad_B, call the following
    //  parmap_n<Eigen::Index, col_multiply>(0, B.cols(), col_multiply(&ret,&colMult));
}
#endif
