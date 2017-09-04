// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "par_utils.h"

using namespace EdgeML;

template<class INT_T, class UnaryFunction>
void EdgeML::map_n(const INT_T& begin, const INT_T& end, UnaryFunction f)
{
  assert(end >= begin);
  for (INT_T pos = begin; pos < end; ++pos)
    f(pos);
}


template<class INT_T, class UnaryFunction>
void EdgeML::parmap_n(const INT_T& begin, const INT_T& end, UnaryFunction f, const INT_T seq_base)
{
  assert(end >= begin);
  if (end - begin <= seq_base)
    map_n<INT_T, UnaryFunction>(begin, end, f);
  else {
    cilk_spawn
      parmap_n<INT_T, UnaryFunction>(begin, (begin + end) / 2, f, seq_base);
    parmap_n<INT_T, UnaryFunction>((begin + end) / 2, end, f, seq_base);
    cilk_sync;
  }
}

void EdgeML::parallelExp(MatrixXuf& D)
{
  if (D.IsRowMajor)
    parmap_n<Eigen::Index, rowExponentiate<Eigen::Index> >
    ((Eigen::Index)0, D.rows(), rowExponentiate<Eigen::Index>(&D), SEQ_BASE);
  else
    parmap_n<Eigen::Index, colExponentiate<Eigen::Index> >
    ((Eigen::Index)0, D.cols(), colExponentiate<Eigen::Index>(&D), SEQ_BASE);
}

struct col_multiply
{
  MatrixXuf* _mat;
  VectorXf* _colMultiplier;
  col_multiply(MatrixXuf *mat, VectorXf *colMultiplier) : _mat(mat), _colMultiplier(colMultiplier) {}
  void operator() (const int& c) { _mat->col(c).noalias() = _mat->col(c) * (*_colMultiplier)(c); }
};
