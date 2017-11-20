// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "Data.h"

namespace EdgeML
{
  namespace FileIO
  {
    struct Data
    {
      int COL_LABEL, COL_FEATURE, NUM_COLS, NUM_FEATURES, NUM_LABELS;

      Data(
        std::string filename,
        MatrixXuf& data,
        MatrixXuf& label,
        dataCount_t maxEntries,
        featureCount_t _COL_LABEL,
        featureCount_t _COL_FEATURE,
        featureCount_t _NUM_COLS,
        featureCount_t _NUM_FEATURES,
        featureCount_t _NUM_LABELS,
        EdgeML::DataFormat& formatType);

      Data(
        std::string filename,
        SparseMatrixuf& data,
        SparseMatrixuf& label,
        dataCount_t maxEntries,
        featureCount_t _COL_LABEL,
        featureCount_t _COL_FEATURE,
        featureCount_t _NUM_COLS,
        featureCount_t _NUM_FEATURES,
        featureCount_t _NUM_LABELS,
        EdgeML::DataFormat& formatType);

      size_t fillEntries(
        char*buf,
        MatrixXuf& data,
        MatrixXuf& label,
        dataCount_t maxEntries,
        featureCount_t numCols,
        uint64_t fileSize,
        EdgeML::DataFormat& formatType);

      size_t libsvmFillEntries(
        char*buf,
        MatrixXuf& data,
        MatrixXuf& label,
        dataCount_t maxEntries,
        featureCount_t numCols,
        uint64_t fileSize,
        EdgeML::DataFormat& formatType);

      size_t libsvmFillEntries(
        char*buf,
        SparseMatrixuf& data,
        SparseMatrixuf& label,
        dataCount_t maxEntries,
        featureCount_t numCols,
        uint64_t fileSize,
        EdgeML::DataFormat& formatType);

    };


    struct membuf : std::streambuf
    {
      membuf(char* begin, char* end)
      {
        this->setg(begin, begin, end);
      }
    };
  }
}
