// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "ProtoNN.h"

using namespace EdgeML;
using namespace EdgeML::ProtoNN;

// WARNING
// Always ensure that the default values in TLC export are the same as here
//
ProtoNNModel::ProtoNNHyperParams::ProtoNNHyperParams()
{
  problemType = undefinedProblem;
  initializationType = undefinedInitialization;
  normalizationType = none;

  seed = 42;

  ntrain = 0;
  nvalidation = 0;

  iters = 20;
  epochs = 20;
  batchSize = 1024;

  d = 15;
  m = 20;
  k = 0;

  D = 0;
  l = 0;

  gammaNumerator = 1.0;
  gamma = -1.0; // will be set to 2.5*gammaNumerator/median(dist b/w points and init prototypes)

  lambdaW = 1.0;
  lambdaZ = 1.0;
  lambdaB = 1.0;

  isHyperParamInitialized = false;
}

ProtoNNModel::ProtoNNHyperParams::~ProtoNNHyperParams()
{}

std::string ProtoNNModel::ProtoNNHyperParams::subdirName() const
{
  std::string name
    = "pd_" + std::to_string(d)
    + "_protPerClass_" + std::to_string(k)
    + "_prot_" + std::to_string(m)

    + "_spW_" + std::to_string(lambdaW)
    + "_spZ_" + std::to_string(lambdaZ)
    + "_spB_" + std::to_string(lambdaB)
    + "_gammaNumer_" + std::to_string(gammaNumerator)
    + "_normal_" + std::to_string(normalizationType)
    + "_seed_" + std::to_string(seed)

    + "_bs_" + std::to_string(batchSize)
    + "_it_" + std::to_string(iters)
    + "_ep_" + std::to_string(epochs);

  return name;
}

void ProtoNNModel::ProtoNNHyperParams::finalizeHyperParams()
{
  LOG_INFO("Running sanity checks on hyper-parameter values. If an assert fails, re-check the hyper-parameter value passed as argument to ProtoNN.");

  if (initializationType == perClassKmeans) {
    assert(k >= 1 && "number of prototypes per class should be >= 1");
    m = k*l;
  }
  if (initializationType == predefined && k > 0) {
    m = k*l;
  }

  assert(gammaNumerator > 0 && "gammaNumerator should be >= 1");
  assert(d >= 1 && "projection dimension should be >= 1");
  assert(D >= 1 && "data dimension not specified, please use -D flag");
  assert(l >= 1 && "number of labels not specified, use -l flag");
  assert(batchSize >= 1 && "batch-siez should be >= 1");
  assert(iters >= 1 && "number of iters should be >= 1");
  assert(m >= 1 && "number of prototypes should be >= 1");
  assert(epochs >= 1 && "number of epochs should be >= 1");
  // Following asserts removed to faciliate support for TLC
  // which does not know how many datapoints are going to be fed before-hand!
  // assert(ntrain >= 1);
  // assert(nvalidation >= 0);
  // assert(m <= ntrain);
  if (d > D) {
    LOG_INFO("Passed projection dimension (d) is larger than the original dimension. Setting d = D.");
    d = D;
  }
  assert(lambdaW >= 0.0L && "sparsity ratio of W should be >= 0.0");
  assert(lambdaZ >= 0.0L && "sparsity ratio of Z should be >= 0.0");
  assert(lambdaB >= 0.0L && "sparsity ratio of B should be >= 0.0");
  assert(lambdaW <= 1.0L && "sparsity ratio of W should be <= 1.0");
  assert(lambdaZ <= 1.0L && "sparsity ratio of Z should be <= 1.0");
  assert(lambdaB <= 1.0L && "sparsity ratio of B should be <= 0.0");
  assert(problemType != undefinedProblem && "problem not specified as binary, multiclass or multilabel. Please use -C flag. ");
  assert(normalizationType != undefinedNormalization);

  srand(seed);
  isHyperParamInitialized = true;
  LOG_INFO("Passed.");
}

void ProtoNNModel::ProtoNNHyperParams::setHyperParamsFromArgs(const int argc, const char** argv)
{
  for (int i = 1; i < argc; ++i) {
    if (i % 2 == 1)
      assert(argv[i][0] == '-'); //odd arguments must be specifiers, not values
    else {

      switch (argv[i - 1][1]) {

      case 'P':
        assert(i == 2);
        if (argv[i][0] == '1') initializationType = predefined;
        break;
      case 'R':
        seed = std::stoi(argv[i], NULL);
        break;
      case 'g':
        gammaNumerator = (FP_TYPE)strtod(argv[i], NULL);
        break;
      case 'r':
        ntrain = strtol(argv[i], NULL, 0);
        break;
      case 'v':
        nvalidation = strtol(argv[i], NULL, 0);
        break;
      case 'D':
        D = strtol(argv[i], NULL, 0);
        break;
      case 'l':
        l = strtol(argv[i], NULL, 0);
        break;
      case 'C':
        if (argv[i][0] == '0') problemType = binary;
        else if (argv[i][0] == '1') problemType = multiclass;
        else if (argv[i][0] == '2') problemType = multilabel;
        else exitWithHelp();
        break;
      case 'W':
        lambdaW = (FP_TYPE)strtod(argv[i], NULL);
        break;
      case 'Z':
        lambdaZ = (FP_TYPE)strtod(argv[i], NULL);
        break;
      case 'B':
        lambdaB = (FP_TYPE)strtod(argv[i], NULL);
        break;
      case 'b':
        batchSize = strtol(argv[i], NULL, 0);
        break;
      case 'd':
        d = strtol(argv[i], NULL, 0);
        break;
      case 'm':
        m = strtol(argv[i], NULL, 0);
        if (initializationType == predefined) break;
        assert(initializationType == undefinedInitialization || initializationType == overallKmeans && "specify only one of the -m and -k flags");
        initializationType = overallKmeans;
        break;
      case 'k':
        k = strtol(argv[i], NULL, 0);
        if (initializationType == predefined) break;
        assert(initializationType == undefinedInitialization || initializationType == perClassKmeans && "specify only one of the -m and -k flags");
        initializationType = perClassKmeans;
        break;
      case 'T':
        iters = std::stoi(argv[i], NULL);
        break;
      case 'E':
        epochs = std::stoi(argv[i], NULL);
        break;
      case 'N':
        if (argv[i][0] == '0') normalizationType = none;
        else if (argv[i][0] == '1') normalizationType = minMax;
        else normalizationType = l2;
        break;

      case 'I':
      case 'V':
      case 'O':
      case 'F':
      case 'M':
        break;

      default:
        LOG_INFO("Command line argument not recognized; saw character: " + std::string(1, argv[i - 1][1]));
        exitWithHelp();
        break;
      }
    }
  }

  finalizeHyperParams();
}


void ProtoNNModel::ProtoNNHyperParams::exitWithHelp()
{
  LOG_INFO("Options:");

  LOG_INFO("-P    : [Required] Option to load a predefined model, Visit docs for format. [Default: 0]");
  LOG_INFO("-R    : [Required] A random number seed which can be used to re-generate previously obtained experimental results. [Default: 42]");
  LOG_INFO("-r    : [Required] Number of training points.");
  LOG_INFO("-v    : [Required] Number of validation/test points.");
  LOG_INFO("-D    : [Required] The original dimension of the data.");
  LOG_INFO("-l    : [Required] Number of Classes");
  LOG_INFO("-C    : [Required] Problem Format. Specify one from 0 (binary), 1 (multiclass), 2 (multilabel)");
  LOG_INFO("-d    : [Required] Projection dimension (the dimension into which the data is projected). [Default:  15]");
  LOG_INFO("-m    : [m or k Required] Number of Prototypes. [Default: 20]");
  LOG_INFO("-k    : [m or k Required] Number of Prototypes Per Class.\n");

  LOG_INFO("-g    : [Optional] GammaNumerator, also alters RBF kernel parameter  𝛾 =(2.5⋅𝐺𝑎𝑚𝑚𝑎𝑁𝑢𝑚𝑒𝑟𝑎𝑡𝑜𝑟)/(𝑚𝑒𝑑𝑖𝑎𝑛(||𝐵𝑗,𝑊−𝑋𝑖||22)). [Default: 1.0] ");
  LOG_INFO("-W    : [Optional] Projection sparsity ( 𝜆𝑊 ). [Default:  1.0] ");
  LOG_INFO("-Z    : [Optional] Label Sparsity. [Default:  1.0]");
  LOG_INFO("-B    : [Optional] Prototype sparsity. [Default:  1.0]\n");


  LOG_INFO("-T    : [Optional] Total number of optimization iterations. [Default:  20]");
  LOG_INFO("-E    : [Optional] Number of epochs (complete see-through's) of the data for each iteration, and each parameter. [Default:  20]");
  LOG_INFO("-N    : [Optional] Normalization. Default: 0 (No Normalization), 1 (Min-Max Normalization), 2 (L2-Normalization)\n");

  exit(1);
}
