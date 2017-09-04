/***************************************************************

examples.h

Structure storing parameters for datasets
    - Includes sample datasets for internal usage

Authors: 
    - Chirag Gupta <t-chgupt@microsoft.com>
    - Harsha Vardhan Simhadri <harshasi@microsoft.com>
    - Prateek Jain <prajain@microsoft.com>

Date: 07 June, 2017

****************************************************************/

#include "stdafx.h"

#ifndef __EXAMPLES_H__
#define __EXAMPLES_H__

#include "ProtoNN.h"

#define HOME_PATH "/home/t-chgupt"

struct parameters {
  PROBLEM problem_type;
  INITIALIZATION initialization_type;
  FILE_FORMAT format_type; 
  NORMALIZATION normalization_type;

  int seed; 
  dataCount_t ntot, ntrain, ntest, batch_size;
  int iters, epochs;
  featureCount_t D, d;
  int m, k, l;
  FP_TYPE lambda_W, lambda_Z, lambda_B, gamma; 

  char outdir [100];
  char indir [100];

  void check (void){
    assert(initialization_type != undefined_initialization);
    if(initialization_type == per_class_kmeans){
      assert (k >= 1);
      m = k*l;
      k = -1;
    }

    assert(d >= 1);
    assert(D >= 1);
    assert(batch_size >= 1);
    assert(iters >= 1);
    assert(m >= 1);
    assert(epochs >= 1);
    
    assert(ntrain >= 1);
    assert(ntest >= 1);
    assert(d <= D);
    assert(m <= ntrain);
    assert(lambda_W >= 0.0L);
    assert(lambda_Z >= 0.0L);
    assert(lambda_B >= 0.0L);
    assert(lambda_W <= 1.0L);
    assert(lambda_Z <= 1.0L);
    assert(lambda_B <= 1.0L);
    ntot = ntrain + ntest;
    assert(problem_type != undefined_type);
    assert(initialization_type != undefined_initialization);
    assert(format_type != undefined_format);
    assert(normalization_type != undefined);
    std::cout << "Dataset successfully initialized..." << std::endl;
  }

  void mkdir (void){
    sprintf (outdir, "%s/results/%f_%f_%f_%llu_%d",
	     indir, lambda_W, lambda_Z, lambda_B, d, m);

    char command [100];
    
    try {
      sprintf (command, "mkdir %s/results", indir);
      system(command);
      sprintf (command, "mkdir %s", outdir);
      system (command);
#ifdef DUMP
      sprintf (command, "mkdir %s/dump", outdir);
      system (command);
#endif
#ifdef VERIFY
      sprintf (command, "mkdir %s/verify", outdir);
      system (command);
#endif
    }
    catch (...){
      std::cerr << "One of the directories could not be created... " <<std::endl;
    }
  }  

  void store_params (int argc, char **argv, FP_TYPE* stats){
    char outfile [100];
    sprintf (outfile, "%s/params", outdir);
    std::ofstream f (outfile);
    f       << "d = " << d << std::endl
	    << "m = " << m << std::endl
	    << "lambda_W = " << lambda_W << std::endl
	    << "lambda_Z = " << lambda_Z << std::endl
	    << "lambda_B = " << lambda_B << std::endl
	    << "gamma = " << gamma << std::endl
	    << "batch-size = " << batch_size << std::endl
	    << "epochs = " << epochs << std::endl
	    << "iters = " << iters << std::endl
	    << "seed = " << seed << std::endl;

    if(initialization_type == per_class_kmeans)
      f << "initialization_type = per_class_kmeans" << std::endl;
    else if(initialization_type == overall_kmeans)
      f << "initialization_type = overall_kmeans" << std::endl;
    else if(initialization_type == sample)
      f << "initialization_type = sample" << std::endl;
    else if(initialization_type == predefined)
      f << "initialization_type = predefined" << std::endl;
    else;
    
    if(normalization_type == l2)
      f << "normalization_type = l2-normalization" << std::endl;
    else if(normalization_type == min_max)
      f << "normalization_type = minmax-normalization" << std::endl;
    else if(normalization_type == none)
      f << "normalization_type = none" << std::endl;
    else;

    f << "Command line call: " << std::endl;
    for (int i=0; i<argc; i++)
      f << argv[i] << " ";
    f << std::endl;

    f << "param | iter | objective, training accuracy, testing accuracy\n";
    for (int i=0; i<iters*3 + 1; i++){
      if (i==0) f << "init  | ";
      else if (i%3 == 1) f << "W     | ";
      else if (i%3 == 2) f << "Z     | ";
      else if (i%3 == 0) f << "B     | ";
      else;
      f << (i-1)/3 << "    | ";
      f << *(stats + i*3) << ", " << *(stats + i*3 + 1) << ", " << *(stats + i*3 + 2) << "\n";
    }
    f << std::endl;
  }
  
  void new_dataset (){
    ntrain = -1;
    ntest = -1;
    
    problem_type = undefined_type;
    initialization_type = undefined_initialization; 
    format_type = undefined_format;
    normalization_type = none;

    seed = 42;
    batch_size = -1;
    iters = -1;
    epochs = -1;
    
    d = -1;
    m = -1;
    k = -1;
   
    D = -1;
    l = -1;

    lambda_W = 1.0;
    lambda_Z = 1.0;
    lambda_B = 1.0;
  }
  
  void office_initialize (){
    assert (false); // no support for ranking datasets in main branch
    
    ntrain = 6666670;
    ntest = 3333330;
    ntot = ntrain + ntest;

    problem_type = binary;  
    initialization_type = per_class_kmeans;
    format_type = office_format;
    normalization_type = min_max;
    
    seed = 42;
    batch_size = 1<<15;
    iters = 10;
    epochs = 1;

    D = 525;
    d = 30;
    l = 2;

    lambda_W = 1.0;
    lambda_Z = 1.0;
    lambda_B = 1.0;

    sprintf (indir, HOME_PATH"/datasets/office/data");    
  }

  void covtype_initialize (){
    ntrain = 522911;
    ntest = 58101;
    ntot = ntrain + ntest;
  
    problem_type = binary;
    initialization_type = per_class_kmeans;
    format_type = covtype_format;
    normalization_type = min_max;
    
    seed = 42;
    batch_size = 1<<10;
    iters = 10;
    epochs = 5;

    D = 54;
    d = 54;
    l = 2;

    lambda_W = 0.7L;
    lambda_Z = 0.7L;
    lambda_B = 0.1L;
    
    sprintf (indir, HOME_PATH"/datasets/covtype/data");    
  }

  void mnist_binary_initialize (){
    ntrain = 60000;
    ntest = 10000;
    ntot = ntrain + ntest;
  
    problem_type = binary;
    initialization_type = per_class_kmeans;
    format_type = mnist_format;
    normalization_type = none;
    
    seed = 42;
    batch_size = 1<<10;
    iters = 1;
    epochs = 1;

    D = 784;
    d = 10;
    l = 2;

    lambda_W = 1.0;
    lambda_Z = 1.0;
    lambda_B = 1.0;

    sprintf (indir, HOME_PATH"/datasets/mnist_binary/data");
  }

  void wiki10_initialize (){
    ntrain = 14146;
    ntest = 6616;
    ntot = ntrain + ntest;
  
    problem_type = multilabel;
    initialization_type = overall_kmeans;
    format_type = libsvm_format;
    normalization_type = min_max;
    
    seed = 42;
    batch_size = 1<<12;
    iters = 20;
    epochs = 5;

    D = 101938;
    d = 75;
    l = 30938;

    lambda_W = 0.4;
    lambda_Z = 0.0005;
    lambda_B = 0.3;

    sprintf (indir, HOME_PATH"/datasets/wiki10/data");
  }

  void eurlex_initialize (){
    ntrain = 15539;
    ntest = 3809;
    ntot = ntrain + ntest;
  
    problem_type = multilabel;
    initialization_type = overall_kmeans;
    format_type = libsvm_format;
    normalization_type = none; 
    
    seed = 42;
    batch_size = 1<<10;
    iters = 20;
    epochs = 2;

    D = 5000;
    d = 250;
    l = 3993;

    lambda_W = 0.7;
    lambda_Z = 0.15;
    lambda_B = 1.0;

    sprintf (indir, HOME_PATH"/datasets/eurlex/data");
  }

  void delicious_initialize (){
    ntrain = 12920;
    ntest = 3185;
    ntot = ntrain + ntest;
  
    problem_type = multilabel;
    initialization_type = overall_kmeans;
    format_type = libsvm_format;
    normalization_type = none;
    
    seed = 42;
    batch_size = 1<<10;
    iters = 20;
    epochs = 2;

    D = 500;
    d = 100;
    l = 983;

    lambda_W = 1.0;
    lambda_Z = 0.05;
    lambda_B = 1.0;

    sprintf (indir, HOME_PATH"/datasets/delicious/data");
  }    

  void mediamill_initialize (){
    ntrain = 29796;
    ntest = 12381;
    ntot = ntrain + ntest;
  
    problem_type = multilabel;
    initialization_type = overall_kmeans;
    format_type = libsvm_format;
    normalization_type = none;
    
    seed = 42;
    batch_size = 1<<10;
    iters = 20;
    epochs = 2;
    
    D = 120;
    d = 30;
    l = 101;

    lambda_W = 1.0;
    lambda_Z = 1.0;
    lambda_B = 1.0;

    sprintf (indir, HOME_PATH"/datasets/mediamill/data");
  }

  void amazon13_initialize (){
    ntrain = 1186239;
    ntest = 306782;
    ntot = ntrain + ntest;
  
    problem_type = multilabel;
    initialization_type = overall_kmeans;
    format_type = libsvm_format;
    normalization_type = min_max;
    
    seed = 42;
    batch_size = 1<<15;
    iters = 50;
    epochs = 1;
    
    D = 203882;
    d = 75;
    d = 200;
    l = 13330;

    lambda_W = 1.0;
    lambda_Z = 0.01;
    lambda_B = 1.0;

    sprintf (indir, HOME_PATH"/datasets/amazon_13k/data");
  }

  void amazon670_initialize (){
    ntrain = 490449;
    ntest = 153025;
    ntot = ntrain + ntest;
  
    problem_type = multilabel;
    initialization_type = overall_kmeans;
    format_type = libsvm_format;
    normalization_type = min_max;
    
    seed = 42;
    batch_size = 1<<15;
    iters = 20;
    epochs = 1;
    
    D = 135909;
    d = 100;
    l = 670091;

    lambda_W = 0.2;
    lambda_Z = 0.0001;
    lambda_B = 0.2;

    sprintf (indir, HOME_PATH"/datasets/amazon_670/data");
  }
};
#endif
