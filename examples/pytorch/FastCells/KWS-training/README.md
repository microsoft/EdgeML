## Training Keyword-Spotting model

This example demonstrates how to train a FastGRNN-based keyword spotting model based on the Google speech commands dataset,
compile it using the ELL compiler and deploy the keyword spotting model on [STM BlueCoin](https://www.st.com/en/evaluation-tools/steval-bcnkt01v1.html).
Follow the steps below to featurize data using ELL, train and export an ONNX model using the EdgeML library,
and prepare a binary that provides prediction capability using the ELL library.   

### Install ELL
[link](https://github.com/microsoft/ELL)

### Download Google speech commands dataset
Download the [dataset](https://storage.cloud.google.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz) and extract data
```
mkdir data_speech_commands_v2
tar xvzf data_speech_commands_v0.02.tar.gz -C data_speech_commands_v2
```

### Export path to dataset and ELL
```
export ELL_ROOT=<path to directory were ELL is installed>
### export ELL_ROOT=/home/user/ELL
export DATASET_PATH=<path to directory were speechcommand dataset is extracted>
### export DATASET_PATH=/mnt/../../data_speech_data_v2
```

### Make training list -
Use `-max n` to over-ride the default limit on the maximum number of samples from each category including `background`. For low false positive rate, train with a large number of negative `background` examples, say 50000 or 250000.
```
python3 $ELL_ROOT/tools/utilities/pythonlibs/audio/training/make_training_list.py -max 50000 --wav_files $DATASET_PATH
```

### Create an ELL featurizer -
```
python $ELL_ROOT/tools/utilities/pythonlibs/audio/training/make_featurizer.py -ws 400 --nfft 512 --iir --log --log_delta 2.220446049250313e-16 --power_spec -fs 32
```

### Compile the ELL featurizer -
```
python $ELL_ROOT/tools/wrap/wrap.py --model_file featurizer.ell --outdir compiled_featurizer --module_name mfcc
cd compiled_featurizer && mkdir build && cd build && cmake .. && make && cd ../..
```

### Pre-Process Dataset:
```
python $ELL_ROOT/tools/utilities/pythonlibs/audio/training/make_dataset.py --list_file $DATASET_PATH/training_list.txt --featurizer compiled_featurizer/mfcc --window_size 98 --shift 98 --multicore
python $ELL_ROOT/tools/utilities/pythonlibs/audio/training/make_dataset.py --list_file $DATASET_PATH/validation_list.txt --featurizer compiled_featurizer/mfcc --window_size 98 --shift 98 --multicore
python $ELL_ROOT/tools/utilities/pythonlibs/audio/training/make_dataset.py --list_file $DATASET_PATH/testing_list.txt --featurizer compiled_featurizer/mfcc --window_size 98 --shift 98 --multicore
```

If you have a background noise clips not containing keywords that you want to fuse with your dataset with, 
place them in a folder `$DATASET_PATH/backgroundNoise` and follow these instructions instead of the ones above. 
```
python $ELL_ROOT/tools/utilities/pythonlibs/audio/training/make_dataset.py --list_file $DATASET_PATH/training_list.txt --featurizer compiled_featurizer/mfcc --window_size 98 --shift 98 --multicore --noise_path $DATASET_PATH/backgroundNoise --max_noise_ratio 0.1 --noise_selection 1
python $ELL_ROOT/tools/utilities/pythonlibs/audio/training/make_dataset.py --list_file $DATASET_PATH/validation_list.txt --featurizer compiled_featurizer/mfcc --window_size 98 --shift 98 --multicore --noise_path $DATASET_PATH/backgroundNoise --max_noise_ratio 0.1 --noise_selection 1
python $ELL_ROOT/tools/utilities/pythonlibs/audio/training/make_dataset.py --list_file $DATASET_PATH/testing_list.txt --featurizer compiled_featurizer/mfcc --window_size 98 --shift 98 --multicore --noise_path $DATASET_PATH/backgroundNoise --max_noise_ratio 0.1 --noise_selection 1
```

### Run model training:
```
python examples/pytorch/FastCells/train_classifier.py \
	--use_gpu  --normalize --rolling --max_rolling_length 235 \
	-a $DATASET_PATH -c $DATASET_PATH/categories.txt --outdir $MODEL_DIR \
	--architecture FastGRNNCUDA --num_layers 2 \
	--epochs 250 --learning_rate 0.005 -bs 128 -hu 128 \
	--lr_min 0.0005 --lr_scheduler CosineAnnealingLR --lr_peaks 0
```
Drop the `--rolling` and `--max_rolling_length` options if you are going to run inference on 1 second clips,
and do not plan to stream data through the model without resettting.

### Convert .onnx model to .ell IR
```
pip install onnx #If you haven't already
python $ELL_ROOT/tools/importers/onnx/onnx_import.py output_model/model.onnx
```


### Compiling model and featurizer header and binary files for ARM Cortex M4 class devices.

These commands will use ELL compiler to generate some files of which 4 are required: featurizer.h, featurizer.S, model.h and model.S

#### For devices with hard FPU, e.g., STM Bluecoin
```
$ELL_ROOT/build/bin/compile -imap model.ell -cfn Predict -cmn completemodel --bitcode -od . --fuseLinearOps True --header --blas false --optimize true --target custom --numBits 32 --cpu cortex-m4 --triple armv6m-gnueabi --features +vfp4,+d16
/usr/lib/llvm-8/bin/opt model.bc -o model.opt.bc -O3
/usr/lib/llvm-8/bin/llc model.opt.bc -o model.S -O3 -filetype=asm -mtriple=armv6m-gnueabi -mcpu=cortex-m4 -relocation-model=pic -float-abi=hard -mattr=+vfp4,+d16
$ELL_ROOT/build/bin/compile -imap featurizer.ell -cfn Filter -cmn mfcc --bitcode -od . --fuseLinearOps True --header --blas false --optimize true --target custom --numBits 32 --cpu cortex-m4 --triple armv6m-gnueabi --features +vfp4,+d16
/usr/lib/llvm-8/bin/opt featurizer.bc -o featurizer.opt.bc -O3
/usr/lib/llvm-8/bin/llc featurizer.opt.bc -o featurizer.S -O3 -filetype=asm -mtriple=armv6m-gnueabi -mcpu=cortex-m4 -relocation-model=pic -float-abi=hard -mattr=+vfp4,+d16
```

#### For M4 class devices without hard FPU, e.g., MXchip
```
$ELL_ROOT/build/bin/compile -imap model.ell -cfn Predict -cmn completemodel --bitcode -od . --fuseLinearOps True --header --blas false --optimize true --target custom --numBits 32 --cpu cortex-m4 --triple armv6m-gnueabi --features +vfp4,+d16,+soft-float
/usr/lib/llvm-8/bin/opt model.bc -o model.opt.bc -O3
/usr/lib/llvm-8/bin/llc model.opt.bc -o model.S -O3 -filetype=asm -mtriple=armv6m-gnueabi -mcpu=cortex-m4 -relocation-model=pic -float-abi=soft -mattr=+vfp4,+d16
$ELL_ROOT/build/bin/compile -imap featurizer.ell -cfn Filter -cmn mfcc --bitcode -od . --fuseLinearOps True --header --blas false --optimize true --target custom --numBits 32 --cpu cortex-m4 --triple armv6m-gnueabi --features +vfp4,+d16,+soft-float
/usr/lib/llvm-8/bin/opt featurizer.bc -o featurizer.opt.bc -O3
/usr/lib/llvm-8/bin/llc featurizer.opt.bc -o featurizer.S -O3 -filetype=asm -mtriple=armv6m-gnueabi -mcpu=cortex-m4 -relocation-model=pic -float-abi=soft -mattr=+vfp4,+d16
```
