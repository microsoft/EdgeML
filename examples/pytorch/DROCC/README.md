# Deep Robust One-Class Classification 
In this directory we present examples of how to use the `DROCCTrainer` to replicate results in [paper](https://proceedings.icml.cc/book/4293.pdf).

`DROCCTrainer` is part of the `edgeml_pytorch` package. Please install the `edgeml_pytorch` package as follows:
```
git clone https://github.com/microsoft/EdgeML
cd EdgeML/pytorch
pip install -e .
``` 

## Tabular Experiments
Data is expected in the following format:
```
train_data.npy: features of train data
test_data.npy: features of test data
train_labels.npy: labels for train data (Normal Class Labelled as 1)
test_labels.npy: labels for test data
```

### Arrhythmia and Thyroid
* Download the datasets from the ODDS Repository, [Arrhythmia](http://odds.cs.stonybrook.edu/arrhythmia-dataset/) and [Thyroid](http://odds.cs.stonybrook.edu/annthyroid-dataset/). This will consist of `arrhythmia.mat` or `annthyroid.mat`.
* The data is divided for training as presented in previous works: [DAGMM](https://openreview.net/forum?id=BJJLHbb0-) and [GOAD](https://openreview.net/forum?id=H1lK_lBtvS).
* To generate the training and test data, use the `data_process_scripts/process_odds.py` script as follows 
```
python data_process_scripts/process_odds.py -d <path/to/downloaded_data/file_name.mat> -o <output path>
```
The output path is referred to as "root_data" in the following section.

### Abalone
* Download the `abalone.data` file from the UCI Repository [here](http://archive.ics.uci.edu/ml/datasets/Abalone).
* To generate the training and test data, use the `data_process_scripts/process_abalone.py` script as follows 
```
python data_process_scripts/process_abalone.py -d <path/to/data/abalone.data> -o <output path>
```
The output path is referred to as "root_data" in the following section.

### Command to run experiments to reproduce results
#### Arrhythmia
```
python3 main_tabular.py --hd 128 --lr 0.0001 --lamda 1 --gamma 2 --ascent_step_size 0.001 --radius 16 --batch_size 256 --epochs 200 --optim 0 --restore 0 --metric F1 -d "root_data"
```

#### Thyroid
```
python3 main_tabular.py --hd 128 --lr 0.001 --lamda 1 --gamma 2 --ascent_step_size 0.001 --radius 2.5 --batch_size 256 --epochs 100 --optim 0 --restore 0 --metric F1 -d "root_data"
```

#### Abalone 
```
python3 main_tabular.py --hd 128 --lr 0.001 --lamda 1 --gamma 2 --ascent_step_size 0.001 --radius 3 --batch_size 256 --epochs 200 --optim 0 --restore 0 --metric F1 -d "root_data"
```


## Time-Series Experiments

### Data Processing
### Epilepsy
* Download the dataset from the UCI Repository [here](https://archive.ics.uci.edu/ml/datasets/Epileptic+Seizure+Recognition). This will consists of a `data.csv` file. 
* To generate the training and test data, use the `data_process_scripts/process_epilepsy.py` script as follows

```
python data_process_scripts/process_epilepsy.py -d <path/to/data/data.csv> -o <output path>
```
The output path is referred to as "root_data" in the following section.


### Example Usage for Epilepsy Dataset
```
python3  main_timeseries.py --hd 128 --lr 0.00001 --lamda 0.5 --gamma 2 --ascent_step_size 0.1 --radius 10 --batch_size 256 --epochs 200  --optim 0 --restore 0 --metric AUC -d "root_data"
```

## CIFAR Experiments
```
python3  main_cifar.py  --lamda 1  --radius 8 --lr 0.001 --gamma 1 --ascent_step_size 0.001 --batch_size 256 --epochs 40 --optim 0 --normal_class 0
```


### Arguments Detail
normal_class => CIFAR10 class to be considered as normal  
lamda => Weightage to the loss from adversarially sampled negative points (\mu in the paper)  
radius => radius corresponding to the definition of set N_i(r)  
hd => LSTM Hidden Dimension  
optim => 0: Adam   1: SGD(M)  
ascent_step_size => step size for gradient ascent to generate adversarial anomalies

