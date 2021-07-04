# Auxiliary Files to help Download and Prepare the Data

## Note
When running commands it is recommended to use the following format to run the files uninterrupted (detached) and log the output.
```
nohup python srcipt_execution args > log.txt &
```
Please replace script_execution with the python commands below.<br/>
Alternately tmux or other commands can be used in place of the above format.

## YouTube Additive Noise
Run the following commands to download the CSV Files to download the YouTube Additive Noise Data (there is no need to use nohup for the wget file) :

```
wget http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/balanced_train_segments.csv
```
Followed by the extraction script to download the actual data :
```
python download_youtube_data.py --csv_file=/path/to/csv_file.csv --target_folder=/path/to/target/folder/
```

Please check [Google's Audioset data page](https://research.google.com/audioset/download.html) for further details.

The downloaded files would need to be converted to 16KHz for our pipeline. Please run the following for the same :
```
python convert_sampling_rate.py --source_folder=/path/to/csv_file.csv --target_folder=/path/to/target/16KHz_folder/ --fs=16000 --log_rate=100
```
The script can convert the sampling rate of any wav file to the specified --fs. But for our applications, we use 16KHz only.<br/>
Choose the log rate for how often the log should be printed for the sample rate converion. This will print a strinng ever log_rate iterations.