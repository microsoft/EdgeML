# Auxiliary Files to help Download and Prepare the Data

## YouTube Additive Noise
Run the following commands to download the CSV Files to download the YouTube Additive Noise Data :

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
Choose the log rate for how often the log should be printed for the sample rate conversion. This will print a string ever log_rate iterations.

Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT license.