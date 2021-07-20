# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

import csv
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--csv_file', default=None, required=True)
parser.add_argument('--target_folder', default=None, required=True)
args = parser.parse_args()

with open(args.csv_file, 'r') as csv_f:
    reader = csv.reader(csv_f, skipinitialspace=True)
    # Skip 3 lines ; Header
    next(reader)
    next(reader)
    next(reader)
    for row in reader:
        # Logging
        print(row, flush=True)
        # Link for the Youtube Video
        YouTube_ID = row[0]                 # "-0RWZT-miFs"
        start_time = int(float(row[1]))     # 420
        end_time = int(float(row[2]))       # 430
        # Construct downloadable link
        YouTube_link = "https://youtu.be/" + YouTube_ID
        # Output Filename
        output_file = f"{args.target_folder}/ID_{YouTube_ID}.wav"
        # Start time in hrs:min:sec format
        start_sec = start_time % 60
        start_min = (start_time // 60) % 60
        start_hrs = start_time // 3600
        # End time in hrs:min:sec format
        end_sec = end_time % 60
        end_min = (end_time // 60) % 60
        end_hrs = end_time // 3600
        # Start and End time args
        time_args = f"-ss {start_hrs}:{start_min}:{start_sec} -to {end_hrs}:{end_min}:{end_sec}"
        # Command Line Execution
        os.system(f"youtube-dl -x -q --audio-format wav --postprocessor-args '{time_args}' {YouTube_link}" + " --exec 'mv {} " + f"{output_file}'")
        print('', flush=True)
