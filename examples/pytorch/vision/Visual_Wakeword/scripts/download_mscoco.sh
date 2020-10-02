# Copyright 2020 Maxim Bonnaerens. All Rights Reserved.
#
# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
# File modified from tensorflow/models/research/slim/datasets/download_mscoco.sh

# Script to download the COCO dataset. See
# http://cocodataset.org/#overview for an overview of the dataset.
#
# usage:
#  bash scripts/download_mscoco.sh path-to-COCO-dataset
#
set -e

YEAR=${2:-2014}
if [ -z "$1" ]; then
  echo "usage download_mscoco.sh [data dir] (2014|2017)"
  exit
fi

if [ "$(uname)" == "Darwin" ]; then
  UNZIP="tar -xf"
else
  UNZIP="unzip -nq"
fi

# Create the output directories.
OUTPUT_DIR="${1%/}"
mkdir -p "${OUTPUT_DIR}"

# Helper function to download and unpack a .zip file.
function download_and_unzip() {
  local BASE_URL=${1}
  local FILENAME=${2}

  if [ ! -f "${FILENAME}" ]; then
    echo "Downloading ${FILENAME} to $(pwd)"
    wget -nd -c "${BASE_URL}/${FILENAME}"
  else
    echo "Skipping download of ${FILENAME}"
  fi
  echo "Unzipping ${FILENAME}"
  ${UNZIP} "${FILENAME}"
  rm "${FILENAME}"
}

cd "${OUTPUT_DIR}"

# Download the images.
BASE_IMAGE_URL="http://images.cocodataset.org/zips"

TRAIN_IMAGE_FILE="train${YEAR}.zip"
download_and_unzip ${BASE_IMAGE_URL} "${TRAIN_IMAGE_FILE}"
TRAIN_IMAGE_DIR="${OUTPUT_DIR}/train${YEAR}"

VAL_IMAGE_FILE="val${YEAR}.zip"
download_and_unzip ${BASE_IMAGE_URL} "${VAL_IMAGE_FILE}"
VAL_IMAGE_DIR="${OUTPUT_DIR}/val${YEAR}"

COMMON_DIR="all$YEAR"
mkdir -p "${COMMON_DIR}"
for i in ${TRAIN_IMAGE_DIR}/*; do cp --symbolic-link "$i" ${COMMON_DIR}/; done
for i in ${VAL_IMAGE_DIR}/*; do cp --symbolic-link "$i" ${COMMON_DIR}/; done

# Download the annotations.
BASE_INSTANCES_URL="http://images.cocodataset.org/annotations"
INSTANCES_FILE="annotations_trainval${YEAR}.zip"
download_and_unzip ${BASE_INSTANCES_URL} "${INSTANCES_FILE}"
