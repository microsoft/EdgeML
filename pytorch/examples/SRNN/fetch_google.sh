#/bin/bash

# If OUT_DIR is modified, please make sure it is reflected in process_google.py
# as well.
OUT_DIR='./GoogleSpeech/'
mkdir -pv $OUT_DIR
mkdir -pv $OUT_DIR/Raw
mkdir -pv $OUT_DIR/Extracted

echo "Downloading dataset."
echo ""
URL='http://download.tensorflow.org/data/speech_commands_v0.01.tar.gz'
cd $OUT_DIR/Raw
wget $URL

if [ $? -eq 0 ]; then
	echo "Download complete. Extracting files . . ."
else
	echo "Fail"
	exit
fi
tar -xzf speech_commands_v0.01.tar.gz
echo "Done. Please run process_google.py for feature extraction"
