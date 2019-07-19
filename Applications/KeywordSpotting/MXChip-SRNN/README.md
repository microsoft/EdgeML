# S-RNN for Speech Command Detection

Here we demonstrate how SRNN can be used to deploy a key-word spotting model on
the [Azure IoT Dev-Kit](https://microsoft.github.io/azure-iot-developer-kit/)
powered by the Coretex M4. The model provided is based on the [Speech Commands Dataset](https://ai.googleblog.com/2017/08/launching-speech-commands-dataset.html). It is trainined to recognise commands in the set: `[go, no, on, up, bed, cat, dog, off, one, six, two, yes]`. When no keyword is detected, the screen will print 'Noise'.

Unit testing and benchmarking code that was used to develop this implementation of S-RNN is provided in the `tests` directory and can be used for debugging purposes.

## Instructions for Deployment 

1. Follow instructions [here](https://github.com/VSChina/devkit-mbedos5-getstarted) 
to set-up the MXChip environment. Verify that the set-up is
working properly by burning the `GettingStarted` example mentioned there. 
2. Clone the EdgeML repository. Let this repository lie in `$EDGEML_HOME`.
3. Change directory to `devkit-mbedos5-getstarted` cloned in step 1.
   ```
      cd devkit-mbedos5-getstarted/
   ```
3. Remove the provided `GetStarted` example and replace it with
   `$EDGEML_HOME/Applications/KeywordSpotting/MXChip-SRNN/`
   ```
      rm -r GetStarted
      cp -r $EDGEML_HOME/Applications/KeywordSpotting/MXChip-SRNN ./
   ```
4. Open `devkit-mbedos5-getstarted/.mbedignore` in your favourite text editor
   and append the following lines:
   ```
      MXChip-SRNN/test/*
   ```
5. Copy the provided build profile file, `develop_custom.json` into the
   `mbed-os` profiles folder:
   ```
      cp MXChip-SRNN/develop_custom.json mbed-os/tools/profiles/
   ```
6. Compile using:
   ```
      mbed compile --profile develop_custom
   ```

7. Upload to MXChip IoT DevKit:
  - Connect the MXChip IoT DevKit with your machine via USB.
  - You will find a removable USB Mass Storage disk named AZ3166.
  - Copy the
  `.\BUILD\AZ3166\GCC_ARM-DEVELOP_CUSTOM\devkit-mbedos5-getstarted.bin` into this disk.
  - The device will reboot and run the application. 

