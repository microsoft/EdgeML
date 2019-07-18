# S-RNN Predictor With Google Dataset

Here we demonstrate how SRNN can be used to deploy a key-word spotting model on
the [Azure IoT Dev-Kit](https://microsoft.github.io/azure-iot-developer-kit/)
powered by the Coretex M4.

## Set-up Instructios

1. Instructions on how to get started quickly with MXChip can be found 
[here])(https://github.com/VSChina/devkit-mbedos5-getstarted). Please follow
those instructions to set-up the environment and verify that everything is
working properly by burning the `GettingStarted`example mentioned there. 
2. Clone the EdgeML repository. Assume this repository lies in `$EDGEML_HOME`.
3. Change directory to `devkit-mbedos5-getstarted` cloned in step 1.
```
   cd devkit-mbedos5-getstarted/
```
3. Remove the provided `GetStarted` example and replace it with
   `$EDGEML_HOME/Applications/KeyWord-MXChip/'
```
   rm -r GetStarted
   cp $EDGEML_HOME/Applications/KeyWord-MXChip/ ./
```
4. Open `devkit-mbedos5-getstarted/.mbedignore` in your favourite text editor
   and append the following lines:
  ```
  Keyword-MXChip/test/*
  ```
5. Copy the provided build profile file, `develop_custom.json` into the
   `mbed-os` profiles folder:
   ```
      cp develop_custom.json mbed-os/toos/
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
