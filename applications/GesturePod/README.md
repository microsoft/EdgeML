GesturePod 
==========

GesturePod is a plug-and-play gesture recognition framework. GesturePod enables detection of gestures locally on tiny-microcontrollers through a ML based gesture recognition pipeline. One exciting application is to detect gestures on white-cane. GesturePod can be clamped onto
any white-cane used by persons with Visually Impairment. Once clamped onto the cane firmly, 
simple and natural gestures performed on the cane can be used to interact with various devices,
for instance a mobile phone. Watch the GesturePod video [here](https://1drv.ms/u/s!AjDloPaG_l0Et7Ikid1voOVFuI116Q).

In this directory, we provide resources for
  1. Running the gesture recognition pipeline on the [Arduino MKR1000](https://store.arduino.cc/usa/arduino-mkr1000), in the `onMKR1000` directory, and
  2. Running the gesture recognition pipeline on your computer in the `onComputer` directory.
  3. The pipeline to a) collect data, b) label the data, and c) extract features  is available in the `training` directory.

To build your own GesturePod refer to our [instructable](https://microsoft.github.io/EdgeML/Projects/GesturePod/instructable.html). 

To learn more about GesturePod, refer our [UIST'19 publication](https://github.com/microsoft/EdgeML/blob/master/docs/publications/GesturePod-UIST19.pdf).

## GesturePod Dataset

The benchmark dataset for Gesture recognition can be downloaded [here](https://www.microsoft.com/en-us/research/uploads/prod/2018/05/dataTR_v1.tar.gz) [MIT Open source license].

_If you are using the dataset please [cite](https://dl.acm.org/doi/10.1145/3332165.3347881) GesturePod._
