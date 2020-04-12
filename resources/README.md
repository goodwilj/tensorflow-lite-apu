## Building CNN Inference TF Lite App

### Cross Compiling the TF Lite Application
Download the cross-compilation toolchain for aarch64:
```bash
  sudo apt-get update
  sudo apt-get install crossbuild-essential-arm64
```

Clone this fork of the TF repository and cd into ```tensorflow-lite-apu``` directory. To build the TF Lite binaries, run:
```bash
 ./tensorflow/lite/tools/make/build_aarch64_lib.sh
```
This will compile the applications in the examples folder:
```bash
./tensorflow/lite/examples
```
The ```cnn_inference``` folder contains the source code for this application.
The aarch64 binary is located in:
```bash
./tensorflow/lite/tools/make/gen/linux_aarch64/bin/cnn_inference
```

This should also compile a static library in:
```bash
./tensorflow/lite/tools/make/gen/linux_aarch64/lib/libtensorflow-lite.a
```
You can link against this library for new applications.

### Running Leda-G board
```scp``` the aarch64 ```cnn_inference``` binary to the board.  Also, ```scp``` the mobilenet tflite file and grace_hopper.bmp file located in the ```./resources``` folder.
Use ```./cnn_inference -h``` for info on command-line options.
