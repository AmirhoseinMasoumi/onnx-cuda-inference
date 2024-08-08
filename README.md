# ONNX Inference with CUDA in C++
This project demonstrates how to perform inference using ONNX models with CUDA acceleration in C++. The project uses ONNX Runtime, OpenCV, and CUDA to process images and run inference on them.

## Table of Contents
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Details about Inference and CUDA](#details-about-inference-and-cuda)
- [Contributing](#contributing)
- [License](#license)

## Prerequisites
Before you begin, ensure you have met the following requirements:

- CMake (version 3.18 or higher)
- ONNX Runtime (version 1.18.1)
- OpenCV with CUDA support
- CUDA Toolkit (version 12.x)
- cuDNN (version 9.x)
- Visual Studio (for Windows users)

## Installation

#### 1. Clone the Repository
```sh
git clone https://github.com/yourusername/onnxInference.git
cd onnxInference
 ```

#### 2. Set up Directories
Modify the paths in CMakeLists.txt to match your local setup:
```cmake
set(ONNXRUNTIME_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/onnxruntime/include")
set(ONNXRUNTIME_LIBRARY_DIR "${CMAKE_SOURCE_DIR}/onnxruntime/lib")

set(OPENCV_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/opencv_cuda/include")
set(OPENCV_LIBRARY_DIR "${CMAKE_SOURCE_DIR}/opencv_cuda/x64/vc17/lib")

set(CUDA_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/cuda/v12.4/include")
set(CUDA_LIBRARY_DIR "${CMAKE_SOURCE_DIR}/cuda/v12.4/lib/x64")
 ```

#### 3. Build the Project
Create a build directory and run CMake:
```sh
mkdir build
cd build
cmake ..
cmake --build .
 ```

## Usage
Prepare Models and Test Images

Ensure your ONNX models and test images are placed in the appropriate directories as specified in main.cpp:

```cpp
std::string modelsPath = "D:/Temp/trainModel/models";
std::string modelPathLowRes = modelsPath + "/segmentation_model_320x288.onnx";
std::string modelPathHighRes = modelsPath + "/segmentation_model_640x576.onnx";
std::string testImageFolder = "D:/Temp/trainModel/test_images/";
 ```

## Run the Inference

Execute the program with optional high-resolution flag:

```sh
./onnxInference [--high-res]
 ```

## View Results

The program will display the predicted masks and overlapped images. Press ESC to exit.

## Project Structure
- **CMakeLists.txt:** Configuration file for CMake.
- **main.cpp:** Contains the main function that processes images and runs inference.
- **onnxInference.h:** Header file for the ONNXInference class.
- **onnxInference.cpp:** Implementation of the ONNXInference class.

## Details about Inference and CUDA
### ONNX Inference with ONNX Runtime
ONNX Runtime is a cross-platform, high-performance scoring engine for Open Neural Network Exchange (ONNX) models. It enables the acceleration of machine learning inferencing across various hardware configurations.

In this project:
- The ONNX model is loaded using Ort::Session.
- Inference is run using session.Run().
- Input preprocessing involves resizing and normalizing images.
- Output postprocessing involves thresholding and resizing the output to match the input image size.

## CUDA Acceleration
CUDA (Compute Unified Device Architecture) is a parallel computing platform and application programming interface (API) model created by NVIDIA. It allows developers to use NVIDIA GPUs for general-purpose processing.

In this project:
- CUDA is used to accelerate the inference process.
- OrtCUDAProviderOptions is configured to enable CUDA as the execution provider in ONNX Runtime.
- The ONNXRuntime session is configured to use CUDA for running inference, which significantly improves performance on compatible hardware.
  
## Key Components and Methods
**`ONNXInference::runInference`**  
This method handles the entire process of:
1. Preprocessing the input image (resizing, normalizing).
2. Creating an input tensor suitable for the ONNX model.
3. Running the inference using ONNX Runtime with CUDA support.
4. Postprocessing the output (thresholding, resizing).
 
**`ONNXInference::logGpuProperties`**  
This method logs GPU properties such as:
1. Device name
2. Total global memory
3. Shared memory per block
4. Number of multiprocessors
5. Clock rate, etc.
Logging these properties helps in understanding the capabilities and performance characteristics of the GPU being used.

## Contributing
Contributions are welcome! Please open an issue or submit a pull request for any changes.
Fork the Project
1. Create your Feature Branch (git checkout -b feature/AmazingFeature)
2. Commit your Changes (git commit -m 'Add some AmazingFeature')
3. Push to the Branch (git push origin feature/AmazingFeature)
4. Open a Pull Request
   
## License
Distributed under the MIT License. See [LICENSE](LICENSE) for more information.
