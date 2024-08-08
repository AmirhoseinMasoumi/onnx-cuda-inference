cmake_minimum_required(VERSION 3.18)

project(onnxInference LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Define the relative path to the ONNX Runtime installation
set(ONNXRUNTIME_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/onnxruntime/include")
set(ONNXRUNTIME_LIBRARY_DIR "${CMAKE_SOURCE_DIR}/onnxruntime/lib")

# Define the relative path to the OpenCV installation
set(OPENCV_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/opencv_cuda/include")
set(OPENCV_LIBRARY_DIR "${CMAKE_SOURCE_DIR}/opencv_cuda/x64/vc17/lib")

# Define the relative path to the CUDA installation
set(CUDA_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/cuda/v12.4/include")
set(CUDA_LIBRARY_DIR "${CMAKE_SOURCE_DIR}/cuda/v12.4/lib/x64")

# Define your executable
add_executable(onnxInference main.cpp onnxinference.cpp onnxinference.h)

# Add include directories
include_directories(${OPENCV_INCLUDE_DIR})
include_directories(${CUDA_INCLUDE_DIR})
include_directories(${ONNXRUNTIME_INCLUDE_DIR})

# Set the linking directories
link_directories(${OPENCV_LIBRARY_DIR})
link_directories(${CUDA_LIBRARY_DIR})
link_directories(${ONNXRUNTIME_LIBRARY_DIR})

# Explicitly link required OpenCV libraries
target_link_libraries(onnxInference
    "${OPENCV_LIBRARY_DIR}/opencv_core4100.lib"
    "${OPENCV_LIBRARY_DIR}/opencv_imgcodecs4100.lib"
    "${OPENCV_LIBRARY_DIR}/opencv_imgproc4100.lib"
    "${OPENCV_LIBRARY_DIR}/opencv_highgui4100.lib"
    "${OPENCV_LIBRARY_DIR}/opencv_cudaarithm4100.lib"
    "${OPENCV_LIBRARY_DIR}/opencv_cudaimgproc4100.lib"
    "${OPENCV_LIBRARY_DIR}/opencv_cudafilters4100.lib"
    "${OPENCV_LIBRARY_DIR}/opencv_cudawarping4100.lib"
    "${OPENCV_LIBRARY_DIR}/opencv_dnn4100.lib"

    "${CUDA_LIBRARY_DIR}/cublas.lib"
    "${CUDA_LIBRARY_DIR}/cublasLt.lib"
    "${CUDA_LIBRARY_DIR}/cudart.lib"

    "${ONNXRUNTIME_LIBRARY_DIR}/onnxruntime.lib"
    "${ONNXRUNTIME_LIBRARY_DIR}/onnxruntime_providers_cuda.lib"
)

include(GNUInstallDirs)
install(TARGETS onnxInference
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
