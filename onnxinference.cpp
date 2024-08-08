#include "ONNXInference.h"
#include <Windows.h>
#include <iostream>
#include <cuda_runtime.h>
#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>

std::wstring ONNXInference::convertUtf8ToUtf16(const std::string &utf8Str) {
    if (utf8Str.empty()) {
        return std::wstring();
    }
    int size_needed = MultiByteToWideChar(CP_UTF8, 0, utf8Str.c_str(), static_cast<int>(utf8Str.size()), NULL, 0);
    std::wstring utf16Str(size_needed, 0);
    MultiByteToWideChar(CP_UTF8, 0, utf8Str.c_str(), static_cast<int>(utf8Str.size()), &utf16Str[0], size_needed);
    return utf16Str;
}

ONNXInference::ONNXInference(const std::string &modelPath)
    : env(ORT_LOGGING_LEVEL_WARNING, "test"), sessionOptions(), session(nullptr), outputName(std::nullopt) {
    try {
        std::cout << "Initializing ONNXInference with model path: " << modelPath << std::endl;

        sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        sessionOptions.SetExecutionMode(ExecutionMode::ORT_PARALLEL);
        sessionOptions.SetIntraOpNumThreads(4);  // Adjust based on your system's capability

        OrtCUDAProviderOptions options;
        options.device_id = 0;
        Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_CUDA(sessionOptions, options.device_id));

        int deviceId;
        cudaGetDevice(&deviceId);
        logGpuProperties(deviceId);

        std::wstring wideModelPath = convertUtf8ToUtf16(modelPath);
        session = Ort::Session(env, wideModelPath.c_str(), sessionOptions);

        // Get output name
        Ort::AllocatorWithDefaultOptions allocator;
        outputName = session.GetOutputNameAllocated(0, allocator);

        std::cout << "ONNXInference initialization complete" << std::endl;

    } catch (const Ort::Exception& e) {
        std::cerr << "ONNX Runtime exception: " << e.what() << std::endl;
        throw;
    } catch (const std::exception& e) {
        std::cerr << "Standard exception: " << e.what() << std::endl;
        throw;
    } catch (...) {
        std::cerr << "Unknown exception occurred during ONNXInference initialization" << std::endl;
        throw;
    }
}

cv::Mat ONNXInference::runInference(const cv::Mat &inputImage, bool useHighRes) {
    cv::Mat processedImage;

    if (inputImage.channels() != 3) {
        cv::cvtColor(inputImage, processedImage, cv::COLOR_GRAY2BGR);
    } else {
        processedImage = inputImage;
    }

    cv::Size targetSize = useHighRes ? cv::Size(640, 576) : cv::Size(320, 288);
    cv::Mat resizedImage;
    cv::resize(processedImage, resizedImage, targetSize);

    // Convert to float and normalize to [0, 1]
    resizedImage.convertTo(resizedImage, CV_32F, 1.0 / 255.0);

    // MobileNetV3 preprocessing: scale from [0, 1] to [0, 255]
    resizedImage = resizedImage * 255.0;

    // Flatten the image data to a vector in NHWC format
    std::vector<float> inputTensorValues;
    inputTensorValues.assign((float*)resizedImage.data, (float*)resizedImage.data + resizedImage.total() * resizedImage.channels());

    // Create input tensor with correct shape for NHWC format
    std::array<int64_t, 4> inputShape = useHighRes ?
                                            std::array<int64_t, 4>{1, 576, 640, 3} :
                                            std::array<int64_t, 4>{1, 288, 320, 3};

    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value inputTensor = Ort::Value::CreateTensor<float>(memoryInfo, inputTensorValues.data(), inputTensorValues.size(), inputShape.data(), inputShape.size());

    // Run the session
    const char* inputNames[] = {"input"};
    const char* outputNames[] = {outputName->get()};
    auto outputTensors = session.Run(Ort::RunOptions{nullptr}, inputNames, &inputTensor, 1, outputNames, 1);
    auto& outputTensor = outputTensors.front();

    // Retrieve output tensor data
    float* outputData = outputTensor.GetTensorMutableData<float>();
    cv::Mat output(useHighRes ? 576 : 288, useHighRes ? 640 : 320, CV_32FC1, outputData);

    // Apply a threshold to create a binary mask
    cv::Mat mask;
    cv::threshold(output, mask, 0.5, 1.0, cv::THRESH_BINARY);

    // Resize output to original image size
    cv::resize(mask, mask, inputImage.size(), 0, 0, cv::INTER_LINEAR);

    return mask;
}

void ONNXInference::visualize(const cv::Mat &originalImage, const cv::Mat &predictedMask) {
    cv::Mat maskGray;

    if (predictedMask.channels() > 1) {
        cv::cvtColor(predictedMask, maskGray, cv::COLOR_BGR2GRAY);
    } else {
        maskGray = predictedMask.clone();
    }

    double minVal, maxVal;
    cv::minMaxLoc(maskGray, &minVal, &maxVal);
    maskGray.convertTo(maskGray, CV_8U, 255.0 / maxVal);

    cv::imshow("Predicted Mask", maskGray);

    cv::Mat coloredMask;
    cv::cvtColor(maskGray, coloredMask, cv::COLOR_GRAY2BGR);
    cv::Mat overlappedImage;
    cv::addWeighted(originalImage, 0.6, coloredMask, 0.4, 0, overlappedImage);

    cv::imshow("Mask Overlap", overlappedImage);
    cv::waitKey(0);
}

void ONNXInference::logGpuProperties(int deviceId) {
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, deviceId);
    std::cout << "GPU Device: " << deviceId << std::endl;
    std::cout << "  Name: " << deviceProp.name << std::endl;
    std::cout << "  Total Global Memory: " << deviceProp.totalGlobalMem / (1024 * 1024) << " MB" << std::endl;
    std::cout << "  Shared Memory per Block: " << deviceProp.sharedMemPerBlock / 1024 << " KB" << std::endl;
    std::cout << "  Registers per Block: " << deviceProp.regsPerBlock << std::endl;
    std::cout << "  Warp Size: " << deviceProp.warpSize << std::endl;
    std::cout << "  Max Threads per Block: " << deviceProp.maxThreadsPerBlock << std::endl;
    std::cout << "  Max Threads Dimension: [" << deviceProp.maxThreadsDim[0] << ", " << deviceProp.maxThreadsDim[1] << ", " << deviceProp.maxThreadsDim[2] << "]" << std::endl;
    std::cout << "  Max Grid Size: [" << deviceProp.maxGridSize[0] << ", " << deviceProp.maxGridSize[1] << "," << deviceProp.maxGridSize[2] << "]" << std::endl;
    std::cout << "  Clock Rate: " << deviceProp.clockRate / 1000 << " MHz" << std::endl;
    std::cout << "  Total Constant Memory: " << deviceProp.totalConstMem / 1024 << " KB" << std::endl;
    std::cout << "  Multiprocessor Count: " << deviceProp.multiProcessorCount << std::endl;
    std::cout << "  L2 Cache Size: " << deviceProp.l2CacheSize / 1024 << " KB" << std::endl;
}
