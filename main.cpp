#include "ONNXInference.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <filesystem>
#include <vector>

namespace fs = std::filesystem;

std::string modelsPath = "D:/Temp/trainModel/models/v5";
std::string modelPathLowRes = modelsPath + "/segmentation_model_320x288.onnx";
std::string modelPathHighRes = modelsPath + "/segmentation_model_640x576.onnx";
std::string testImageFolder = "D:/Temp/trainModel/test_images/";

bool useHighRes = false;  // Flag to control resolution
bool useGPU = false;      // Flag to control CPU/GPU

std::vector<cv::Mat> processImage(const std::string& imagePath, ONNXInference& onnxInference, bool useHighRes) {
    cv::Mat inputImage = cv::imread(imagePath);
    if (inputImage.empty()) {
        std::cerr << "Input image not found: " << imagePath << std::endl;
        return {};
    }

    cv::Mat resizedImage;
    cv::Size targetSize = useHighRes ? cv::Size(640, 576) : cv::Size(320, 288);
    cv::resize(inputImage, resizedImage, targetSize);

    if (resizedImage.channels() != 3) {
        std::cerr << "Input image must have 3 channels (RGB): " << imagePath << std::endl;
        return {};
    }

    cv::Mat predictedMask = onnxInference.runInference(resizedImage, useHighRes);

    if (predictedMask.empty()) {
        std::cerr << "Predicted mask is empty for: " << imagePath << std::endl;
        return {};
    }

    cv::Mat predictedMaskGray;
    if (predictedMask.channels() > 1)
        cv::cvtColor(predictedMask, predictedMaskGray, cv::COLOR_BGR2GRAY);
    else
        predictedMaskGray = predictedMask.clone();

    double minVal, maxVal;
    cv::minMaxLoc(predictedMaskGray, &minVal, &maxVal);
    if (maxVal > 0) {
        predictedMaskGray.convertTo(predictedMaskGray, CV_8U, 255.0 / maxVal);
    }

    cv::Mat coloredMask;
    cv::cvtColor(predictedMaskGray, coloredMask, cv::COLOR_GRAY2BGR);

    cv::Mat overlappedImage;
    cv::addWeighted(resizedImage, 0.6, coloredMask, 0.4, 0, overlappedImage);

    return {predictedMaskGray, overlappedImage};
}

int main(int argc, char** argv) {
    // Check command line arguments for resolution and device flags
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--high-res") {
            useHighRes = true;
        } else if (std::string(argv[i]) == "--use-gpu") {
            useGPU = true;
        }
    }

    try {
        ONNXInference onnxInference(useHighRes ? modelPathHighRes : modelPathLowRes, useGPU);

        for (const auto& entry : fs::directory_iterator(testImageFolder)) {
            if (entry.path().extension() == ".png" || entry.path().extension() == ".jpg") {
                std::cout << "Processing: " << entry.path().string() << std::endl;

                auto start = std::chrono::high_resolution_clock::now();
                auto results = processImage(entry.path().string(), onnxInference, useHighRes);
                auto end = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double> processingTime = end - start;

                if (!results.empty()) {
                    std::cout << "Time taken to process image: " << processingTime.count() << " seconds" << std::endl;

                    cv::imshow("Predicted Mask", results[0]);
                    cv::imshow("Mask Overlap", results[1]);

                    int key = cv::waitKey(0);
                    if (key == 27) // ESC key
                        break;
                }
            }
        }
    }
    catch (const std::exception& e) {
        std::cerr << "Standard exception: " << e.what() << std::endl;
        return -1;
    }
    catch (const Ort::Exception& e) {
        std::cerr << "ONNX Runtime exception: " << e.what() << std::endl;
        return -1;
    }
    catch (...) {
        std::cerr << "Unknown exception occurred" << std::endl;
        return -1;
    }

    return 0;
}
