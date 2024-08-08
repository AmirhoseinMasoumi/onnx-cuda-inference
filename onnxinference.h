#ifndef ONNXINFERENCE_H
#define ONNXINFERENCE_H

#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

class ONNXInference {
public:
    ONNXInference(const std::string &modelPath);
    cv::Mat runInference(const cv::Mat &inputImage, bool useHighRes);
    void visualize(const cv::Mat &originalImage, const cv::Mat &predictedMask);

private:
    Ort::Env env;
    Ort::Session session;
    Ort::SessionOptions sessionOptions;
    Ort::AllocatorWithDefaultOptions allocator;
    std::optional<Ort::AllocatedStringPtr> outputName;

    std::wstring convertUtf8ToUtf16(const std::string &utf8Str);
    void logGpuProperties(int deviceId);
};

#endif // ONNXINFERENCE_H
