#ifndef CONVERTER_H
#define CONVERTER_H

#include <vector>
#include <opencv2/core/core.hpp>
#include <Eigen/Dense>
// #include"Thirdparty/g2o/g2o/types/types_six_dof_expmap.h"
// #include"Thirdparty/g2o/g2o/types/types_seven_dof_expmap.h"

class Converter
{
public:
    static Eigen::Matrix<float, 3, 3> ToMatrix3f(const cv::Mat& cvMat);

    static std::vector<float> ToQuaternion(const cv::Mat& M);
};

#endif