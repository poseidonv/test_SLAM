#include "Converter.h"

Eigen::Matrix<float, 3, 3> Converter::ToMatrix3f(const cv::Mat& cvMat)
{
    Eigen::Matrix<float, 3 ,3> eigMat;
    eigMat<<cvMat.at<float>(0,0), cvMat.at<float>(0,1), cvMat.at<float>(0,2),
        cvMat.at<float>(1,0), cvMat.at<float>(1,1), cvMat.at<float>(1,2),
        cvMat.at<float>(2,0), cvMat.at<float>(2,1), cvMat.at<float>(2,2);
    return eigMat;
}

std::vector<float> Converter::ToQuaternion(const cv::Mat& cvMat)
{
    Eigen::Matrix<float, 3, 3> eigMat = ToMatrix3f(cvMat);
    Eigen::Quaternionf q(eigMat);

    std::vector<float> vq(4, 0);
    vq[0] = q.x();
    vq[1] = q.y();
    vq[2] = q.z();
    vq[3] = q.w();

    return vq;
}