#include "Reprojection.h"

ReprojectionError::ReprojectionError(const cv::Mat K)
{
    mfx = K.at<float>(0,0);
    mcx = K.at<float>(0,2);
    mfy = K.at<float>(1,1);
    mcy = K.at<float>(1,2);
}

void ReprojectionError::SetMeasurement(cv::Point2f p)
{
    mu = p.x;
    mv = p.y;
}

void ReprojectionError::SetRt(cv::Mat R, cv::Mat t, cv::Point3f pw)
{
    mR = R.clone();
    mt = t.clone();

    cv::Mat pWorld = (cv::Mat_<float>(3,1)<<pw.x, pw.y, pw.z);
    mpCamera = mR*pWorld + mt;
}

cv::Mat ReprojectionError::Jocobian()
{
    // cv::Mat pWorld = (cv::Mat_<float>(3,1)<<pw.x, pw.y, pw.z);
    // cv::Mat pCamera = mR*pWorld + mt;
    float x = mpCamera.at<float>(0,0);
    float y = mpCamera.at<float>(1,0);
    float z = mpCamera.at<float>(2,0);
    float z2 = z*z;
    cv::Mat J_e_pc = -(cv::Mat_<float>(2,3)<<mfx/z, 0, -mfx*x/z2, 0, mfy/z, -mfy*y/z2);

    cv::Mat J_pc_ksi = (cv::Mat_<float>(3,6)<<1, 0, 0, 0, z, -y,
                                                0, 1, 0, -z, 0, x,
                                                0, 0, 1, y, -x, 0);
    cv::Mat J_e_ksi = J_e_pc*J_pc_ksi;

    cv::Mat J_e_pw = J_e_pc*mR;

    // cv::Mat J(2, 9, CV_32FC1);
    cv::Mat_<float> J(2, 9);
    J_e_ksi.copyTo( J.rowRange(0,2).colRange(0,6) );
    J_e_pw.copyTo( J.rowRange(0,2).colRange(6,9) );

    // std::cout<<"J_e_ksi: "<<std::endl<<J_e_ksi<<std::endl;
    // std::cout<<"J_test: "<<std::endl<<J_test<<std::endl;

    return J;
    // cout<<"J_e_pw: "<<endl<<J_e_pw<<endl;
    // cout<<"J: "<<endl<<J<<endl;
}

cv::Mat ReprojectionError::ComputeError()
{
    float x = mpCamera.at<float>(0,0);
    float y = mpCamera.at<float>(1,0);
    float z = mpCamera.at<float>(2,0);

    cv::Mat error(2,1, CV_32FC1);

    error.at<float>(0,0) = mu - (mfx*x/z + mcx);
    error.at<float>(1,0) = mv - (mfy*y/z + mcy);

    return error;
}