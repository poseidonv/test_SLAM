#include <vector>
#include <map>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv/cvaux.h>
#include <opencv/cxcore.hpp>

#include "Converter.h"

using namespace std;

#define HALF_PATCH_SIZE 15
#define MAX_ITERATIONS 10

void EstimatePose(string filename1, string filename2, cv::Mat& R, cv::Mat& t);

void GenerateMatches(cv::Mat& img1, vector<cv::KeyPoint>& keypoints1, cv::Mat& img2, vector<cv::KeyPoint>& keypoints2, vector<cv::DMatch>& good_matches);

vector<vector<size_t>> GenerateEightPoints(int N);

void GetTruthPose(cv::Mat& R, cv::Mat& t);

cv::Mat FindFundamental(vector<cv::KeyPoint>& points1, vector<cv::KeyPoint>& points2, vector<cv::DMatch>& good_matches,vector<vector<size_t>>& sets);

void DecomposeE(cv::Mat& E, cv::Mat& R, cv::Mat& t, cv::Mat& K, vector<cv::KeyPoint>& keypoints_1, vector<cv::KeyPoint>& keypoints_2, vector<cv::DMatch>& good_matches, vector<cv::Point3f>& vP3D1);

int CheckRT(cv::Mat& R, cv::Mat& t, cv::Mat& K, vector<cv::KeyPoint>& keypoints_1, vector<cv::KeyPoint>& keypoints_2, vector<cv::DMatch>& good_matches, vector<cv::Point3f>& vP3D1);

void Normalize(vector<cv::KeyPoint>& vKeys, vector<cv::Point2f>& normalizedKeys, cv::Mat& T);

cv::Mat ComputeF21(vector<cv::Point2f>& vPn1i, vector<cv::Point2f>& vPn2i);

float CheckFundamental(vector<cv::KeyPoint>& points1, vector<cv::KeyPoint>& points2, vector<cv::DMatch>& good_matches, cv::Mat& F21, vector<bool>& vbCurrentInliers, float sigma);

void Triangulate(cv::Point2f& kp1, cv::Point2f& kp2, cv::Mat& P1, cv::Mat& P2, cv::Mat& p3dC1);

void GetPredict(vector<cv::KeyPoint>& keypoints_1, vector<cv::KeyPoint>& keypoints_2, vector<cv::KeyPoint>& keypoints_2_hat,vector<cv::DMatch>& good_matches, cv::Mat& R, cv::Mat& t, vector<cv::Point3f>& vP3D1);

void BundleAdjustment(const vector<cv::KeyPoint>& keypoints_2, const cv::Mat& K, vector<cv::Point3f>& vP3D1, cv::Mat& R, cv::Mat& t, int iters,vector<cv::DMatch>& good_matches);

Eigen::VectorXf ComputeUpdate(cv::Mat& H, cv::Mat& b, int nCamera, int nPoints);

void SaveKeyFrameTrajectoryTUM(const string& filename, vector<pair<cv::Mat, cv::Mat>>& poses);

// template<typename _Matrix_Type_> 
// _Matrix_Type_ pseudoInverse(const _Matrix_Type_ &a, double epsilon = 
//     std::numeric_limits<double>::epsilon());
//     // {
//     //     Eigen::JacobiSVD< _Matrix_Type_ > svd(a, Eigen::ComputeFullU | Eigen::ComputeFullV);  
//     //     double tolerance = epsilon * std::max(a.cols(), a.rows()) *svd.singularValues().array().abs()(0);  
//     //     return svd.matrixV() *  (svd.singularValues().array().abs() > tolerance).select(svd.singularValues().array().inverse(), 0).matrix().asDiagonal() * svd.matrixU().adjoint(); 
//     // }