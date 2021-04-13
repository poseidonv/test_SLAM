#include <iostream>
#include <unordered_map>
#include <ctime>
#include <iomanip>
#include <typeinfo>
#include <cstdlib>
#include <sstream>

#include "SLAM_base.h"
#include "Reprojection.h"

using namespace std;

#define N_PICTURES 800

int main(int argc, char **argv)
{
    vector<string> filenames;
    string filedir = "/home/poseidon/Documents/00/image_0/";
    for (int i = 0; i < N_PICTURES; i++)
    {
        stringstream ss;
        ss << std::setw(6) << std::setfill('0') << i;
        filenames.push_back(filedir + ss.str() + ".png");
    }
    vector<pair<cv::Mat, cv::Mat>> poses;
    poses.push_back(make_pair(cv::Mat::eye(3, 3, CV_32FC1), cv::Mat::zeros(3, 1, CV_32FC1)));

    cv::Mat totalR, totalt;
    totalR = cv::Mat::eye(3, 3, CV_32FC1);
    totalt = cv::Mat::zeros(3, 1, CV_32FC1);
    for (int i = 1; i < N_PICTURES; i++)
    {
        cv::Mat Rcw, tcw;
        cv::Mat Rwc, twc;

        cv::Mat resultR, resultt;
        // /home/poseidon/Documents/00/image_0/000201.png /home/poseidon/Documents/00/image_0/000202.png
        EstimatePose(filenames[i - 1], filenames[i], Rcw, tcw);
        // cout<<filenames[i - 1]<<" "<<filenames[i]<<endl;
        Rwc = Rcw.t();
        twc = -Rwc * tcw;
        totalt = totalt + totalR*twc;
        totalR = totalR*Rwc;
        // totalR = totalR*Rcw;

        // Rwc = Rcw.t();

        resultR = totalR.clone();
        resultt = totalt.clone();

        poses.push_back(make_pair(resultR, resultt));
        // cout << "R: \n"<< Rwc << endl<< "t:\n"<< twc << endl;
    }

    string filename = "TEST_KITTI.txt";
    SaveKeyFrameTrajectoryTUM(filename, poses);

    return 0;
}