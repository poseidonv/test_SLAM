#include <opencv2/opencv.hpp>
#include <vector>

class ReprojectionError
{
public:
    ReprojectionError() {};

    ReprojectionError(const cv::Mat K);

    void SetMeasurement(cv::Point2f p);

    void SetRt(cv::Mat R, cv::Mat t, cv::Point3f pw);

    cv::Mat Jocobian();

    cv::Mat ComputeError();
    
    ~ReprojectionError() {};

private:
    float mfx, mfy, mcx, mcy;
    float mu, mv;

    cv::Mat mR, mt;
    cv::Mat mpCamera;
};