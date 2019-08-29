#include <iostream>
#include <string>
#include <vector>
#include <chrono>
#include <opencv2/opencv.hpp>
#include "myORBextractor.h"
 
using namespace std;
 
//设置特征点提取需要的一些参数
int nFeatures = 1000;//图像金字塔上特征点的数量
int nLevels = 8;//图像金字塔层数
float fScaleFactor = 1.2;//金字塔比例因子
int fIniThFAST = 20;//检测fast角点阈值
int fMinThFAST = 8; //最低阈值
 
int main(int argc, char** argv) {
    cv::Mat image = cv::imread(argv[1],0);
    vector<cv::KeyPoint> Keypoints;
    cv::Mat mDescriptors;
 
    myORB::ORBextractor ORBextractor =  myORB::ORBextractor(nFeatures,fScaleFactor,nLevels,fIniThFAST,fMinThFAST);
 
    cout << endl  << "ORB Extractor Parameters: " << endl;
    cout << "- Number of Features: " << nFeatures << endl;
    cout << "- Scale Levels: " << nLevels << endl;
    cout << "- Scale Factor: " << fScaleFactor << endl;
    cout << "- Initial Fast Threshold: " << fIniThFAST << endl;
    cout << "- Minimum Fast Threshold: " << fMinThFAST << endl;
 
    ORBextractor(image, cv::Mat(), Keypoints, mDescriptors);
 
    cv::Mat outimg;
    cv::drawKeypoints(image, Keypoints, outimg);
    cv::imshow("ORB features", outimg);
    cv::waitKey(0);
    return 0;
 
}

