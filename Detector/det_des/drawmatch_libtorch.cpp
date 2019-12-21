#include <cstdlib>
#include <iostream>
#include <vector>
#include <bitset>
#include <fstream>
#include <sstream>

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv/cv.hpp>


#include "Thirdparty/DBoW2/DBoW2/TemplatedVocabulary.h"
#include "Thirdparty/DBoW2/DBoW2/BowVector.h"
#include "Thirdparty/DBoW2/DBoW2/FeatureVector.h"
#include "Thirdparty/DBoW2/DBoW2/FLFNET.h"
#include "Thirdparty/DBoW2/DBoW2/FDLF.h"

#include "DLFhandler.h"
#include "omp.h"

using namespace DBoW2;
using namespace cv;
using namespace std;

vector<DMatch> ransac(vector<DMatch> matches,vector<KeyPoint> queryKeyPoint,vector<KeyPoint> trainKeyPoint)
{
    //定义保存匹配点对坐标
    vector<Point2f> srcPoints(matches.size()),dstPoints(matches.size());
    //保存从关键点中提取到的匹配点对的坐标
    for(int i=0;i<matches.size();i++)
    {
        srcPoints[i]=queryKeyPoint[matches[i].queryIdx].pt;
        dstPoints[i]=trainKeyPoint[matches[i].trainIdx].pt;
    }
    //保存计算的单应性矩阵
    Mat homography;
    //保存点对是否保留的标志
    vector<unsigned char> inliersMask(srcPoints.size());
    //匹配点对进行RANSAC过滤
    homography = findHomography(srcPoints,dstPoints,RANSAC,5,inliersMask);
    //RANSAC过滤后的点对匹配信息
    vector<DMatch> matches_ransac;
    //手动的保留RANSAC过滤后的匹配点对
    for(int i=0;i<inliersMask.size();i++)
    {
        if(inliersMask[i])
        {
            matches_ransac.push_back(matches[i]);
            //cout<<"第"<<i<<"对匹配："<<endl;
            //cout<<"queryIdx:"<<matches[i].queryIdx<<"\ttrainIdx:"<<matches[i].trainIdx<<endl;
            //cout<<"imgIdx:"<<matches[i].imgIdx<<"\tdistance:"<<matches[i].distance<<endl;
        }
    }
    //返回RANSAC过滤后的点对匹配信息
    return matches_ransac;
}

int main() {
    //0. 参数
    string strAssociationFilename = "/home/wang/workspace/data/rgbd_dataset_freiburg1_room/1.txt";
    string datapath = "/home/wang/workspace/data/rgbd_dataset_freiburg1_room/rgb";
    vector<double> vTimestamps;

    const int kpts_num = 1000;//超参数,可以根据情况修改 每张照片提取特征点的个数 1000个,目前是跟ORB本身的个数是一样的.
    float ffeats[kpts_num][128] = {0};//定义一个1000的矩阵，用于存放feats数据

    vector <KeyPoint> mvKpts;
    vector <vector<float>> dspts;
    typedef TemplatedVocabulary <FDLF::TDescriptor, FDLF> FDLFVocabulary;
    //!!Note: 9 and 3 is the parameters you may want to change
    FDLFVocabulary DLFvoc(10, 6, TF_IDF, L2_NORM);
    vector < vector < vector < float >> > features;

    //1. load features
    ifstream fAssociation;
    fAssociation.open(strAssociationFilename.c_str());
    while (!fAssociation.eof()) {
        string s;
        getline(fAssociation, s);
        if (!s.empty()) {
            stringstream ss;
            ss << s;
            double t;
            string sRGB, sD;
            ss >> t;
            vTimestamps.push_back(t);
        }
    }
    int nImages = vTimestamps.size();// 一共多少照片
    cout << "all images number = " << nImages << endl;

    ORB_SLAM2::DLF DLFextract("/home/wang/workspace/RFNet_JIT/NASNet_0.1/des.pt",
                              "/home/wang/workspace/RFNet_JIT/NASNet_0.1/kpt.pt", 32);


    vector <KeyPoint> kp1;
    double tframe = vTimestamps[1];
    string strTimeStamp = to_string(tframe);
    cv::Mat imRGB = cv::imread(datapath + "/" + strTimeStamp + ".png", CV_LOAD_IMAGE_UNCHANGED);
    cv::Mat imRGB1 = cv::imread(datapath + "/" + strTimeStamp + ".png", CV_LOAD_IMAGE_UNCHANGED);
    cv::Mat imGray;
    cvtColor(imRGB, imGray, CV_RGB2GRAY);
    cv::Mat img_float;
    imGray.convertTo(img_float, CV_32FC1, 1.f / 255.f, 0);
    torch::Tensor patches1;
    auto kps1 = DLFextract.get_kps(img_float, patches1);
    auto des_tensor1 = DLFextract.get_des(patches1);
    kp1.resize(kpts_num);
    for (int i = 0; i < kpts_num; i++)//定义行循环
    {
        kp1[i].pt.x = kps1[i][1].item().toFloat();
        kp1[i].pt.y = kps1[i][0].item().toFloat();
    }
    torch::Tensor des_tensor1_cpu = des_tensor1.cpu();
    cv::Mat des1(des_tensor1_cpu.size(0), des_tensor1_cpu.size(1), CV_32F, des_tensor1_cpu.data<float_t>());

    vector <KeyPoint> kp2;
    tframe = vTimestamps[20];
    strTimeStamp = to_string(tframe);
    imRGB = cv::imread(datapath + "/" + strTimeStamp + ".png", CV_LOAD_IMAGE_UNCHANGED);
    cv::Mat imRGB2 = cv::imread(datapath + "/" + strTimeStamp + ".png", CV_LOAD_IMAGE_UNCHANGED);
    cvtColor(imRGB, imGray, CV_RGB2GRAY);
    imGray.convertTo(img_float, CV_32FC1, 1.f / 255.f, 0);
    torch::Tensor patches2;

    auto kps2 = DLFextract.get_kps(img_float, patches2);
    auto des_tensor2 = DLFextract.get_des(patches2);
    kp2.resize(kpts_num);
    for (int i = 0; i < kpts_num; i++)//定义行循环
    {
        kp2[i].pt.x = kps2[i][1].item().toFloat();
        kp2[i].pt.y = kps2[i][0].item().toFloat();
    }
    torch::Tensor des_tensor2_cpu = des_tensor2.cpu();
    cv::Mat des2(des_tensor2_cpu.size(0), des_tensor2_cpu.size(1), CV_32FC1, des_tensor2_cpu.data<float>());


    BFMatcher matcher(NORM_L2);
    vector <vector<DMatch>> knnmatches;
    vector <DMatch> matches;
    const float minRatio = 1.f / 1.5f;
    matcher.knnMatch(des1, des2, knnmatches, 2);
    for (size_t i = 0; i < knnmatches.size(); i++) {
        const DMatch &bestMatch = knnmatches[i][0];
        const DMatch &betterMatch = knnmatches[i][1];
        float distanceRatio = bestMatch.distance / betterMatch.distance;
        if (distanceRatio < minRatio)
            matches.push_back(bestMatch);
    }

    std::vector <DMatch> good_matches = ransac(matches, kp1, kp2);

// 画出匹配图+
    Mat imageMatches;
    drawMatches(imRGB1, kp1, imRGB2, kp2, good_matches, imageMatches, Scalar(255, 0, 0));//进行绘制

    imshow("image2", imageMatches);
    waitKey();
}