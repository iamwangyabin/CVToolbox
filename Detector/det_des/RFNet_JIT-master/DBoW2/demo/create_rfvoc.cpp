#include <iostream>
#include <vector>
#include <torch/script.h> // One-stop header.
#include <torch/torch.h>
#include <ATen/Tensor.h>

#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <opencv2/features2d.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>

// DBoW2
#include "DBoW2.h" // defines OrbVocabulary and OrbDatabase
#include "RFNetHandle.h"

using namespace DBoW2;
using namespace std;

const int NIMAGES = 4;

void loadFeatures(vector<vector<cv::Mat > > &features);
void changeStructure(const cv::Mat &plain, vector<cv::Mat> &out);

void loadFeatures(vector<vector<cv::Mat > > &features)
{
    features.clear();
    features.reserve(NIMAGES);
    string des_module_path = "/home/wang/workspace/RFNET/rfnet/des.pt";
    string det_module_path = "/home/wang/workspace/RFNET/rfnet/det.pt";
    RFNet rfnet(des_module_path,det_module_path,15,512,32,8,3,100,100,0,320,240);
    torch::Tensor patches;

    cout << "Extracting rfnet features..." << endl;
    for(int i = 0; i < NIMAGES; ++i)
    {
        stringstream ss;
        ss << "images/image" << i << ".png";
        torch::Tensor kpts = rfnet.get_rfkps(ss.str(),patches);
        torch::Tensor des = rfnet.get_rfdes(patches);
//        cv::Mat descriptors ;
        cv::Mat descriptors = cv::Mat::zeros(512,128,CV_32F);

//        cout<<des<<endl;
        std::memcpy(des.data_ptr(),descriptors.data, sizeof(float)*des.numel());
        features.push_back(vector<cv::Mat >());
        changeStructure(descriptors, features.back());
    }
}


void changeStructure(const cv::Mat &plain, vector<cv::Mat> &out)
{
    out.resize(plain.rows);

    for(int i = 0; i < plain.rows; ++i)
    {
        out[i] = plain.row(i);
    }
}

int main() {
    vector<vector<cv::Mat > > features;

    loadFeatures(features);
    cout<<"ok"<<endl;
}

