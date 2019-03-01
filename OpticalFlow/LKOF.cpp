//
// Created by wang on 19-2-6.
//


#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <math.h>
#include <fstream>
#include <iostream>

bool isInsideImage(int y, int x, cv::Mat &m){
    int width = m.cols;
    int height = m.rows;
    if(x >= 0 && x < width && y >= 0 && y < height) return true;
    else return false;
}
double get_Sum9(cv::Mat &m, int y, int x){
    if(x < 0 || x >= m.cols) return 0;
    if(y < 0 || y >= m.rows) return 0;

    double val = 0.0;
    int tmp = 0;
    if(isInsideImage(y - 1, x - 1, m)){
        ++ tmp;
        val += m.at<double>(y - 1, x - 1);
    }
    if(isInsideImage(y - 1, x, m)){
        ++ tmp;
        val += m.at<double>(y - 1, x);
    }
    if(isInsideImage(y - 1, x + 1, m)){
        ++ tmp;
        val += m.at<double>(y - 1, x + 1);
    }
    if(isInsideImage(y, x - 1, m)){
        ++ tmp;
        val += m.at<double>(y, x - 1);
    }
    if(isInsideImage(y, x, m)){
        ++ tmp;
        val += m.at<double>(y, x);
    }
    if(isInsideImage(y, x + 1, m)){
        ++ tmp;
        val += m.at<double>(y, x + 1);
    }
    if(isInsideImage(y + 1, x - 1, m)){
        ++ tmp;
        val += m.at<double>(y + 1, x - 1);
    }
    if(isInsideImage(y + 1, x, m)){
        ++ tmp;
        val += m.at<double>(y + 1, x);
    }
    if(isInsideImage(y + 1, x + 1, m)){
        ++ tmp;
        val += m.at<double>(y + 1, x + 1);
    }
    if(tmp == 9) return val;
    else return m.at<double>(y, x) * 9;
}

cv::Mat get_Sum9_Mat(cv::Mat &m){
    cv::Mat res = cv::Mat::zeros(m.rows, m.cols, CV_64FC1);
    for(int i = 1; i < m.rows - 1; i++){
        for(int j = 1; j < m.cols - 1; j++){
            res.at<double>(i, j) = get_Sum9(m, i, j);
        }
    }
    return res;
}

cv::Mat get_fx(cv::Mat& src1, cv::Mat& src2){
    cv::Mat fx;
    cv::Mat kernel = cv::Mat::ones(2,2,CV_64FC1);
    kernel.at<double>(0,0) = -1.0;
    kernel.at<double>(1,0) = -1.0;

    cv::Mat dst1,dst2;
    cv::filter2D(src1,dst1,-1,kernel);
    cv::filter2D(src2,dst2,-1,kernel);

    fx = dst1+dst2;
    return fx;
}

cv::Mat get_fy(cv::Mat &src1, cv::Mat &src2){
    cv::Mat fy;
    cv::Mat kernel = cv::Mat::ones(2, 2, CV_64FC1);
    kernel.at<double>(0, 0) = -1.0;
    kernel.at<double>(0, 1) = -1.0;

    cv::Mat dst1, dst2;
    filter2D(src1, dst1, -1, kernel);
    filter2D(src2, dst2, -1, kernel);

    fy = dst1 + dst2;
    return fy;
}

cv::Mat get_ft(cv::Mat &src1, cv::Mat &src2){
    cv::Mat ft;
    cv::Mat kernel = cv::Mat::ones(2, 2, CV_64FC1);
    kernel = kernel.mul(-1);

    cv::Mat dst1, dst2;
    filter2D(src1, dst1, -1, kernel);
    kernel = kernel.mul(-1);
    filter2D(src2, dst2, -1, kernel);

    ft = dst1 + dst2;
    return ft;
}

void saveMat(cv::Mat &M, std::string s){
    s += ".txt";
    FILE *pOut = fopen(s.c_str(), "w+");
    for(int i=0; i<M.rows; i++){
        for(int j=0; j<M.cols; j++){
            fprintf(pOut, "%lf", M.at<double>(i, j));
            if(j == M.cols - 1) fprintf(pOut, "\n");
            else fprintf(pOut, " ");
        }
    }
    fclose(pOut);
}


void getLucasKanadeOpticalFlow(cv::Mat &img1, cv::Mat &img2, cv::Mat &u, cv::Mat &v){

    cv::Mat fx = get_fx(img1, img2);
    cv::Mat fy = get_fy(img1, img2);
    cv::Mat ft = get_ft(img1, img2);

    cv::Mat fx2 = fx.mul(fx);
    cv::Mat fy2 = fy.mul(fy);
    cv::Mat fxfy = fx.mul(fy);
    cv::Mat fxft = fx.mul(ft);
    cv::Mat fyft = fy.mul(ft);

    cv::Mat sumfx2 = get_Sum9_Mat(fx2);
    cv::Mat sumfy2 = get_Sum9_Mat(fy2);
    cv::Mat sumfxft = get_Sum9_Mat(fxft);
    cv::Mat sumfxfy = get_Sum9_Mat(fxfy);
    cv::Mat sumfyft = get_Sum9_Mat(fyft);

    cv::Mat tmp = sumfx2.mul(sumfy2) - sumfxfy.mul(sumfxfy);
    u = sumfxfy.mul(sumfyft) - sumfy2.mul(sumfxft);
    v = sumfxft.mul(sumfxfy) - sumfx2.mul(sumfyft);
    divide(u, tmp, u);
    divide(v, tmp, v);

}




int main(){
    cv::Mat img1 = cv::imread("1.png", 0);
    cv::Mat img2 = cv::imread("2.png", 0);

    img1.convertTo(img1, CV_64FC1, 1.0/255, 0);
    img2.convertTo(img2, CV_64FC1, 1.0/255, 0);

    cv::Mat u = cv::Mat::zeros(img1.rows, img1.cols, CV_64FC1);
    cv::Mat v = cv::Mat::zeros(img1.rows, img1.cols, CV_64FC1);

    getLucasKanadeOpticalFlow(img1, img2, u, v);
    saveMat(u, "U");
    saveMat(v, "V");
    return 0;
}