//
// Created by wang on 19-1-28.
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

/**
 * Calculate left right high low pixel between X
 * */
double getAverage(cv::Mat &m, int y, int x){
    if(x < 0 || x >= m.cols) return 0;
    if(y < 0 || y >= m.rows) return 0;
    double val = 0.0;
    int temp = 0;
    if(isInsideImage(y-1,x,m)){
        ++temp;
        val+=m.at<double>(y-1,x);
    }
    if(isInsideImage(y+1,x,m)){
        ++temp;
        val+=m.at<double>(y+1,x);
    }
    if(isInsideImage(y,x-1,m)){
        ++temp;
        val+=m.at<double>(y,x-1);
    }
    if(isInsideImage(y,x+1,m)){
        ++temp;
        val+=m.at<double>(y,x+1);
    }
    return val/temp;
}

cv::Mat get_Average4_Mat(cv::Mat &m){
    cv::Mat res = cv::Mat::zeros(m.rows, m.cols, CV_64FC1);
    for(int i = 0; i < m.rows; i++){
        for(int j = 0; j < m.cols; j++){
            res.at<double>(i, j) = getAverage(m, i, j);
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

void getHornSchunckOpticalFlow(cv::Mat img1, cv::Mat img2){
    double lambda = 0.05;
    cv::Mat u = cv::Mat::zeros(img1.rows,img1.cols,CV_64FC1);
    cv::Mat v = cv::Mat::zeros(img1.rows,img1.cols,CV_64FC1);

    cv::Mat fx = get_fx(img1, img2);
    cv::Mat fy = get_fy(img1, img2);
    cv::Mat ft = get_ft(img1, img2);

    int i=0;
    double last = 0.0;
    while(1){
        cv::Mat Uav = get_Average4_Mat(u);
        cv::Mat Vav = get_Average4_Mat(v);
        cv::Mat P = fx.mul(Uav) + fy.mul(Vav) + ft;
        cv::Mat D = fx.mul(fx) + fy.mul(fy) + lambda;

        cv::Mat temp;
        cv::divide(P,D,temp);

        cv::Mat utemp,vtemp;
        utemp=Uav-fx.mul(temp);
        vtemp=Vav-fy.mul(temp);

        cv::Mat eq = fx.mul(utemp)+fy.mul(vtemp)+ft;
        double thistime = cv::mean(eq)[0];
        std::cout<<"i = "<<i<<", mean = "<<thistime<<std::endl;
        if(i!=0&&fabs(last)<=fabs(thistime))
            break;
        i++;
        last = thistime;
        u = utemp;
        v = vtemp;
    }
    saveMat(u, "U");
    saveMat(v, "V");
//    cv::imshow("v2", v);100

//    cv::waitKey(20000);
}







int main(){
    cv::Mat img1 = cv::imread("1.png", 0);
    cv::Mat img2 = cv::imread("2.png", 0);

    img1.convertTo(img1, CV_64FC1, 1.0/255, 0);
    img2.convertTo(img2, CV_64FC1, 1.0/255, 0);

    getHornSchunckOpticalFlow(img1, img2);
    return 0;
}