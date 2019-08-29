#ifndef ORBEXTRACTOR_H
#define ORBEXTRACTOR_H
 
#include <vector>
#include <list>
#include <opencv/cv.h>
 
 
namespace myORB
{
 
class ExtractorNode
{
public:
    ExtractorNode():bNoMore(false){}
 
    void DivideNode(ExtractorNode &n1, ExtractorNode &n2, ExtractorNode &n3, ExtractorNode &n4);
 
    std::vector<cv::KeyPoint> vKeys;
    cv::Point2i UL, UR, BL, BR;
    std::list<ExtractorNode>::iterator lit;
    bool bNoMore;
};
 
class ORBextractor
{
public:
    
    enum {HARRIS_SCORE=0, FAST_SCORE=1 };
 
    ORBextractor(int nfeatures, float scaleFactor, int nlevels,
                 int iniThFAST, int minThFAST);
 
    ~ORBextractor(){}
 
    // Compute the ORB features and descriptors on an image.
    // ORB are dispersed on the image using an octree.
    // Mask is ignored in the current implementation.
    // 重载（）运算符，作为对外接口
    void operator()( cv::InputArray image, cv::InputArray mask,
      std::vector<cv::KeyPoint>& keypoints,
      cv::OutputArray descriptors);
 
    int inline GetLevels(){
        return nlevels;}
 
    float inline GetScaleFactor(){
        return scaleFactor;}
 
    std::vector<float> inline GetScaleFactors(){
        return mvScaleFactor;
    }
 
    std::vector<float> inline GetInverseScaleFactors(){
        return mvInvScaleFactor;
    }
 
    std::vector<float> inline GetScaleSigmaSquares(){
        return mvLevelSigma2;
    }
 
    std::vector<float> inline GetInverseScaleSigmaSquares(){
        return mvInvLevelSigma2;
    }
 
    //存放各层级图片
    std::vector<cv::Mat> mvImagePyramid;
 
protected:
    //计算高斯金字塔
    void ComputePyramid(cv::Mat image);
 
    ///对图像金字塔中的每一层图像进行特征点的计算。具体的计算过程是将图像网格分割为小区域，
    /// 每一个小区域独立使用 FAST 角点检测。检测完成之后使用 DistributeOctTree 函数对检测得到的所有角点进行筛选，使得角点分布均匀。
    void ComputeKeyPointsOctTree(std::vector<std::vector<cv::KeyPoint> >& allKeypoints);
    //将关键点分配到四叉树
    std::vector<cv::KeyPoint> DistributeOctTree(const std::vector<cv::KeyPoint>& vToDistributeKeys, const int &minX,
                                           const int &maxX, const int &minY, const int &maxY, const int &nFeatures, const int &level);
 
    void ComputeKeyPointsOld(std::vector<std::vector<cv::KeyPoint> >& allKeypoints);
    //存储关键点附近patch的点对
    std::vector<cv::Point> pattern;
 
    int nfeatures; //特征点的个数
    double scaleFactor;  //相邻层图像的比例系数
    int nlevels;    //构造金字塔的层数
    int iniThFAST;  //检测fast角点阈值
    int minThFAST;  //没有检测到角点的前提下降低阈值
 
    //每层特征点的数量
    std::vector<int> mnFeaturesPerLevel;
    //patch圆的最大坐标
    std::vector<int> umax;
    //每层相对于原始图像的缩放比例
    std::vector<float> mvScaleFactor;
    //每层相对于原始图像缩放比例的倒数
    std::vector<float> mvInvScaleFactor;
    //每层相对于原始图像的缩放比例的平方
    std::vector<float> mvLevelSigma2;
    std::vector<float> mvInvLevelSigma2;
};
 
} //namespace ORB_SLAM
 
#endif

