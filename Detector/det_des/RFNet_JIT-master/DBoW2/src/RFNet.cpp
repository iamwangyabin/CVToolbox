#include <vector>
#include <string>
#include <sstream>

#include "FClass.h"
#include "RFNet.h"

using namespace std;

namespace DBoW2 {

// --------------------------------------------------------------------------

void RFNet_128::meanValue(const std::vector<RFNet_128::pDescriptor> &descriptors, 
  RFNet_128::TDescriptor &mean)
{
  mean.resize(0);
  mean.resize(RFNet_128::L, 0);
  
  float s = descriptors.size();
  
  vector<RFNet_128::pDescriptor>::const_iterator it;
  for(it = descriptors.begin(); it != descriptors.end(); ++it)
  {
    const RFNet_128::TDescriptor &desc = **it;
    for(int i = 0; i < RFNet_128::L; i += 4)
    {
      mean[i  ] += desc[i  ] / s;
      mean[i+1] += desc[i+1] / s;
      mean[i+2] += desc[i+2] / s;
      mean[i+3] += desc[i+3] / s;
    }
  }
}

// --------------------------------------------------------------------------
  
double RFNet_128::distance(const RFNet_128::TDescriptor &a, const RFNet_128::TDescriptor &b)
{
  double sqd = 0.;
  for(int i = 0; i < RFNet_128::L; i += 4)
  {
    sqd += (a[i  ] - b[i  ])*(a[i  ] - b[i  ]);
    sqd += (a[i+1] - b[i+1])*(a[i+1] - b[i+1]);
    sqd += (a[i+2] - b[i+2])*(a[i+2] - b[i+2]);
    sqd += (a[i+3] - b[i+3])*(a[i+3] - b[i+3]);
  }
  return sqd;
}

// --------------------------------------------------------------------------

std::string RFNet_128::toString(const RFNet_128::TDescriptor &a)
{
  stringstream ss;
  for(int i = 0; i < RFNet_128::L; ++i)
  {
    ss << a[i] << " ";
  }
  return ss.str();
}

// --------------------------------------------------------------------------
  
void RFNet_128::fromString(RFNet_128::TDescriptor &a, const std::string &s)
{
  a.resize(RFNet_128::L);
  
  stringstream ss(s);
  for(int i = 0; i < RFNet_128::L; ++i)
  {
    ss >> a[i];
  }
}

// --------------------------------------------------------------------------

void RFNet_128::toMat32F(const std::vector<TDescriptor> &descriptors, 
    cv::Mat &mat)
{
  if(descriptors.empty())
  {
    mat.release();
    return;
  }
  
  const int N = descriptors.size();
  const int L = RFNet_128::L;
  
  mat.create(N, L, CV_32F);
  
  for(int i = 0; i < N; ++i)
  {
    const TDescriptor& desc = descriptors[i];
    float *p = mat.ptr<float>(i);
    for(int j = 0; j < L; ++j, ++p)
    {
      *p = desc[j];
    }
  } 
}

// --------------------------------------------------------------------------

} // namespace DBoW2

