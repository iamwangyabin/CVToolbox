//
// Created by wang on 19-7-26.
//

#ifndef DBOW2_RFNETHANDLE_H
#define DBOW2_RFNETHANDLE_H

#include <iostream>
#include <vector>
#include <torch/script.h> // One-stop header.
#include <torch/torch.h>
#include <ATen/Tensor.h>

#include <opencv/cv.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv/highgui.h>
#include <opencv2/highgui/highgui.hpp>

class RFNet {
public:
    RFNet(std::string des_path,
          std::string det_path,
          int ksize,
          int topk,
          int PSIZE,
          int radius,
          float com_strength,
          float com_strength1,
          float com_strength2,
          float thresh,
          int output_h,
          int output_w);

    torch::Tensor soft_nms_3d(torch::Tensor scale_logits);

    torch::Tensor filtbordmask(torch::Tensor &imscore);

    torch::Tensor nms(torch::Tensor input);

    torch::Tensor topk_mask(torch::Tensor maps);

    void soft_max_and_argmax_1d(torch::Tensor input, torch::Tensor &orint_maps,
                                torch::Tensor &score_map, torch::Tensor &scale_map);

    torch::Tensor
    clip_patch(torch::Tensor &kpts_byxc, torch::Tensor kpts_scale, torch::Tensor &kpts_ori,torch::Tensor &image);

    torch::Tensor get_rfdes(torch::Tensor &patches);

    torch::Tensor get_rfkps(std::string img_path,torch::Tensor &patches);


private:
    std::string des_module_path;
    std::string det_module_path;
    std::shared_ptr<torch::jit::script::Module> det_module;
    std::shared_ptr<torch::jit::script::Module> des_module;

    int ksize; //15
    int topk;
    int PSIZE;
    int radius; //8

    int img_resize_height;
    int img_resize_width;
    int img_orig_height;
    int img_orig_width;

    float com_strength;  //3.0
    float com_strength1; //100
    float com_strength2; //100
    float thresh;
    int output_h;
    int output_w;
    int new_h;
    int new_w;
    float sh;
    float sw;

    torch::Tensor scale_list;
    int dim;
    bool keepdim;

    torch::Tensor im_info;
    torch::DeviceType device_type = torch::kCUDA;
    torch::Device device = torch::Device(device_type);
    std::vector<float> scale_vector{3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0, 17.0, 19.0, 21.0};

    void im_rescale(cv::Mat im, cv::Mat &out_img);
};


#endif //DBOW2_RFNETHANDLE_H
