#include <utility>

//
// Created by wang on 19-7-26.
//

#include "RFNetHandle.h"


void RFNet::im_rescale(cv::Mat im, cv::Mat &out_img) {
    cv::resize(im, out_img, cv::Size(new_w, new_h));
}

torch::Tensor RFNet::soft_nms_3d(torch::Tensor scale_logits) {
    int num_scales = scale_logits.sizes()[3];
    torch::Tensor max_each_scale = torch::max_pool2d(scale_logits.permute({0, 3, 1, 2}), {ksize, ksize}, {1},
                                                     {int(ksize / 2)}).permute({0, 2, 3, 1});
    torch::Tensor max_all_scale = std::get<0>(torch::max(max_each_scale, -1, true));
    torch::Tensor exp_maps = torch::exp(com_strength * (scale_logits - max_all_scale));
    torch::Tensor input = exp_maps.permute({0, 3, 1, 2});
    torch::Tensor weight = torch::full({1, num_scales, ksize, ksize}, 1).to(exp_maps.device());
    torch::Tensor bias = torch::ones({1}).to(exp_maps.device());
    auto sum_exp = std::get<0>(
            torch::thnn_conv2d_forward(input, weight, {ksize, ksize}, bias, {1}, {int(ksize / 2)})).permute(
            {0, 2, 3, 1});
    torch::Tensor probs = exp_maps / (sum_exp + 1e-8);
    return probs;
}

torch::Tensor RFNet::filtbordmask(torch::Tensor &imscore) {
    int height = imscore.sizes()[1];
    int width = imscore.sizes()[2];
    torch::Tensor mask = torch::ones({1, height - 2 * radius, width - 2 * radius, 1});
    return torch::constant_pad_nd(mask, {0, 0, radius, radius, radius, radius, 0, 0}, 0).to(imscore.device());
}

torch::Tensor RFNet::nms(torch::Tensor input) {
    int height = input.sizes()[1];
    int width = input.sizes()[2];
    torch::Tensor zeros = torch::zeros_like(input);
    int pad = int(ksize / 2);
    input = torch::where(input < thresh, zeros, input);
    torch::Tensor input_pad = torch::constant_pad_nd(input, {0, 0, 2 * pad, 2 * pad, 2 * pad, 2 * pad, 0, 0}, 0);
    torch::Tensor slice_map;
    for (int i = 0; i < ksize; ++i) {
        for (int j = 0; j < ksize; ++j) {
            torch::Tensor slice = input_pad.slice(1, i, height + 2 * pad + i).slice(2, j, width + 2 * pad + j);
            if (i == 0 && j == 0) {
                slice_map = slice;
            } else {
                slice_map = torch::cat({slice_map, slice}, -1);
            }
        }
    }
    torch::Tensor max_slice = std::get<0>(torch::max(slice_map, -1, true));
    torch::Tensor center_map = slice_map.slice(3, int(slice_map.sizes()[3] / 2), int(slice_map.sizes()[3] / 2) + 1);
    torch::Tensor mask = torch::ge(center_map, max_slice);
    mask = mask.slice(1, pad, height + pad).slice(2, pad, width + pad);
    return mask.type_as(input);
}

torch::Tensor RFNet::topk_mask(torch::Tensor maps) {
    int batch = maps.sizes()[0];
    int height = maps.sizes()[1];
    int width = maps.sizes()[2];
    torch::Tensor maps_flat = maps.view({batch, -1});
    torch::Tensor indices = std::get<1>(maps_flat.sort(-1, true)).slice(1, 0, topk);
    torch::Tensor batch_idx = torch::arange(0, batch).unsqueeze(-1).repeat({1, topk}).to(indices.device()).to(
            indices.dtype());
    batch_idx = batch_idx.view(-1);
    torch::Tensor row_index = indices.contiguous().view(-1);
    torch::Tensor topk_mask_flat = torch::zeros(maps_flat.sizes()).to(torch::kUInt8).to(maps.device());
    for (int i = 0; i < batch_idx.sizes()[0]; ++i) {
        topk_mask_flat[batch_idx[i]][row_index[i]] = 1;
    }
    torch::Tensor mask = topk_mask_flat.view({batch, height, width, -1});
    return mask;
}

torch::Tensor
RFNet::clip_patch(torch::Tensor &kpts_byxc, torch::Tensor kpts_scale, torch::Tensor &kpts_ori,
                  torch::Tensor &image) {
    assert(kpts_byxc.sizes()[0] == kpts_scale.sizes()[0]);
    int out_width = PSIZE;
    int out_height = PSIZE;
    int B = image.sizes()[0];
    int im_height = image.sizes()[1];
    int im_width = image.sizes()[2];
    int num_kp = kpts_byxc.sizes()[0];
    int max_y = int(im_height - 1);
    int max_x = int(im_width - 1);

    auto temp = torch::meshgrid({torch::linspace(-1, 1, out_height).to(torch::kFloat32).to(device),
                                 torch::linspace(-1, 1, out_width).to(torch::kFloat32).to(device)});
    torch::Tensor y_t = temp[0];
    torch::Tensor x_t = temp[1];
    torch::Tensor one_t = torch::ones_like(x_t);
    x_t = x_t.contiguous().view(-1);
    y_t = y_t.contiguous().view(-1);
    one_t = one_t.view(-1);
    torch::Tensor grid = torch::stack({x_t, y_t, one_t});
    grid = grid.view(-1).repeat(num_kp).view({num_kp, 3, -1});

    torch::Tensor thetas = torch::eye(2, 3).to(torch::kFloat).to(device);
    thetas = thetas.unsqueeze(0).repeat({num_kp, 1, 1});
    kpts_scale = kpts_scale.view({B,-1})/im_info[0];
    kpts_scale = kpts_scale.view(-1)/2.0;
    thetas = thetas * kpts_scale.unsqueeze(-1).unsqueeze(-1);
    torch::Tensor ones (torch::tensor({0,0,1}).unsqueeze(0).unsqueeze(0));
    ones = ones.to(torch::kFloat).to(device).repeat({num_kp,1,1});
    thetas = torch::cat({thetas,ones},1);

    torch::Tensor cos = kpts_ori.slice(1,0,1);
    torch::Tensor sin = kpts_ori.slice(1,1,2);
    torch::Tensor zeros = torch::zeros_like(cos);
    torch::Tensor _ones = torch::ones_like(cos);
    torch::Tensor R = torch::cat({cos,-sin,zeros,sin,cos,zeros,zeros,zeros,_ones},-1);
    R = R.view({-1,3,3});
    thetas = torch::matmul(thetas,R);

    torch::Tensor T_g = torch::matmul(thetas,grid);
    torch::Tensor x = T_g.slice(1,0,1).squeeze(1);
    torch::Tensor y = T_g.slice(1,1,2).squeeze(1);
    torch::Tensor kp_x_ofst = kpts_byxc.slice(1,2,3).view({B,-1}).to(torch::kFloat32)/im_info[0];
    kp_x_ofst = kp_x_ofst.view({-1,1});

    torch::Tensor kp_y_ofst = kpts_byxc.slice(1,1,2).view({B,-1}).to(torch::kFloat32)/im_info[0];
    kp_y_ofst = kp_y_ofst.view({-1,1});
    x = x+kp_x_ofst;
    y = y+kp_y_ofst;
    x= x.view({-1});
    y=y.view({-1});
    torch::Tensor x0 = x.floor();
    torch::Tensor x1 = x0+1;
    torch::Tensor y0 = y.floor();
    torch::Tensor y1 = y0+1;
    x0 = x0.clamp(0,max_x);
    x1 = x1.clamp(0,max_x);
    y0 = y0.clamp(0,max_y);
    y1 = y0.clamp(0,max_y);
    int dim2 = im_width;
    int dim1 = im_width*im_height;
    torch::Tensor batch_inds = kpts_byxc.slice(1,0,1);
    torch::Tensor base = batch_inds.repeat({1,out_height*out_width}).to(torch::kFloat32);
    base = base.view({-1})*dim1;
    torch::Tensor base_y0 = base+y0*dim2;
    torch::Tensor base_y1 = base+y1*dim2;
    torch::Tensor im_flat = image.view({-1});
    torch::Tensor idx_a = torch::_cast_Long((base_y0+x0).to(device));
    torch::Tensor idx_b = torch::_cast_Long((base_y1+x0).to(device));
    torch::Tensor idx_c = torch::_cast_Long((base_y0+x1).to(device));
    torch::Tensor idx_d = torch::_cast_Long((base_y1+x1).to(device));
    torch::Tensor Ia = torch::gather(im_flat,0,idx_a).to(device);
    torch::Tensor Ib = torch::gather(im_flat,0,idx_b).to(device);
    torch::Tensor Ic = torch::gather(im_flat,0,idx_c).to(device);
    torch::Tensor Id = torch::gather(im_flat,0,idx_d).to(device);
    torch::Tensor x0_f = x0.to(torch::kFloat);
    torch::Tensor x1_f = x1.to(torch::kFloat);
    torch::Tensor y0_f = y0.to(torch::kFloat);
    torch::Tensor y1_f = y1.to(torch::kFloat);
    torch::Tensor wa = (x1_f -x)*(y1_f -y);
    torch::Tensor wb = (x1_f -x )*(y-y0_f);
    torch::Tensor wc = (x - x0_f) * (y1_f - y);
    torch::Tensor wd = (x - x0_f) * (y - y0_f);
    torch::Tensor output = wa * Ia + wb * Ib + wc * Ic + wd * Id;
    output = output.view({num_kp, out_height, out_width});
    return output.unsqueeze(1);

}

RFNet::RFNet(std::string des_path,
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
             int output_w) {
    des_module_path = std::move(des_path);
    det_module_path = std::move(det_path);

    RFNet::ksize = ksize;   //15
    RFNet::topk = topk;     //512
    RFNet::PSIZE = PSIZE;   //32
    RFNet::radius = radius; //8
    RFNet::img_resize_height = output_h;  //320
    RFNet::img_resize_width = output_w;  //240
    RFNet::img_orig_height = 640; //640
    RFNet::img_orig_width = 480;  //480
    RFNet::com_strength = com_strength; //3
    RFNet::thresh = thresh;  //0
    RFNet::output_h = output_h; //320
    RFNet::output_w = output_w; //240

    new_h = output_h;
    new_w = output_w;
    RFNet::sh = float(output_h) / img_orig_height;
    RFNet::sw = float(output_w) / img_orig_width;
    dim = -1;
    keepdim = true;
    im_info = torch::tensor({sh,sw});
    det_module = torch::jit::load(det_module_path);
    des_module = torch::jit::load(des_module_path);
    scale_list = torch::tensor(at::ArrayRef < float >
                               ({ scale_vector[0], scale_vector[1], scale_vector[2], scale_vector[3], scale_vector[4], scale_vector[5], scale_vector[6], scale_vector[7], scale_vector[8], scale_vector[9] })).to(device);
    std::cout<<"rf info:"<<std::endl;
    std::cout<<"img_resize_height:"<<img_resize_height<<std::endl;
    std::cout<<"img_resize_width:"<<img_resize_width<<std::endl;
    std::cout<<"img_orig_height:"<<img_orig_height<<std::endl;
    std::cout<<"img_orig_width:"<<img_orig_width<<std::endl;
    std::cout<<"output_h:"<<output_h<<std::endl;
    std::cout<<"output_w:"<<output_w<<std::endl;
    std::cout<<"new_h:"<<new_h<<std::endl;
    std::cout<<"new_w:"<<new_w<<std::endl;
    std::cout<<"sh:"<<sh<<std::endl;
    std::cout<<"sw:"<<sw<<std::endl;


}

torch::Tensor RFNet::get_rfkps(std::string img_path, torch::Tensor &patches) {
    cv::Mat image = cv::imread(img_path, CV_LOAD_IMAGE_UNCHANGED);
    cv::Mat img_float;
    image.convertTo(img_float, CV_32FC1, 1.f / 255.f, 0);
    cv::Mat resize_img;
    im_rescale(img_float, resize_img);
    torch::Tensor img_resize_tensor = torch::from_blob(resize_img.data, {1, 240, 320, 1}, at::kFloat).to(device);
    torch::Tensor imgfloat_tensor = torch::from_blob(img_float.data, {1, img_orig_width, img_orig_height, 1}, at::kFloat).to(device);
    img_resize_tensor = img_resize_tensor.permute({0, 3, 1, 2});
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(img_resize_tensor);
    auto output = det_module->forward(inputs).toTuple();
    torch::Tensor score_maps = output->elements()[0].toTensor();
    torch::Tensor orint_maps = output->elements()[1].toTensor();
    torch::Tensor scale_probs = soft_nms_3d(score_maps);
    torch::Tensor score_map, scale_map;
    soft_max_and_argmax_1d(scale_probs, orint_maps, score_map, scale_map);
    torch::Tensor imscore = score_map * filtbordmask(score_map);
    imscore = imscore * nms(imscore);
    torch::Tensor im_topk = topk_mask(imscore);
    torch::Tensor kpts = im_topk.nonzero();
    auto temp_ori = orint_maps.squeeze().chunk(2, -1);
    torch::Tensor cos = temp_ori[0];
    torch::Tensor sin = temp_ori[1];
    cos = cos.masked_select(im_topk);
    sin = sin.masked_select(im_topk);
    torch::Tensor im_orint = torch::cat({cos.unsqueeze(-1), sin.unsqueeze(-1)}, -1);
    std::cout<<imgfloat_tensor.sizes()<<std::endl;
    patches = clip_patch(kpts,torch::masked_select(scale_map,im_topk),im_orint,imgfloat_tensor);
    return kpts;
}

void RFNet::soft_max_and_argmax_1d(torch::Tensor input, torch::Tensor &orint_maps, torch::Tensor &score_map,
                                   torch::Tensor &scale_map) {
    torch::Tensor input_exp1 = torch::exp(com_strength1 * (input - std::get<0>(torch::max(input, dim, true))));
    torch::Tensor input_softmax1 = input_exp1 / (torch::sum(input_exp1, dim, true) + 1e-8);
    torch::Tensor input_exp2 = torch::exp(com_strength2 * (input - std::get<0>(torch::max(input, dim, true))));
    torch::Tensor input_softmax2 = input_exp2 / (torch::sum(input_exp2, dim, true) + 1e-8);
    score_map = torch::sum(input * input_softmax1, dim, keepdim);
    scale_list.view({1, 1, 1, -1}).to(input_softmax2.device());
    scale_map = torch::sum(scale_list * input_softmax2, dim, keepdim);
    orint_maps = torch::sum((orint_maps * input_softmax1.unsqueeze(-1)), dim - 1, keepdim);
    orint_maps = orint_maps / torch::norm(orint_maps, 2, -1, true);
}

torch::Tensor RFNet::get_rfdes(torch::Tensor &patches) {
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(patches);
    auto output = des_module->forward(inputs).toTensor();
    return output;
}
