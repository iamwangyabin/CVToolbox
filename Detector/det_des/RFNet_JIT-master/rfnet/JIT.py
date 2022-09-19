import torch
from model.rf_det import RFDet
from model.rf_des import HardNetNeiMask
from config import cfg

# resume = "/home/wang/workspace/RFNET/runs/07_23_14_07/model/e001_NN_0.327_NNT_0.409_NNDR_0.812_MeanMS_0.516_det.pth.tar"
#
# det = RFDet(cfg.TRAIN.score_com_strength,
#             cfg.TRAIN.scale_com_strength,
#             cfg.TRAIN.NMS_THRESH,
#             cfg.TRAIN.NMS_KSIZE,
#             cfg.TRAIN.TOPK,
#             cfg.MODEL.GAUSSIAN_KSIZE,
#             cfg.MODEL.GAUSSIAN_SIGMA,
#             cfg.MODEL.KSIZE,
#             cfg.MODEL.padding,
#             cfg.MODEL.dilation,
#             cfg.MODEL.scale_list,
#             cfg.LOSS.SCORE,
#             cfg.LOSS.PAIR,
#             )
#
# device = torch.device("cuda:0")
# model = det.to(device)
# checkpoint = torch.load(resume)
# det.load_state_dict(checkpoint["state_dict"])
#
# img = torch.from_numpy(img.transpose((2, 0, 1)))[None, :].to(
#     device, dtype=torch.float
# )
# score_maps, orint_maps = det(img)
#
# traced_script_module = torch.jit.trace(det, img)
# traced_script_module.save("det.pt")

resume = "/home/wang/workspace/RFNET/runs/07_25_14_05/model/e001_NN_0.283_NNT_0.337_NNDR_0.692_MeanMS_0.437_des.pth.tar"
des = HardNetNeiMask(cfg.HARDNET.MARGIN, cfg.MODEL.COO_THRSH)
device = torch.device("cuda")
des = des.to(device).eval()
checkpoint = torch.load(resume)
des.load_state_dict(checkpoint["state_dict"])

device = torch.device("cuda:0")
im_patches= torch.randn((512,1,32,32)).to(device)
im_des = des(im_patches)
traced_script_module2 = torch.jit.trace(des, im_patches)
traced_script_module2.save("des.pt")

