#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
import torch.nn as nn
from utils.flow_utils import flow_points_project
from utils.depth_utils import get_scaled_shifted_depth

def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)
    
def get_depth_loss(pred, gt, mask=None):
    pred = get_scaled_shifted_depth(pred, gt, mask)
    if mask is None:
        mask = torch.ones_like(pred)
    loss = torch.sum(torch.abs(pred - gt) * mask) / torch.sum(mask)
    return loss

# def get_flow_loss(img_flow, flow_pkg):
#     _, K, R, T, flow, flow_vis = flow_pkg

#     flow_vis: torch.Tensor = (flow_vis > 0.5) & (flow[0] <= flow.shape[2] - 1.0) & (flow[0] >= 0.0) & (flow[1] <= flow.shape[1] - 1.0) & (flow[1] >= 0.0)
#     if not torch.nonzero(flow_vis).any():
#         return 0.0
#     loss = torch.abs(img_flow[:, flow_vis] - flow[:, flow_vis])
#     loss = torch.cat([loss[:1] / flow.shape[2], loss[1:] / flow.shape[1]], dim=0)
#     loss = torch.mean(torch.sum(loss, dim=0))
#     return loss

def get_flow_loss(img_flow, flow_pkg, img_opacity=None, dist=1e-3):
    _, K, R, T, flow, flow_vis = flow_pkg
    H, W = flow.shape[1:]

    flow_vis: torch.Tensor = (flow_vis > 0.5) & (flow[0] <= flow.shape[2] - 1.0) & (flow[0] >= 0.0) & (flow[1] <= flow.shape[1] - 1.0) & (flow[1] >= 0.0)
    selected_coord = torch.nonzero(flow_vis, as_tuple=True)
    if selected_coord[0].numel() == 0:
        return 0.0
    flow_vis = flow_vis.float()
    if img_opacity is not None:
        flow_vis = flow_vis * img_opacity
    img_flow = torch.permute(img_flow[:, selected_coord[0], selected_coord[1]], (1, 0))  # N, 3
    flow = torch.permute(flow[:, selected_coord[0], selected_coord[1]], (1, 0))  # N, 3
    flow_vis = flow_vis[selected_coord[0], selected_coord[1]]  # N,
    img_flow, mask = flow_points_project(img_flow, K, R, T, dist=dist)
    flow_vis = flow_vis * mask.float()  # N,
    
    loss = torch.abs(img_flow - flow) * flow_vis[..., None]
    loss = torch.cat([loss[..., :1] / W, loss[..., 1:] / H], dim=-1)
    loss = torch.mean(torch.sum(loss, dim=-1))
    return loss