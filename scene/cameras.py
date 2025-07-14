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
from torch import nn
import numpy as np
from utils.graphics_utils import getWorld2View2, getProjectionMatrix

class Camera(nn.Module):
    def __init__(
            self, 
            colmap_id,
            cam_id,
            R, 
            T, 
            FoVx, 
            FoVy, 
            image, 
            gt_alpha_mask,
            image_name, 
            uid, 
            time, 
            fid,
            trans=np.array([0.0, 0.0, 0.0]), 
            scale=1.0, 
            data_device = "cuda", 
            depth=None, 
            semantic=None, 
            sky=None,
            flow=None
        ):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.cam_id = cam_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name
        self.time = time
        self.fid = fid

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        self.original_image = image.clamp(0.0, 1.0)
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]
        self.depth = depth
        self.semantic = semantic
        self.sky = sky
        self.flow = flow

        if gt_alpha_mask is not None:
            self.original_image *= gt_alpha_mask

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1)
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1)
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

        self.to(self.data_device)
    
    def to(self, device):
        self.data_device = device

        self.original_image = self.original_image.to(device)
        self.depth = self.depth.to(device) if self.depth is not None else None
        self.semantic = self.semantic.to(device) if self.semantic is not None else None
        self.sky = self.sky.to(device) if self.sky is not None else None
        if self.flow is not None:
            self.flow = [[a.to(device) if torch.is_tensor(a) else a for a in b] for b in self.flow]

        self.world_view_transform = self.world_view_transform.to(device)
        self.projection_matrix = self.projection_matrix.to(device)
        self.full_proj_transform = self.full_proj_transform.to(device)
        self.camera_center = self.camera_center.to(device)

    def cuda(self):
        self.to('cuda')

class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform, time_stamp):
        self.image_width = width
        self.image_height = height    
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]
        self.time = time_stamp

