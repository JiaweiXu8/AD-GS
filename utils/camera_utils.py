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

from scene.cameras import Camera
import numpy as np
from utils.general_utils import PILtoTorch
from utils.graphics_utils import fov2focal
import torch
from torch.nn.functional import interpolate
from scene.dataset_readers import CameraInfo

WARNED = False

def loadCam(args, id, cam_info: CameraInfo, resolution_scale):
    orig_w, orig_h = cam_info.image.size

    if args.resolution in [1, 2, 4, 8]:
        resolution = round(orig_w/(resolution_scale * args.resolution)), round(orig_h/(resolution_scale * args.resolution))
    else:  # should be a type that converts to float
        if args.resolution == -1:
            if orig_w > 1600:
                global WARNED
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                        "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))

    resized_image_rgb = PILtoTorch(cam_info.image, resolution)

    gt_image = resized_image_rgb[:3, ...]
    loaded_mask, depth, semantic, sky, lidar_depth, flow = None, None, None, None, None, None

    if cam_info.depth is not None:
        depth = interpolate(torch.tensor(cam_info.depth, dtype=torch.float32)[None, None, ...], [resolution[1], resolution[0]], mode='bilinear').squeeze()

    if cam_info.semantic is not None:
        semantic = torch.tensor(cam_info.semantic, dtype=torch.long)
        if semantic.shape[0] != resolution[1] or semantic.shape[1] != resolution[0]:
            idxh = torch.linspace(0, semantic.shape[0] - 1, resolution[1], dtype=torch.int32)
            idxw = torch.linspace(0, semantic.shape[1] - 1, resolution[0], dtype=torch.int32)
            semantic = semantic[idxh[:, None], idxw]

    if cam_info.sky is not None:
        sky = torch.tensor(cam_info.sky, dtype=torch.float32)
        sky = interpolate(torch.tensor(cam_info.sky, dtype=torch.float32)[None, None, ...], [resolution[1], resolution[0]], mode='bilinear').squeeze()
        sky = (sky > 0.5).float()
    
    if cam_info.flow is not None:
        flow = [[b[0]] + [torch.tensor(a, dtype=torch.float32) for a in b[1:]] for b in cam_info.flow]

    if resized_image_rgb.shape[1] == 4:
        loaded_mask = resized_image_rgb[3:4, ...]

    return Camera(
        colmap_id=cam_info.uid,
        cam_id=cam_info.cam_id,
        R=cam_info.R, 
        T=cam_info.T, 
        FoVx=cam_info.FovX, 
        FoVy=cam_info.FovY, 
        image=gt_image, 
        gt_alpha_mask=loaded_mask, 
        fid=cam_info.fid, 
        time=cam_info.time,
        image_name=cam_info.image_name, 
        uid=id, 
        data_device=args.data_device if not args.lazy_load_to_gpu else 'cpu',
        depth=depth, 
        semantic=semantic,
        sky=sky,
        flow=flow
    )

def cameraList_from_camInfos(cam_infos, resolution_scale, args):
    camera_list = []

    for id, c in enumerate(cam_infos):
        camera_list.append(loadCam(args, id, c, resolution_scale))

    return camera_list

def camera_to_JSON(id, camera : Camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        'id' : id,
        'img_name' : camera.image_name,
        'width' : camera.width,
        'height' : camera.height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        'fy' : fov2focal(camera.FovY, camera.height),
        'fx' : fov2focal(camera.FovX, camera.width)
    }
    return camera_entry
