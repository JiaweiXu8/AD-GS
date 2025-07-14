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
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from utils.system_utils import put_color
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args, get_config
from gaussian_renderer import GaussianModel
from scene.env import EnvironmentMap
from scene.cameras import Camera
import imageio
import numpy as np
import random
import copy
from scene.dataset_readers import storePly
import time
from utils.image_utils import psnr
from utils.loss_utils import ssim
from lpipsPyTorch import lpips
import json
from utils.graphics_utils import fov2focal
from utils.flow_utils import get_img_flow, flow_to_img

to8b = lambda x: (255.0 * np.clip(x, 0, 1)).astype(np.uint8)

def render_set(model_path, name, iteration, views, scene, pipeline, output_video, cam_order, cal_metrics=True):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    psnrs, ssims, lpipses, lpips_alex = [], [], [], []

    total_time = 0.0
    renderings = dict()
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        t = time.time()
        rendering = torch.clip(render(view, scene.gaussians, scene.env_map, pipeline)["render"], 0.0, 1.0)
        total_time += time.time() - t
        gt = torch.clip(view.original_image.to('cuda'), 0.0, 1.0)

        if cal_metrics:
            psnrs.append(psnr(rendering[None], gt[None]).item())
            ssims.append(ssim(rendering[None], gt[None]).item())
            lpipses.append(lpips(rendering[None], gt[None], net_type='vgg').item())
            lpips_alex.append(lpips(rendering[None], gt[None], net_type='alex').item())

        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
        if output_video:
            try:
                renderings[view.cam_id].append(to8b(torch.permute(rendering, (1, 2, 0)).detach().cpu().numpy()))
            except KeyError:
                renderings[view.cam_id] = [to8b(torch.permute(rendering, (1, 2, 0)).detach().cpu().numpy())]
    
    if output_video:
        video = []
        for cam_id, cam_renderings in renderings.items():
            cam_renderings = np.stack(cam_renderings, axis=0)
            renderings[cam_id] = cam_renderings
        if len(cam_order) == 0:
            cam_order = sorted(renderings.keys())
        for cam_id in cam_order:
            video.append(renderings[cam_id])
        video = np.concatenate(video, axis=2)  # F, H, W, C
        video_path = os.path.join(model_path, name, "ours_{}".format(iteration), 'video.mp4')
        imageio.mimwrite(video_path, video, fps=10, quality=8)

    if cal_metrics:
        fps = len(views) / total_time
        psnrs, ssims, lpipses, lpips_alex = np.mean(psnrs), np.mean(ssims), np.mean(lpipses), np.mean(lpips_alex)
        print(name)
        print("  SSIM : {}".format(ssims))
        print("  PSNR : {}".format(psnrs))
        print("  LPIPS(VGG): {}".format(lpipses))
        print("  LPIPS(ALEX): {}".format(lpips_alex))
        print("  FPS  : {}".format(fps))
        print("")
        res = {
            "ours_{}".format(iteration): {
                "SSIM": ssims,
                "PSNR": psnrs,
                "LPIPS(VGG)": lpipses,
                "LPIPS(ALEX)": lpips_alex,
                "FPS": fps,
            }
        }
        res_path = os.path.join(model_path, "results.json" if name == 'test' else "results-train.json")
        with open(res_path, 'w') as fp:
                json.dump(res, fp, indent=True)

def render_deform(model_path, name, iteration, views, scene: Scene, pipeline, output_video, cam_order):
    deform_path = os.path.join(model_path, name, "ours_{}".format(iteration), "deform")
    makedirs(deform_path, exist_ok=True)

    renderings = dict()
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        xyz1 = scene.gaussians.get_deformed_xyz(view.time)
        xyz2 = scene.gaussians.get_deformed_xyz(view.time + 1.0 / len(views))
        deform = torch.abs(xyz2 - xyz1) * len(views)
        deform = (deform - torch.min(deform)) / (torch.max(deform) - torch.min(deform))
        rendering = render(view, scene.gaussians, scene.env_map, pipeline, override_color=deform.clip(0.0, 1.0))["foreground"]
        torchvision.utils.save_image(rendering, os.path.join(deform_path, '{0:05d}'.format(idx) + ".png"))
        if output_video:
            try:
                renderings[view.cam_id].append(to8b(torch.permute(rendering, (1, 2, 0)).detach().cpu().numpy()))
            except KeyError:
                renderings[view.cam_id] = [to8b(torch.permute(rendering, (1, 2, 0)).detach().cpu().numpy())]
    
    if output_video:
        video = []
        for cam_id, cam_renderings in renderings.items():
            cam_renderings = np.stack(cam_renderings, axis=0)
            renderings[cam_id] = cam_renderings
        if len(cam_order) == 0:
            cam_order = sorted(renderings.keys())
        for cam_id in cam_order:
            video.append(renderings[cam_id])
        video = np.concatenate(video, axis=2)  # F, H, W, C
        video_path = os.path.join(model_path, name, "ours_{}".format(iteration), 'deform.mp4')
        imageio.mimwrite(video_path, video, fps=10, quality=8)


GENERAL_RENDER_FUNC = {
    'render': render_set,
    'deform': render_deform,
}

def get_env_point_cloud(model_path, scene: Scene):
    env_save_path = os.path.join(model_path, 'env', "ours_{}".format(scene.loaded_iter))
    os.makedirs(env_save_path, exist_ok=True)
    scene.env_map.extract_env_map(os.path.join(env_save_path, 'env_map.ply'))
    print('Save to', os.path.join(env_save_path, 'env_map.ply'))

POINT_CLOUD_FUNC = {
    'env': get_env_point_cloud,
}

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, mode : str, output_video : bool, cam_order : list):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree, dataset.order_args)
        env_map = EnvironmentMap(**dataset.env_args)
        scene = Scene(dataset, gaussians, env_map, load_iteration=iteration, shuffle=False)
        print(put_color("[Render Mode] " + mode, color='violet'))

        if mode in GENERAL_RENDER_FUNC.keys():
            render_func = GENERAL_RENDER_FUNC[mode]
            if not skip_train:
                render_func(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), scene, pipeline, output_video, cam_order)
            if not skip_test:
                render_func(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), scene, pipeline, output_video, cam_order)
        elif mode in POINT_CLOUD_FUNC.keys():
            POINT_CLOUD_FUNC[mode](dataset.model_path, scene)
        elif mode == 'time':
            num_frames = 150
            views = scene.getTrainCameras()
            view = views[random.randint(0, len(views) - 1)]
            views = []
            for i in range(0, num_frames):
                cam = copy.deepcopy(view)
                cam.fid = i
                cam.time = i / num_frames
                views.append(cam)
            render_set(dataset.model_path, "interp_time", scene.loaded_iter, views, scene, pipeline, output_video, cam_order=[], cal_metrics=False)
        else:
            raise "Not support mode: " + mode

if __name__ == "__main__":

    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    parser.add_argument('--config', '-c', type=str, default=None)
    config_path = parser.parse_known_args()[0].config
    if config_path is not None and os.path.exists(config_path):
        print("Find Config:", config_path)
        config = get_config(config_path)
    else:
        config = None
    model = ModelParams(parser, config, sentinel=True)
    pipeline = PipelineParams(parser, config)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--mode", default='render', type=str)
    parser.add_argument("--video", '-v', action='store_true')
    parser.add_argument("--cam_order", nargs="+", type=int, default=[])
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    args.data_device = "cuda:0" if args.data_device == 'cuda' else args.data_device
    torch.cuda.set_device(args.data_device)

    # Initialize system state (RNG)
    safe_state(args.quiet)
    # print(vars(args))

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, args.mode, args.video, args.cam_order)