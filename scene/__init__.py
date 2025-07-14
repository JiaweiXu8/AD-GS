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

import os
import random
import json
import numpy as np
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON
from scene.env import EnvironmentMap
from utils.graphics_utils import get_bound_diagonal_distance
from scene.dataset_readers import storePly

class Scene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, env_map: EnvironmentMap, load_iteration=None, shuffle=True, resolution_scales=[1.0]):
        """
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians
        self.env_map = env_map

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}

        if os.path.exists(os.path.join(args.source_path, "poses.npz")):
            print("Found poses.npz, assuming KITTI or vKITTI data set!")
            scene_info = sceneLoadTypeCallbacks["KITTI"](args.source_path, args.use_colmap, split_mode=args.split_mode, num_cam=args.num_cam)
        elif os.path.exists(os.path.join(args.source_path, "cameras.npz")):
            print("Found cameras.npz file, assuming Waymo data set!")
            scene_info = sceneLoadTypeCallbacks["Waymo"](args.source_path, args.use_colmap, args.num_cam)
        elif os.path.exists(os.path.join(args.source_path, "meta.npz")):
            print("Found meta.npz file, assuming nuScenes data set!")
            scene_info = sceneLoadTypeCallbacks["nuScenes"](args.source_path, args.use_colmap, args.num_cam)
        else:
            assert False, "Could not recognize scene type!"

        if not self.loaded_iter:
            storePly(
                os.path.join(self.model_path, "input.ply"), 
                scene_info.point_cloud.points, 
                scene_info.point_cloud.colors * 255.0,
                scene_info.point_cloud.time,
                scene_info.point_cloud.obj_id
            )
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]
        if scene_info.bound is not None:
            self.scene_extent = np.sqrt(np.sum((scene_info.bound[1] - scene_info.bound[0]) ** 2))
        else:
            self.scene_extent = get_bound_diagonal_distance(scene_info.point_cloud.points)
        self.frame_gap = scene_info.frame_gap
        print(scene_info.nerf_normalization)
        print("Cameras extent:", self.cameras_extent, "Scene extent:", self.scene_extent, "Frame gap:", self.frame_gap)
        print("PCD Bound:", np.min(scene_info.point_cloud.points, axis=0), "<->", np.max(scene_info.point_cloud.points, axis=0))
        print("Train Camera:", len(scene_info.train_cameras), "Test Camera:", len(scene_info.test_cameras))

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)

        if self.loaded_iter:
            ckpt_path = os.path.join(self.model_path, "point_cloud", "iteration_" + str(self.loaded_iter))
            self.gaussians.load_ply(os.path.join(ckpt_path, "point_cloud.ply"))
            self.env_map.load_weights(os.path.join(ckpt_path, "env.pth"))
        else:
            self.gaussians.create_from_pcd(
                scene_info.point_cloud,
                self.scene_extent,
                self.cameras_extent,
                self.frame_gap,
                args.default_order_downsample_ratio,
            )

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
        self.env_map.save_weights(os.path.join(point_cloud_path, "env.pth"))

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]