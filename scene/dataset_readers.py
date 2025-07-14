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
import sys
import cv2
import torch
import open3d as o3d
from tqdm import tqdm
from PIL import Image
from typing import NamedTuple
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud
from utils.general_utils import PILtoTorch


class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    depth: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    fid: int
    time: float
    semantic: np.array
    sky: np.array
    cam_id: int = 0
    flow: list = None

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    frame_gap: float = None
    bound: list = None
    others: dict = None

def get_val_frames(num_frames, test_every=None, train_every=None):
    assert train_every is None or test_every is None
    if train_every is None:
        val_frames = set(np.arange(test_every, num_frames, test_every))
    else:
        train_frames = set(np.arange(0, num_frames, train_every))
        val_frames = (set(np.arange(num_frames)) - train_frames) if train_every > 1 else train_frames

    return list(val_frames)

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

def fetchPly(path, return_tuple=False):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    try:
        normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    except:
        normals = None

    try:
        t = np.array(vertices['t'])[..., None]
    except:
        t = None

    try:
        obj_mask = np.array(vertices['obj'])[..., None]
    except:
        obj_mask = None


    if return_tuple:
        return BasicPointCloud(points=positions, colors=colors, normals=normals, time=t, obj_mask=obj_mask)
    return positions, colors, normals, t, obj_mask

def storePly(path, xyz, rgb, t=None, obj_id=None):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    if t is not None:
        dtype.append(('t', 'f4'))
    if obj_id is not None:
        dtype.append(('obj', 'f4'))
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    if t is not None:
        attributes = np.concatenate([attributes, t], axis=-1)
    if obj_id is not None:
        attributes = np.concatenate([attributes, obj_id], axis=-1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def readKITTIInfo(path, use_colmap, split_mode='nvs-75', num_cam: int = 2):
    meta = np.load(os.path.join(path, "poses.npz"), allow_pickle=True)
    time_stamp = meta['time_stamp']
    R = meta['R']
    T = meta['T']
    height = int(meta['height'])
    width = int(meta['width'])
    focal = float(meta['focal'])
    FovX=focal2fov(focal, width)
    FovY=focal2fov(focal, height)
    frame_gap = num_cam / time_stamp.shape[0]
    max_fid = np.max(time_stamp)
    min_fid = np.min(time_stamp)
    time_scale_func = lambda x: ((x - min_fid) / (max_fid - min_fid))
    if split_mode == 'nvs-25':
        i_test = get_val_frames(time_stamp.shape[0] // num_cam, train_every=4)
        frame_gap *= 4
    elif split_mode == 'nvs-50':
        i_test = get_val_frames(time_stamp.shape[0] // num_cam, test_every=2)
        frame_gap *= 2
    elif split_mode == 'nvs-75':
        i_test = get_val_frames(time_stamp.shape[0] // num_cam, test_every=4)
    else:
        raise ValueError("No such split method: " + split_mode)
    
    # print(sorted(i_test))
    train_cameras = []
    test_cameras = []
    for idx, (img_path, fid) in enumerate(zip(sorted(os.listdir(os.path.join(path, "image"))), time_stamp)):
        depth_path = os.path.join(path, "depth", img_path.split(".")[0] + ".npy")
        flow_path = os.path.join(path, "flow", split_mode, img_path.split(".")[0] + ".npz")
        semantic_path = os.path.join(path, "semantic", 'mask_' + img_path.split(".")[0] + ".npy")
        sky_path = os.path.join(path, "sky", 'mask_' + img_path.split(".")[0] + ".npy")
        img_path = os.path.join(path, "image", img_path)
        img = Image.open(img_path)
        assert img.size[0] == width and img.size[1] == height
        flow = np.load(flow_path, allow_pickle=True)['flow'] if os.path.exists(flow_path) else None
        if flow is not None:
            for i in range(len(flow)):
                flow[i][0] = time_scale_func(flow[i][0])
        cam = CameraInfo(
            uid=idx,
            cam_id=idx % num_cam,
            fid=fid,
            R=R[idx, :3, :3],
            T=T[idx, :3],
            FovX=FovX,
            FovY=FovY,
            width=width,
            height=height,
            image_path=img_path,
            depth=np.load(depth_path).squeeze(-1),
            image_name=img_path.split("/")[-1],
            image=img,
            time=time_scale_func(fid),
            semantic=np.load(semantic_path).astype(np.int32),
            sky=np.load(sky_path) != 0,
            flow=flow,
        )
        if idx // num_cam in i_test:
            test_cameras.append(cam)
        else:
            if not os.path.exists(flow_path):
                print(f'[WARNING] Frame {fid} has no flow data. Image {img_path} might have no object, or fail to run prepare-flow.sh')
            # assert os.path.exists(flow_path)
            train_cameras.append(cam)
        
    assert len(test_cameras) == len(i_test) * num_cam, "Wrong Test Cam Number: find {}, but need {}".format(len(test_cameras), len(i_test) * 2)
    nerf_normalization = getNerfppNorm(train_cameras)

    ply_path = os.path.join(path, "points3d-{}.ply".format(split_mode[-2:]))
    assert os.path.join(ply_path), 'Cannot Find PCD for initialization: {}'.format(ply_path)
    xyz, rgb, _, tim, obj_id = fetchPly(ply_path)
    bound = [np.min(xyz, axis=0), np.max(xyz, axis=0)]
    print("Load PCD:", ply_path)
    tim = time_scale_func(tim)
    if use_colmap:
        colmap_ply_path = os.path.join(path, 'colmap-{}.ply'.format(split_mode[-2:]))
        assert os.path.exists(colmap_ply_path), 'Cannot find SfM point cloud: ' + colmap_ply_path
        colmap_xyz, colmap_rgb, _, _, _ = fetchPly(colmap_ply_path)
        obj_id = np.concatenate([obj_id, np.zeros((colmap_xyz.shape[0], 1), dtype=np.float32)], axis=0)
        tim = np.concatenate([tim, np.full((colmap_xyz.shape[0], 1), fill_value=-1, dtype=np.float32)], axis=0)
        xyz = np.concatenate([xyz, colmap_xyz], axis=0)
        rgb = np.concatenate([rgb, colmap_rgb], axis=0)
        print("Load SfM PCD:", colmap_ply_path)
    
    scene_mask = obj_id[..., 0] <= 0.5
    obj_mask = np.bitwise_not(scene_mask)
    scene_xyz, scene_rgb = xyz[scene_mask], rgb[scene_mask]
    obj_xyz, obj_rgb, obj_tim, obj_id = xyz[obj_mask], rgb[obj_mask], tim[obj_mask], obj_id[obj_mask]

    scene_pcd = o3d.geometry.PointCloud()
    scene_pcd.points = o3d.utility.Vector3dVector(scene_xyz)
    scene_pcd.colors = o3d.utility.Vector3dVector(scene_rgb)
    scene_pcd = scene_pcd.voxel_down_sample(voxel_size=0.5)
    # scene_pcd, _ = scene_pcd.remove_radius_outlier(nb_points=10, radius=0.5)
    scene_xyz = np.asarray(scene_pcd.points, dtype=np.float32)
    scene_rgb = np.asarray(scene_pcd.colors, dtype=np.float32)
    
    obj_pts_num = int(obj_xyz.shape[0] * 0.1)
    rand_choice = np.random.permutation(obj_xyz.shape[0])[: obj_pts_num]
    obj_xyz, obj_rgb, obj_tim, obj_id = obj_xyz[rand_choice], obj_rgb[rand_choice], obj_tim[rand_choice], obj_id[rand_choice]
    
    xyz = np.concatenate([scene_xyz, obj_xyz], axis=0)
    rgb = np.concatenate([scene_rgb, obj_rgb], axis=0)
    tim = np.concatenate([np.full((scene_xyz.shape[0], 1), fill_value=-1, dtype=np.float32), obj_tim], axis=0)
    obj_id = np.concatenate([np.zeros((scene_xyz.shape[0], 1), dtype=np.float32), obj_id], axis=0)
    pcd = BasicPointCloud(points=xyz, colors=rgb, time=tim, obj_id=obj_id)

    scene_info = SceneInfo(
        point_cloud=pcd,
        train_cameras=train_cameras,
        test_cameras=test_cameras,
        nerf_normalization=nerf_normalization,
        frame_gap=frame_gap,
        bound=bound,
    )
    return scene_info

def readWaymoInfo(path, use_colmap=False, num_cam: int = 1):
    train_cam_infos, test_cam_infos = [], []
    meta = np.load(os.path.join(path, "cameras.npz"), allow_pickle=True)
    K, R, T = meta['K'], meta['R'], meta['T']
    time_stamps = meta['time_stamps']
    is_val_list = meta['is_val_list']
    frame_gap = num_cam / time_stamps.shape[0]
    time_scale_func = lambda x: ((x - np.min(time_stamps)) / (np.max(time_stamps) - np.min(time_stamps)))
    
    for idx, (img_path, fid) in enumerate(zip(sorted(os.listdir(os.path.join(path, "image"))), time_stamps)):
        depth_path = os.path.join(path, "depth", img_path.split(".")[0] + ".npy")
        flow_path = os.path.join(path, "flow", img_path.split(".")[0] + ".npz")
        semantic_path = os.path.join(path, "semantic", 'mask_' + img_path.split(".")[0] + ".npy")
        sky_path = os.path.join(path, "sky", 'mask_' + img_path.split(".")[0] + ".npy")
        img_path = os.path.join(path, "image", img_path)
        img = Image.open(img_path)
        flow = np.load(flow_path, allow_pickle=True)['flow'] if os.path.exists(flow_path) else None
        if flow is not None:
            for i in range(len(flow)):
                flow[i][0] = time_scale_func(flow[i][0])
        cam = CameraInfo(
            uid=idx,
            cam_id=idx % num_cam,
            fid=fid,
            R=R[idx, :3, :3],
            T=T[idx, :3],
            FovX=focal2fov(K[idx, 0], K[idx, 2] * 2),
            FovY=focal2fov(K[idx, 1], K[idx, 3] * 2),
            width=img.size[0],
            height=img.size[1],
            image_path=img_path,
            depth=np.load(depth_path).squeeze(-1),
            image_name=img_path.split("/")[-1],
            image=img,
            time=time_scale_func(fid),
            semantic=np.load(semantic_path).astype(np.int32),
            sky=np.load(sky_path) != 0,
            flow=flow,
        )
        if is_val_list[idx]:
            test_cam_infos.append(cam)
        else:
            if not os.path.exists(flow_path):
                print(f'[WARNING] Frame {fid} has no flow data. Image {img_path} might have no object, or fail to run prepare-flow.sh')
            # assert os.path.exists(flow_path)
            train_cam_infos.append(cam)
        
    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    assert os.path.join(ply_path), 'Cannot Find PCD for initialization: {}'.format(ply_path)
    xyz, rgb, _, tim, obj_id = fetchPly(ply_path)
    bound = [np.min(xyz, axis=0), np.max(xyz, axis=0)]
    print("Load PCD:", ply_path)
    tim = time_scale_func(tim)
    if use_colmap:
        colmap_ply_path = os.path.join(path, 'colmap.ply')
        assert os.path.exists(colmap_ply_path), 'Cannot find SfM point cloud: ' + colmap_ply_path
        colmap_xyz, colmap_rgb, _, _, _ = fetchPly(colmap_ply_path)
        obj_id = np.concatenate([obj_id, np.zeros((colmap_xyz.shape[0], 1), dtype=np.float32)], axis=0)
        tim = np.concatenate([tim, np.full((colmap_xyz.shape[0], 1), fill_value=-1, dtype=np.float32)], axis=0)
        xyz = np.concatenate([xyz, colmap_xyz], axis=0)
        rgb = np.concatenate([rgb, colmap_rgb], axis=0)
        print("Load SfM PCD:", colmap_ply_path)
    
    scene_mask = obj_id[..., 0] <= 0.5
    obj_mask = np.bitwise_not(scene_mask)
    scene_xyz, scene_rgb = xyz[scene_mask], rgb[scene_mask]
    obj_xyz, obj_rgb, obj_tim, obj_id = xyz[obj_mask], rgb[obj_mask], tim[obj_mask], obj_id[obj_mask]

    scene_pcd = o3d.geometry.PointCloud()
    scene_pcd.points = o3d.utility.Vector3dVector(scene_xyz)
    scene_pcd.colors = o3d.utility.Vector3dVector(scene_rgb)
    scene_pcd = scene_pcd.voxel_down_sample(voxel_size=0.2)
    # scene_pcd, _ = scene_pcd.remove_radius_outlier(nb_points=10, radius=0.5)
    scene_xyz = np.asarray(scene_pcd.points, dtype=np.float32)
    scene_rgb = np.asarray(scene_pcd.colors, dtype=np.float32)
    
    obj_pts_num = int(obj_xyz.shape[0] * 0.3)
    rand_choice = np.random.permutation(obj_xyz.shape[0])[: obj_pts_num]
    obj_xyz, obj_rgb, obj_tim, obj_id = obj_xyz[rand_choice], obj_rgb[rand_choice], obj_tim[rand_choice], obj_id[rand_choice]
    
    xyz = np.concatenate([scene_xyz, obj_xyz], axis=0)
    rgb = np.concatenate([scene_rgb, obj_rgb], axis=0)
    tim = np.concatenate([np.full((scene_xyz.shape[0], 1), fill_value=-1, dtype=np.float32), obj_tim], axis=0)
    obj_id = np.concatenate([np.zeros((scene_xyz.shape[0], 1), dtype=np.float32), obj_id], axis=0)
    pcd = BasicPointCloud(points=xyz, colors=rgb, time=tim, obj_id=obj_id)

    scene_info = SceneInfo(
        point_cloud=pcd,
        train_cameras=train_cam_infos,
        test_cameras=test_cam_infos,
        nerf_normalization=nerf_normalization,
        frame_gap=frame_gap,
        bound=bound,
    )
    return scene_info

def readnuScenesInfo(path, use_colmap=False, num_cam: int = 3):
    train_cam_infos, test_cam_infos = [], []
    meta = np.load(os.path.join(path, "meta.npz"), allow_pickle=True)
    K, R, T = meta['K'], meta['R'], meta['T']
    time_stamps = meta['time_stamps']
    is_val_list = meta['is_val_list']
    frame_gap = num_cam / time_stamps.shape[0]  # num_cam / time.shape[0]
    time_scale_func = lambda x: ((x - np.min(time_stamps)) / (np.max(time_stamps) - np.min(time_stamps)))

    for idx, (img_path, fid) in enumerate(zip(sorted(os.listdir(os.path.join(path, "image"))), time_stamps)):
        depth_path = os.path.join(path, "depth", img_path.split(".")[0] + ".npy")
        flow_path = os.path.join(path, "flow", img_path.split(".")[0] + ".npz")
        semantic_path = os.path.join(path, "semantic", 'mask_' + img_path.split(".")[0] + ".npy")
        sky_path = os.path.join(path, "sky", 'mask_' + img_path.split(".")[0] + ".npy")
        img_path = os.path.join(path, "image", img_path)
        img = Image.open(img_path)
        flow = np.load(flow_path, allow_pickle=True)['flow'] if os.path.exists(flow_path) else None
        if flow is not None:
            for i in range(len(flow)):
                flow[i][0] = time_scale_func(flow[i][0])
        cam = CameraInfo(
            uid=idx,
            cam_id=idx % num_cam,
            fid=fid,
            R=R[idx, :3, :3],
            T=T[idx, :3],
            FovX=focal2fov(K[idx, 0, 0], K[idx, 0, 2] * 2),
            FovY=focal2fov(K[idx, 1, 1], K[idx, 1, 2] * 2),
            width=img.size[0],
            height=img.size[1],
            image_path=img_path,
            depth=np.load(depth_path).squeeze(-1),
            image_name=img_path.split("/")[-1],
            image=img,
            time=time_scale_func(fid),
            semantic=np.load(semantic_path).astype(np.int32),
            sky=np.load(sky_path) != 0,
            flow=flow,
        )
        if is_val_list[idx]:
            test_cam_infos.append(cam)
        else:
            if not os.path.exists(flow_path):
                print(f'[WARNING] Frame {fid} has no flow data. Image {img_path} might have no object, or fail to run prepare-flow.sh')
            # assert os.path.exists(flow_path)
            train_cam_infos.append(cam)

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    assert os.path.join(ply_path), 'Cannot Find PCD for initialization: {}'.format(ply_path)
    xyz, rgb, _, tim, obj_id = fetchPly(ply_path)
    bound = [np.min(xyz, axis=0), np.max(xyz, axis=0)]
    print("Load PCD:", ply_path)
    tim = time_scale_func(tim)
    if use_colmap:
        colmap_ply_path = os.path.join(path, 'colmap.ply')
        assert os.path.exists(colmap_ply_path), 'Cannot find SfM point cloud: ' + colmap_ply_path
        colmap_xyz, colmap_rgb, _, _, _ = fetchPly(colmap_ply_path)
        obj_id = np.concatenate([obj_id, np.zeros((colmap_xyz.shape[0], 1), dtype=np.float32)], axis=0)
        tim = np.concatenate([tim, np.full((colmap_xyz.shape[0], 1), fill_value=-1, dtype=np.float32)], axis=0)
        xyz = np.concatenate([xyz, colmap_xyz], axis=0)
        rgb = np.concatenate([rgb, colmap_rgb], axis=0)
        print("Load SfM PCD:", colmap_ply_path)
    
    scene_mask = obj_id[..., 0] <= 0.5
    obj_mask = np.bitwise_not(scene_mask)
    scene_xyz, scene_rgb = xyz[scene_mask], rgb[scene_mask]
    obj_xyz, obj_rgb, obj_tim, obj_id = xyz[obj_mask], rgb[obj_mask], tim[obj_mask], obj_id[obj_mask]

    scene_pcd = o3d.geometry.PointCloud()
    scene_pcd.points = o3d.utility.Vector3dVector(scene_xyz)
    scene_pcd.colors = o3d.utility.Vector3dVector(scene_rgb)
    scene_pcd = scene_pcd.voxel_down_sample(voxel_size=0.15)
    # scene_pcd, _ = scene_pcd.remove_radius_outlier(nb_points=10, radius=0.5)
    scene_xyz = np.asarray(scene_pcd.points, dtype=np.float32)
    scene_rgb = np.asarray(scene_pcd.colors, dtype=np.float32)
    
    obj_pts_num = int(obj_xyz.shape[0] * 0.5)
    rand_choice = np.random.permutation(obj_xyz.shape[0])[: obj_pts_num]
    obj_xyz, obj_rgb, obj_tim, obj_id = obj_xyz[rand_choice], obj_rgb[rand_choice], obj_tim[rand_choice], obj_id[rand_choice]
    
    xyz = np.concatenate([scene_xyz, obj_xyz], axis=0)
    rgb = np.concatenate([scene_rgb, obj_rgb], axis=0)
    tim = np.concatenate([np.full((scene_xyz.shape[0], 1), fill_value=-1, dtype=np.float32), obj_tim], axis=0)
    obj_id = np.concatenate([np.zeros((scene_xyz.shape[0], 1), dtype=np.float32), obj_id], axis=0)
    pcd = BasicPointCloud(points=xyz, colors=rgb, time=tim, obj_id=obj_id)

    scene_info = SceneInfo(
        point_cloud=pcd,
        train_cameras=train_cam_infos,
        test_cameras=test_cam_infos,
        nerf_normalization=nerf_normalization,
        frame_gap=frame_gap,
        bound=bound,
    )
    return scene_info


sceneLoadTypeCallbacks = {
    'KITTI': readKITTIInfo,
    'Waymo': readWaymoInfo,
    'nuScenes': readnuScenesInfo,
}