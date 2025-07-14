import os
import argparse
import numpy as np
from plyfile import PlyData, PlyElement
from typing import NamedTuple
import torch
from torch.nn.functional import grid_sample
import tqdm
from PIL import Image
import imageio
from itertools import combinations

class BasicPointCloud(NamedTuple):
    points : np.array
    colors : np.array
    normals : np.array
    time : np.array = None

def fetchPly(path, scale=1.0, downsample=1.0, time_scale_func=None):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T / scale
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    try:
        normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    except:
        normals = np.zeros(positions.shape)

    try:
        t = np.array(vertices['t'])[..., None]
        if time_scale_func is not None:
            t = time_scale_func(t)
    except:
        t = None

    assert downsample > 0.0 and downsample <= 1.0
    if downsample < 1.0:
        rand_choice = np.random.permutation(positions.shape[0])[:int(positions.shape[0] * downsample)]
        positions = positions[rand_choice]
        colors = colors[rand_choice]
        normals = normals[rand_choice]
        t = t[rand_choice] if t is not None else None

    return BasicPointCloud(points=positions, colors=colors, normals=normals, time=t)

def storePly(path, xyz, rgb, t=None, obj_mask=None):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    if t is not None:
        dtype.append(('t', 'f4'))
    if obj_mask is not None:
        dtype.append(('obj', 'f4'))
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    if t is not None:
        attributes = np.concatenate([attributes, t], axis=-1)
    if obj_mask is not None:
        attributes = np.concatenate([attributes, obj_mask], axis=-1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def get_val_frames(num_frames, test_every=None, train_every=None):
    assert train_every is None or test_every is None
    if train_every is None:
        val_frames = set(np.arange(test_every, num_frames, test_every))
    else:
        train_frames = set(np.arange(0, num_frames, train_every))
        val_frames = (set(np.arange(num_frames)) - train_frames) if train_every > 1 else train_frames

    return list(val_frames)

def segment_waymo_pcd(path):
    pcd_path = os.path.join(path, 'points3d.ply')
    pcd = fetchPly(pcd_path)
    pcd_points = torch.tensor(pcd.points, dtype=torch.float32, device='cuda')
    pcd_obj_segment = torch.zeros((pcd.points.shape[0],), dtype=torch.float32, device='cuda')
    pcd_timestamp = torch.tensor(pcd.time, dtype=torch.float32, device='cuda').squeeze(-1)

    meta = np.load(os.path.join(path, "cameras.npz"), allow_pickle=True)
    Ks, Rs, Ts = meta['K'], meta['R'], meta['T']
    time_stamps = meta['time_stamps']
    is_val_list = meta['is_val_list']
    for idx, (semantic_path, fid) in enumerate(zip(sorted(os.listdir(os.path.join(path, "semantic"))), time_stamps)):
        if is_val_list[idx]:
            continue
        semantic_path = os.path.join(path, 'semantic', semantic_path)
        K, R, T = Ks[idx], Rs[idx], Ts[idx]
        K = np.array([
            [K[0], 0.0, K[2]],
            [0.0, K[1], K[3]],
            [0.0, 0.0, 1.0],
        ], dtype=np.float32)
        W, H = K[0, 2] * 2, K[1, 2] * 2
        K = torch.tensor(K, dtype=torch.float32, device='cuda')
        R = torch.tensor(R, dtype=torch.float32, device='cuda')
        T = torch.tensor(T, dtype=torch.float32, device='cuda')

        selected_mask = (pcd_timestamp == fid) & (pcd_obj_segment == 0.0)
        semantic_map = torch.tensor(np.load(semantic_path).astype(np.float32), dtype=torch.float32, device='cuda')
        proj_points = (K @ (R @ pcd_points[selected_mask][..., None] + T[..., None])).squeeze(-1)
        proj_mask = proj_points[..., 2] > 0.0
        proj_points = proj_points[..., :2] / proj_points[..., 2:]
        proj_mask = proj_mask & (proj_points[..., 0] > 0.0) & (proj_points[..., 0] < W)
        proj_mask = proj_mask & (proj_points[..., 1] > 0.0) & (proj_points[..., 1] < H)
        proj_points[..., 0] /= W
        proj_points[..., 1] /= H
        proj_points = proj_points * 2.0 - 1.0
        obj_segment = grid_sample(semantic_map[None, None], proj_points[None, None], mode='nearest', align_corners=True).squeeze()
        obj_segment[~proj_mask] = 0.0
        pcd_obj_segment[selected_mask] = obj_segment
    
    print('Total points:', pcd.points.shape[0], 'Get object points:', torch.sum(pcd_obj_segment > 0).item())
    pcd_obj_segment = pcd_obj_segment[..., None].detach().cpu().numpy()
    storePly(pcd_path, pcd.points, pcd.colors * 255.0, pcd.time, pcd_obj_segment)

def segment_kitti_pcd(path, split_mode='nvs-75'):
    pcd_path = os.path.join(path, 'points3d-{}.ply'.format(split_mode.split('-')[-1]))
    pcd = fetchPly(pcd_path)
    pcd_points = torch.tensor(pcd.points, dtype=torch.float32, device='cuda')
    pcd_obj_segment = torch.zeros((pcd.points.shape[0],), dtype=torch.float32, device='cuda')
    pcd_timestamp = torch.tensor(pcd.time, dtype=torch.float32, device='cuda').squeeze(-1)

    meta = np.load(os.path.join(path, 'poses.npz'), allow_pickle=True)
    Rs, Ts = meta['R'], meta['T']
    H = int(meta['height'])
    W = int(meta['width'])
    focal = float(meta['focal'])
    time_stamp = meta['time_stamp']
    K = torch.tensor([
        [focal, 0.0, W / 2.0],
        [0.0, focal, H / 2.0],
        [0.0, 0.0, 1.0],
    ], dtype=torch.float32, device='cuda')

    if split_mode == 'nvs-25':
        i_test = get_val_frames(time_stamp.shape[0] // 2, train_every=4)
    elif split_mode == 'nvs-50':
        i_test = get_val_frames(time_stamp.shape[0] // 2, test_every=2)
    elif split_mode == 'nvs-75':
        i_test = get_val_frames(time_stamp.shape[0] // 2, test_every=4)
    else:
        raise ValueError("No such split method: " + split_mode)
    

    for idx, (semantic_path, fid) in enumerate(zip(sorted(os.listdir(os.path.join(path, "semantic"))), time_stamp)):
        if idx // 2 in i_test:
            continue
        semantic_path = os.path.join(path, 'semantic', semantic_path)
        R, T = Rs[idx], Ts[idx]
        R = torch.tensor(R, dtype=torch.float32, device='cuda')
        T = torch.tensor(T, dtype=torch.float32, device='cuda')

        selected_mask = (pcd_timestamp == fid) & (pcd_obj_segment == 0.0)
        semantic_map = torch.tensor(np.load(semantic_path).astype(np.float32), dtype=torch.float32, device='cuda')
        proj_points = (K @ (R @ pcd_points[selected_mask][..., None] + T[..., None])).squeeze(-1)
        proj_mask = proj_points[..., 2] > 0.0
        proj_points = proj_points[..., :2] / proj_points[..., 2:]
        proj_mask = proj_mask & (proj_points[..., 0] > 0.0) & (proj_points[..., 0] < W)
        proj_mask = proj_mask & (proj_points[..., 1] > 0.0) & (proj_points[..., 1] < H)
        proj_points[..., 0] /= W
        proj_points[..., 1] /= H
        proj_points = proj_points * 2.0 - 1.0
        obj_segment = grid_sample(semantic_map[None, None], proj_points[None, None], mode='nearest', align_corners=True).squeeze()
        obj_segment[~proj_mask] = 0.0
        pcd_obj_segment[selected_mask] = obj_segment
    
    print('Total points:', pcd.points.shape[0], 'Get object points:', torch.sum(pcd_obj_segment > 0).item())
    pcd_obj_segment = pcd_obj_segment[..., None].detach().cpu().numpy()
    storePly(pcd_path, pcd.points, pcd.colors * 255.0, pcd.time, pcd_obj_segment)

def segment_nuscenes_pcd(path):
    pcd_path = os.path.join(path, 'points3d.ply')
    pcd = fetchPly(pcd_path)
    pcd_points = torch.tensor(pcd.points, dtype=torch.float32, device='cuda')
    pcd_obj_segment = torch.zeros((pcd.points.shape[0],), dtype=torch.float32, device='cuda')
    pcd_timestamp = torch.tensor(pcd.time, dtype=torch.float32, device='cuda').squeeze(-1)

    meta = np.load(os.path.join(path, "meta.npz"), allow_pickle=True)
    Ks, Rs, Ts = meta['K'], meta['R'], meta['T']
    time_stamps = meta['time_stamps']
    is_val_list = meta['is_val_list']
    for idx, (semantic_path, fid) in enumerate(zip(sorted(os.listdir(os.path.join(path, "semantic"))), time_stamps)):
        if is_val_list[idx]:
            continue
        semantic_path = os.path.join(path, 'semantic', semantic_path)
        K, R, T = Ks[idx], Rs[idx], Ts[idx]
        K = torch.tensor(K, dtype=torch.float32, device='cuda')
        R = torch.tensor(R, dtype=torch.float32, device='cuda')
        T = torch.tensor(T, dtype=torch.float32, device='cuda')

        selected_mask = (pcd_timestamp == fid) & (pcd_obj_segment == 0.0)
        semantic_map = torch.tensor(np.load(semantic_path).astype(np.float32), dtype=torch.float32, device='cuda')
        H, W = semantic_map.shape[-2], semantic_map.shape[-1]
        proj_points = (K @ (R @ pcd_points[selected_mask][..., None] + T[..., None])).squeeze(-1)
        proj_mask = proj_points[..., 2] > 0.0
        proj_points = proj_points[..., :2] / proj_points[..., 2:]
        proj_mask = proj_mask & (proj_points[..., 0] > 0.0) & (proj_points[..., 0] < W)
        proj_mask = proj_mask & (proj_points[..., 1] > 0.0) & (proj_points[..., 1] < H)
        proj_points[..., 0] /= W
        proj_points[..., 1] /= H
        proj_points = proj_points * 2.0 - 1.0
        obj_segment = grid_sample(semantic_map[None, None], proj_points[None, None], mode='nearest', align_corners=True).squeeze()
        obj_segment[~proj_mask] = 0.0
        pcd_obj_segment[selected_mask] = obj_segment
    
    print('Total points:', pcd.points.shape[0], 'Get object points:', torch.sum(pcd_obj_segment > 0).item())
    pcd_obj_segment = pcd_obj_segment[..., None].detach().cpu().numpy()
    storePly(pcd_path, pcd.points, pcd.colors * 255.0, pcd.time, pcd_obj_segment)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('path')
    parser.add_argument('--split_mode', default='nvs-75')
    args = parser.parse_args()
    if os.path.exists(os.path.join(args.path, "cameras.npz")):
        print("Found cameras.npz file, assuming Waymo data set!")
        segment_waymo_pcd(args.path)
        # merge_waymo_segment(args.path)
    elif os.path.exists(os.path.join(args.path, 'poses.npz')):
        print("Found poses.npz file, assuming KITTI or vKITTI data set!")
        segment_kitti_pcd(args.path, args.split_mode)
    elif os.path.exists(os.path.join(args.path, 'meta.npz')):
        print("Found meta.npz file, assuming nuScenes data set!")
        segment_nuscenes_pcd(args.path)
    else:
        assert False, 'Could not recognize scene type!'