import numpy as np
import os
import argparse
import json
from nuscenes.nuscenes import NuScenes
import shutil
from plyfile import PlyData, PlyElement
import tqdm
import torch
from PIL import Image
import sys

def get_nearest_lidar_token(sorted_lidar_token, timestamp, start=0, end=-1):
    if end == -1:
        end = len(sorted_lidar_token)
    assert end - start >= 2
    if end - start == 2:
        return sorted_lidar_token[start][1] if np.abs(timestamp - sorted_lidar_token[start][0]) < np.abs(timestamp - sorted_lidar_token[start + 1][0]) else sorted_lidar_token[start + 1][1]
    mid = (start + end) // 2
    if sorted_lidar_token[mid][0] == timestamp:
        return sorted_lidar_token[mid][1]
    elif sorted_lidar_token[mid][0] > timestamp:
        return get_nearest_lidar_token(sorted_lidar_token, timestamp, start=start, end=mid + 1)
    else:
        return get_nearest_lidar_token(sorted_lidar_token, timestamp, start=mid, end=end)

def build_rotation(r):
    norm = np.sqrt(r[0]*r[0] + r[1]*r[1] + r[2]*r[2] + r[3]*r[3])

    q = r / norm

    r = q[0]
    x = q[1]
    y = q[2]
    z = q[3]
    R = np.stack([
        1 - 2 * (y*y + z*z), 2 * (x*y - r*z), 2 * (x*z + r*y),
        2 * (x*y + r*z), 1 - 2 * (x*x + z*z), 2 * (y*z - r*x),
        2 * (x*z - r*y), 2 * (y*z + r*x), 1 - 2 * (x*x + y*y),
    ], axis=-1).reshape(3, 3)
    return R

def storePly(path, xyz, rgb, t=None):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    if t is not None:
        dtype.append(('t', 'f4'))
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    if t is not None:
        attributes = np.concatenate([attributes, t], axis=-1)
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

parser = argparse.ArgumentParser()
parser.add_argument('src')
parser.add_argument('dst')
parser.add_argument('scene', type=str)
parser.add_argument('--first_frame', default=10, type=int)
parser.add_argument('--last_frame', default=69, type=int)
parser.add_argument('--downsample_ratio', '-r', default=1.0, type=float)
parser.add_argument('--version', type=str, default='v1.0-trainval', choices=['v1.0-mini', 'v1,0-trainval'])
parser.add_argument('--use_color', action='store_true')
parser.add_argument('--use_depth', action='store_true')
args = parser.parse_args()
dst_path = os.path.join(args.dst, args.scene)
src_path = args.src
first_frame = args.first_frame
last_frame = args.last_frame
downsample_ratio = args.downsample_ratio
dst_image_path = os.path.join(dst_path, "image")
depth_folder = os.path.join(dst_path, 'lidar_depth')

with open(os.path.join(src_path, args.version, 'scene.json'), 'rb') as f:
    scene_token = None
    for scene in json.load(f):
        if scene['name'] == args.scene:
            scene_token = scene['token']
            break
    assert scene_token is not None, 'Cannot find scene: {}'.format(args.scene)

os.makedirs(dst_path, exist_ok=True)
os.makedirs(dst_image_path, exist_ok=True)
if args.use_depth:
    os.makedirs(depth_folder, exist_ok=True)

nusc = NuScenes(version=args.version, dataroot=src_path, verbose=True)
scene = nusc.get('scene', scene_token)
assert scene['name'] == args.scene
print("Parsing {}, token {}".format(args.scene, scene_token))

SENSORS = [
    # 'CAM_BACK',
    # 'CAM_BACK_LEFT',
    # 'CAM_BACK_RIGHT',
    'CAM_FRONT',
    'CAM_FRONT_LEFT',
    'CAM_FRONT_RIGHT'
]

iter_sample = nusc.get('sample', scene['first_sample_token'])

lidar_iter = nusc.get('sample_data', iter_sample['data']['LIDAR_TOP'])
lidar_tokens = [(lidar_iter['timestamp'], lidar_iter)]
while lidar_iter['next'] != "":
    lidar_iter = nusc.get('sample_data', lidar_iter['next'])
    lidar_tokens.append((lidar_iter['timestamp'], lidar_iter))
# print("Get lidar timestamp:", [x[0] for x in lidar_tokens])
sorted(lidar_tokens, key=lambda x: x[0])

Ks = []
RTs = []
time_stamps = []
pointcloud = []
pcd_rgb = []
cameras_iter = [nusc.get('sample_data', iter_sample['data'][cam]) for cam in SENSORS]
global2ego0 = None
val_fid_list = get_val_frames(last_frame - first_frame + 1, 4)
is_val_list = []
process_bar = tqdm.tqdm(range(last_frame - first_frame + 1), desc="Processing")
for idx in range(last_frame + 1):
    if idx < first_frame or idx > last_frame:
        cameras_iter = [nusc.get('sample_data', i['next']) for i in cameras_iter]
        continue

    # In nuScenes dataset, the lidar(20Hz) and cameras(12Hz) have different frequencies. We use the nearest lidar to align with the cameras.
    lidar_iter = get_nearest_lidar_token(lidar_tokens, timestamp=cameras_iter[0]['timestamp'])
    lidar_path = os.path.join(src_path, lidar_iter['filename'])
    pcd = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 5)[..., :3, None]

    lidar2ego = nusc.get('calibrated_sensor', lidar_iter['calibrated_sensor_token'])
    lidar2ego_r = build_rotation(np.array(lidar2ego['rotation']))
    lidar2ego_t = np.array(lidar2ego['translation'])
    pcd = lidar2ego_r @ pcd + lidar2ego_t[..., None]

    ego2global = nusc.get('ego_pose', lidar_iter['ego_pose_token'])
    ego2global_r = build_rotation(np.array(ego2global['rotation']))
    ego2global_t = np.array(ego2global['translation'])
    ego2global_rt = np.eye(4)
    ego2global_rt[:3, :3] = ego2global_r
    ego2global_rt[:3, 3] = ego2global_t
    if global2ego0 is None:
        global2ego0 = np.linalg.inv(ego2global_rt)
    ego2global_rt = global2ego0 @ ego2global_rt
    pcd = ego2global_rt[:3, :3] @ pcd + ego2global_rt[:3, 3:]

    total_mask = np.zeros((pcd.shape[0],), dtype=np.bool8)
    counts = 0
    points_color = np.zeros((pcd.shape[0], 3), dtype=np.float32)

    is_val = (idx - first_frame) in val_fid_list

    for data in cameras_iter:
        img_path , _ , cam_intrinsic = nusc.get_sample_data(data['token'])

        image_id = len(Ks)
        shutil.copyfile(img_path, os.path.join(dst_image_path, "{:06d}.png".format(image_id)))
        K = np.array(cam_intrinsic, dtype=np.float32).reshape(3, 3)
        Ks.append(K)
        time_stamps.append(idx - first_frame)
        is_val_list.append(is_val)

        ego2global = nusc.get('ego_pose', data['ego_pose_token'])
        ego2global_r = build_rotation(np.array(ego2global['rotation']))
        ego2global_t = np.array(ego2global['translation'])
        ego2global_rt = np.eye(4)
        ego2global_rt[:3,:3] = ego2global_r
        ego2global_rt[:3,3] = ego2global_t
        if global2ego0 is None:
            global2ego0 = np.linalg.inv(ego2global_rt)
        ego2global_rt = global2ego0 @ ego2global_rt

        cam2ego = nusc.get('calibrated_sensor', data['calibrated_sensor_token'])
        cam2ego_r = build_rotation(np.array(cam2ego['rotation']))
        cam2ego_t = np.array(cam2ego['translation'])
        cam2ego_rt = np.eye(4)
        cam2ego_rt[:3, :3] = cam2ego_r
        cam2ego_rt[:3, 3] = cam2ego_t

        RT = np.linalg.inv(ego2global_rt @ cam2ego_rt)
        RTs.append(RT)

        H, W = data['height'], data['width']
        proj_pcd = (K @ (RT[:3, :3] @ pcd + RT[:3, 3:]))[..., 0]
        depth = proj_pcd[..., 2]
        mask = proj_pcd[..., 2] > 0.0
        proj_pcd = proj_pcd[..., :2] / proj_pcd[..., 2:]
        mask = np.bitwise_and(mask, np.bitwise_and(proj_pcd[..., 0] >= 0, proj_pcd[..., 0] <= W - 1))
        mask = np.bitwise_and(mask, np.bitwise_and(proj_pcd[..., 1] >= 0, proj_pcd[..., 1] <= H - 1))

        if args.use_depth:
            proj_uv = np.round(proj_pcd[mask]).astype(np.int32)
            depth = depth[mask]
            depth_map = np.zeros((H, W), dtype=np.float32)
            depth_mask = np.zeros((H, W), dtype=np.bool_)
            depth_map[proj_uv[:, 1], proj_uv[:, 0]] = depth
            depth_mask[proj_uv[:, 1], proj_uv[:, 0]] = True
            np.savez(os.path.join(depth_folder, '{:06d}.npz'.format(image_id)), depth=depth_map, mask=depth_mask)
            depth_map = (depth_map - np.min(depth_map)) / (np.max(depth_map) - np.min(depth_map))
            Image.fromarray(np.uint8(np.repeat(depth_map[..., None], axis=-1, repeats=3) * 255.0)).save(os.path.join(depth_folder, '{:06d}.png'.format(image_id)))
        
        if not is_val:
            total_mask = np.bitwise_or(mask, total_mask)
            if args.use_color:
                proj_pts = torch.tensor(proj_pcd, dtype=torch.float32)
                proj_pts[..., 0] /= W
                proj_pts[..., 1] /= H
                proj_pts = proj_pts * 2.0 - 1.0
                img = np.array(Image.open(img_path)) / 255.0
                img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1)
                points_color[mask] += torch.nn.functional.grid_sample(img[None], proj_pts[None, None], align_corners=True).squeeze().permute(1, 0).detach().cpu().numpy()[mask]
                counts += np.float32(mask)


    if not is_val:
        pcd = pcd[total_mask][..., :3, 0]
        if args.use_color:
            points_color = points_color[total_mask]
            counts = counts[total_mask]
        if downsample_ratio < 1.0:
            choice = np.random.permutation(pcd.shape[0])[:int(pcd.shape[0] * downsample_ratio)]
            pcd = pcd[choice]
            if args.use_color:
                points_color = points_color[choice]
                counts = counts[choice]
        pcd = np.concatenate([pcd, np.full((pcd.shape[0], 1), dtype=np.float32, fill_value=idx - first_frame)], axis=-1)
        pointcloud.append(pcd)
        if args.use_color:
            pcd_rgb.append(points_color / counts[..., None])

    cameras_iter = [nusc.get('sample_data', i['next']) for i in cameras_iter]
    process_bar.update(1)

process_bar.close()
pointcloud = np.concatenate(pointcloud, axis=0)
RTs = np.stack(RTs, axis=0)
Ks = np.stack(Ks, axis=0)
time_stamps = np.array(time_stamps, dtype=np.float32)
is_val_list = np.array(is_val_list, dtype=np.bool_)

if args.use_color:
    pcd_rgb = np.concatenate(pcd_rgb, axis=0) * 255
else:
    pcd_rgb = np.random.random((pointcloud.shape[0], 3)) * 255

storePly(os.path.join(dst_path, "points3d.ply"), pointcloud[..., :3], pcd_rgb, t=pointcloud[..., 3:])
np.savez(
    os.path.join(dst_path, 'meta.npz'),
    R = RTs[..., :3, :3],
    T = RTs[..., :3, 3],
    K = Ks,
    time_stamps = time_stamps,
    is_val_list = is_val_list,
)

print("Get PCD:", pointcloud.shape)
print("Get Images and RTs:", RTs.shape[0])
