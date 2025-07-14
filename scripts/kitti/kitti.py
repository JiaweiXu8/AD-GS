import os, shutil
import argparse
import numpy as np
from plyfile import PlyData, PlyElement
from PIL import Image
import torch
import torchvision

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

def get_rotation(roll, pitch, heading):
    s_heading = np.sin(heading)
    c_heading = np.cos(heading)
    rot_z = np.array([[c_heading, -s_heading, 0], [s_heading, c_heading, 0], [0, 0, 1]])

    s_pitch = np.sin(pitch)
    c_pitch = np.cos(pitch)
    rot_y = np.array([[c_pitch, 0, s_pitch], [0, 1, 0], [-s_pitch, 0, c_pitch]])

    s_roll = np.sin(roll)
    c_roll = np.cos(roll)
    rot_x = np.array([[1, 0, 0], [0, c_roll, -s_roll], [0, s_roll, c_roll]])

    rot = np.matmul(rot_z, np.matmul(rot_y, rot_x))

    return rot

def get_val_frames(num_frames, test_every=None, train_every=None):
    assert train_every is None or test_every is None
    if train_every is None:
        val_frames = set(np.arange(test_every, num_frames, test_every))
    else:
        train_frames = set(np.arange(0, num_frames, train_every))
        val_frames = (set(np.arange(num_frames)) - train_frames) if train_every > 1 else train_frames

    return list(val_frames)

parser = argparse.ArgumentParser()
parser.add_argument("src")
parser.add_argument("dst")
parser.add_argument("scene", type=str)
parser.add_argument("--part", default="training")
parser.add_argument("--first_frame", default=65, type=int)
parser.add_argument("--last_frame", default=120, type=int)
parser.add_argument("--downsample_ratio", '-r', default=1.0, type=float)
parser.add_argument("--use_depth", action="store_true")
parser.add_argument("--use_color", action="store_true")
args = parser.parse_args()
dst_path = os.path.join(args.dst, args.scene)
first_frame = args.first_frame
last_frame = args.last_frame
downsample_ratio = args.downsample_ratio

left_image_path = os.path.join(args.src, "data_tracking_image_2", args.part, "image_02", args.scene)
right_image_path = os.path.join(args.src, "data_tracking_image_3", args.part, "image_03", args.scene)
object_meta_path = os.path.join(args.src, "data_tracking_label_2", args.part, "label_02", args.scene + ".txt")
oxts_meta_path = os.path.join(args.src, "data_tracking_oxts", args.part, "oxts", args.scene + ".txt")
calib_meta_path = os.path.join(args.src, "data_tracking_calib", args.part, "calib", args.scene + ".txt")
velodyne_meta_path = os.path.join(args.src, "data_tracking_velodyne", args.part, "velodyne", args.scene)

assert os.path.exists(left_image_path), "Cannot Find: " + left_image_path
assert os.path.exists(right_image_path), "Cannot Find: " + right_image_path
assert os.path.exists(object_meta_path), "Cannot Find: " + object_meta_path
assert os.path.exists(oxts_meta_path), "Cannot Find: " + oxts_meta_path
assert os.path.exists(calib_meta_path), "Cannot Find: " + calib_meta_path
assert os.path.exists(velodyne_meta_path), "Cannot Find: " + velodyne_meta_path

os.makedirs(dst_path, exist_ok=True)
with open(calib_meta_path) as f:
    calib_str = f.read().splitlines()
    calibs = []
    for calibration in calib_str:
        calibs.append(np.array([float(val) for val in calibration.split()[1:]]))

    P0 = np.reshape(calibs[0], [3, 4])
    P1 = np.reshape(calibs[1], [3, 4])
    P2 = np.reshape(calibs[2], [3, 4])
    P3 = np.reshape(calibs[3], [3, 4])

    Tr_cam2camrect = np.eye(4)
    R_rect = np.reshape(calibs[4], [3, 3])
    Tr_cam2camrect[:3, :3] = R_rect
    Tr_velo2cam = np.concatenate([np.reshape(calibs[5], [3, 4]), np.array([[0., 0., 0., 1.]])], axis=0)
    Tr_imu2velo = np.concatenate([np.reshape(calibs[6], [3, 4]), np.array([[0., 0., 0., 1.]])], axis=0)

    camera_meta = {'P0': P0, 'P1': P1, 'P2': P2, 'P3': P3, 'Tr_cam2camrect': Tr_cam2camrect,
            'Tr_velo2cam': Tr_velo2cam, 'Tr_imu2velo': Tr_imu2velo}
focal = camera_meta['P2'][0, 0]
img_temp = Image.open(os.path.join(left_image_path, "000000.png"))
W, H = img_temp.size
print("Height:", H, "Width:", W, "Focal:", focal)

K = np.array([
    [focal, 0.0, W / 2.0],
    [0.0, focal, H / 2.0],
    [0.0, 0.0, 1.0],
], dtype=np.float64)

camrect_cam2 = np.linalg.inv(K) @ camera_meta["P2"]
camrect_cam3 = np.linalg.inv(K) @ camera_meta["P3"]

i_test_25 = get_val_frames(last_frame - first_frame + 1, train_every=4)
i_test_50 = get_val_frames(last_frame - first_frame + 1, test_every=2)
i_test_75 = get_val_frames(last_frame - first_frame + 1, test_every=4)

time_stamp = []
pcd_75 = []
pcd_50 = []
pcd_25 = []
color_75 = []
color_50 = []
color_25 = []
RT = []
rt_0 = None
oxts_tracking = np.loadtxt(oxts_meta_path)
scale = np.cos(oxts_tracking[0][0] * np.pi / 180)
dst_image_path = os.path.join(dst_path, "image")
dst_depth_path = os.path.join(dst_path, "lidar_depth")
os.makedirs(dst_image_path, exist_ok=True)
if args.use_depth:
    os.makedirs(dst_depth_path, exist_ok=True)
T_imu2cam = camera_meta['Tr_cam2camrect'] @ camera_meta['Tr_velo2cam'] @ camera_meta['Tr_imu2velo']
for idx, (left_image, right_image, velodyne, oxt) in enumerate(zip(
    sorted(os.listdir(left_image_path)),
    sorted(os.listdir(right_image_path)),
    sorted(os.listdir(velodyne_meta_path)),
    oxts_tracking
)):
    if idx < first_frame or idx > last_frame:
        continue
    
    shutil.copyfile(os.path.join(left_image_path, left_image), os.path.join(dst_image_path, "{:06d}.png".format(int(2 * (idx - first_frame)))))
    shutil.copyfile(os.path.join(right_image_path, right_image), os.path.join(dst_image_path, "{:06d}.png".format(int(2 * (idx - first_frame) + 1))))

    r = 6378137.0
    x = scale * r * ((np.pi * oxt[1]) / 180)
    y = scale * r * np.log(np.tan((np.pi * (90 + oxt[0])) / 360))
    translation = np.array([x, y, oxt[2]])
    rotation = get_rotation(oxt[3], oxt[4], oxt[5])
    rt_inv = np.eye(4)
    rt_inv[:3] = np.concatenate([rotation, translation[:, None]], axis=-1)
    if idx == first_frame:
        rt_0 = np.linalg.inv(rt_inv)

    rt_inv = np.matmul(rt_0, rt_inv)
    rt = T_imu2cam @ np.linalg.inv(rt_inv)
    # rt[2:3, 3] *= -1

    RT.append(camrect_cam2 @ rt)
    RT.append(camrect_cam3 @ rt)
    time_stamp.append(idx - first_frame)
    time_stamp.append(idx - first_frame)

    vel = np.fromfile(os.path.join(velodyne_meta_path, velodyne), dtype=np.float32, count=-1).reshape(-1, 4)
    vel[..., 3] = 1.0

    points_color = np.zeros((vel.shape[0], 3), dtype=np.float32)
    proj_vel = (camera_meta["P2"] @ camera_meta["Tr_cam2camrect"] @ camera_meta['Tr_velo2cam'] @ vel[..., None]).squeeze(-1)
    left_mask = proj_vel[..., 2] > 0
    depth = proj_vel[..., 2]
    proj_vel = proj_vel[..., :2] / proj_vel[..., 2:]
    left_mask = left_mask & (proj_vel[..., 0] <= W - 1) & (proj_vel[..., 1] <= H - 1) & (proj_vel[..., 0] >= 0.0) & (proj_vel[..., 1] >= 0.0)
    if args.use_depth:
        proj_uv = np.round(proj_vel[left_mask]).astype(np.int32)
        depth = depth[left_mask]
        depth_map = np.zeros((H, W), dtype=np.float32)
        depth_mask = np.zeros((H, W), dtype=np.bool_)
        depth_map[proj_uv[:, 1], proj_uv[:, 0]] = depth
        depth_mask[proj_uv[:, 1], proj_uv[:, 0]] = True
        np.savez(os.path.join(dst_depth_path, "{:06d}.npz".format(int(2 * (idx - first_frame)))), depth=depth_map, mask=depth_mask)
    if args.use_color:
        proj_pts = torch.tensor(proj_vel, dtype=torch.float32, device='cuda')
        proj_pts[..., 0] /= W
        proj_pts[..., 1] /= H
        proj_pts = proj_pts * 2.0 - 1.0
        img = np.array(Image.open(os.path.join(right_image_path, right_image))) / 255.0
        img = torch.tensor(img, dtype=torch.float32, device='cuda').permute(2, 0, 1)
        points_color[left_mask] = torch.nn.functional.grid_sample(img[None], proj_pts[None, None], align_corners=True).squeeze().permute(1, 0).detach().cpu().numpy()[left_mask]

    proj_vel = (camera_meta["P3"] @ camera_meta["Tr_cam2camrect"] @ camera_meta['Tr_velo2cam'] @ vel[..., None]).squeeze(-1)
    right_mask = proj_vel[..., 2] > 0
    depth = proj_vel[..., 2]
    proj_vel = proj_vel[..., :2] / proj_vel[..., 2:]
    right_mask = right_mask & (proj_vel[..., 0] <= W - 1) & (proj_vel[..., 1] <= H - 1) & (proj_vel[..., 0] >= 0.0) & (proj_vel[..., 1] >= 0.0)
    if args.use_depth:
        proj_uv = np.round(proj_vel[right_mask]).astype(np.int32)
        depth = depth[right_mask]
        depth_map = np.zeros((H, W), dtype=np.float32)
        depth_mask = np.zeros((H, W), dtype=np.bool_)
        depth_map[proj_uv[:, 1], proj_uv[:, 0]] = depth
        depth_mask[proj_uv[:, 1], proj_uv[:, 0]] = True
        np.savez(os.path.join(dst_depth_path, "{:06d}.npz".format(int(2 * (idx - first_frame) + 1))), depth=depth_map, mask=depth_mask)
    if args.use_color:
        proj_pts = torch.tensor(proj_vel, dtype=torch.float32, device='cuda')
        proj_pts[..., 0] /= W
        proj_pts[..., 1] /= H
        proj_pts = proj_pts * 2.0 - 1.0
        img = np.array(Image.open(os.path.join(right_image_path, right_image))) / 255.0
        img = torch.tensor(img, dtype=torch.float32, device='cuda').permute(2, 0, 1)
        points_color[right_mask] += torch.nn.functional.grid_sample(img[None], proj_pts[None, None], align_corners=True).squeeze().permute(1, 0).detach().cpu().numpy()[right_mask]

    points_color = points_color / np.clip(np.float32(left_mask) + np.float32(right_mask), a_min=1, a_max=None)[..., None]
    mask = left_mask | right_mask
    vel = vel[mask]
    points_color = points_color[mask]

    # vel = np.linalg.inv(rt) @ camera_meta['Tr_cam2camrect'] @ camera_meta['Tr_velo2cam'] @ vel[..., None]
    vel = (rt_inv @ np.linalg.inv(camera_meta['Tr_imu2velo']) @ vel[..., None]).squeeze(-1)[..., :3]
    if downsample_ratio < 1.0:
        choice = np.random.permutation(vel.shape[0])[:int(vel.shape[0] * downsample_ratio)]
        vel = vel[choice]
        points_color = points_color[choice]
    vel = np.concatenate([vel, np.full((vel.shape[0], 1), fill_value=idx - first_frame, dtype=np.float32)], axis=-1)

    if (idx - first_frame) not in i_test_75:
        pcd_75.append(vel)
        color_75.append(points_color)
    if (idx - first_frame) not in i_test_50:
        pcd_50.append(vel)
        color_50.append(points_color)
    if (idx - first_frame) not in i_test_25:
        pcd_25.append(vel)
        color_25.append(points_color)

RT = np.stack(RT, axis=0)
pcd_75 = np.concatenate(pcd_75, axis=0)
pcd_50 = np.concatenate(pcd_50, axis=0)
pcd_25 = np.concatenate(pcd_25, axis=0)
if args.use_color:
    color_75 = np.concatenate(color_75, axis=0)
    color_50 = np.concatenate(color_50, axis=0)
    color_25 = np.concatenate(color_25, axis=0)
else:
    color_75 = np.random.random((pcd_75.shape[0], 3))
    color_50 = np.random.random((pcd_50.shape[0], 3))
    color_25 = np.random.random((pcd_25.shape[0], 3))
np.savez(
    os.path.join(dst_path, "poses.npz"),
    R = RT[..., :3, :3],
    T = RT[..., :3, 3],
    focal = focal,
    height = H,
    width = W,
    time_stamp = np.array(time_stamp, dtype=np.float64),
)


print("Get PCD:", pcd_75.shape, pcd_50.shape, pcd_25.shape)
print("Get Images and RTs:", RT.shape[0])

storePly(os.path.join(dst_path, "points3d-75.ply"), pcd_75[..., :3], color_75 * 255, t=pcd_75[..., 3:])
storePly(os.path.join(dst_path, "points3d-50.ply"), pcd_50[..., :3], color_50 * 255, t=pcd_50[..., 3:])
storePly(os.path.join(dst_path, "points3d-25.ply"), pcd_25[..., :3], color_25 * 255, t=pcd_25[..., 3:])
