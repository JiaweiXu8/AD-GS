import os
import argparse
import numpy as np
import torch
from plyfile import PlyData, PlyElement
import tensorflow as tf
from waymo_open_dataset.utils import range_image_utils, transform_utils
from waymo_open_dataset.utils.frame_utils import parse_range_image_and_camera_projection
from waymo_open_dataset import dataset_pb2
import tqdm
from PIL import Image
os.environ['CUDA_VISIBLE_DEVICES'] = ""

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

def parse_range_image_and_camera_projection(frame):
    """Parse range images and camera projections given a frame.

    Args:
    frame: open dataset frame proto

    Returns:
    range_images: A dict of {laser_name,
        [range_image_first_return, range_image_second_return]}.
    camera_projections: A dict of {laser_name,
        [camera_projection_from_first_return,
        camera_projection_from_second_return]}.
    seg_labels: segmentation labels, a dict of {laser_name,
        [seg_label_first_return, seg_label_second_return]}
    range_image_top_pose: range image pixel pose for top lidar.
    """
    range_images = {}
    range_image_top_pose = None
    for laser in frame.lasers:
        if len(laser.ri_return1.range_image_compressed) > 0:  # pylint: disable=g-explicit-length-test
            range_image_str_tensor = tf.io.decode_compressed(
                laser.ri_return1.range_image_compressed, 'ZLIB')
            ri = dataset_pb2.MatrixFloat()
            ri.ParseFromString(bytearray(range_image_str_tensor.numpy()))
            range_images[laser.name] = [ri]

            if laser.name == dataset_pb2.LaserName.TOP:
                range_image_top_pose_str_tensor = tf.io.decode_compressed(
                    laser.ri_return1.range_image_pose_compressed, 'ZLIB')
                range_image_top_pose = dataset_pb2.MatrixFloat()
                range_image_top_pose.ParseFromString(
                    bytearray(range_image_top_pose_str_tensor.numpy()))

        if len(laser.ri_return2.range_image_compressed) > 0:  # pylint: disable=g-explicit-length-test
            range_image_str_tensor = tf.io.decode_compressed(
                laser.ri_return2.range_image_compressed, 'ZLIB')
            ri = dataset_pb2.MatrixFloat()
            ri.ParseFromString(bytearray(range_image_str_tensor.numpy()))
            range_images[laser.name].append(ri)
    return range_images, range_image_top_pose

def convert_range_image_to_point_cloud_flow(
    frame,
    range_images,
    range_image_top_pose,
    ri_index=0,
):
    """
    Modified from the codes of Waymo Open Dataset.
    Convert range images to point cloud.
    Convert range images flow to scene flow.
    Args:
        frame: open dataset frame
        range_images: A dict of {laser_name, [range_image_first_return, range_image_second_return]}.
        range_imaages_flow: A dict similar to range_images.
        camera_projections: A dict of {laser_name,
            [camera_projection_from_first_return, camera_projection_from_second_return]}.
        range_image_top_pose: range image pixel pose for top lidar.
        ri_index: 0 for the first return, 1 for the second return.

    Returns:
        points: {[N, 3]} list of 3d lidar points of length 5 (number of lidars).
        points_flow: {[N, 3]} list of scene flow vector of each point.
        cp_points: {[N, 6]} list of camera projections of length 5 (number of lidars).
    """
    calibrations = sorted(frame.context.laser_calibrations, key=lambda c: c.name)
    points = []

    frame_pose = tf.convert_to_tensor(
        np.reshape(np.array(frame.pose.transform), [4, 4])
    )
    # [H, W, 6]
    range_image_top_pose_tensor = tf.reshape(
        tf.convert_to_tensor(range_image_top_pose.data), range_image_top_pose.shape.dims
    )
    # [H, W, 3, 3]
    range_image_top_pose_tensor_rotation = transform_utils.get_rotation_matrix(
        range_image_top_pose_tensor[..., 0],
        range_image_top_pose_tensor[..., 1],
        range_image_top_pose_tensor[..., 2],
    )
    range_image_top_pose_tensor_translation = range_image_top_pose_tensor[..., 3:]
    range_image_top_pose_tensor = transform_utils.get_transform(
        range_image_top_pose_tensor_rotation, range_image_top_pose_tensor_translation
    )
    for c in calibrations:
        range_image = range_images[c.name][ri_index]
        if len(c.beam_inclinations) == 0:  # pylint: disable=g-explicit-length-test
            beam_inclinations = range_image_utils.compute_inclination(
                tf.constant([c.beam_inclination_min, c.beam_inclination_max]),
                height=range_image.shape.dims[0],
            )
        else:
            beam_inclinations = tf.constant(c.beam_inclinations)

        beam_inclinations = tf.reverse(beam_inclinations, axis=[-1])
        extrinsic = np.reshape(np.array(c.extrinsic.transform), [4, 4])

        range_image_tensor = tf.reshape(
            tf.convert_to_tensor(range_image.data), range_image.shape.dims
        )
        pixel_pose_local = None
        frame_pose_local = None
        if c.name == dataset_pb2.LaserName.TOP:
            pixel_pose_local = range_image_top_pose_tensor
            pixel_pose_local = tf.expand_dims(pixel_pose_local, axis=0)
            frame_pose_local = tf.expand_dims(frame_pose, axis=0)
        range_image_mask = range_image_tensor[..., 0] > 0

        mask_index = tf.where(range_image_mask)

        (points_cartesian,) = extract_point_cloud_from_range_image(
            tf.expand_dims(range_image_tensor[..., 0], axis=0),
            tf.expand_dims(extrinsic, axis=0),
            tf.expand_dims(tf.convert_to_tensor(beam_inclinations), axis=0),
            pixel_pose=pixel_pose_local,
            frame_pose=frame_pose_local,
        )
        # points_cartesian = tf.squeeze(points_cartesian, axis=0)

        points_tensor = tf.gather_nd(points_cartesian, mask_index)

        points.append(points_tensor.numpy())

    return points

def extract_point_cloud_from_range_image(
    range_image,
    extrinsic,
    inclination,
    pixel_pose=None,
    frame_pose=None,
    dtype=tf.float32,
    scope=None,
):
    """Extracts point cloud from range image.

    Args:
      range_image: [B, H, W] tensor. Lidar range images.
      extrinsic: [B, 4, 4] tensor. Lidar extrinsic.
      inclination: [B, H] tensor. Inclination for each row of the range image.
        0-th entry corresponds to the 0-th row of the range image.
      pixel_pose: [B, H, W, 4, 4] tensor. If not None, it sets pose for each range
        image pixel.
      frame_pose: [B, 4, 4] tensor. This must be set when pixel_pose is set. It
        decides the vehicle frame at which the cartesian points are computed.
      dtype: float type to use internally. This is needed as extrinsic and
        inclination sometimes have higher resolution than range_image.
      scope: the name scope.

    Returns:
      range_image_points: [B, H, W, 3] with {x, y, z} as inner dims in vehicle frame.
      range_image_origins: [B, H, W, 3] with {x, y, z}, the origin of the range image
    """
    with tf.compat.v1.name_scope(
        scope,
        "ExtractPointCloudFromRangeImage",
        [range_image, extrinsic, inclination, pixel_pose, frame_pose],
    ):
        range_image_polar = range_image_utils.compute_range_image_polar(
            range_image, extrinsic, inclination, dtype=dtype
        )
        range_image_points_cartesian = compute_range_image_cartesian(
            range_image_polar,
            extrinsic,
            pixel_pose=pixel_pose,
            frame_pose=frame_pose,
            dtype=dtype,
        )
        return range_image_points_cartesian
    
def compute_range_image_cartesian(
    range_image_polar,
    extrinsic,
    pixel_pose=None,
    frame_pose=None,
    dtype=tf.float32,
    scope=None,
):
    """Computes range image cartesian coordinates from polar ones.

    Args:
      range_image_polar: [B, H, W, 3] float tensor. Lidar range image in polar
        coordinate in sensor frame.
      extrinsic: [B, 4, 4] float tensor. Lidar extrinsic.
      pixel_pose: [B, H, W, 4, 4] float tensor. If not None, it sets pose for each
        range image pixel.
      frame_pose: [B, 4, 4] float tensor. This must be set when pixel_pose is set.
        It decides the vehicle frame at which the cartesian points are computed.
      dtype: float type to use internally. This is needed as extrinsic and
        inclination sometimes have higher resolution than range_image.
      scope: the name scope.

    Returns:
      range_image_cartesian: [B, H, W, 3] cartesian coordinates.
    """
    range_image_polar_dtype = range_image_polar.dtype
    range_image_polar = tf.cast(range_image_polar, dtype=dtype)
    extrinsic = tf.cast(extrinsic, dtype=dtype)
    if pixel_pose is not None:
        pixel_pose = tf.cast(pixel_pose, dtype=dtype)
    if frame_pose is not None:
        frame_pose = tf.cast(frame_pose, dtype=dtype)

    with tf.compat.v1.name_scope(
        scope,
        "ComputeRangeImageCartesian",
        [range_image_polar, extrinsic, pixel_pose, frame_pose],
    ):
        azimuth, inclination, range_image_range = tf.unstack(range_image_polar, axis=-1)

        cos_azimuth = tf.cos(azimuth)
        sin_azimuth = tf.sin(azimuth)
        cos_incl = tf.cos(inclination)
        sin_incl = tf.sin(inclination)

        # [B, H, W].
        x = cos_azimuth * cos_incl * range_image_range
        y = sin_azimuth * cos_incl * range_image_range
        z = sin_incl * range_image_range

        # [B, H, W, 3]
        range_image_points = tf.stack([x, y, z], -1)
        # [B, 3, 3]
        rotation = extrinsic[..., 0:3, 0:3]
        # translation [B, 1, 3]
        translation = tf.expand_dims(tf.expand_dims(extrinsic[..., 0:3, 3], 1), 1)

        # To vehicle frame.
        # [B, H, W, 3]
        range_image_points = (
            tf.einsum("bkr,bijr->bijk", rotation, range_image_points) + translation
        )
        if pixel_pose is not None:
            # To global frame.
            # [B, H, W, 3, 3]
            pixel_pose_rotation = pixel_pose[..., 0:3, 0:3]
            # [B, H, W, 3]
            pixel_pose_translation = pixel_pose[..., 0:3, 3]
            # [B, H, W, 3]
            range_image_points = (
                tf.einsum("bhwij,bhwj->bhwi", pixel_pose_rotation, range_image_points)
                + pixel_pose_translation
            )

            if frame_pose is None:
                raise ValueError("frame_pose must be set when pixel_pose is set.")
            # To vehicle frame corresponding to the given frame_pose
            # [B, 4, 4]
            world_to_vehicle = tf.linalg.inv(frame_pose)
            world_to_vehicle_rotation = world_to_vehicle[:, 0:3, 0:3]
            world_to_vehicle_translation = world_to_vehicle[:, 0:3, 3]
            # [B, H, W, 3]
            range_image_points = (
                tf.einsum(
                    "bij,bhwj->bhwi", world_to_vehicle_rotation, range_image_points
                )
                + world_to_vehicle_translation[:, tf.newaxis, tf.newaxis, :]
            )

        range_image_points = tf.cast(range_image_points, dtype=range_image_polar_dtype)
        return range_image_points


parser = argparse.ArgumentParser()
parser.add_argument("src")
parser.add_argument("dst")
parser.add_argument("--part", default="training")
parser.add_argument("--first_frame", default=65, type=int)
parser.add_argument("--last_frame", default=120, type=int)
parser.add_argument("--downsample_ratio", '-r', default=1.0, type=float)
parser.add_argument("--select_camera", default=[0], type=int, nargs='+')
parser.add_argument("--use_color", action='store_true')
parser.add_argument("--use_depth", action='store_true')
args = parser.parse_args()
first_frame = args.first_frame
last_frame = args.last_frame
dst_path = args.dst
downsample_ratio = args.downsample_ratio
OPENCV2DATASET = np.array([[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]])

image_folder = os.path.join(dst_path, 'image')
os.makedirs(image_folder, exist_ok=True)

if args.use_depth:
    depth_folder = os.path.join(dst_path, 'lidar_depth')
    os.makedirs(depth_folder, exist_ok=True)

ego_0 = None
RTs = []
Ks = []
pcd = []
pcd_rgb = []
time_stamps = []
is_val_list = []
dataset = tf.data.TFRecordDataset(args.src, compression_type="")
if last_frame == -1:
    last_frame = len([_ for _ in dataset]) - 1
val_fid_list = get_val_frames(last_frame - first_frame + 1, 4)
process_bar = tqdm.tqdm(range(last_frame - first_frame + 1), desc="Processing")
for fid, data in enumerate(dataset):
    if fid < first_frame or (last_frame != -1 and fid > last_frame):
        continue

    frame = dataset_pb2.Frame()
    frame.ParseFromString(bytearray(data.numpy()))
    ego_to_world = np.array(frame.pose.transform).reshape(4, 4)
    if fid == first_frame:
        ego_0 = np.linalg.inv(ego_to_world)
    ego_to_world = ego_0 @ ego_to_world
    is_val = (fid - first_frame) in val_fid_list

    range_images, range_image_top_pose = parse_range_image_and_camera_projection(frame)

    # https://github.com/waymo-research/waymo-open-dataset/blob/master/src/waymo_open_dataset/protos/segmentation.proto
    if range_image_top_pose is None:
        print("the camera only split doesn't contain lidar points.")
        continue

    # collect first return only
    points = convert_range_image_to_point_cloud_flow(
        frame,
        range_images,
        range_image_top_pose,
        ri_index=0,
    )
    points = np.concatenate(points, axis=0)
    points = ego_to_world[:3, :3] @ points[..., None] + ego_to_world[:3, 3:]

    mask_total = np.full((points.shape[0],), dtype=np.bool_, fill_value=False)
    points_color = np.zeros((points.shape[0], 3), dtype=np.float32)
    counts = 0
    # for idx, (img, cam) in enumerate(zip(frame.images, frame.context.camera_calibrations)):
    for idx, img in enumerate(frame.images):
        # if idx not in args.select_camera:
        if img.name - 1 not in args.select_camera:
            continue

        cam = [c for c in frame.context.camera_calibrations if c.name == img.name][0]

        image_id = len(RTs)
        img_path = os.path.join(image_folder, '{:06d}.jpg'.format(image_id))
        with open(img_path, 'wb') as f:
            f.write(img.image)
        Ks.append(np.array(cam.intrinsic))
        K = np.array([
            [cam.intrinsic[0], 0.0, cam.intrinsic[2]],
            [0.0, cam.intrinsic[1], cam.intrinsic[3]],
            [0.0, 0.0, 1.0],
        ], dtype=np.float32)

        # cam_ego_to_world = np.array(img.pose.transform).reshape(4, 4)
        # cam_ego_to_world = ego_0 @ cam_ego_to_world
        # RT_inv = cam_ego_to_world @ np.array(cam.extrinsic.transform).reshape(4, 4) @ OPENCV2DATASET

        RT_inv = ego_to_world @ np.array(cam.extrinsic.transform).reshape(4, 4) @ OPENCV2DATASET
        RT = np.linalg.inv(RT_inv)
        RTs.append(RT)
        time_stamps.append(fid - first_frame)
        is_val_list.append(is_val)

        proj_pts = (K @ (RT[:3, :3] @ points + RT[:3, 3:])).squeeze(-1)
        mask = (proj_pts[:, 2] > 0.0)
        depth = proj_pts[..., 2]
        proj_pts = proj_pts[..., :2] / proj_pts[..., 2:]
        W, H = Image.open(img_path).size
        mask = np.bitwise_and(mask, np.bitwise_and(proj_pts[..., 0] >= 0.0, proj_pts[..., 0] <= W - 1))
        mask = np.bitwise_and(mask, np.bitwise_and(proj_pts[..., 1] >= 0.0, proj_pts[..., 1] <= H - 1))
        if args.use_depth:
            proj_uv = np.round(proj_pts[mask]).astype(np.int32)
            depth = depth[mask]
            depth_map = np.zeros((H, W), dtype=np.float32)
            depth_mask = np.zeros((H, W), dtype=np.bool_)
            depth_map[proj_uv[:, 1], proj_uv[:, 0]] = depth
            depth_mask[proj_uv[:, 1], proj_uv[:, 0]] = True
            np.savez(os.path.join(depth_folder, '{:06d}.npz'.format(image_id)), depth=depth_map, mask=depth_mask)

        if not is_val:
            mask_total = np.bitwise_or(mask, mask_total)
            if args.use_color:
                proj_pts = torch.tensor(proj_pts, dtype=torch.float32)
                proj_pts[..., 0] /= W
                proj_pts[..., 1] /= H
                proj_pts = proj_pts * 2.0 - 1.0
                img = np.array(Image.open(img_path)) / 255.0
                img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1)
                points_color[mask] += torch.nn.functional.grid_sample(img[None], proj_pts[None, None], align_corners=True).squeeze().permute(1, 0).detach().cpu().numpy()[mask]
                counts += np.float32(mask)

    if not is_val:
        points = points.squeeze(-1)[mask_total]
        if args.use_color:
            points_color = points_color[mask_total]
            counts = counts[mask_total]
        if downsample_ratio < 1.0:
            choice = np.random.permutation(points.shape[0])[:int(points.shape[0] * downsample_ratio)]
            points = points[choice]
            if args.use_color:
                points_color = points_color[choice]
                counts = counts[choice]
        points = np.concatenate([points, np.full((points.shape[0], 1), dtype=np.float32, fill_value=fid - first_frame)], axis=-1)
        pcd.append(points)
        if args.use_color:
            pcd_rgb.append(points_color / counts[..., None])
    process_bar.update(1)
process_bar.close()

pcd = np.concatenate(pcd, axis=0)
RTs = np.stack(RTs, axis=0)
Ks = np.stack(Ks, axis=0)
is_val_list = np.array(is_val_list, dtype=np.bool_)
time_stamps = np.array(time_stamps, dtype=np.float32)

if args.use_color:
    pcd_rgb = np.concatenate(pcd_rgb, axis=0) * 255.0
else:
    pcd_rgb = np.random.random((pcd.shape[0], 3)) * 255.0
storePly(os.path.join(dst_path, "points3d.ply"), pcd[..., :3], pcd_rgb, t=pcd[..., 3:])
np.savez(
    os.path.join(dst_path, 'cameras.npz'),
    R = RTs[..., :3, :3],
    T = RTs[..., :3, 3],
    K = Ks,
    time_stamps = time_stamps,
    is_val_list = is_val_list
)

print("Get PCD:", pcd.shape)
print("Get Images and RTs:", RTs.shape[0])