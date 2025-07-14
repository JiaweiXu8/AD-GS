import os
import struct
import shutil
import sqlite3
import argparse
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation
from plyfile import PlyElement, PlyData
from tqdm import tqdm

# try "conda install colmap -c conda-forge" if you have several problem in installing colmap.
# this script is mainly borrowed from StreetGS. https://github.com/zju3dv/street_gaussians

def print_notice(text):
    print("\033[32m{}\033[0m".format(text))

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('path')
    parser.add_argument('--cmd', default='colmap', help='command for colmap')
    parser.add_argument('--use_gpu', action='store_true')
    parser.add_argument('--split_mode', default='nvs-75')
    parser.add_argument('--cam', type=int, default=1)
    args = parser.parse_args()
    return args

def storePly(path, xyz, rgb, t=None, dynamic=None):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    if t is not None:
        dtype.append(('t', 'f4'))
    if dynamic is not None:
        dtype.append(('dy', 'f4'))
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    if t is not None:
        attributes = np.concatenate([attributes, t], axis=-1)
    if dynamic is not None:
        attributes = np.concatenate([attributes, dynamic], axis=-1)
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

def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    """Read and unpack the next bytes from a binary file.
    :param fid:
    :param num_bytes: Sum of combination of {2, 4, 8}, e.g. 2, 6, 16, 30, etc.
    :param format_char_sequence: List of {c, e, f, d, h, H, i, I, l, L, q, Q}.
    :param endian_character: Any of {@, =, <, >, !}
    :return: Tuple of read and unpacked values.
    """
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)

def read_points3D_binary(path_to_model_file):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadPoints3DBinary(const std::string& path)
        void Reconstruction::WritePoints3DBinary(const std::string& path)
    """


    with open(path_to_model_file, "rb") as fid:
        num_points = read_next_bytes(fid, 8, "Q")[0]

        xyzs = np.empty((num_points, 3))
        rgbs = np.empty((num_points, 3))
        errors = np.empty((num_points, 1))

        for p_id in range(num_points):
            binary_point_line_properties = read_next_bytes(
                fid, num_bytes=43, format_char_sequence="QdddBBBd")
            xyz = np.array(binary_point_line_properties[1:4])
            rgb = np.array(binary_point_line_properties[4:7])
            error = np.array(binary_point_line_properties[7])
            track_length = read_next_bytes(
                fid, num_bytes=8, format_char_sequence="Q")[0]
            track_elems = read_next_bytes(
                fid, num_bytes=8*track_length,
                format_char_sequence="ii"*track_length)
            xyzs[p_id] = xyz
            rgbs[p_id] = rgb
            errors[p_id] = error
    return xyzs, rgbs, errors

def run_colmap(colmap_path, camera_meta, colmap_cmd='colmap', use_gpu=False, cam_num=1):
    mask_dir = os.path.join(colmap_path, 'masks')
    image_dir = os.path.join(colmap_path, 'images')
    assert os.path.exists(image_dir), 'Cannot find ' + image_dir
    assert os.path.exists(mask_dir), 'Cannot find ' + mask_dir
    print(image_dir)

    print_notice('Feature Extraction')
    ret = os.system(
        f'{colmap_cmd} feature_extractor \
        --ImageReader.mask_path {mask_dir} \
        --ImageReader.camera_model SIMPLE_PINHOLE  \
        --ImageReader.single_camera_per_folder 1 \
        --database_path {colmap_path}/database.db \
        --image_path {image_dir}' + \
        ' --SiftExtraction.use_gpu 0' if not use_gpu else ''
    )
    assert ret == 0, 'There might be several mistakes in feature extraction.'
    print_notice('Feature Extraction Done')

    print_notice('Process camera meta')
    model_dir = os.path.join(colmap_path, 'created/sparse/model')
    os.makedirs(model_dir, exist_ok=True)

    db_connect = sqlite3.connect(os.path.join(colmap_path, 'database.db'))
    c = db_connect.cursor()
    c.execute('SELECT * FROM images')
    with open(os.path.join(model_dir, 'images.txt'), 'w') as f:
        R = camera_meta['R']
        T = camera_meta['T']
        for meta in c.fetchall():
            img_id, img_name, cam_id = meta[0], meta[1], meta[2]
            idx = int(img_name.split('/')[-1].split('.')[0])
            R_quat = Rotation.from_matrix(R[idx]).as_quat()  # The returned value is in scalar-last (x, y, z, w) format.
            R_quat[0], R_quat[1], R_quat[2], R_quat[3] = R_quat[3], R_quat[0], R_quat[1], R_quat[2]
            rt = np.concatenate([R_quat, T[idx]], axis=0)
            f.write(f'{img_id} ' + ' '.join([str(a) for a in rt.tolist()]) + f' {idx % cam_num} {img_name}\n\n')
    
    with open(os.path.join(model_dir, 'cameras.txt'), 'w') as f:
        for cam_id in range(cam_num):
            cx = camera_meta['cx'][cam_id]
            cy = camera_meta['cy'][cam_id]
            fx = camera_meta['fx'][cam_id]
            f.write(f'{cam_id} SIMPLE_PINHOLE {int(cx * 2)} {int(cy * 2)} {fx} {cx} {cy}\n')
            params = np.array([fx, cx, cy]).astype(np.float64)
            c.execute("UPDATE cameras SET params = ? WHERE camera_id = ?", (params.tostring(), cam_id + 1))  # colmap start cam_id from 1
        
    db_connect.commit()
    db_connect.close()
    print_notice('Process camera meta Done')

    print_notice('Exhaustive Match')
    ret = os.system(
        f'{colmap_cmd} exhaustive_matcher \
        --database_path {colmap_path}/database.db' + \
        ' --SiftMatching.use_gpu 0' if not use_gpu else ""
    )
    assert ret == 0, 'There might be several mistakes in exhaustive match'
    print_notice('Exhaustive Match Done')

    print_notice('Point Triangulate')
    triangulated_dir = os.path.join(colmap_path, 'triangulated/sparse/model')
    os.makedirs(triangulated_dir, exist_ok=True)
    os.system('touch {}'.format(os.path.join(model_dir, 'points3D.txt')))
    ret = os.system(f'{colmap_cmd} point_triangulator \
        --database_path {colmap_path}/database.db \
        --image_path {image_dir} \
        --input_path {model_dir} \
        --output_path {triangulated_dir} \
        --Mapper.ba_refine_focal_length 0 \
        --Mapper.ba_refine_principal_point 0 \
        --Mapper.max_extra_param 0 \
        --clear_points 0 \
        --Mapper.ba_global_max_num_iterations 30 \
        --Mapper.filter_max_reproj_error 4 \
        --Mapper.filter_min_tri_angle 0.5 \
        --Mapper.tri_min_angle 0.5 \
        --Mapper.tri_ignore_two_view_tracks 1 \
        --Mapper.tri_complete_max_reproj_error 4 \
        --Mapper.tri_continue_max_angle_error 4')
    assert ret == 0, 'There might be several mistakes in point triangulate'
    print_notice('Point Triangulate Done')

def prepare_colmap_meta_waymo(path, colmap_path, num_cam=1):
    colmap_image_dir = os.path.join(colmap_path, 'images')
    os.makedirs(colmap_image_dir, exist_ok=True)
    colmap_mask_dir = os.path.join(colmap_path, 'masks')
    os.makedirs(colmap_mask_dir, exist_ok=True)

    for i in range(num_cam):
        os.makedirs(os.path.join(colmap_image_dir, f"{i}"), exist_ok=True)
        os.makedirs(os.path.join(colmap_mask_dir, f"{i}"), exist_ok=True)

    image_path = os.path.join(path, 'image')
    meta = np.load(os.path.join(path, "cameras.npz"), allow_pickle=True)
    K, R, T = meta['K'], meta['R'], meta['T']
    time_stamps = meta['time_stamps']
    is_val_list = meta['is_val_list']
    cur_idx = 0
    for idx, img_path in enumerate(tqdm(list(sorted(os.listdir(image_path))), desc='Reading')):
        if is_val_list[idx]:
            continue
        cam_id = idx % num_cam
        shutil.copy(os.path.join(image_path, img_path), os.path.join(colmap_image_dir, f"{cam_id}", '{:06d}.jpg'.format(cur_idx)))
        semantic_path = os.path.join(path, "semantic", 'mask_' + img_path.split(".")[0] + ".npy")
        sky_path = os.path.join(path, "sky", 'mask_' + img_path.split(".")[0] + ".npy")
        semantic_mask = np.load(semantic_path) == 0
        sky_mask = np.load(sky_path) == 0
        mask = np.logical_and(semantic_mask, sky_mask)[..., None]
        # mask = semantic_mask[..., None]
        mask = np.uint8(np.repeat(mask, 3, axis=-1) * 255)
        Image.fromarray(mask).save(os.path.join(colmap_mask_dir, f"{cam_id}", '{:06d}.jpg'.format(cur_idx)))
        cur_idx += 1
    select_list = np.logical_not(is_val_list)
    return {
        'cx': K[select_list, 2],
        'cy': K[select_list, 3],
        'fx': K[select_list, 0],
        'fy': K[select_list, 1],
        'R': R[select_list],
        'T': T[select_list],
    }

def prepare_colmap_meta_kitti(path, colmap_path, split_mode='nvs-75', num_cam=2):
    colmap_image_dir = os.path.join(colmap_path, 'images')
    os.makedirs(colmap_image_dir, exist_ok=True)
    colmap_mask_dir = os.path.join(colmap_path, 'masks')
    os.makedirs(colmap_mask_dir, exist_ok=True)

    for i in range(num_cam):
        os.makedirs(os.path.join(colmap_image_dir, f"{i}"), exist_ok=True)
        os.makedirs(os.path.join(colmap_mask_dir, f"{i}"), exist_ok=True)

    image_path = os.path.join(path, 'image')
    meta = np.load(os.path.join(path, "poses.npz"), allow_pickle=True)
    R, T = meta['R'], meta['T']
    height = float(meta['height'])
    width = float(meta['width'])
    focal = meta['focal']

    if split_mode == 'nvs-25':
        i_test = get_val_frames(R.shape[0] // 2, train_every=4)
    elif split_mode == 'nvs-50':
        i_test = get_val_frames(R.shape[0] // 2, test_every=2)
    elif split_mode == 'nvs-75':
        i_test = get_val_frames(R.shape[0] // 2, test_every=4)
    else:
        raise ValueError("No such split method: " + split_mode)
    
    indices = []
    cur_idx = 0
    for idx, img_path in enumerate(tqdm(list(sorted(os.listdir(image_path))), desc='Reading')):
        if idx // 2 in i_test:
            continue
        cam_id = idx % num_cam
        shutil.copy(os.path.join(image_path, img_path), os.path.join(colmap_image_dir, f"{cam_id}", '{:06d}.png'.format(cur_idx)))
        semantic_path = os.path.join(path, "semantic", 'mask_' + img_path.split(".")[0] + ".npy")
        sky_path = os.path.join(path, "sky", 'mask_' + img_path.split(".")[0] + ".npy")
        semantic_mask = np.load(semantic_path) == 0
        sky_mask = np.load(sky_path) == 0
        mask = np.logical_and(semantic_mask, sky_mask)[..., None]
        # mask = semantic_mask[..., None]
        mask = np.uint8(np.repeat(mask, 3, axis=-1) * 255)
        Image.fromarray(mask).save(os.path.join(colmap_mask_dir, f"{cam_id}", '{:06d}.png'.format(cur_idx)))
        cur_idx += 1
        indices.append(idx)

    
    return {
        'cx': np.full((len(indices)), fill_value=width / 2),
        'cy': np.full((len(indices)), fill_value=height / 2),
        'fx': np.full((len(indices)), fill_value=focal),
        'fy': np.full((len(indices)), fill_value=focal),
        'R': R[indices],
        'T': T[indices],
    }

def prepare_colmap_meta_nuscenes(path, colmap_path, num_cam=3):
    colmap_image_dir = os.path.join(colmap_path, 'images')
    os.makedirs(colmap_image_dir, exist_ok=True)
    colmap_mask_dir = os.path.join(colmap_path, 'masks')
    os.makedirs(colmap_mask_dir, exist_ok=True)

    for i in range(num_cam):
        os.makedirs(os.path.join(colmap_image_dir, f"{i}"), exist_ok=True)
        os.makedirs(os.path.join(colmap_mask_dir, f"{i}"), exist_ok=True)

    image_path = os.path.join(path, 'image')
    meta = np.load(os.path.join(path, "meta.npz"), allow_pickle=True)
    K, R, T = meta['K'], meta['R'], meta['T']
    time_stamps = meta['time_stamps']
    is_val_list = meta['is_val_list']
    cur_idx = 0
    for idx, img_path in enumerate(tqdm(list(sorted(os.listdir(image_path))), desc='Reading')):
        if is_val_list[idx]:
            continue
        cam_id = idx % num_cam
        shutil.copy(os.path.join(image_path, img_path), os.path.join(colmap_image_dir, f"{cam_id}", '{:06d}.png'.format(cur_idx)))
        semantic_path = os.path.join(path, "semantic", 'mask_' + img_path.split(".")[0] + ".npy")
        sky_path = os.path.join(path, "sky", 'mask_' + img_path.split(".")[0] + ".npy")
        semantic_mask = np.load(semantic_path) == 0
        sky_mask = np.load(sky_path) == 0
        mask = np.logical_and(semantic_mask, sky_mask)[..., None]
        # mask = semantic_mask[..., None]
        mask = np.uint8(np.repeat(mask, 3, axis=-1) * 255)
        Image.fromarray(mask).save(os.path.join(colmap_mask_dir, f"{cam_id}", '{:06d}.png'.format(cur_idx)))
        cur_idx += 1
    select_list = np.logical_not(is_val_list)
    return {
        'cx': K[select_list, 0, 2],
        'cy': K[select_list, 1, 2],
        'fx': K[select_list, 0, 0],
        'fy': K[select_list, 1, 1],
        'R': R[select_list],
        'T': T[select_list],
    }

if __name__ == '__main__':
    args = get_args()
    colmap_dir = os.path.join(args.path, 'colmap')

    ply_path = os.path.join(args.path, 'colmap.ply')
    if os.path.exists(os.path.join(args.path, "cameras.npz")):
        os.makedirs(colmap_dir, exist_ok=True)
        print("Found cameras.npz file, assuming Waymo data set!")
        camera_meta = prepare_colmap_meta_waymo(args.path, colmap_dir, num_cam=args.cam)
    elif os.path.exists(os.path.join(args.path, 'poses.npz')):
        print('Found poses.npz file, assuming KITTI or vKITTI data set!')
        colmap_dir = os.path.join(args.path, 'colmap-{}'.format(args.split_mode.split('-')[-1]))
        os.makedirs(colmap_dir, exist_ok=True)
        camera_meta = prepare_colmap_meta_kitti(args.path, colmap_dir, split_mode=args.split_mode, num_cam=args.cam)
        ply_path = os.path.join(args.path, 'colmap-{}.ply'.format(args.split_mode.split('-')[-1]))
    elif os.path.exists(os.path.join(args.path, "meta.npz")):
        os.makedirs(colmap_dir, exist_ok=True)
        print("Found meta.npz file, assuming nuScenes data set!")
        camera_meta = prepare_colmap_meta_nuscenes(args.path, colmap_dir, num_cam=args.cam)
    else:
        assert False, 'Could not recognize scene type!'

    run_colmap(colmap_path=colmap_dir, camera_meta=camera_meta, colmap_cmd=args.cmd, use_gpu=args.use_gpu, cam_num=args.cam)
    xyz, rgb, _ = read_points3D_binary(os.path.join(colmap_dir, 'triangulated/sparse/model/points3D.bin'))
    storePly(ply_path, xyz=xyz, rgb=rgb)
    print('SfM pointcloud:', ply_path, 'pts:', xyz.shape[0])