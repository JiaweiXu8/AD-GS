
import torch.hub
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import numpy as np
import imageio
import torch

from matplotlib import cm
import torch.nn.functional as F
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import argparse
import tqdm
import flow_vis

def get_val_frames(num_frames, test_every=None, train_every=None):
    assert train_every is None or test_every is None
    if train_every is None:
        val_frames = set(np.arange(test_every, num_frames, test_every))
    else:
        train_frames = set(np.arange(0, num_frames, train_every))
        val_frames = (set(np.arange(num_frames)) - train_frames) if train_every > 1 else train_frames

    return list(val_frames)

def read_video_from_path(path):
    try:
        reader = imageio.get_reader(path)
    except Exception as e:
        print("Error opening video file: ", e)
        return None
    frames = []
    for i, im in enumerate(reader):
        frames.append(np.array(im))
    return np.stack(frames)


def draw_circle(rgb, coord, radius, color=(255, 0, 0), visible=True, color_alpha=None):
    # Create a draw object
    draw = ImageDraw.Draw(rgb)
    # Calculate the bounding box of the circle
    left_up_point = (coord[0] - radius, coord[1] - radius)
    right_down_point = (coord[0] + radius, coord[1] + radius)
    # Draw the circle
    color = tuple(list(color) + [color_alpha if color_alpha is not None else 255])

    draw.ellipse(
        [left_up_point, right_down_point],
        fill=tuple(color) if visible else None,
        outline=tuple(color),
    )
    return rgb


def draw_line(rgb, coord_y, coord_x, color, linewidth):
    draw = ImageDraw.Draw(rgb)
    draw.line(
        (coord_y[0], coord_y[1], coord_x[0], coord_x[1]),
        fill=tuple(color),
        width=linewidth,
    )
    return rgb


def add_weighted(rgb, alpha, original, beta, gamma):
    return (rgb * alpha + original * beta + gamma).astype("uint8")


class Visualizer:
    def __init__(
        self,
        save_dir: str = "./results",
        grayscale: bool = False,
        pad_value: int = 0,
        fps: int = 10,
        mode: str = "rainbow",  # 'cool', 'optical_flow'
        linewidth: int = 2,
        show_first_frame: int = 10,
        tracks_leave_trace: int = 0,  # -1 for infinite
    ):
        self.mode = mode
        self.save_dir = save_dir
        if mode == "rainbow":
            self.color_map = cm.get_cmap("gist_rainbow")
        elif mode == "cool":
            self.color_map = cm.get_cmap(mode)
        self.show_first_frame = show_first_frame
        self.grayscale = grayscale
        self.tracks_leave_trace = tracks_leave_trace
        self.pad_value = pad_value
        self.linewidth = linewidth
        self.fps = fps

    def visualize(
        self,
        video: torch.Tensor,  # (B,T,C,H,W)
        tracks: torch.Tensor,  # (B,T,N,2)
        visibility: torch.Tensor = None,  # (B, T, N, 1) bool
        gt_tracks: torch.Tensor = None,  # (B,T,N,2)
        segm_mask: torch.Tensor = None,  # (B,1,H,W)
        filename: str = "video",
        writer=None,  # tensorboard Summary Writer, used for visualization during training
        step: int = 0,
        query_frame=0,
        save_video: bool = True,
        compensate_for_camera_motion: bool = False,
        opacity: float = 1.0,
    ):
        if compensate_for_camera_motion:
            assert segm_mask is not None
        if segm_mask is not None:
            coords = tracks[0, query_frame].round().long()
            segm_mask = segm_mask[0, query_frame][coords[:, 1], coords[:, 0]].long()

        video = F.pad(
            video,
            (self.pad_value, self.pad_value, self.pad_value, self.pad_value),
            "constant",
            255,
        )
        color_alpha = int(opacity * 255)
        tracks = tracks + self.pad_value

        if self.grayscale:
            transform = transforms.Grayscale()
            video = transform(video)
            video = video.repeat(1, 1, 3, 1, 1)

        res_video = self.draw_tracks_on_video(
            video=video,
            tracks=tracks,
            visibility=visibility,
            segm_mask=segm_mask,
            gt_tracks=gt_tracks,
            query_frame=query_frame,
            compensate_for_camera_motion=compensate_for_camera_motion,
            color_alpha=color_alpha,
        )
        if save_video:
            self.save_video(res_video, filename=filename, writer=writer, step=step)
        return res_video

    def save_video(self, video, filename, writer=None, step=0):
        if writer is not None:
            writer.add_video(
                filename,
                video.to(torch.uint8),
                global_step=step,
                fps=self.fps,
            )
        else:
            os.makedirs(self.save_dir, exist_ok=True)
            wide_list = list(video.unbind(1))
            wide_list = [wide[0].permute(1, 2, 0).cpu().numpy() for wide in wide_list]

            # Prepare the video file path
            save_path = os.path.join(self.save_dir, f"{filename}.mp4")

            # Create a writer object
            video_writer = imageio.get_writer(save_path, fps=self.fps)

            # Write frames to the video file
            for frame in wide_list[2:-1]:
                video_writer.append_data(frame)

            video_writer.close()

            print(f"Video saved to {save_path}")

    def draw_tracks_on_video(
        self,
        video: torch.Tensor,
        tracks: torch.Tensor,
        visibility: torch.Tensor = None,
        segm_mask: torch.Tensor = None,
        gt_tracks=None,
        query_frame=0,
        compensate_for_camera_motion=False,
        color_alpha: int = 255,
    ):
        B, T, C, H, W = video.shape
        _, _, N, D = tracks.shape

        assert D == 2
        assert C == 3
        video = video[0].permute(0, 2, 3, 1).byte().detach().cpu().numpy()  # S, H, W, C
        tracks = tracks[0].long().detach().cpu().numpy()  # S, N, 2
        if gt_tracks is not None:
            gt_tracks = gt_tracks[0].detach().cpu().numpy()

        res_video = []

        # process input video
        for rgb in video:
            res_video.append(rgb.copy())
        vector_colors = np.zeros((T, N, 3))

        if self.mode == "optical_flow":
            import flow_vis

            vector_colors = flow_vis.flow_to_color(tracks - tracks[query_frame][None])
        elif segm_mask is None:
            if self.mode == "rainbow":
                y_min, y_max = (
                    tracks[query_frame, :, 1].min(),
                    tracks[query_frame, :, 1].max(),
                )
                norm = plt.Normalize(y_min, y_max)
                for n in range(N):
                    if isinstance(query_frame, torch.Tensor):
                        query_frame_ = query_frame[n]
                    else:
                        query_frame_ = query_frame
                    color = self.color_map(norm(tracks[query_frame_, n, 1]))
                    color = np.array(color[:3])[None] * 255
                    vector_colors[:, n] = np.repeat(color, T, axis=0)
            else:
                # color changes with time
                for t in range(T):
                    color = np.array(self.color_map(t / T)[:3])[None] * 255
                    vector_colors[t] = np.repeat(color, N, axis=0)
        else:
            if self.mode == "rainbow":
                vector_colors[:, segm_mask <= 0, :] = 255

                y_min, y_max = (
                    tracks[0, segm_mask > 0, 1].min(),
                    tracks[0, segm_mask > 0, 1].max(),
                )
                norm = plt.Normalize(y_min, y_max)
                for n in range(N):
                    if segm_mask[n] > 0:
                        color = self.color_map(norm(tracks[0, n, 1]))
                        color = np.array(color[:3])[None] * 255
                        vector_colors[:, n] = np.repeat(color, T, axis=0)

            else:
                # color changes with segm class
                segm_mask = segm_mask.cpu()
                color = np.zeros((segm_mask.shape[0], 3), dtype=np.float32)
                color[segm_mask > 0] = np.array(self.color_map(1.0)[:3]) * 255.0
                color[segm_mask <= 0] = np.array(self.color_map(0.0)[:3]) * 255.0
                vector_colors = np.repeat(color[None], T, axis=0)

        #  draw tracks
        if self.tracks_leave_trace != 0:
            for t in range(query_frame + 1, T):
                first_ind = (
                    max(0, t - self.tracks_leave_trace)
                    if self.tracks_leave_trace >= 0
                    else 0
                )
                curr_tracks = tracks[first_ind : t + 1]
                curr_colors = vector_colors[first_ind : t + 1]
                if compensate_for_camera_motion:
                    diff = (
                        tracks[first_ind : t + 1, segm_mask <= 0]
                        - tracks[t : t + 1, segm_mask <= 0]
                    ).mean(1)[:, None]

                    curr_tracks = curr_tracks - diff
                    curr_tracks = curr_tracks[:, segm_mask > 0]
                    curr_colors = curr_colors[:, segm_mask > 0]

                res_video[t] = self._draw_pred_tracks(
                    res_video[t],
                    curr_tracks,
                    curr_colors,
                )
                if gt_tracks is not None:
                    res_video[t] = self._draw_gt_tracks(
                        res_video[t], gt_tracks[first_ind : t + 1]
                    )

        #  draw points
        for t in range(T):
            img = Image.fromarray(np.uint8(res_video[t]))
            for i in range(N):
                coord = (tracks[t, i, 0], tracks[t, i, 1])
                visibile = True
                if visibility is not None:
                    visibile = visibility[0, t, i]
                if coord[0] != 0 and coord[1] != 0:
                    if not compensate_for_camera_motion or (
                        compensate_for_camera_motion and segm_mask[i] > 0
                    ):
                        img = draw_circle(
                            img,
                            coord=coord,
                            radius=int(self.linewidth * 2),
                            color=vector_colors[t, i].astype(int),
                            visible=visibile,
                            color_alpha=color_alpha,
                        )
            res_video[t] = np.array(img)

        #  construct the final rgb sequence
        if self.show_first_frame > 0:
            res_video = [res_video[0]] * self.show_first_frame + res_video[1:]
        return torch.from_numpy(np.stack(res_video)).permute(0, 3, 1, 2)[None].byte()

    def _draw_pred_tracks(
        self,
        rgb: np.ndarray,  # H x W x 3
        tracks: np.ndarray,  # T x 2
        vector_colors: np.ndarray,
        alpha: float = 0.5,
    ):
        T, N, _ = tracks.shape
        rgb = Image.fromarray(np.uint8(rgb))
        for s in range(T - 1):
            vector_color = vector_colors[s]
            original = rgb.copy()
            alpha = (s / T) ** 2
            for i in range(N):
                coord_y = (int(tracks[s, i, 0]), int(tracks[s, i, 1]))
                coord_x = (int(tracks[s + 1, i, 0]), int(tracks[s + 1, i, 1]))
                if coord_y[0] != 0 and coord_y[1] != 0:
                    rgb = draw_line(
                        rgb,
                        coord_y,
                        coord_x,
                        vector_color[i].astype(int),
                        self.linewidth,
                    )
            if self.tracks_leave_trace > 0:
                rgb = Image.fromarray(
                    np.uint8(
                        add_weighted(
                            np.array(rgb), alpha, np.array(original), 1 - alpha, 0
                        )
                    )
                )
        rgb = np.array(rgb)
        return rgb

    def _draw_gt_tracks(
        self,
        rgb: np.ndarray,  # H x W x 3,
        gt_tracks: np.ndarray,  # T x 2
    ):
        T, N, _ = gt_tracks.shape
        color = np.array((211, 0, 0))
        rgb = Image.fromarray(np.uint8(rgb))
        for t in range(T):
            for i in range(N):
                gt_tracks = gt_tracks[t][i]
                #  draw a red cross
                if gt_tracks[0] > 0 and gt_tracks[1] > 0:
                    length = self.linewidth * 3
                    coord_y = (int(gt_tracks[0]) + length, int(gt_tracks[1]) + length)
                    coord_x = (int(gt_tracks[0]) - length, int(gt_tracks[1]) - length)
                    rgb = draw_line(
                        rgb,
                        coord_y,
                        coord_x,
                        color,
                        self.linewidth,
                    )
                    coord_y = (int(gt_tracks[0]) - length, int(gt_tracks[1]) + length)
                    coord_x = (int(gt_tracks[0]) + length, int(gt_tracks[1]) - length)
                    rgb = draw_line(
                        rgb,
                        coord_y,
                        coord_x,
                        color,
                        self.linewidth,
                    )
        rgb = np.array(rgb)
        return rgb

@torch.no_grad()
def batchify(model, img, points, batch_size=2**15):
    pred_tracks, pred_visibility = [], []
    for p in torch.split(points, batch_size, dim=0):
        pkg = model(img[None], queries=p[None])
        pred_tracks.append(pkg[0][0])
        pred_visibility.append(pkg[1][0])
    pred_tracks = torch.cat(pred_tracks, dim=1)
    pred_visibility = torch.cat(pred_visibility, dim=1)
    return pred_tracks, pred_visibility[..., 0]


@torch.no_grad()
def generate_waymo_flow(path, downsample, slide_window, num_cams=1):
    flow_folder = os.path.join(path, 'flow')
    os.makedirs(flow_folder, exist_ok=True)

    images = []
    masks = []
    meta = np.load(os.path.join(path, "cameras.npz"), allow_pickle=True)
    K, R, T, time_stamps, is_val_list = meta['K'], meta['R'], meta['T'], meta['time_stamps'], meta['is_val_list']
    img_list = list(sorted(os.listdir(os.path.join(path, "image"))))
    indices = []
    for idx, img_path in enumerate(tqdm.tqdm(img_list, desc='Reading')):
        if is_val_list[idx]:
            continue
        semantic_path = os.path.join(path, "semantic", 'mask_' + img_path.split(".")[0] + ".npy")
        img_path = os.path.join(path, 'image', img_path)
        img = np.array(Image.open(img_path))
        mask = (np.load(semantic_path) > 0).astype(np.float32)
        masks.append(mask)
        images.append(img)
        indices.append(idx)
    K, R, T, time_stamps = K[~is_val_list], R[~is_val_list], T[~is_val_list], time_stamps[~is_val_list]
    images = np.stack(images, axis=0)
    masks = np.stack(masks, axis=0)
    images = torch.tensor(images, dtype=torch.float32).permute(0, 3, 1, 2)
    masks = torch.tensor(masks, dtype=torch.float32)
    if downsample is not None and downsample > 1:
        images = F.interpolate(images, size=(images.shape[2] // downsample, images.shape[3] // downsample), mode='bilinear')
        masks = F.interpolate(masks[:, None], size=(masks.shape[1] // downsample, masks.shape[2] // downsample), mode='bilinear')[:, 0]
        K = K / downsample
    assert K.shape[0] == images.shape[0]

    H, W = images.shape[2], images.shape[3]
    grid = torch.stack(torch.meshgrid(
        torch.arange(0, W, dtype=torch.float32, device='cuda'),
        torch.arange(0, H, dtype=torch.float32, device='cuda'),
        indexing='xy',
    ), dim=-1) # H, W, 2
    flow_video = []

    lis_temp = np.arange(0, slide_window + 1, dtype=np.int32) * num_cams
    for idx in tqdm.tqdm(range(images.shape[0]), desc='Processing'):
        selected_coord = torch.nonzero(masks[idx].cuda() > 0.5, as_tuple=True)
        pts = grid[selected_coord]
        if pts.numel() == 0:
            print(f'[WARNING] Image {indices[idx]} has no object detected.')
            continue
        pts = torch.cat([torch.zeros((pts.shape[0], 1), dtype=torch.float32, device='cuda'), pts], dim=-1)
        flow = []
        if idx // num_cams < images.shape[0] // num_cams - slide_window:
            img = images[lis_temp + idx]
            forward_flow_pts, forward_flow_vis_pts = batchify(cotracker, img.cuda(), points=pts)
            forward_flow_pts, forward_flow_vis_pts = forward_flow_pts[-1], forward_flow_vis_pts[-1]
            forward_flow = grid.clone()
            forward_flow_vis = torch.zeros((H, W), dtype=torch.float32, device='cuda')
            forward_flow[selected_coord] = forward_flow_pts
            forward_flow_vis[selected_coord] = forward_flow_vis_pts.float()

            forward_flow = torch.permute(forward_flow, (2, 0, 1))
            forward_flow = forward_flow.detach().cpu().numpy()  # 2, H, W
            forward_flow_vis = forward_flow_vis.detach().cpu().numpy()  # H, W

            K_select = K[idx + slide_window * num_cams]
            K_select = np.array([
                [K_select[0], 0.0, K_select[2]],
                [0.0, K_select[1], K_select[3]],
                [0.0, 0.0, 1.0],
            ], dtype=np.float32)
            flow.append([time_stamps[idx + slide_window * num_cams], K_select, R[idx + slide_window * num_cams], T[idx + slide_window * num_cams], forward_flow, forward_flow_vis])

            flow_img = flow_vis.flow_to_color(np.transpose(forward_flow, (1, 2, 0)) - grid.detach().cpu().numpy())
            flow_video.append(np.uint8(flow_img))
            # Image.fromarray(np.uint8(img)).save(os.path.join(flow_folder, '{:06d}.png'.format(idx)))
        if idx // num_cams >= slide_window:
            img = images[idx - lis_temp]
            backward_flow_pts, backward_flow_vis_pts = batchify(cotracker, img.cuda(), points=pts)
            backward_flow_pts, backward_flow_vis_pts = backward_flow_pts[-1], backward_flow_vis_pts[-1]
            backward_flow = grid.clone()
            backward_flow_vis = torch.zeros((H, W), dtype=torch.float32, device='cuda')
            backward_flow[selected_coord] = backward_flow_pts
            backward_flow_vis[selected_coord] = backward_flow_vis_pts.float()

            backward_flow = torch.permute(backward_flow, (2, 0, 1))
            backward_flow = backward_flow.detach().cpu().numpy()  # 2, H, W
            backward_flow_vis = backward_flow_vis.detach().cpu().numpy()  # H, W

            K_select = K[idx - slide_window * num_cams]
            K_select = np.array([
                [K_select[0], 0.0, K_select[2]],
                [0.0, K_select[1], K_select[3]],
                [0.0, 0.0, 1.0],
            ], dtype=np.float32)
            flow.append([time_stamps[idx - slide_window * num_cams], K_select, R[idx - slide_window * num_cams], T[idx - slide_window * num_cams], backward_flow, backward_flow_vis])
        np.savez(os.path.join(flow_folder, '{:06d}.npz'.format(indices[idx])), flow=flow)
    flow_video = np.stack(flow_video, axis=0)
    return flow_video

def get_track_waymo_video(path, downsample=None, start_frame=0, end_frame=-1, num_cams=1, cam_id=0):
    images = []
    pts = None
    for idx, img_path in enumerate(sorted(os.listdir(os.path.join(path, "image")))):
        if idx % num_cams != cam_id:
            continue
        if idx // num_cams < start_frame or (end_frame != -1 and idx // num_cams > end_frame):
            continue
        semantic_path = os.path.join(path, "semantic", 'mask_' + img_path.split(".")[0] + ".npy")
        img_path = os.path.join(path, "image", img_path)
        img = Image.open(img_path)
        if downsample is not None:
            w, h = img.size[0] // downsample, img.size[1] // downsample
            img = img.resize((w, h))
        images.append(torch.tensor(np.array(img), dtype=torch.float32).permute(2, 0, 1))

        if idx // num_cams == start_frame:
            W, H = img.size
            grid = torch.stack(torch.meshgrid(
                torch.arange(0, W, dtype=torch.float32, device='cuda'),
                torch.arange(0, H, dtype=torch.float32, device='cuda'),
                indexing='xy',
            ), dim=-1) # H, W, 2
            mask = torch.tensor((np.load(semantic_path) > 0).astype(np.float32), device='cuda', dtype=torch.float32)
            mask = F.interpolate(mask[None, None], size=(H, W), mode='bilinear').squeeze()
            selected_coord = torch.nonzero(mask > 0.5, as_tuple=True)
            pts = grid[selected_coord]
            pts = torch.cat([torch.zeros((pts.shape[0], 1), dtype=torch.float32, device='cuda'), pts], dim=-1)

    images = torch.stack(images, dim=0)[None].cuda()
    # images = torch.flip(images, dims=[1])
    pts = pts[torch.randperm(pts.shape[0], device='cuda')[:1000]]
    pred_tracks, pred_visibility = cotracker(images, queries=pts[None]) # grid_size=40) # B T N 2,  B T N 1
    vis = Visualizer(save_dir=os.path.join(path, "flow_videos"), pad_value=120, linewidth=3)
    vis.visualize(images, pred_tracks, pred_visibility)
    vis.save_video(images.byte(), 'origin')

def generate_kitti_flow(path, downsample, slide_window, split_mode='nvs-75', num_cams=2):
    flow_folder = os.path.join(path, 'flow', split_mode)
    os.makedirs(flow_folder, exist_ok=True)

    images = []
    masks = []
    meta = np.load(os.path.join(path, "poses.npz"), allow_pickle=True)
    R, T, time_stamps = meta['R'], meta['T'], meta['time_stamp']
    H, W, focal = int(meta['height']), int(meta['width']), float(meta['focal'])
    K = np.array([
        [focal, 0.0, W / 2.0],
        [0.0, focal, H / 2.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float32)
    if split_mode == 'nvs-25':
        i_test = get_val_frames(time_stamps.shape[0] // num_cams, train_every=4)
    elif split_mode == 'nvs-50':
        i_test = get_val_frames(time_stamps.shape[0] // num_cams, test_every=2)
    elif split_mode == 'nvs-75':
        i_test = get_val_frames(time_stamps.shape[0] // num_cams, test_every=4)
    else:
        raise ValueError("No such split method: " + split_mode)

    img_list = list(sorted(os.listdir(os.path.join(path, "image"))))
    indices = []
    for idx, img_path in enumerate(tqdm.tqdm(img_list, desc='Reading')):
        if idx // num_cams in i_test:
            continue
        indices.append(idx)
        semantic_path = os.path.join(path, "semantic", 'mask_' + img_path.split(".")[0] + ".npy")
        img_path = os.path.join(path, 'image', img_path)
        img = np.array(Image.open(img_path))
        mask = (np.load(semantic_path) > 0).astype(np.float32)
        masks.append(mask)
        images.append(img)
    R, T, time_stamps = R[indices], T[indices], time_stamps[indices]
    images = np.stack(images, axis=0)
    masks = np.stack(masks, axis=0)
    images = torch.tensor(images, dtype=torch.float32).permute(0, 3, 1, 2)
    masks = torch.tensor(masks, dtype=torch.float32)
    assert H == images.shape[2] and W == images.shape[3]
    if downsample is not None and downsample > 1:
        images = F.interpolate(images, size=(images.shape[2] // downsample, images.shape[3] // downsample), mode='bilinear')
        masks = F.interpolate(masks[:, None], size=(masks.shape[1] // downsample, masks.shape[2] // downsample), mode='bilinear')[:, 0]
        K = K / downsample
        K[-1, -1] = 1.0
    assert T.shape[0] == images.shape[0]
    
    grid = torch.stack(torch.meshgrid(
        torch.arange(0, W, dtype=torch.float32, device='cuda'),
        torch.arange(0, H, dtype=torch.float32, device='cuda'),
        indexing='xy',
    ), dim=-1) # H, W, 2
    flow_video = []

    lis_temp = np.arange(0, slide_window + 1, dtype=np.int32) * num_cams
    for idx in tqdm.tqdm(range(images.shape[0]), desc='Processing'):
        selected_coord = torch.nonzero(masks[idx].cuda() > 0.5, as_tuple=True)
        pts = grid[selected_coord]
        if pts.numel() == 0:
            print(f'[WARNING] Image {indices[idx]} has no object detected.')
            continue
        pts = torch.cat([torch.zeros((pts.shape[0], 1), dtype=torch.float32, device='cuda'), pts], dim=-1)
        flow = []
        
        if idx // num_cams < images.shape[0] // num_cams - slide_window:
            img = images[lis_temp + idx]
            forward_flow_pts, forward_flow_vis_pts = batchify(cotracker, img.cuda(), points=pts)
            forward_flow_pts, forward_flow_vis_pts = forward_flow_pts[-1], forward_flow_vis_pts[-1]
            forward_flow = grid.clone()
            forward_flow_vis = torch.zeros((H, W), dtype=torch.float32, device='cuda')
            forward_flow[selected_coord] = forward_flow_pts
            forward_flow_vis[selected_coord] = forward_flow_vis_pts.float()

            forward_flow = torch.permute(forward_flow, (2, 0, 1))
            forward_flow = forward_flow.detach().cpu().numpy()  # 2, H, W
            forward_flow_vis = forward_flow_vis.detach().cpu().numpy()  # H, W

            flow.append([time_stamps[idx + slide_window * num_cams], K, R[idx + slide_window * num_cams], T[idx + slide_window * num_cams], forward_flow, forward_flow_vis])

            flow_img = flow_vis.flow_to_color(np.transpose(forward_flow, (1, 2, 0)) - grid.detach().cpu().numpy())
            flow_video.append(np.uint8(flow_img))
            # Image.fromarray(np.uint8(img)).save(os.path.join(flow_folder, '{:06d}.png'.format(idx)))
        if idx // num_cams >= slide_window:
            img = images[idx - lis_temp]
            backward_flow_pts, backward_flow_vis_pts = batchify(cotracker, img.cuda(), points=pts)
            backward_flow_pts, backward_flow_vis_pts = backward_flow_pts[-1], backward_flow_vis_pts[-1]
            backward_flow = grid.clone()
            backward_flow_vis = torch.zeros((H, W), dtype=torch.float32, device='cuda')
            backward_flow[selected_coord] = backward_flow_pts
            backward_flow_vis[selected_coord] = backward_flow_vis_pts.float()

            backward_flow = torch.permute(backward_flow, (2, 0, 1))
            backward_flow = backward_flow.detach().cpu().numpy()  # 2, H, W
            backward_flow_vis = backward_flow_vis.detach().cpu().numpy()  # H, W

            flow.append([time_stamps[idx - slide_window * num_cams], K, R[idx - slide_window * num_cams], T[idx - slide_window * num_cams], backward_flow, backward_flow_vis])
        np.savez(os.path.join(flow_folder, '{:06d}.npz'.format(indices[idx])), flow=flow)
    flow_video = np.stack(flow_video, axis=0)
    return flow_video

def get_track_kitti_video(path, downsample=None, start_frame=0, end_frame=-1, num_cams=2, cam_id=0):
    images = []
    pts = None
    for idx, img_path in enumerate(sorted(os.listdir(os.path.join(path, "image")))):
        if idx % num_cams != cam_id:
            continue
        if idx // num_cams < start_frame or (end_frame != -1 and idx // num_cams > end_frame):
            continue
        semantic_path = os.path.join(path, "semantic", 'mask_' + img_path.split(".")[0] + ".npy")
        img_path = os.path.join(path, "image", img_path)
        img = Image.open(img_path)
        if downsample is not None:
            w, h = img.size[0] // downsample, img.size[1] // downsample
            img = img.resize((w, h))
        images.append(torch.tensor(np.array(img), dtype=torch.float32).permute(2, 0, 1))

        if idx // num_cams == start_frame:
            W, H = img.size
            grid = torch.stack(torch.meshgrid(
                torch.arange(0, W, dtype=torch.float32, device='cuda'),
                torch.arange(0, H, dtype=torch.float32, device='cuda'),
                indexing='xy',
            ), dim=-1) # H, W, 2
            mask = torch.tensor((np.load(semantic_path) > 0).astype(np.float32), device='cuda', dtype=torch.float32)
            mask = F.interpolate(mask[None, None], size=(H, W), mode='bilinear').squeeze()
            selected_coord = torch.nonzero(mask > 0.5, as_tuple=True)
            pts = grid[selected_coord]
            pts = torch.cat([torch.zeros((pts.shape[0], 1), dtype=torch.float32, device='cuda'), pts], dim=-1)

    images = torch.stack(images, dim=0)[None].cuda()
    # images = torch.flip(images, dims=[1])
    pts = pts[torch.randperm(pts.shape[0], device='cuda')[:1000]]
    pred_tracks, pred_visibility = cotracker(images, queries=pts[None]) # grid_size=40) # B T N 2,  B T N 1
    vis = Visualizer(save_dir=os.path.join(path, "flow_videos"), pad_value=120, linewidth=3)
    vis.visualize(images, pred_tracks, pred_visibility)
    vis.save_video(images.byte(), 'origin')

def generate_nuscenes_flow(path, downsample, slide_window, num_cams=3):
    flow_folder = os.path.join(path, 'flow')
    os.makedirs(flow_folder, exist_ok=True)

    images = []
    masks = []
    meta = np.load(os.path.join(path, "meta.npz"), allow_pickle=True)
    K, R, T, time_stamps, is_val_list = meta['K'], meta['R'], meta['T'], meta['time_stamps'], meta['is_val_list']
    img_list = list(sorted(os.listdir(os.path.join(path, "image"))))
    indices = []
    for idx, img_path in enumerate(tqdm.tqdm(img_list, desc='Reading')):
        if is_val_list[idx]:
            continue
        semantic_path = os.path.join(path, "semantic", 'mask_' + img_path.split(".")[0] + ".npy")
        img_path = os.path.join(path, 'image', img_path)
        img = np.array(Image.open(img_path))
        mask = (np.load(semantic_path) > 0).astype(np.float32)
        masks.append(mask)
        images.append(img)
        indices.append(idx)
    K, R, T, time_stamps = K[~is_val_list], R[~is_val_list], T[~is_val_list], time_stamps[~is_val_list]
    images = np.stack(images, axis=0)
    masks = np.stack(masks, axis=0)
    images = torch.tensor(images, dtype=torch.float32).permute(0, 3, 1, 2)
    masks = torch.tensor(masks, dtype=torch.float32)
    if downsample is not None and downsample > 1:
        images = F.interpolate(images, size=(images.shape[2] // downsample, images.shape[3] // downsample), mode='bilinear')
        masks = F.interpolate(masks[:, None], size=(masks.shape[1] // downsample, masks.shape[2] // downsample), mode='bilinear')[:, 0]
        K = K / downsample
        K[-1, -1] = 1.0
    assert K.shape[0] == images.shape[0]

    H, W = images.shape[2], images.shape[3]
    grid = torch.stack(torch.meshgrid(
        torch.arange(0, W, dtype=torch.float32, device='cuda'),
        torch.arange(0, H, dtype=torch.float32, device='cuda'),
        indexing='xy',
    ), dim=-1) # H, W, 2
    flow_video = []

    lis_temp = np.arange(0, slide_window + 1, dtype=np.int32) * num_cams
    for idx in tqdm.tqdm(range(images.shape[0]), desc='Processing'):
        # if os.path.exists(os.path.join(flow_folder, '{:06d}.npz'.format(indices[idx]))):
        #     continue
        selected_coord = torch.nonzero(masks[idx].cuda() > 0.5, as_tuple=True)
        pts = grid[selected_coord]
        if pts.numel() == 0:
            print(f'[WARNING] Image {indices[idx]} has no object detected.')
            continue
        pts = torch.cat([torch.zeros((pts.shape[0], 1), dtype=torch.float32, device='cuda'), pts], dim=-1)
        flow = []

        if idx // num_cams < images.shape[0] // num_cams - slide_window:
            img = images[lis_temp + idx]
            forward_flow_pts, forward_flow_vis_pts = batchify(cotracker, img.cuda(), points=pts)
            forward_flow_pts, forward_flow_vis_pts = forward_flow_pts[-1], forward_flow_vis_pts[-1]
            forward_flow = grid.clone()
            forward_flow_vis = torch.zeros((H, W), dtype=torch.float32, device='cuda')
            forward_flow[selected_coord] = forward_flow_pts
            forward_flow_vis[selected_coord] = forward_flow_vis_pts.float()

            forward_flow = torch.permute(forward_flow, (2, 0, 1))
            forward_flow = forward_flow.detach().cpu().numpy()  # 2, H, W
            forward_flow_vis = forward_flow_vis.detach().cpu().numpy()  # H, W

            K_select = K[idx + slide_window * num_cams]
            flow.append([time_stamps[idx + slide_window * num_cams], K_select, R[idx + slide_window * num_cams], T[idx + slide_window * num_cams], forward_flow, forward_flow_vis])

            flow_img = flow_vis.flow_to_color(np.transpose(forward_flow, (1, 2, 0)) - grid.detach().cpu().numpy())
            flow_video.append(np.uint8(flow_img))
            # Image.fromarray(np.uint8(img)).save(os.path.join(flow_folder, '{:06d}.png'.format(idx)))
        if idx // num_cams >= slide_window:
            img = images[idx - lis_temp]
            backward_flow_pts, backward_flow_vis_pts = batchify(cotracker, img.cuda(), points=pts)
            backward_flow_pts, backward_flow_vis_pts = backward_flow_pts[-1], backward_flow_vis_pts[-1]
            backward_flow = grid.clone()
            backward_flow_vis = torch.zeros((H, W), dtype=torch.float32, device='cuda')
            backward_flow[selected_coord] = backward_flow_pts
            backward_flow_vis[selected_coord] = backward_flow_vis_pts.float()

            backward_flow = torch.permute(backward_flow, (2, 0, 1))
            backward_flow = backward_flow.detach().cpu().numpy()  # 2, H, W
            backward_flow_vis = backward_flow_vis.detach().cpu().numpy()  # H, W

            K_select = K[idx - slide_window * num_cams]
            flow.append([time_stamps[idx - slide_window * num_cams], K_select, R[idx - slide_window * num_cams], T[idx - slide_window * num_cams], backward_flow, backward_flow_vis])
        np.savez(os.path.join(flow_folder, '{:06d}.npz'.format(indices[idx])), flow=flow)
    flow_video = np.stack(flow_video, axis=0)
    return flow_video

def get_track_nuscenes_video(path, downsample=None, start_frame=0, end_frame=-1, num_cams=3, cam_id=0):
    images = []
    pts = None
    for idx, img_path in enumerate(sorted(os.listdir(os.path.join(path, "image")))):
        if idx % num_cams != cam_id:
            continue
        if idx // num_cams < start_frame or (end_frame != -1 and idx // num_cams > end_frame):
            continue
        semantic_path = os.path.join(path, "semantic", 'mask_' + img_path.split(".")[0] + ".npy")
        img_path = os.path.join(path, "image", img_path)
        img = Image.open(img_path)
        if downsample is not None:
            w, h = img.size[0] // downsample, img.size[1] // downsample
            img = img.resize((w, h))
        images.append(torch.tensor(np.array(img), dtype=torch.float32).permute(2, 0, 1))

        if idx // num_cams == start_frame:
            W, H = img.size
            grid = torch.stack(torch.meshgrid(
                torch.arange(0, W, dtype=torch.float32, device='cuda'),
                torch.arange(0, H, dtype=torch.float32, device='cuda'),
                indexing='xy',
            ), dim=-1) # H, W, 2
            mask = torch.tensor((np.load(semantic_path) > 0).astype(np.float32), device='cuda', dtype=torch.float32)
            mask = F.interpolate(mask[None, None], size=(H, W), mode='bilinear').squeeze()
            selected_coord = torch.nonzero(mask > 0.5, as_tuple=True)
            pts = grid[selected_coord]
            pts = torch.cat([torch.zeros((pts.shape[0], 1), dtype=torch.float32, device='cuda'), pts], dim=-1)

    images = torch.stack(images, dim=0)[None].cuda()
    pts = pts[torch.randperm(pts.shape[0], device='cuda')[:1000]]
    pred_tracks, pred_visibility = cotracker(images, queries=pts[None]) # grid_size=40) # B T N 2,  B T N 1
    vis = Visualizer(save_dir=os.path.join(path, "flow_videos"), pad_value=120, linewidth=3)
    vis.visualize(images, pred_tracks, pred_visibility)
    vis.save_video(images.byte(), 'origin')

if __name__ == '__main__':
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

    parser = argparse.ArgumentParser()
    parser.add_argument('path')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--downsample', default=1, type=int)
    parser.add_argument('--video', '-v', action='store_true')
    parser.add_argument('--start', default=0, type=int)
    parser.add_argument('--end', default=-1, type=int)
    parser.add_argument('--step', default=4, type=int)
    parser.add_argument('--cam', default=0, type=int)  # only use to generate video
    parser.add_argument('--fps', default=10, type=int)
    parser.add_argument('--split_mode', default='nvs-75')
    args = parser.parse_args()
    torch.cuda.set_device(args.device)
    src_path = args.path

    cotracker = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline").cuda()

    # for cached model
    # cotracker = torch.hub.load("/root/.cache/torch/hub/facebookresearch_co-tracker_main", "cotracker3_offline", trust_repo=True, source='local').cuda()


    flow_video = None
    flow_video_path = os.path.join(args.path, 'flow.mp4')
    if os.path.exists(os.path.join(args.path, "cameras.npz")):
        print("Found cameras.npz file, assuming Waymo data set!")
        if args.video:
            get_track_waymo_video(args.path, args.downsample, args.start, args.end, cam_id=args.cam)
        else:
            flow_video = generate_waymo_flow(args.path, args.downsample, args.step)
    elif os.path.exists(os.path.join(args.path, 'poses.npz')):
        print("Found poses.npz, assuming KITTI or vKITTI data set!")
        if args.video:
            get_track_kitti_video(args.path, args.downsample, args.start, args.end, cam_id=args.cam)
        else:
            flow_video = generate_kitti_flow(args.path, args.downsample, args.step, split_mode=args.split_mode)
            flow_video_path = os.path.join(args.path, 'flow-{}.mp4'.format(args.split_mode.split('-')[-1]))
    elif os.path.exists(os.path.join(args.path, "meta.npz")):
        print("Found meta.npz file, assuming nuScenes data set!")
        if args.video:
            get_track_nuscenes_video(args.path, args.downsample, args.start, args.end, cam_id=args.cam)
        else:
            flow_video = generate_nuscenes_flow(args.path, args.downsample, args.step)
    else:
        assert False, 'Could not recognize scene type!'
        
        
    if flow_video is not None:
        imageio.mimwrite(flow_video_path, flow_video, fps=args.fps, quality=8)
        print("Save a visualized video:", flow_video_path)