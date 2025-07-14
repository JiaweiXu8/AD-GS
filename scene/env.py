import torch, os
import numpy as np
import torch.nn as nn
from utils.graphics_utils import theta_to_vector, vector_to_theta
from torch.nn.functional import grid_sample, normalize
from utils.system_utils import searchForMaxIteration
from scene.cameras import Camera
from utils.graphics_utils import fov2focal
import open3d as o3d

def get_image_cam_rays(focal, height, width):
    K = torch.tensor([
        [focal, 0.0, width / 2],
        [0.0, focal, height / 2],
        [0.0, 0.0, 1.0],
    ], dtype=torch.float32, device='cuda')
    K_inv = torch.inverse(K)
    grid = torch.stack(torch.meshgrid(
        torch.arange(0, width, dtype=torch.float32, device='cuda'),
        torch.arange(0, height, dtype=torch.float32, device='cuda'),
        indexing='xy',
    ), dim=-1)  # H, W, 2
    grid_pts = torch.cat([grid, torch.ones((grid.shape[0], grid.shape[1], 1), dtype=torch.float32, device='cuda')], dim=-1)[..., None]
    pix_cam_ray = (K_inv @ grid_pts)
    pix_cam_ray = normalize(pix_cam_ray[..., 0], p=2, dim=-1)
    return pix_cam_ray, grid


class EnvironmentMap:
    def __init__(self, resolution, num_channel=3, use_cache=True):
        self.resolution = resolution
        grid_map = (torch.rand((1, num_channel, resolution, resolution), dtype=torch.float32, device='cuda') * 2.0 - 1.0) * 1e-4
        self.grid_map = nn.Parameter(grid_map.requires_grad_(True))
        self.scale = torch.tensor([1.0 / torch.pi, 2.0 / torch.pi], dtype=torch.float32, device='cuda')

        self.optimizer = None
        self.image_cam_rays = dict()
        self.grid = dict()
        self.use_cache = use_cache
    
    def set_image_cam_ray(self, focal, height, width):
        self.image_cam_rays, self.grid = get_image_cam_rays(focal, height, width)

    def get_image_background(self, cam: Camera, use_cache=True, return_grid=False):
        use_cache = use_cache and self.use_cache
        if not use_cache:
            focal = fov2focal(cam.FoVx, cam.image_width)
            image_cam_rays, grid = get_image_cam_rays(focal, cam.image_height, cam.image_width)
        else:
            try:
                image_cam_rays = self.image_cam_rays[cam.cam_id]
                grid = self.grid[cam.cam_id]
            except KeyError:
                focal = fov2focal(cam.FoVx, cam.image_width)
                image_cam_rays, grid = get_image_cam_rays(focal, cam.image_height, cam.image_width)
                self.image_cam_rays[cam.cam_id] = image_cam_rays  # H, W, 3
                self.grid[cam.cam_id] = grid
        # image_cam_rays = (cam.world_view_transform[:3, :3].cuda().transpose(0, 1) @ image_cam_rays[..., None]).squeeze(-1)
        image_cam_rays = (cam.world_view_transform[:3, :3].cuda() @ image_cam_rays[..., None]).squeeze(-1)  # the matrix has already been rotated.

        background_image = self.get_env_color(image_cam_rays)

        if return_grid:
            return background_image, grid
        return background_image

    def get_env_color(self, view, input_angle=False):
        if not input_angle:
            view = normalize(view, p=2, dim=-1)
            angle = vector_to_theta(view) # H, W, 2
        else:
            angle = view  # H, W, 2
        angle = angle * self.scale
        rgb = grid_sample(self.grid_map, grid=angle[None, ...], align_corners=True)  # 1, C, H, W
        rgb = torch.sigmoid(rgb).squeeze(0)  # 3, H, W
        return rgb
    
    def training_setup(self, training_args):
        l = [
            {'params': [self.grid_map], 'lr': training_args.env_lr, "name": "env"},
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

    def save_weights(self, weights_path):
        torch.save(self.grid_map, weights_path)

    def load_weights(self, weights_path):
        grid_map = torch.load(weights_path, map_location='cuda')
        self.grid_map = nn.Parameter(grid_map.requires_grad_(True))

    def extract_env_map(self, path, num_pts=50_0000):
        pts = torch.cat([
            (torch.rand((num_pts, 1), device="cuda") * 2.0 - 1.0) * torch.pi,
            (torch.rand((num_pts, 1), device="cuda") * 2.0 - 1.0) * torch.pi / 2.0,
        ], dim=-1)
        rgb = self.get_env_color(pts[None], input_angle=True).squeeze(1).transpose(1, 0)
        pcd = o3d.geometry.PointCloud()
        pts = theta_to_vector(pts)
        pcd.points = o3d.utility.Vector3dVector(pts.detach().cpu().numpy())
        pcd.colors = o3d.utility.Vector3dVector(rgb.detach().cpu().numpy())
        o3d.io.write_point_cloud(path, pcd)