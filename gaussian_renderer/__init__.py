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
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from scene.env import EnvironmentMap

def render(viewpoint_camera, pc : GaussianModel, env_map : EnvironmentMap, pipe, scaling_modifier = 1.0, override_color = None, flow_pkg=None, render_objmask=False):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=torch.tensor([0.0, 0.0, 0.0], device='cuda', dtype=torch.float32),
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform.cuda(),
        projmatrix=viewpoint_camera.full_proj_transform.cuda(),
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center.cuda(),
        prefiltered=False,
        inv_depth=pipe.inv_depth,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    flow_points = None
    if flow_pkg is not None:
        flow_time, K, R, T, _, _ = flow_pkg
        flow_points = pc.get_deformed_xyz(flow_time)
        # flow_points = pc.get_flow_proj(flow_pkg)


    deform_pkg = pc.get_deformed_pkg(viewpoint_camera.time)
    
    means3D = deform_pkg['xyz']
    opacity = deform_pkg['opacity']
    shs = deform_pkg['shs'] if override_color is None else None
    rotations = deform_pkg['rotation']

    scales = pc.get_scaling
    means2D = screenspace_points

    semantic = None
    if render_objmask:
        semantic = pc.get_obj_mask.float()[..., None]

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    foreground, radii, depth, img_opacity, img_flow, img_semantic = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = override_color,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        flow_points = flow_points, # N, 3
        semantic = semantic, # N, K, K <= 32
    )
    
    # semantic 'zero' for background
    # background, flow_bg = env_map.get_image_background(viewpoint_camera, return_grid=True)
    # rendered_image = foreground + (1.0 - img_opacity) * background
    # img_flow = img_flow + (1.0 - img_opacity.detach()) * torch.permute(flow_bg, (2, 0, 1))

    background = env_map.get_image_background(viewpoint_camera)
    rendered_image = foreground + (1.0 - img_opacity) * background
    

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    res = {
        "render": rendered_image,
        "viewspace_points": screenspace_points,
        "visibility_filter" : radii > 0,
        "radii": radii,
        "depth": depth.squeeze(0),
        "opacity": opacity,
        'img_opacity': img_opacity.squeeze(0),
        'foreground': foreground,
        'background': background,
        
        'img_flow': img_flow if flow_points is not None else None,
        'img_semantic': img_semantic if semantic is not None else None,
    }
    if deform_pkg is not None:
        res.update(deform_pkg)
    return res
