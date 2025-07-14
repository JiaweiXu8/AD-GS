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
import sys
from datetime import datetime
import numpy as np
import random

def inverse_sigmoid(x):
    return torch.log(x/(1-x))

def PILtoTorch(pil_image, resolution):
    resized_image_PIL = pil_image.resize(resolution)
    resized_image = torch.from_numpy(np.array(resized_image_PIL)) / 255.0
    if len(resized_image.shape) == 3:
        return resized_image.permute(2, 0, 1)
    else:
        return resized_image.unsqueeze(dim=-1).permute(2, 0, 1)

def get_expon_lr_func(
    lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000
):
    """
    Copied from Plenoxels

    Continuous learning rate decay function. Adapted from JaxNeRF
    The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
    is log-linearly interpolated elsewhere (equivalent to exponential decay).
    If lr_delay_steps>0 then the learning rate will be scaled by some smooth
    function of lr_delay_mult, such that the initial learning rate is
    lr_init*lr_delay_mult at the beginning of optimization but will be eased back
    to the normal learning rate when steps>lr_delay_steps.
    :param conf: config subtree 'lr' or similar
    :param max_steps: int, the number of steps during optimization.
    :return HoF which takes step as input
    """

    def helper(step):
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            # Disable this parameter
            return 0.0
        if lr_delay_steps > 0:
            # A kind of reverse cosine decay.
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip(step / max_steps, 0, 1)
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return delay_rate * log_lerp

    return helper

def strip_lowerdiag(L):
    uncertainty = torch.stack([L[:, 0, 0], L[:, 0, 1], L[:, 0, 2], L[:, 1, 1], L[:, 1, 2], L[:, 2, 2]], dim=-1)
    return uncertainty

def strip_symmetric(sym):
    return strip_lowerdiag(sym)

def fill_symmetric(sym):
    mat = torch.stack([
        sym[..., 0], sym[..., 1], sym[..., 2],
        sym[..., 1], sym[..., 3], sym[..., 4],
        sym[..., 2], sym[..., 4], sym[..., 5],
    ], dim=-1).reshape(-1, 3, 3)
    return mat

def build_rotation(r, normalized=False):
    if not normalized:
        norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])
        q = r / norm[:, None]
    else:
        q = r

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]
    R = torch.stack([
        1 - 2 * (y*y + z*z), 2 * (x*y - r*z), 2 * (x*z + r*y),
        2 * (x*y + r*z), 1 - 2 * (x*x + z*z), 2 * (y*z - r*x),
        2 * (x*z - r*y), 2 * (y*z + r*x), 1 - 2 * (x*x + y*y),
    ], dim=-1).reshape(-1, 3, 3)
    return R

def build_scaling_rotation(s, r):
    L = torch.diag_embed(s)
    R = build_rotation(r)
    L = R @ L
    return L

def quaternion_multiply(q1, q2):
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    return torch.stack((w, x, y, z), dim=-1)

def quaternion_conjugate(q):
    return q * torch.tensor([1.0, -1.0, -1.0, -1.0], dtype=torch.float32, device=q.device)

def quaternion_log(q, dim=-1):
    q_norm = torch.clamp_min(torch.linalg.norm(q, dim=dim, keepdim=True), 1e-5)
    s, v = torch.split(q, (1, 3), dim=dim)
    v = torch.nn.functional.normalize(v, dim=dim)
    res = torch.cat([torch.log(q_norm), v * torch.arccos(s / q_norm)], dim=dim)
    return res

def quaternion_exp(q, dim=-1):
    s, v = torch.split(q, (1, 3), dim=dim)
    v_norm = torch.linalg.norm(v, dim=dim, keepdim=True)
    res = torch.cat([torch.cos(v_norm), torch.sin(v_norm) * v / torch.clamp_min(v_norm, 1e-5)], dim=dim)
    res = torch.exp(s) * res
    return res

def safe_state(silent):
    old_f = sys.stdout
    class F:
        def __init__(self, silent):
            self.silent = silent

        def write(self, x):
            if not self.silent:
                if x.endswith("\n"):
                    old_f.write(x.replace("\n", " [{}]\n".format(str(datetime.now().strftime("%d/%m %H:%M:%S")))))
                else:
                    old_f.write(x)

        def flush(self):
            old_f.flush()

    sys.stdout = F(silent)

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    # torch.cuda.set_device(torch.device("cuda:0"))

def ellipse_surface(s):
    return s[..., 0:1] * s[..., 1:2] + s[..., 0:1] * s[..., 2:3] + s[..., 1:2] * s[..., 2:3]

