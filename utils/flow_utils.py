import torch
from torch.nn.functional import interpolate
from flow_vis import flow_to_color

def flow_points_project(flow_pts, K, R, T, dist=1e-3):
    # flow_pts: N, 3
    proj_pts = (K @ (R @ flow_pts[..., None] + T[..., None]))[..., 0]  # N, 3
    mask = proj_pts[..., 2] > dist
    proj_pts = proj_pts[..., :2] / torch.clamp_min(proj_pts[..., 2:], dist)
    return proj_pts, mask

def get_img_flow(img_flow, flow_pkg, dist=1e-3):
    _, K, R, T, flow, flow_vis = flow_pkg
    _, H, W = flow.shape
    grid = torch.stack(torch.meshgrid(
        torch.arange(0, W, dtype=torch.float32, device=flow.device),
        torch.arange(0, H, dtype=torch.float32, device=flow.device),
        indexing='xy',
    ), dim=-1)

    flow_vis: torch.Tensor = (flow_vis > 0.5) & (flow[0] <= flow.shape[2] - 1.0) & (flow[0] >= 0.0) & (flow[1] <= flow.shape[1] - 1.0) & (flow[1] >= 0.0)
    selected_coord = torch.nonzero(flow_vis, as_tuple=True)
    if selected_coord[0].numel() == 0:
        return torch.permute(grid, (2, 0, 1))
    img_flow = torch.permute(img_flow[:, selected_coord[0], selected_coord[1]], (1, 0))  # N, 3
    img_flow, mask = flow_points_project(img_flow, K, R, T, dist=dist)
    grid[selected_coord[0][mask], selected_coord[1][mask]] = img_flow[mask]
    return torch.permute(grid, (2, 0, 1))

def flow_to_img(flow, mask=None):
    _, H, W = flow.shape
    grid = torch.stack(torch.meshgrid(
        torch.arange(0, W, dtype=torch.float32, device=flow.device),
        torch.arange(0, H, dtype=torch.float32, device=flow.device),
        indexing='xy',
    ), dim=-1)
    delta_flow = flow.permute(1, 2, 0) - grid
    if mask is not None:
        delta_flow = mask[..., None] * delta_flow
    delta_flow = torch.clip(delta_flow, -max(H, W), max(H, W))
    img = flow_to_color(delta_flow.detach().cpu().numpy())
    img = torch.tensor(img).permute(2, 0, 1) / 255.0

    # delta_flow = flow - grid.permute(2, 0, 1)
    # img = torch.linalg.norm(delta_flow, dim=0)
    # if mask is not None:
    #     delta_flow = mask[None] * delta_flow
    # img = (img - img.min()) / (img.max() - img.min())
    # img = img[None].repeat(3,1,1)
    return img