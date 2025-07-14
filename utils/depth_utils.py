import torch

def get_scaled_shifted_depth(prediction, target, mask=None):
    # input: H, W
    scale, shift = normalized_depth_scale_and_shift(prediction, target, mask)
    depthmap = scale * prediction + shift
    return depthmap

def normalized_depth_scale_and_shift(prediction, target, mask=None):
    """
    More info here: https://arxiv.org/pdf/2206.00665.pdf supplementary section A2 Depth Consistency Loss
    This function computes scale/shift required to normalizes predicted depth map,
    to allow for using normalized depth maps as input from monocular depth estimation networks.
    These networks are trained such that they predict normalized depth maps.

    Solves for scale/shift using a least squares approach with a closed form solution:
    Based on:
    https://github.com/autonomousvision/monosdf/blob/d9619e948bf3d85c6adec1a643f679e2e8e84d4b/code/model/loss.py#L7
    Args:
        prediction: predicted depth map
        target: ground truth depth map
        mask: mask of valid pixels
    Returns:
        scale and shift for depth prediction
    """
    if mask is None:
        mask = torch.ones_like(prediction)
        
    # system matrix: A = [[a_00, a_01], [a_10, a_11]]
    a_00 = torch.sum(mask * prediction * prediction)
    a_01 = torch.sum(mask * prediction)
    a_11 = torch.sum(mask)

    # right hand side: b = [b_0, b_1]
    b_0 = torch.sum(mask * prediction * target)
    b_1 = torch.sum(mask * target)

    # solution: x = A^-1 . b = [[a_11, -a_01], [-a_10, a_00]] / (a_00 * a_11 - a_01 * a_10) . b
    det = a_00 * a_11 - a_01 * a_01
    if det == 0:
        return 0.0, 0.0

    scale = (a_11 * b_0 - a_01 * b_1) / det
    shift = (-a_01 * b_0 + a_00 * b_1) / det

    return scale, shift