num_cam = 1
order_args = dict(
    # bspline(ctrl_pts, order - 1), poly, fft, slerp(ctrl_pts, order - 1)
    xyz = [None, 5, 0, 6, 0, 0],
    rotation=[0, 0, 0, 0, None, 5],
    shs=[0, 0, 0, 6, 0, 0],
    background=[0, 0, 0, 0, 0, 0],
)