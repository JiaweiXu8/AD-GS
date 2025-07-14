num_cam = 3
order_args = dict(
    # bspline(ctrl_pts, order - 1), poly, fft, slerp(ctrl_pts, order - 1)
    xyz = [None, 5, 0, 6, 0, 0],
    rotation=[0, 0, 0, 0, None, 5],
    shs=[0, 0, 0, 6, 0, 0],
    background=[None, 5, 0, 6, 0, 0],
)