order_args = dict(
    # bspline(ctrl_pts, order - 1), poly, fft, slerp(ctrl_pts, order - 1)
    xyz=[None, 2, 0, 6, 0, 0],
    rotation=[0, 0, 0, 0, None, 2],
    shs=[0, 0, 0, 6, 0, 0],
    background=[None, 2, 0, 6, 0, 0],
)

num_cam = 2
obj_deform_lr_scale = 0.1
object_extent = 5.0
min_camera_extent = 5.0