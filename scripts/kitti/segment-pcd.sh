for split_mode in nvs-75 nvs-50 nvs-25
do
    python scripts/segment_pcd.py ./data/kitti/0001 --split_mode $split_mode
    python scripts/segment_pcd.py ./data/kitti/0002 --split_mode $split_mode
    python scripts/segment_pcd.py ./data/kitti/0006 --split_mode $split_mode
done