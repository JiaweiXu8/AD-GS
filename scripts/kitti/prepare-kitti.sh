echo $1

python scripts/kitti/kitti.py $1 ./data/kitti 0001 --first_frame 380 --last_frame 431 --use_color
python scripts/kitti/kitti.py $1 ./data/kitti 0002 --first_frame 140 --last_frame 224 --use_color
python scripts/kitti/kitti.py $1 ./data/kitti 0006 --first_frame 65 --last_frame 120 --use_color

