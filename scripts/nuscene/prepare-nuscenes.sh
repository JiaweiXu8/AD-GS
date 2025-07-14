echo $1

python scripts/nuscene/nuscene.py $1 ./data/nuscenes scene-0230 --use_color
python scripts/nuscene/nuscene.py $1 ./data/nuscenes scene-0242 --use_color
python scripts/nuscene/nuscene.py $1 ./data/nuscenes scene-0255 --use_color
python scripts/nuscene/nuscene.py $1 ./data/nuscenes scene-0295 --use_color
python scripts/nuscene/nuscene.py $1 ./data/nuscenes scene-0518 --use_color
python scripts/nuscene/nuscene.py $1 ./data/nuscenes scene-0749 --use_color
