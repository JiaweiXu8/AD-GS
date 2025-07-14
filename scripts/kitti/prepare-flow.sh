python scripts/flow.py ./data/kitti/0001 --split_mode nvs-75 --step 4
python scripts/flow.py ./data/kitti/0002 --split_mode nvs-75 --step 4
python scripts/flow.py ./data/kitti/0006 --split_mode nvs-75 --step 4

python scripts/flow.py ./data/kitti/0001 --split_mode nvs-50 --step 2
python scripts/flow.py ./data/kitti/0002 --split_mode nvs-50 --step 2
python scripts/flow.py ./data/kitti/0006 --split_mode nvs-50 --step 2

python scripts/flow.py ./data/kitti/0001 --split_mode nvs-25 --step 1
python scripts/flow.py ./data/kitti/0002 --split_mode nvs-25 --step 1
python scripts/flow.py ./data/kitti/0006 --split_mode nvs-25 --step 1