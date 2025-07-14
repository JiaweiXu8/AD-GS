echo $1

for mode in 75 50 25
do
    for scene in 0001 0002 0006
    do
        python train.py -c ./arguments/kitti-$mode.py -s ./data/kitti/$scene -m ./output/kitti/nvs-$mode/$scene --split_mode nvs-$mode --data_device $1
        python render.py -c ./arguments/kitti-$mode.py -m ./output/kitti/nvs-$mode/$scene --data_device $1 -v
    done
done
