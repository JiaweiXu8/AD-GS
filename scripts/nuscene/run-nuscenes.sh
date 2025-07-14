echo $1

for scene in scene-0230 scene-0242 scene-0255 scene-0295 scene-0518 scene-0749
do
    python train.py -c ./arguments/nuscenes.py -s ./data/nuscenes/$scene -m ./output/nuscenes/$scene --data_device $1
    python render.py -c ./arguments/nuscenes.py -m ./output/nuscenes/$scene --data_device $1 -v --cam_order 1 0 2
done
