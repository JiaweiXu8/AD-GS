echo $1

for scene in scene006 scene026 scene090 scene105 scene108 scene134 scene150 scene181
do
    python train.py -c ./arguments/waymo.py -s ./data/waymo/$scene -m ./output/waymo/$scene --data_device $1
    python render.py -c ./arguments/waymo.py -m ./output/waymo/$scene --data_device $1 -v
done
