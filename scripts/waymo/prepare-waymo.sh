echo $1

python scripts/waymo/waymo.py $1/individual_files_validation_segment-10448102132863604198_472_000_492_000_with_camera_labels.tfrecord ./data/waymo/scene006 --first_frame 0 --last_frame 85 --use_color
python scripts/waymo/waymo.py $1/individual_files_validation_segment-12374656037744638388_1412_711_1432_711_with_camera_labels.tfrecord ./data/waymo/scene026 --first_frame 0 --last_frame 100 --use_color
python scripts/waymo/waymo.py $1/individual_files_validation_segment-17612470202990834368_2800_000_2820_000_with_camera_labels.tfrecord ./data/waymo/scene090 --first_frame 0 --last_frame 102 --use_color
python scripts/waymo/waymo.py $1/individual_files_validation_segment-1906113358876584689_1359_560_1379_560_with_camera_labels.tfrecord ./data/waymo/scene105 --first_frame 20 --last_frame 186 --use_color
python scripts/waymo/waymo.py $1/individual_files_validation_segment-2094681306939952000_2972_300_2992_300_with_camera_labels.tfrecord ./data/waymo/scene108 --first_frame 20 --last_frame 115 --use_color
python scripts/waymo/waymo.py $1/individual_files_validation_segment-4246537812751004276_1560_000_1580_000_with_camera_labels.tfrecord ./data/waymo/scene134 --first_frame 106 --last_frame 198 --use_color
python scripts/waymo/waymo.py $1/individual_files_validation_segment-5372281728627437618_2005_000_2025_000_with_camera_labels.tfrecord ./data/waymo/scene150 --first_frame 96 --last_frame 197 --use_color
python scripts/waymo/waymo.py $1/individual_files_validation_segment-8398516118967750070_3958_000_3978_000_with_camera_labels.tfrecord ./data/waymo/scene181 --first_frame 0 --last_frame 160 --use_color
