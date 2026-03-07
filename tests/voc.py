import undata as ud
from IPython import embed

dataset = ud.UNDataset.read_voc(
    annotations_dir="/home/alenic/datasets/DroneDetectionDataset/train/Drone_TrainSet_XMLs",
    images_dir="/home/alenic/datasets/DroneDetectionDataset/train/Drone_TrainSet"
)

embed()