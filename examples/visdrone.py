import undata as ud
import os

"""
  <bbox_left>,<bbox_top>,<bbox_width>,<bbox_height>,<score>,<object_category>,<truncation>,<occlusion>

    Name                                                  Description
-------------------------------------------------------------------------------------------------------------------------------     
 <bbox_left>	     The x coordinate of the top-left corner of the predicted bounding box

 <bbox_top>	     The y coordinate of the top-left corner of the predicted object bounding box

 <bbox_width>	     The width in pixels of the predicted object bounding box

<bbox_height>	     The height in pixels of the predicted object bounding box

   <score>	     The score in the DETECTION file indicates the confidence of the predicted bounding box enclosing 
                     an object instance.
                     The score in GROUNDTRUTH file is set to 1 or 0. 1 indicates the bounding box is considered in evaluation, 
                     while 0 indicates the bounding box will be ignored.
                      
<object_category>    The object category indicates the type of annotated object, (i.e., ignored regions(0), pedestrian(1), 
                     people(2), bicycle(3), car(4), van(5), truck(6), tricycle(7), awning-tricycle(8), bus(9), motor(10), 
                     others(11))
                      
<truncation>	     The score in the DETECTION result file should be set to the constant -1.
                     The score in the GROUNDTRUTH file indicates the degree of object parts appears outside a frame 
                     (i.e., no truncation = 0 (truncation ratio 0%), and partial truncation = 1 (truncation ratio 1% ~ 50%)).
                      
<occlusion>	     The score in the DETECTION file should be set to the constant -1.
                     The score in the GROUNDTRUTH file indicates the fraction of objects being occluded (i.e., no occlusion = 0 
                     (occlusion ratio 0%), partial occlusion = 1 (occlusion ratio 1% ~ 50%), and heavy occlusion = 2 
                     (occlusion ratio 50% ~ 100%)).
"""


data_root = os.environ["DATASET_ROOT"]

data_dir = os.path.join(data_root, "visdrone", "VisDrone2019-DET-test-dev")
images_dir = os.path.join(data_dir, "images")
annotations_dir = os.path.join(data_dir, "annotations")


dataset = ud.UNDataset(
    rootdir=images_dir,
    labels_map=[
        "pedestrian",
        "people",
        "bicycle",
        "car",
        "van",
        "truck",
        "tricycle",
        "awning-tricycle",
        "bus",
        "motor",
    ],  # 1 to 10
)

image_files = os.listdir(images_dir)

for img_file in image_files:
    image_path = os.path.join(images_dir, img_file)

    if not os.path.isfile(image_path):
        print("Skipping", image_path)
        continue

    name, ext = os.path.splitext(image_path)
    ann_path = os.path.join(annotations_dir, img_file.replace(ext, ".txt"))

    # Create bboxes
    bbox_list = []
    lines = open(ann_path, "r").readlines()
    fail = False
    for line in lines:
        try:
            x, y, w, h, score, cat, trunc, occ = map(int, line.split(","))
            cat = cat - 1  # mat from 0 to C-1

            if score == 1 and cat <= 9:
                bb = ud.UNBBox(coords=[x, y, w, h], format="xywh", label_id=cat)
                bbox_list.append(bb)

        except:
            print("Error in bbox", ann_path)
            fail = True
            break

    if not fail:
        sample = ud.UNSample(image_path=img_file, bbox=bbox_list)

        dataset.append(sample)

dataset.compute_image_wh()
df = dataset.to_dataframe()
print(df)

# df = dataset.as_dataframe()
#


# Export to yolo
# yolo_anns_dir = os.path.join(data_dir, "anns")
# dataset.to_yolo(ann_path=yolo_anns_dir, exist_ok=True)

# Export to undataset
dataset.to_json(os.path.join(data_dir, "VisDrone2019-DET-val-undataset.json"))
