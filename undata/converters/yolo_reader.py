import os
import yaml
from tqdm import tqdm
from typing import TYPE_CHECKING

from undata.converters.base import UNDatasetReader

if TYPE_CHECKING:
    from undata.undataset import UNDataset


class YOLOReader(UNDatasetReader):

    def read(
        self,
        classes_path: str,
        anns_root: str,
        images_root: str,
        images_lead: bool = True,
    ):
        if not os.path.exists(anns_root):
            raise ValueError(f"Annotation path: {anns_root} does not exists")

        if not os.path.exists(images_root):
            raise ValueError(f"Images path: {images_root} does not exists")

        if not os.path.exists(classes_path):
            raise ValueError(f"Classes path: {classes_path} does not exists")

        if classes_path.endswith(".yaml"):
            with open(classes_path, "r") as fp:
                classes = yaml.safe_load(fp)

        elif classes_path.endswith(".txt"):
            with open(classes_path, "r") as fp:
                classes = {"names": [line.strip() for line in fp.readlines()]}
        else:
            raise ValueError("Invalid classes_path, you must provide .txt or .yaml")

        from undata.undataset import UNDataset
        from undata.unsample import UNSample

        labels_map = {idx: name for idx, name in enumerate(classes["names"])}
        undataset = UNDataset(rootdir=images_root, labels_map=labels_map)
        # Get the filenames
        images_list = os.listdir(images_root)
        anns_list = os.listdir(anns_root)

        image_names = dict([(os.path.splitext(i)[0], i) for i in images_list])
        annotation_names = dict([(os.path.splitext(a)[0], a) for a in anns_list])

        if images_lead:
            for img_name, img_filename in tqdm(
                image_names.items(), desc=f"Loading from yolo images"
            ):

                # image_path is relative to rootdir
                sample = UNSample(
                    image_path=img_filename,
                )

                # Check if exists a related annotation
                annotation_global_path = os.path.join(
                    anns_root,
                    img_name + ".txt",
                )

                if os.path.exists(annotation_global_path):
                    with open(annotation_global_path, "r") as fp:
                        yolo_lines = fp.readlines()

                    sample.yolo_loads(yolo_lines)
                else:
                    # If annotations doesn't exist, it means that it a background image
                    pass
                sample.compute_image_wh(undataset.rootdir)
                undataset.append(sample)
        else:
            for ann_name, ann_filename in tqdm(
                annotation_names.items(), desc=f"Loading from yolo annotations"
            ):

                # Check if the associated image exists
                image_glob_path = os.path.join(
                    images_root,
                    image_names[ann_name],
                )
                # It's a double check
                if not os.path.exists(image_glob_path):
                    raise ValueError(f"Image path {image_glob_path} does not exists")

                sample = UNSample(
                    image_path=image_names[ann_name],
                )

                annotation_global_path = os.path.join(
                    anns_root,
                    ann_name + ".txt",
                )

                with open(annotation_global_path, "r") as fp:
                    yolo_lines = fp.readlines()

                sample.yolo_loads(yolo_lines)
                sample.compute_image_wh(undataset.rootdir)

                undataset.append(sample)

        return undataset
