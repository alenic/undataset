from __future__ import annotations

import os
import yaml
from tqdm import tqdm
from typing import TYPE_CHECKING, List

from undata.unbbox import UNBBox
from undata.datasets.od.odsample import ODSample

if TYPE_CHECKING:
    from undata.datasets.od.oddataset import ODDataset


class YOLOReader:
    @classmethod
    def read_sample(cls, yolo_lines: List[str]) -> "ODSample":

        bboxes = []
        for l in yolo_lines:
            try:
                label_id, cx, cy, w, h = list(map(float, l.replace("\n", "").split()))
                label_id = int(label_id)
                bboxes.append(
                    UNBBox(coords=[cx, cy, w, h], format="yolo", label_id=label_id)
                )
            except:
                print("Error in parsing line:", l)

        return ODSample(bbox=bboxes)

    @classmethod
    def read(
        cls,
        classes_path: str,
        annotations_dir: str,
        images_dir: str,
        dataset_cls: type[ODDataset],
        images_lead: bool = True,
    ) -> ODDataset:
        if not os.path.exists(annotations_dir):
            raise ValueError(f"Annotation path: {annotations_dir} does not exists")

        if not os.path.exists(images_dir):
            raise ValueError(f"Images path: {images_dir} does not exists")

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

        labels_map = {idx: name for idx, name in enumerate(classes["names"])}
        undataset = dataset_cls(rootdir=images_dir, labels_map=labels_map)
        # Get the filenames
        images_list = os.listdir(images_dir)
        anns_list = os.listdir(annotations_dir)

        image_names = dict([(os.path.splitext(i)[0], i) for i in images_list])
        annotation_names = dict([(os.path.splitext(a)[0], a) for a in anns_list])

        if images_lead:
            for img_name, img_filename in tqdm(
                image_names.items(), desc=f"Loading from yolo images"
            ):

                # image_path is relative to rootdir
                sample = ODSample(
                    image_path=img_filename,
                )

                # Check if exists a related annotation
                annotation_global_path = os.path.join(
                    annotations_dir,
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
                    images_dir,
                    image_names[ann_name],
                )
                # It's a double check
                if not os.path.exists(image_glob_path):
                    raise ValueError(f"Image path {image_glob_path} does not exists")

                sample = ODSample(
                    image_path=image_names[ann_name],
                )

                annotation_global_path = os.path.join(
                    annotations_dir,
                    ann_name + ".txt",
                )

                with open(annotation_global_path, "r") as fp:
                    yolo_lines = fp.readlines()

                sample = cls.read_sample(yolo_lines)
                sample.compute_image_wh(undataset.rootdir)

                undataset.append(sample)

        return undataset
