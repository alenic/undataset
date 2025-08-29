import copy
import os
import glob
import yaml
from typing import List, Dict, Optional, Union
from collections import defaultdict

import pandas as pd
from pydantic import BaseModel, Field
from tqdm import tqdm

from undata.unsample import UNSample


class UNDataset(BaseModel):
    sample: Dict[int, UNSample] = Field(default_factory=dict)
    rootdir: str = "."
    labels_map: Optional[Dict[int, str]] = None
    tags_map: Optional[Dict[int, str]] = None

    def add_sample(self, sample: UNSample):
        self.sample[len(self.sample)] = sample

    def get_sample(self, idx: int) -> UNSample:
        if idx not in self.sample:
            raise IndexError()

        return self.sample[idx]

    def set_labels_map(self, labels_map: Union[Dict[int, str], List[str]]):
        if isinstance(labels_map, list):
            self.labels_map = {k: labels_map[k] for k in range(len(labels_map))}
        else:
            self.labels_map = copy.deepcopy(labels_map)

    def set_tags_map(self, tags_map: Union[Dict[int, str], List[str]]):
        if isinstance(tags_map, list):
            self.tags_map = {k: tags_map[k] for k in range(len(tags_map))}
        else:
            self.tags_map = copy.deepcopy(tags_map)

    def set_rootdir(self, rootdir: str):
        self.rootdir = rootdir

    # =================== Utility ===============================
    def remap_labels(
        self, remap_dict: Dict[int, Union[int, None]], new_labels_map: Dict[int, str]
    ):
        # Assert that everything is coherent
        for key in remap_dict:
            assert key in self.labels_map.keys()

        for value in remap_dict.values():
            if value is not None:
                assert value in self.labels_map.keys()

        for key, value in remap_dict.items():
            if value is None:
                print(f"Remapping {key}:{self.labels_map[key]} -> {value}")
            else:
                print(
                    f"Remapping {key}:{self.labels_map[key]} -> {value}:{new_labels_map[value]}"
                )

        for idx in tqdm(self.sample.keys(), desc="Remapping Labels"):
            self.sample[idx].remap_label_ids(remap_dict)

        self.set_labels_map(new_labels_map)

        return self

    def compute_image_wh(self):
        for idx in tqdm(self.sample.keys(), desc="Computing Image Width and Height"):
            self.sample[idx].compute_image_wh(self.rootdir)

    def find_duplicate_images(self):
        import hashlib

        def compute_image_hash(image_path: str, hash_algo="md5") -> str:
            hash_func = hashlib.new(hash_algo)
            with open(image_path, "rb") as f:
                while chunk := f.read(8192):
                    hash_func.update(chunk)
            return hash_func.hexdigest()

        hashes = {}
        for idx in tqdm(self.sample.keys(), desc=f"Find duplicated images"):
            path = os.path.join(self.rootdir, self.sample[idx].image_path)
            try:
                image_hash = compute_image_hash(path)
            except Exception as e:
                print(f"Error processing {path}: {e}")
                continue

            if image_hash in hashes:
                hashes[image_hash].append(idx)
            else:
                hashes[image_hash] = [idx]

        return list(hashes.values())

    def bbox_convert(self, to_format: str):
        for idx in tqdm(self.sample.keys(), desc=f"Converting BBoxes to {to_format}"):
            self.sample[idx].bbox_convert(to_format, inplace=True)

    def get_label_counts(self):
        label_counts = defaultdict(int)
        for idx in tqdm(self.sample.keys(), desc=f"Counting BBoxes labels"):
            sample_label_counts = self.sample[idx].get_label_counts()
            for cidx, count_value in sample_label_counts.items():
                label_counts[cidx] += count_value
        return label_counts

    # =================== Import/Export =========================
    def export_to_json(self, json_file: str, indent: int = 2):
        with open(json_file, "w") as fp:
            fp.write(self.model_dump_json(indent=indent))

    def load_from_json(self, json_file: str):
        with open(json_file, "r") as fp:
            self = self.model_validate_json(fp.read())

        return self

    def as_dataframe(self):
        df = pd.DataFrame()
        frames = []
        for idx in tqdm(self.sample.keys(), desc="Convert UNDataset as DataFrame"):
            sample_df = self.sample[idx].as_dataframe()
            sample_df["index"] = idx
            frames.append(sample_df)

        df = pd.concat(frames, ignore_index=True)

        return df.reset_index(drop=True)

    def from_dataframe(self, df):
        for idx, group in tqdm(
            df.groupby("index"),
            desc="Loading from Dataframe",
        ):
            sample = UNSample()
            sample.from_dataframe(group)
            self.sample[idx] = sample

        return self

    def export_to_yolo(self, ann_path: str, exist_ok: bool = True):
        export = False
        if os.path.exists(ann_path):
            accept = input(
                f"{ann_path} already exists, do you want to conitnue? (y/n): "
            )
            if (accept.lower().strip() == "y") | (accept.lower().strip() == "yes"):
                export = True
            else:
                export = False
        else:
            os.makedirs(ann_path, exist_ok=exist_ok)
            export = True

        if export:
            for idx in tqdm(self.sample.keys(), desc=f"Exporting to yolo annotations"):
                yolo_str = self.sample[idx].yolo_dumps()
                image_name = os.path.basename(self.sample[idx].image_path)
                image_name, _ = os.path.splitext(image_name)
                with open(os.path.join(ann_path, image_name + ".txt"), "w") as fp:
                    fp.write(yolo_str)

    def load_from_yolo(self, classes_path: str, ann_path: str, images_path: str):
        if not os.path.exists(ann_path):
            raise ValueError(f"Annotation path: {ann_path} does not exists")

        if not os.path.exists(images_path):
            raise ValueError(f"Images path: {images_path} does not exists")

        if not os.path.exists(classes_path):
            raise ValueError(f"Classes path: {classes_path} does not exists")

        if classes_path.endswith(".yaml"):
            with open(classes_path, "r") as fp:
                classes = yaml.safe_load(fp)

        elif classes_path.endswith(".txt"):
            with open(classes_path, "r") as fp:
                classes = {"names": [line.strip() for line in fp.readlines()]}

        self.rootdir = images_path
        self.set_labels_map(classes["names"])

        annotations_glob = glob.glob(os.path.join(ann_path, "*.txt"))
        images = os.listdir(images_path)
        images_name_ext = [os.path.splitext(p) for p in images]
        images_name = [p[0] for p in images_name_ext]
        images_ext = [p[1] for p in images_name_ext]
        for ann in tqdm(annotations_glob, desc=f"Loading from yolo annotations"):
            filename, _ = os.path.splitext(os.path.basename(ann))

            if filename not in images_name:
                raise ValueError(f"{filename} is not present in images")

            image_name_idx = images_name.index(filename)
            image_path = images_name[image_name_idx] + images_ext[image_name_idx]

            image_glob_path = os.path.join(
                images_path,
                images_name[image_name_idx] + images_ext[image_name_idx],
            )
            # It's a double check
            if not os.path.exists(image_glob_path):
                raise ValueError(f"Image path {image_glob_path} does not exists")

            sample = UNSample(
                image_path=image_path,
            )

            with open(ann, "r") as fp:
                yolo_lines = fp.readlines()

            sample.yolo_loads(yolo_lines)
            sample.compute_image_wh(self.rootdir)

            self.add_sample(sample)

        return self

    def export_to_voc(self, ann_path: str):
        export = False
        if os.path.exists(ann_path):
            accept = input(
                f"{ann_path} already exists, do you want to conitnue? (y/n): "
            )
            if (accept.lower().strip() == "y") | (accept.lower().strip() == "yes"):
                export = True
            else:
                export = False
        else:
            os.makedirs(ann_path, exist_ok=True)
            export = True

        if export:
            for idx in tqdm(self.sample.keys(), desc=f"Exporting to yolo annotations"):
                voc_str = self.sample[idx].voc_dumps(self.labels_map)
                image_name = os.path.basename(self.sample[idx].image_path)
                image_name, _ = os.path.splitext(image_name)
                with open(os.path.join(ann_path, image_name + ".txt"), "w") as fp:
                    fp.write(voc_str)

    def __iter__(self):
        for i in range(len(self.sample)):
            yield self.get_sample(i)

    def __len__(self) -> int:
        return len(self.sample)

    def __getitem__(self, idx: int) -> UNSample:
        return self.get_sample(idx)
