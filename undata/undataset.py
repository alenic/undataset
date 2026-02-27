import copy
import os
import hashlib
import yaml
from typing import List, Dict, Optional, Union
from collections import defaultdict

import pandas as pd
from pydantic import BaseModel, PrivateAttr, Field
from tqdm import tqdm

from undata.unsample import UNSample
from undata.untypes import BBoxFormatType
from undata.converters.yolo import YOLOConverter


class UNDataset(BaseModel):
    rootdir: str = "."
    labels_map: Optional[Union[Dict[int, str], List[str]]] = None
    tags_map: Optional[Dict[int, str]] = None

    sample: Dict[int, UNSample] = Field(default_factory=dict)
    _path_to_id_map: Dict[str, int] = PrivateAttr(default_factory=dict)  # initial value

    def append(self, sample: UNSample) -> "UNDataset":
        new_id = len(self.sample)
        while new_id in self.sample:
            new_id += 1

        self.sample[new_id] = sample
        # Keep track of the id, given an image path
        self._path_to_id_map[sample.image_path] = new_id
        return self

    def get_paths_to_ids(self):
        for idx, sample in self.sample.items():
            self._path_to_id_map[sample.image_path] = idx

    def reset_index(self, sort_by_image_path: bool = True) -> "UNDataset":
        if sort_by_image_path:
            sorted_sample = {
                k: v
                for k, v in sorted(
                    self.sample.items(), key=lambda item: item[1].image_path
                )
            }
            self.sample = sorted_sample
        else:
            self.sample = {i: v for i, v in enumerate(self.sample.values())}

        return self

    def get_sample(self, idx: int, inplace: bool = False) -> UNSample:
        if idx not in self.sample:
            raise IndexError()

        s = self.sample[idx]
        return s.model_copy(deep=True) if not inplace else s

    def get_image_paths(self) -> List[str]:
        image_paths = []

        for k, s in self.sample.items():
            image_paths.append(os.path.join(self.rootdir, s.image_path))

        return image_paths

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
    def remap_labels(self, remap_dict: Dict[int, int], new_labels_map: Dict[int, str]):
        if self.labels_map is not None and remap_dict is not None:
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

    def filter_bbox_labels(self, keep_ids: List, inplace=False):
        if not inplace:
            dataset_res = self.model_copy(deep=True)
        for idx in tqdm(self.sample.keys(), desc="Filtering BBoxes"):
            if inplace:
                self.sample[idx].filter_bbox_labels(keep_ids=keep_ids, inplace=True)
            else:

                dataset_res.get_sample(idx, inplace=True).filter_bbox_labels(
                    keep_ids=keep_ids, inplace=True
                )
        if not inplace:
            return dataset_res

        return self

    def compute_image_wh(self):
        for idx in tqdm(self.sample.keys(), desc="Computing Image Width and Height"):
            self.sample[idx].compute_image_wh(self.rootdir)

    def find_duplicate_images(self) -> List[int]:

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

    def bbox_convert(self, to_format: BBoxFormatType, inplace: bool = False):
        # Return a copy of the dataset
        if not inplace:
            new_dataset = copy.deepcopy(self)
            new_dataset.bbox_convert(to_format=to_format, inplace=True)
            return new_dataset

        for idx in tqdm(self.sample.keys(), desc=f"Converting BBoxes to {to_format}"):
            self.sample[idx].bbox_convert(to_format, inplace=inplace)

        return self

    def get_labels_counts(self):
        label_counts = defaultdict(int)
        for idx in tqdm(self.sample.keys(), desc=f"Counting BBoxes labels"):
            sample_label_counts = self.sample[idx].get_labels_counts()
            for cidx, count_value in sample_label_counts.items():
                label_counts[cidx] += count_value
        return label_counts

    def get_stats(self):
        stats = {
            "num_samples": len(self.sample),
            "num_bboxes": 0,
            "num_bboxes_per_label": {},
            "num_labels": 0,
        }
        image_widths = []
        image_heights = []

        # Bounding box stats
        bbox_widths = []
        bbox_heights = []
        bbox_areas = []

        label_counts = self.get_labels_counts()
        label_counts = dict(label_counts)
        sorted_dict = dict(sorted(label_counts.items()))
        stats["num_bboxes_per_label_id"] = sorted_dict
        if self.labels_map:
            stats["num_bboxes_per_label_name"] = {
                self.labels_map[k]: v for k, v in sorted_dict.items()
            }
        stats["num_labels"] = len(label_counts)
        stats["num_bboxes"] = sum(label_counts.values())

        for idx in tqdm(self.sample.keys(), desc=f"Computing dataset stats"):
            sample = self.sample[idx]
            if sample.image_w is not None:
                image_widths.append(sample.image_w)
            if sample.image_h is not None:
                image_heights.append(sample.image_h)
            # --- BBox stats ---

            if sample.bbox is not None:
                sample_c = sample.bbox_convert("rel_xywh", rounded=True, inplace=False)
                for bbox in sample_c.bbox:
                    # Example: bbox = (x_min, y_min, x_max, y_max)
                    _, _, w, h = bbox.coords
                    bbox_widths.append(w)
                    bbox_heights.append(h)
                    bbox_areas.append(w * h)

        stats["min_image_width"] = min(image_widths) if image_widths else None
        stats["max_image_width"] = max(image_widths) if image_widths else None
        stats["avg_image_width"] = (
            sum(image_widths) / len(image_widths) if image_widths else None
        )

        stats["min_image_height"] = min(image_heights) if image_heights else None
        stats["max_image_height"] = max(image_heights) if image_heights else None
        stats["avg_image_height"] = (
            sum(image_heights) / len(image_heights) if image_heights else None
        )

        # Bounding box stats
        stats["min_bbox_width"] = min(bbox_widths) if bbox_widths else None
        stats["max_bbox_width"] = max(bbox_widths) if bbox_widths else None
        stats["avg_bbox_width"] = (
            sum(bbox_widths) / len(bbox_widths) if bbox_widths else None
        )

        stats["min_bbox_height"] = min(bbox_heights) if bbox_heights else None
        stats["max_bbox_height"] = max(bbox_heights) if bbox_heights else None
        stats["avg_bbox_height"] = (
            sum(bbox_heights) / len(bbox_heights) if bbox_heights else None
        )

        stats["min_bbox_area"] = min(bbox_areas) if bbox_areas else None
        stats["max_bbox_area"] = max(bbox_areas) if bbox_areas else None
        stats["avg_bbox_area"] = (
            sum(bbox_areas) / len(bbox_areas) if bbox_areas else None
        )

        return stats

    # =================== Filters ===============================
    def filter_image_size(
        self,
        min_width: Optional[int] = None,
        max_width: Optional[int] = None,
        min_height: Optional[int] = None,
        max_height: Optional[int] = None,
    ):
        """
        Returns a new UNDataset containing only samples where
        min_width <= image_w <= max_width and min_height <= image_h <= max_height.
        """
        filtered = UNDataset(
            rootdir=self.rootdir,
            labels_map=copy.deepcopy(self.labels_map),
            tags_map=copy.deepcopy(self.tags_map),
        )
        for idx, sample in self.sample.items():
            w = getattr(sample, "image_w", None)
            h = getattr(sample, "image_h", None)
            if w is None or h is None:
                continue
            if min_width is not None and w < min_width:
                continue
            if max_width is not None and w > max_width:
                continue
            if min_height is not None and h < min_height:
                continue
            if max_height is not None and h > max_height:
                continue
            filtered.append(copy.deepcopy(sample))
        return filtered

    def filter_bbox_size(
        self,
        min_width: Optional[float] = None,
        max_width: Optional[float] = None,
        min_height: Optional[float] = None,
        max_height: Optional[float] = None,
    ):
        """
        Returns a new UNDataset where each sample contains only bounding boxes
        with min_width <= bbox_width <= max_width and min_height <= bbox_height <= max_height.
        Samples with no remaining bboxes are excluded.
        """
        filtered = UNDataset(
            rootdir=self.rootdir,
            labels_map=copy.deepcopy(self.labels_map),
            tags_map=copy.deepcopy(self.tags_map),
        )
        for idx, sample in self.sample.items():
            # Deepcopy to avoid modifying the original sample
            sample_copy = copy.deepcopy(sample)
            sample_copy.bbox_convert(to_format="rel_xywh")
            if sample_copy.bbox is not None:
                filtered_bboxes = []
                for bbox in sample_copy.bbox:
                    # Assume bbox.coords = (x, y, w, h) or similar
                    _, _, w, h = bbox.coords
                    if min_width is not None and w < min_width:
                        continue
                    if max_width is not None and w > max_width:
                        continue
                    if min_height is not None and h < min_height:
                        continue
                    if max_height is not None and h > max_height:
                        continue
                    filtered_bboxes.append(bbox)
                if filtered_bboxes:
                    sample_copy.bbox = filtered_bboxes
                    filtered.append(sample_copy)
            else:
                # If no bbox, skip sample
                continue
        return filtered

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
        grouped_df = df.groupby("index")
        for idx, group in tqdm(
            grouped_df,
            desc="Loading from Dataframe",
        ):
            self.sample[idx] = UNSample.from_dataframe(group)

        return self

    # ========= YOLO Converter ===========
    def export_to_yolo(self, ann_path: str, exist_ok: bool = True):
        return YOLOConverter.write(
            self,
            ann_path,
            exist_ok,
        )

    def load_from_yolo(
        self,
        classes_path: str,
        anns_root: str,
        images_root: str,
        images_lead: bool = True,
    ):
        return YOLOConverter.read(
            classes_path,
            anns_root,
            images_root,
            images_lead,
        )
    # ==================================

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
