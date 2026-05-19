import copy
import os
import hashlib
from typing import List, Dict, Optional, Union, Tuple

from collections import defaultdict

import pandas as pd
from pydantic import BaseModel, PrivateAttr, Field, field_validator
from tqdm import tqdm

from undata.unsample import UNSample
from undata.untypes import BBoxFormatType
from undata.converters import YOLOReader, YOLOWriter, VOCReader, VOCWriter

from PIL import Image


class UNDataset(BaseModel):
    rootdir: str = "."
    labels_map: Optional[Dict[int, str]] = Field(
        default=None, exclude_if=lambda x: x is None
    )
    tags_map: Optional[Dict[int, str]] = Field(
        default=None, exclude_if=lambda x: x is None
    )

    sample: Dict[int, UNSample] = Field(default_factory=dict)
    _next_sample_id: int = PrivateAttr(default=0)

    # ===================== Internal Operations ===================

    @field_validator("labels_map", "tags_map", mode="before")
    @classmethod
    def _normalize_map(cls, v):
        if v is None:
            return None
        if isinstance(v, list):
            return {i: name for i, name in enumerate(v)}
        if isinstance(v, dict):
            # normalize keys to int in case they come as "0", "1", ...
            return {int(k): str(val) for k, val in v.items()}
        raise TypeError("Expected list[str] or dict[int, str]")

    def _refresh_next_sample_id(self):
        if not self.sample:
            self._next_sample_id = 0
            return
        self._next_sample_id = max(self.sample.keys()) + 1

    def model_post_init(self, __context):
        self._refresh_next_sample_id()

    # ===================== Basic Operations ======================

    def append(self, sample: UNSample) -> "UNDataset":
        """
        Append a sample to the dataset. You have to create an UNSample first

        Parameters
        ----------
        sample : UNSample

        Returns
        -------
        The same UNDataset
        """
        if not isinstance(sample, UNSample):
            raise TypeError()

        if self._next_sample_id in self.sample:
            self._refresh_next_sample_id()

        new_id = self._next_sample_id
        self.sample[new_id] = sample
        self._next_sample_id = new_id + 1
        return self

    def delete(self, idx: int) -> "UNDataset":
        """
        Delete a specific sample, given its id

        Parameters
        ----------
        idx : int
        The id of the sample that has to be deleted

        Returns
        -------
        The same UNDataset
        """
        if idx not in self.sample:
            raise IndexError(f"Index {idx} does not exists")

        del self.sample[idx]
        return self

    def get_sample(self, idx: int, inplace: bool = False) -> UNSample:
        if idx not in self.sample:
            raise IndexError(f"Index {idx} does not exists")

        sample = self.sample[idx]
        return sample.model_copy(deep=True) if not inplace else sample

    def items(self, inplace: bool = False):
        for idx in self.sample.keys():
            yield idx, self.get_sample(idx, inplace=inplace)

    def __iter__(self):
        for idx in self.sample.keys():
            yield self.get_sample(idx)

    def __getitem__(self, idx: int) -> UNSample:
        return self.get_sample(idx, inplace=True)

    def __len__(self) -> int:
        return len(self.sample)

    def set_rootdir(self, rootdir: str, check: bool = False):
        if check:
            if not os.path.exists(rootdir):
                raise FileNotFoundError(f"Root directory does not exist: {rootdir}")
            if not os.path.isdir(rootdir):
                raise NotADirectoryError(f"Root path is not a directory: {rootdir}")

        self.rootdir = rootdir

    def set_labels_map(self, labels_map: Union[Dict[int, str], List[str]]):
        self.labels_map = self._normalize_map(copy.deepcopy(labels_map))

    def set_tags_map(self, tags_map: Union[Dict[int, str], List[str]]):
        self.tags_map = self._normalize_map(copy.deepcopy(tags_map))

    def reset_index(self) -> "UNDataset":
        sorted_index = sorted(self.sample.keys())
        self.sample = {i: v for i, v in enumerate(sorted_index)}
        self._refresh_next_sample_id()
        return self

    # =================== Helpers ===============================
    def image_paths(self) -> List[str]:
        image_paths = []

        for sample in self.sample.values():
            image_paths.append(os.path.join(self.rootdir, sample.image_path))

        return image_paths

    def remove_labels(
        self, label_ids: List[int], keep_labels_map: bool = False, inplace=False
    ) -> "UNDataset":
        """
        Remove bounding boxes with the given label ids from every sample.

        Parameters
        ----------
        label_ids : List[int]
            Label ids to remove across the dataset.

        inplace : bool, default=False
            If `True`, modify the current dataset. Otherwise, return a deep-copied
            dataset with the matching bounding boxes removed from each sample.

        Returns
        -------
        UNDataset
            The updated dataset instance.
        """
        if not inplace:
            dataset_res = self.model_copy(deep=True)
            dataset_res.remove_labels(
                label_ids=label_ids,
                keep_labels_map=keep_labels_map,
                inplace=True,
            )
            return dataset_res

        for idx in tqdm(self.sample.keys(), desc="Removing Labels"):
            self.sample[idx].remove_labels(label_ids=label_ids, inplace=True)

        if not keep_labels_map:
            self.normalize_labels_map(ascending_order=False, inplace=True)

        return self

    def merge_labels(
        self,
        map_merge: Dict[int, List[int]],
        new_labels_map: Dict[int, str],
        inplace=False,
    ) -> "UNDataset":
        """
        Merge multiple source label ids into a smaller set of target label ids.
        Labels not included in `map_merge` are preserved and appended with new
        incremental ids after the merged ones. You must provide the names for
        the merged target ids in `new_labels_map`.

        Parameters
        ----------
        map_merge : Dict[int, List[int]]
            Mapping from each merged target id to the list of original label ids
            that should be collapsed into it. For example:

            {
                0: [1, 4],
                1: [0],
                2: [2, 3],
            }

            A source label id cannot appear in more than one merge group.

        new_labels_map : Dict[int, str]
            Names for the merged target ids. Its keys must exactly match the
            keys of `map_merge`. For example:

            {
                0: "cat",
                1: "dog",
                2: "other",
            }

        inplace : bool, default=False
            If `True`, modify the current dataset. Otherwise, return a
            deep-copied dataset with the merged labels applied.

        Returns
        -------
        UNDataset
            The updated dataset instance.
        """
        if not inplace:
            dataset_res = self.model_copy(deep=True)
            dataset_res.merge_labels(
                map_merge=map_merge,
                new_labels_map=new_labels_map,
                inplace=True,
            )
            return dataset_res

        if self.labels_map is None:
            raise ValueError("labels_map is required to merge labels")

        normalized_new_labels_map = self._normalize_map(copy.deepcopy(new_labels_map))

        merge_keys = set(map_merge.keys())
        new_label_keys = set(normalized_new_labels_map.keys())
        if merge_keys != new_label_keys:
            raise ValueError(
                "new_labels_map keys must exactly match the merged label ids"
            )

        seen_old_ids = set()
        remap_dict: Dict[int, int] = {}

        for new_id, old_ids in map_merge.items():
            if not old_ids:
                raise ValueError(f"Merged label {new_id} must contain at least one id")

            for old_id in old_ids:
                if old_id not in self.labels_map:
                    raise ValueError(f"Unknown label id in merge map: {old_id}")
                if old_id in seen_old_ids:
                    raise ValueError(
                        f"Label id {old_id} appears in more than one merge group"
                    )
                seen_old_ids.add(old_id)
                remap_dict[old_id] = new_id

        full_labels_map = copy.deepcopy(normalized_new_labels_map)
        next_new_id = max(full_labels_map.keys(), default=-1) + 1

        for old_id in sorted(self.labels_map.keys()):
            if old_id in seen_old_ids:
                continue
            remap_dict[old_id] = next_new_id
            full_labels_map[next_new_id] = self.labels_map[old_id]
            next_new_id += 1

        for idx in tqdm(self.sample.keys(), desc="Merging Labels"):
            self.sample[idx].remap_label_ids(remap_dict)

        self.set_labels_map(full_labels_map)
        return self

    def remap_labels(
        self, remap_dict: Dict[int, Optional[int]], new_labels_map: Dict[int, str]
    ):
        """
        Remap label ids across the whole dataset.

        Labels present in `remap_dict` are rewritten to the provided target ids.
        If a label is mapped to `None`, all bounding boxes using that label are
        removed. Labels not present in `remap_dict` keep their original ids.

        Parameters
        ----------
        remap_dict : Dict[int, Optional[int]]
            Mapping from original label ids to new label ids. Use `None` as the
            target value to drop a label entirely.

        new_labels_map : Dict[int, str]
            The final label map after remapping. Its keys must exactly match the
            set of label ids that remain after applying `remap_dict`, including
            preserved labels that were not explicitly remapped.

        Returns
        -------
        UNDataset
            The same dataset instance.
        """
        if self.labels_map is None:
            raise ValueError("labels_map is required to remap labels")

        normalized_new_labels_map = self._normalize_map(copy.deepcopy(new_labels_map))

        for old_id in remap_dict:
            if old_id not in self.labels_map:
                raise ValueError(f"Unknown source label id in remap: {old_id}")

        resulting_label_ids = set()
        for old_id in self.labels_map:
            new_id = remap_dict.get(old_id, old_id)
            if new_id is not None:
                resulting_label_ids.add(new_id)

        new_label_keys = set(normalized_new_labels_map.keys())
        if resulting_label_ids != new_label_keys:
            raise ValueError(
                "new_labels_map keys must exactly match the resulting label ids"
            )

        for key, value in remap_dict.items():
            if value is None:
                print(f"Remapping {key}:{self.labels_map[key]} -> {value}")
            else:
                print(
                    f"Remapping {key}:{self.labels_map[key]} -> {value}:{normalized_new_labels_map[value]}"
                )

        for idx in tqdm(self.sample.keys(), desc="Remapping Labels"):
            self.sample[idx].remap_label_ids(remap_dict)

        self.set_labels_map(normalized_new_labels_map)
        return self

    def normalize_labels_map(
        self, ascending_order: bool = False, inplace: bool = False
    ) -> "UNDataset":
        """
        Rebuild `labels_map` so that every used label id is mapped to a
        contiguous id range starting from 0.

        Existing label names are preserved when available. Used label ids that
        are missing from `labels_map` are assigned a fallback name equal to
        their original id string. Entries present in `labels_map` but unused in
        the dataset are dropped.

        When `ascending_order` is `True`, mapped labels are reordered by the
        ascending lexicographic order of their names. Used label ids without a
        corresponding `labels_map` entry are appended afterward, ordered by
        their original numeric id.

        Parameters
        ----------
        ascending_order : bool, default=False
            If `True`, remap labels using the ascending lexicographic order of
            the available label names. If some used label ids are not present in
            `labels_map`, they are appended after the mapped labels.

        inplace : bool, default=False
            If `True`, modify the current dataset. Otherwise, return a
            deep-copied dataset with normalized label ids and `labels_map`.

        Returns
        -------
        UNDataset
            The updated dataset instance.
        """
        if not inplace:
            dataset_res = self.model_copy(deep=True)
            dataset_res.normalize_labels_map(
                ascending_order=ascending_order, inplace=True
            )
            return dataset_res

        label_check = self.check_labels()
        used_label_ids = label_check["used_label_ids"]
        unmapped_label_ids = label_check["unmapped_label_ids"]
        unused_label_map_ids = label_check["unused_label_map_ids"]
        current_labels_map = copy.deepcopy(self.labels_map or {})

        if not used_label_ids:
            self.set_labels_map({})
            return self

        if ascending_order:
            mapped_used_label_ids = [
                label_id
                for label_id in used_label_ids
                if label_id in current_labels_map
            ]
            mapped_used_label_ids.sort(
                key=lambda label_id: (current_labels_map[label_id], label_id)
            )
            ordered_used_label_ids = mapped_used_label_ids + sorted(unmapped_label_ids)
        else:
            ordered_used_label_ids = used_label_ids

        remap_dict: Dict[int, Optional[int]] = {}
        new_labels_map: Dict[int, str] = {}

        for new_id, old_id in enumerate(ordered_used_label_ids):
            remap_dict[old_id] = new_id
            new_labels_map[new_id] = current_labels_map.get(old_id, str(old_id))

        if self.labels_map is None or unmapped_label_ids:
            for idx in tqdm(self.sample.keys(), desc="Normalizing Labels Map"):
                self.sample[idx].remap_label_ids(remap_dict)
            self.set_labels_map(new_labels_map)
            return self

        for old_id in unused_label_map_ids:
            remap_dict[old_id] = None

        self.remap_labels(remap_dict=remap_dict, new_labels_map=new_labels_map)
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

    # ======================= Statistics ==========================
    def label_counts(self):
        label_counts = defaultdict(int)
        for idx in tqdm(self.sample.keys(), desc=f"Counting BBoxes labels"):
            sample_label_counts = self.sample[idx].label_counts()
            for cidx, count_value in sample_label_counts.items():
                label_counts[cidx] += count_value

        label_counts = dict(sorted(label_counts.items()))
        return label_counts

    def check_labels(self) -> Dict[str, Union[bool, List[int]]]:
        """
        Check whether `labels_map` is coherent with the label ids used in the
        dataset bounding boxes.

        Returns
        -------
        Dict[str, Union[bool, List[int]]]
            A dictionary containing:
            - `is_coherent`: `True` when every used label id is mapped and there
              are no extra ids in `labels_map`
            - `used_label_ids`: sorted label ids found in the dataset bboxes
            - `unmapped_label_ids`: sorted used label ids missing from `labels_map`
            - `unused_label_map_ids`: sorted ids present in `labels_map` but not
              used by any bbox
        """
        used_label_ids = sorted(self.label_counts().keys())
        mapped_label_ids = sorted(self.labels_map.keys()) if self.labels_map else []

        used_label_ids_set = set(used_label_ids)
        mapped_label_ids_set = set(mapped_label_ids)

        unmapped_label_ids = sorted(used_label_ids_set - mapped_label_ids_set)
        unused_label_map_ids = sorted(mapped_label_ids_set - used_label_ids_set)

        return {
            "is_coherent": not unmapped_label_ids and not unused_label_map_ids,
            "used_label_ids": used_label_ids,
            "unmapped_label_ids": unmapped_label_ids,
            "unused_label_map_ids": unused_label_map_ids,
        }

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

        label_counts = self.label_counts()
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
    def filter_bbox_labels(self, keep_ids: List[int], inplace=False):
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

    # =================== Plot =======================================
    def draw_sample(
        self,
        index: int,
        show_text: bool = True,
        font_size: int = 20,
        color: Optional[Tuple[int, int, int]] = None,
        cmap: str = "tab10",
    ) -> Image.Image:
        from undata.plotter import Plotter

        return Plotter(cmap).draw_sample(
            self.rootdir,
            sample=self.sample[index],
            labels_map=self.labels_map,
            show_text=show_text,
            font_size=font_size,
            color=color,
        )

    def crop_sample(self, index: int, padding_perc: float = 0.0) -> List[Image.Image]:
        from undata.plotter import Plotter

        assert padding_perc >= 0.0

        return Plotter().crop_sample(
            rootdir=self.rootdir, sample=self.sample[index], padding_perc=padding_perc
        )

    # =================== Conversion Methods =========================

    # --------- JSON
    @classmethod
    def read_json(cls, json_file: str) -> "UNDataset":
        with open(json_file, "r") as fp:
            json_str = fp.read()

        undataset = UNDataset().model_validate_json(json_str)

        return undataset

    def to_json(self, json_file: str, indent: int = 2):
        json_str = self.model_dump_json(indent=indent)

        with open(json_file, "w") as fp:
            fp.write(json_str)

    # --------- Pandas Dataframe
    @classmethod
    def read_dataframe(cls, df, rootdir: str = ".") -> "UNDataset":
        undataset = UNDataset(rootdir=rootdir)

        grouped_df = df.groupby("index")
        for idx, group in tqdm(
            grouped_df,
            desc="Loading from Dataframe",
        ):
            undataset.sample[idx] = UNSample.from_dataframe(group)

        undataset._refresh_next_sample_id()
        return undataset

    def to_dataframe(self) -> pd.DataFrame:
        df = pd.DataFrame()
        frames = []
        for idx in tqdm(self.sample.keys(), desc="Convert UNDataset as DataFrame"):
            sample_df = self.sample[idx].as_dataframe()
            sample_df["index"] = idx
            frames.append(sample_df)

        df = pd.concat(frames, ignore_index=True)

        return df.reset_index(drop=True)

    # --------- YOLO
    @classmethod
    def read_yolo(
        cls,
        classes_path: str,
        annotations_dir: str,
        images_dir: str,
        images_lead: bool = True,
    ) -> "UNDataset":

        undataset = YOLOReader.read(
            classes_path,
            annotations_dir,
            images_dir,
            images_lead,
        )
        return undataset

    def to_yolo(self, ann_path: str, exist_ok: bool = True):
        return YOLOWriter().write(
            self,
            ann_path,
            exist_ok,
        )

    # --------- VOC
    @classmethod
    def read_voc(
        cls,
        annotations_dir: str,
        images_dir: str,
        images_lead: bool = True,
    ) -> "UNDataset":

        undataset = VOCReader.read(
            annotations_dir,
            images_dir,
            images_lead,
        )
        return undataset

    def to_voc(self, ann_path: str, exist_ok: bool = True):
        VOCWriter().write(self, ann_path, exist_ok)
