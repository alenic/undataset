import copy
import os
from typing import List, Optional, Dict, Union

import numpy as np
import pandas as pd
from PIL import Image
from pydantic import BaseModel
from collections import defaultdict

from undata.bbox_converter import conversion_map
from undata.untypes import BBoxFormatType
from undata.unbbox import UNBBox


class UNSample(BaseModel):
    image_path: str  # Relative path
    image_w: Optional[int] = None  # Image width
    image_h: Optional[int] = None  # Image height
    bbox: Optional[List[UNBBox]] = None  # BBox
    tag_id: Optional[List[int]] = None  # Tags

    def as_json(self) -> str:
        return self.model_dump_json(indent=2)

    @classmethod
    def from_json(cls, json_str: str, strict: Optional[bool] = None):
        return cls.model_validate_json(json_str, strict=strict)

    def add_bbox(
        self,
        coords: List[Union[float, int]],
        format: BBoxFormatType,
        label_id: Optional[int] = None,
        score: Optional[float] = None,
    ):
        if len(coords) != 4:
            raise ValueError("Bounding box coordinates must have exactly 4 values")

        if self.bbox is None:
            self.bbox = []

        self.bbox.append(
            UNBBox(coords=coords, format=format, label_id=label_id, score=score)
        )

    def filter_bbox_labels(self, keep_ids: List, inplace=False):
        keep_bbox = []
        if inplace:
            if self.bbox is not None:
                for bb in self.bbox:
                    if bb.label_id in keep_ids:
                        keep_bbox.append(bb)
                self.bbox = keep_bbox
            return self
        else:
            sample = copy.deepcopy(self)
            if sample.bbox is not None:
                for bb in sample.bbox:
                    if bb.label_id in keep_ids:
                        keep_bbox.append(bb)
                sample.bbox = keep_bbox
            return sample

    def compute_image_wh(self, rootdir: str):
        path = os.path.join(rootdir, self.image_path)
        if not os.path.exists(path):
            raise ValueError(f"Image {path} does not exists")

        try:
            with Image.open(path) as img:
                w, h = img.size
            self.image_w = w
            self.image_h = h
            return w, h
        except Exception as e:
            raise ValueError(f"Failed to open image {path}: {str(e)}")

    def get_labels_counts(self):
        label_counts = defaultdict(int)
        if self.bbox:
            for bbox in self.bbox:
                if bbox.label_id is not None:
                    label_counts[bbox.label_id] += 1
        return label_counts

    def as_dataframe(self):
        NA = pd.NA
        rows = []
        if not self.bbox:
            rows.append(
                {
                    "image_path": self.image_path,
                    "image_w": self.image_w,
                    "image_h": self.image_h,
                    "bbox_0": NA,
                    "bbox_1": NA,
                    "bbox_2": NA,
                    "bbox_3": NA,
                    "label_id": NA,
                    "format": "rel_xywh",  # or a sensible default for your pipeline
                    "tag_id": self.tag_id,
                }
            )
        else:
            for bb in self.bbox:
                rows.append(
                    {
                        "image_path": self.image_path,
                        "image_w": self.image_w,
                        "image_h": self.image_h,
                        "bbox_0": bb.coords[0],
                        "bbox_1": bb.coords[1],
                        "bbox_2": bb.coords[2],
                        "bbox_3": bb.coords[3],
                        "label_id": bb.label_id if bb.label_id is not None else NA,
                        "format": str(bb.format),
                        "tag_id": self.tag_id,
                    }
                )
        return pd.DataFrame(rows)

    @staticmethod
    def from_dataframe(df: pd.DataFrame) -> "UNSample":
        if df.empty:
            raise ValueError("df is empty")

        sample = UNSample(image_path=str(df.iloc[0]["image_path"]))
        sample.image_w = (
            None if pd.isna(df.iloc[0]["image_w"]) else int(df.iloc[0]["image_w"])
        )
        sample.image_h = (
            None if pd.isna(df.iloc[0]["image_h"]) else int(df.iloc[0]["image_h"])
        )
        tag_id = df.iloc[0].get("tag_id", None)
        sample.tag_id = (
            None if (isinstance(tag_id, float) and pd.isna(tag_id)) else tag_id
        )

        bboxes: List[UNBBox] = []
        for row in df.itertuples(index=False):
            coords = [
                getattr(row, "bbox_0"),
                getattr(row, "bbox_1"),
                getattr(row, "bbox_2"),
                getattr(row, "bbox_3"),
            ]
            lbl = getattr(row, "label_id")
            fmt: BBoxFormatType = getattr(row, "format")

            has_coords = all(not pd.isna(v) for v in coords)
            has_label = (lbl is not None) and (not pd.isna(lbl))
            if not (has_coords and has_label):
                continue

            bboxes.append(
                UNBBox(
                    coords=[float(c) for c in coords],
                    label_id=int(lbl),
                    format=fmt,
                )
            )

        sample.bbox = bboxes or None
        return sample

    def bbox_convert(
        self,
        to_format: BBoxFormatType,
        rounded: bool = False,
        inplace: bool = True,
    ):
        if not self.bbox:
            return (
                self
                if inplace
                else UNSample(
                    image_path=self.image_path,
                    image_h=self.image_h,
                    image_w=self.image_w,
                    bbox=None,
                    tag_id=self.tag_id,
                )
            )

        if self.image_w is None or self.image_h is None:
            raise ValueError("image_w and image_h must be set for bbox conversion")

        if inplace:
            for bb in self.bbox:
                src, dst = bb.format, to_format
                if src == dst:
                    continue
                key = (src, dst)
                if key not in conversion_map:
                    raise ValueError(f"Conversion from {src} to {dst} is not supported")
                new_coords = conversion_map[key](
                    bb.coords, self.image_w, self.image_h, rounded
                )
                bb.coords = new_coords
                bb.format = dst
            return self
        else:
            new_bboxes = []
            for bb in self.bbox:
                src, dst = bb.format, to_format
                if src == dst:
                    new_bboxes.append(copy.deepcopy(bb))
                    continue
                key = (src, dst)
                if key not in conversion_map:
                    raise ValueError(f"Conversion from {src} to {dst} is not supported")
                new_coords = conversion_map[key](
                    bb.coords, self.image_w, self.image_h, rounded
                )
                new_bboxes.append(
                    UNBBox(
                        coords=new_coords,
                        format=dst,
                        label_id=bb.label_id,
                        score=bb.score,
                    )
                )
            return UNSample(
                image_path=self.image_path,
                image_h=self.image_h,
                image_w=self.image_w,
                bbox=new_bboxes,
                tag_id=self.tag_id,
            )

    def remap_label_ids(self, remap_dict_ids: Dict[int, Optional[int]]):
        if not self.bbox:
            return self
        keep = []
        for bb in self.bbox:
            if bb.label_id is None:
                keep.append(bb)
                continue
            new_id = remap_dict_ids.get(bb.label_id, bb.label_id)
            if new_id is None:
                continue  # drop
            bb.label_id = new_id
            keep.append(bb)
        self.bbox = keep if keep else None
        return self


    def voc_dumps(self, labels_map: Optional[Dict[int, str]] = None):
        # TODO: handle image channels
        sample = self.bbox_convert(to_format="xyxy", rounded=True, inplace=False)

        if not sample.bbox:
            return ""  # Return an empty string if no bounding boxes exist

        voc_str = f"""
<annotation>
    <folder></folder>
    <filename>{sample.image_path}</filename>
    <path>{sample.image_path}</path>

    <size>
        <width>{sample.image_w}</width>
        <height>{sample.image_h}</height>
        <depth>3</depth>
    </size>

    <segmented>0</segmented>

"""
        if sample.bbox:
            for bbox in sample.bbox:
                if labels_map:
                    label_name = labels_map[bbox.label_id]
                else:
                    label_name = bbox.label_id

                object_template = f"""
    <object>
        <name>{label_name}</name>
        <pose>Unspecified</pose>
        <truncated>0</truncated>
        <difficult>0</difficult>
        <bndbox>
            <xmin>{bbox.coords[0]}</xmin>
            <ymin>{bbox.coords[1]}</ymin>
            <xmax>{bbox.coords[2]}</xmax>
            <ymax>{bbox.coords[3]}</ymax>
        </bndbox>
    </object>
"""
                voc_str += object_template

        return voc_str + "\n</annotation>"
