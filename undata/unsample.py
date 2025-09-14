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
    image_path: str = None  # Relative path
    image_w: Optional[int] = None  # Image width
    image_h: Optional[int] = None  # Image height
    bbox: Optional[List[UNBBox]] = None  # BBox
    tag_id: Optional[List[int]] = None  # Tags

    def as_json(self) -> str:
        return self.model_dump_json(indent=2)

    def from_json(self, json_str: str, strict: bool = None):
        self.model_validate_json(json_str, strict=strict)

    def add_bbox(
        self,
        coords: List[Union[float, int]],
        format: BBoxFormatType,
        label_id: Optional[int] = None,
        score: Optional[float] = None,
    ):
        if len(coords) != 4:
            raise ValueError("Bounding box coordinates must have exactly 4 values")

        self.bbox.append(
            UNBBox(coords=coords, format=format, label_id=label_id, score=score)
        )

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
        for bbox in self.bbox:
            label_counts[bbox.label_id] += 1

        return label_counts

    def as_dataframe(self):
        data = []
        if not self.bbox:
            data.append(
                {
                    "image_path": self.image_path,
                    "image_w": self.image_w,
                    "image_h": self.image_h,
                    "bbox_0": np.nan,
                    "bbox_1": np.nan,
                    "bbox_2": np.nan,
                    "bbox_3": np.nan,
                    "label_id": np.nan,
                    "format": "rel_xywh",
                    "tag_id": self.tag_id,
                }
            )
        else:
            for bbox in self.bbox:
                data.append(
                    {
                        "image_path": self.image_path,
                        "image_w": self.image_w,
                        "image_h": self.image_h,
                        "bbox_0": bbox.coords[0],
                        "bbox_1": bbox.coords[1],
                        "bbox_2": bbox.coords[2],
                        "bbox_3": bbox.coords[3],
                        "label_id": bbox.label_id,
                        "format": bbox.format,
                        "tag_id": self.tag_id,
                    }
                )
        return pd.DataFrame(data)

    def from_dataframe(self, df: pd.DataFrame):
        if df.empty:
            raise ValueError("df is empty")

        image_path = df.iloc[0]["image_path"]
        image_w = df.iloc[0]["image_w"]
        image_h = df.iloc[0]["image_h"]
        tag_id = df.iloc[0]["tag_id"]
        bboxes = []

        for row in df.itertuples(index=False):
            coords = [row.bbox_0, row.bbox_1, row.bbox_2, row.bbox_3]
            label_id = row.label_id
            format = row.format
            if any(pd.isna(v) for v in coords + [label_id]):
                coords = None
                label_id = None
                format = None
            else:
                bbox = UNBBox(
                    coords=coords,
                    label_id=label_id,
                    format=format,
                )
                bboxes.append(bbox)

        self.image_path = image_path
        self.image_w = image_w
        self.image_h = image_h
        if len(bboxes) == 0:
            self.bbox = None
        else:
            self.bbox = bboxes

        self.tag_id = tag_id

        return self

    def bbox_convert(
        self,
        to_format: BBoxFormatType,
        rounded: bool = False,
        inplace: bool = True,
    ):

        if inplace:
            if self.bbox:
                for bb in self.bbox:
                    key = (bb.format, to_format)

                    if key[0] == key[1]:
                        continue

                    if key not in conversion_map:
                        raise ValueError(
                            f"Conversion from {key[0]} to {key[1]} is not supported"
                        )

                    coords = conversion_map[key](
                        bb.coords, self.image_w, self.image_h, rounded
                    )
                    bb.coords = coords
                    bb.format = to_format
        else:
            bboxes = []
            if self.bbox:
                for bb in self.bbox:
                    key = (bb.format, to_format)

                    if key[0] == key[1]:
                        bboxes.append(copy.deepcopy(bb))
                        continue

                    if key not in conversion_map:
                        raise ValueError(
                            f"Conversion from {key[0]} to {key[1]} is not supported"
                        )

                    coords = conversion_map[key](
                        bb.coords, self.image_w, self.image_h, rounded
                    )
                    bb.coords = coords
                    bb.format = to_format

                    bboxes.append(
                        UNBBox(
                            coords=coords,
                            format=to_format,
                            label_id=bb.label_id,
                            score=bb.score,
                        )
                    )
            sample = UNSample(
                image_path=self.image_path,
                image_h=self.image_h,
                image_w=self.image_w,
                bbox=bboxes,
                tag_id=self.tag_id,
            )
            return sample

        return self

    def remap_label_ids(self, remap_dict_ids: Dict[int, int]):
        if self.bbox:
            bbox_to_remove = []
            for i, bbox in enumerate(self.bbox):
                if remap_dict_ids[bbox.label_id] is None:
                    bbox_to_remove.append(i)
                bbox.label_id = remap_dict_ids[bbox.label_id]

            if len(bbox_to_remove) > 0:
                # Remove in reverse order to avoid index shift
                for i in sorted(bbox_to_remove, reverse=True):
                    self.bbox.pop(i)
        return self

    def yolo_loads(self, yolo_lines: List[str]):
        if self.bbox:
            raise ValueError("Sample has already bboxes")

        bboxes = []
        for l in yolo_lines:
            label_id, cx, cy, w, h = list(map(float, l.replace("\n", "").split()))
            label_id = int(label_id)
            bboxes.append(
                UNBBox(coords=[cx, cy, w, h], format="yolo", label_id=label_id)
            )
        self.bbox = bboxes

        return self

    def yolo_dumps(self):
        sample = self.bbox_convert(to_format="yolo", inplace=False)

        if not sample.bbox:
            return ""  # Return an empty string if no bounding boxes exist

        yolos_str = []
        if sample.bbox:
            for bbox in sample.bbox:
                bbox_str = f"{bbox.label_id} {bbox.coords[0]} {bbox.coords[1]} {bbox.coords[2]} {bbox.coords[3]}"
                yolos_str.append(bbox_str)

        return "\n".join(yolos_str)

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
