from typing import List, Optional, Union
from pydantic import BaseModel
from undata.untypes import BBoxFormatType
from undata.bbox_converter import BBoxConverter

Number = int | float


conversion_map = {
    ("rel_xywh", "xywh"): lambda coords, w, h, r: BBoxConverter.rel_xywh_to_xywh(
        coords, w, h, r
    ),
    ("rel_xywh", "xyxy"): lambda coords, w, h, r: BBoxConverter.rel_xywh_to_xyxy(
        coords, w, h, r
    ),
    ("rel_xywh", "yolo"): lambda coords, w, h, r: BBoxConverter.rel_xywh_to_yolo(
        coords
    ),
    ("xywh", "rel_xywh"): lambda coords, w, h, r: BBoxConverter.xywh_to_rel_xywh(
        coords, w, h
    ),
    ("xywh", "xyxy"): lambda coords, w, h, r: BBoxConverter.xywh_to_xyxy(coords, r),
    ("xywh", "yolo"): lambda coords, w, h, r: BBoxConverter.xywh_to_yolo(coords, w, h),
    ("xyxy", "rel_xywh"): lambda coords, w, h, r: BBoxConverter.xyxy_to_rel_xywh(
        coords, w, h
    ),
    ("xyxy", "xywh"): lambda coords, w, h, r: BBoxConverter.xyxy_to_xywh(coords, r),
    ("xyxy", "yolo"): lambda coords, w, h, r: BBoxConverter.xyxy_to_yolo(coords, w, h),
    ("yolo", "rel_xywh"): lambda coords, w, h, r: BBoxConverter.yolo_to_rel_xywh(
        coords, w, h
    ),
    ("yolo", "xywh"): lambda coords, w, h, r: BBoxConverter.yolo_to_xywh(
        coords, w, h, r
    ),
    ("yolo", "xyxy"): lambda coords, w, h, r: BBoxConverter.yolo_to_xyxy(
        coords, w, h, r
    ),
}

class UNBBox(BaseModel):
    coords: tuple[Number, Number, Number, Number]
    format: BBoxFormatType
    label_id: Optional[int] = None
    score: Optional[float] = None

    def convert(self, format: BBoxFormatType, image_w: int, image_h: int, rounded:bool=True, inplace: bool = False) -> "UNBBox":
        new_coords = conversion_map[(self.format, format)](self.coords, image_w, image_h, rounded)
        if inplace:
            self.coords = new_coords
            self.format = format
            return self
        
        bbox = self.model_copy(deep=True)
        bbox.coords = new_coords
        bbox.format = format
        return bbox

