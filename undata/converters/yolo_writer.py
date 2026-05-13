import os
from typing import TYPE_CHECKING

from tqdm import tqdm

if TYPE_CHECKING:
    from undata.undataset import UNDataset
    from undata.unsample import UNSample


class YOLOWriter:

    def write_sample(self, sample: "UNSample") -> str:
        sample = sample.bbox_convert(to_format="yolo", inplace=False)

        if not sample.bbox:
            return ""  # Return an empty string if no bounding boxes exist

        yolo_string_list = []
        if sample.bbox:
            for bbox in sample.bbox:
                bbox_str = f"{bbox.label_id} {bbox.coords[0]} {bbox.coords[1]} {bbox.coords[2]} {bbox.coords[3]}"
                yolo_string_list.append(bbox_str)

        return "\n".join(yolo_string_list)

    def write(self, dataset: "UNDataset", ann_path: str, exist_ok: bool = True) -> None:
        if os.path.exists(ann_path):
            if not exist_ok:
                raise FileExistsError(f"{ann_path} already exists")
        else:
            os.makedirs(ann_path, exist_ok=exist_ok)

        for idx in tqdm(dataset.sample.keys(), desc=f"Exporting to yolo annotations"):
            yolo_str = self.write_sample(dataset.sample[idx])
            # Get name of the image
            image_name = os.path.basename(dataset.sample[idx].image_path)
            image_name, _ = os.path.splitext(image_name)
            with open(os.path.join(ann_path, image_name + ".txt"), "w") as fp:
                fp.write(yolo_str)
