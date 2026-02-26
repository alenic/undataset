from typing import TYPE_CHECKING
import os
from tqdm import tqdm

from undata.converters.base import UNDatasetWriter

if TYPE_CHECKING:
    from undata.undataset import UNDataset


class YOLOWriter(UNDatasetWriter):
    def write(self, dataset: "UNDataset", ann_path: str, exist_ok: bool = True) -> None:
        if os.path.exists(ann_path):
            if not exist_ok:
                raise FileExistsError(f"{ann_path} already exists")
        else:
            os.makedirs(ann_path, exist_ok=exist_ok)

        for idx in tqdm(
            dataset.sample.keys(), desc=f"Exporting to yolo annotations"
        ):
            yolo_str = dataset.sample[idx].yolo_dumps()
            image_name = os.path.basename(dataset.sample[idx].image_path)
            image_name, _ = os.path.splitext(image_name)
            with open(os.path.join(ann_path, image_name + ".txt"), "w") as fp:
                fp.write(yolo_str)
