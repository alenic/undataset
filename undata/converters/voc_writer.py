from typing import TYPE_CHECKING
import os
from tqdm import tqdm

from undata.converters.base import UNDatasetWriter

if TYPE_CHECKING:
    from undata.undataset import UNDataset, UNSample


class VOCWriter(UNDatasetWriter):

    def write_sample(self, sample: "UNSample") -> str:
        sample = sample.bbox_convert(to_format="yolo", inplace=False)

        if not sample.bbox:
            return "null"  # Return an empty string if no bounding boxes exist

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

        for idx in tqdm(dataset.sample.keys(), desc=f"Exporting to VOC annotations"):
            voc_str = dataset.sample[idx].voc_dumps(self.labels_map)
            image_name = os.path.basename(self.sample[idx].image_path)
            image_name, _ = os.path.splitext(image_name)
            with open(os.path.join(ann_path, image_name + ".txt"), "w") as fp:
                fp.write(voc_str)
