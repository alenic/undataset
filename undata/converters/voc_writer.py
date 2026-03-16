import os
import xml.etree.ElementTree as ET
from typing import TYPE_CHECKING, Dict, Optional

from tqdm import tqdm

from undata.converters.base import UNDatasetWriter

if TYPE_CHECKING:
    from undata.undataset import UNDataset
    from undata.unsample import UNSample


class VOCWriter(UNDatasetWriter):
    @staticmethod
    def _label_name(label_id: Optional[int], labels_map: Optional[Dict[int, str]]) -> str:
        if label_id is None:
            raise ValueError("VOC export requires bbox label_id to be set")

        if labels_map is None:
            return str(label_id)

        if label_id not in labels_map:
            raise KeyError(f"Missing label name for label_id={label_id}")

        return labels_map[label_id]

    @staticmethod
    def _indent_xml(root: ET.Element) -> None:
        # Keep output readable while staying compatible with older Python versions.
        try:
            ET.indent(root, space="    ")
        except AttributeError:
            pass

    def write_sample(
        self,
        sample: "UNSample",
        labels_map: Optional[Dict[int, str]] = None,
        rootdir: Optional[str] = None,
    ) -> str:
        if sample.image_w is None or sample.image_h is None:
            raise ValueError(
                f"VOC export requires image_w and image_h for {sample.image_path}"
            )

        sample_conv = sample.bbox_convert(to_format="xyxy", rounded=True, inplace=False)

        image_relpath = sample_conv.image_path
        image_filename = os.path.basename(image_relpath)
        image_folder = os.path.dirname(image_relpath)
        image_fullpath = (
            os.path.normpath(os.path.join(rootdir, image_relpath))
            if rootdir is not None
            else image_relpath
        )

        annotation = ET.Element("annotation")
        ET.SubElement(annotation, "folder").text = image_folder
        ET.SubElement(annotation, "filename").text = image_filename
        ET.SubElement(annotation, "path").text = image_fullpath

        source = ET.SubElement(annotation, "source")
        ET.SubElement(source, "database").text = "Unknown"

        size = ET.SubElement(annotation, "size")
        ET.SubElement(size, "width").text = str(sample_conv.image_w)
        ET.SubElement(size, "height").text = str(sample_conv.image_h)
        ET.SubElement(size, "depth").text = "3"

        ET.SubElement(annotation, "segmented").text = "0"

        for bbox in sample_conv.bbox or []:
            label_name = self._label_name(bbox.label_id, labels_map)
            obj = ET.SubElement(annotation, "object")
            ET.SubElement(obj, "name").text = label_name
            ET.SubElement(obj, "pose").text = "Unspecified"
            ET.SubElement(obj, "truncated").text = "0"
            ET.SubElement(obj, "difficult").text = "0"

            bndbox = ET.SubElement(obj, "bndbox")
            ET.SubElement(bndbox, "xmin").text = str(int(bbox.coords[0]))
            ET.SubElement(bndbox, "ymin").text = str(int(bbox.coords[1]))
            ET.SubElement(bndbox, "xmax").text = str(int(bbox.coords[2]))
            ET.SubElement(bndbox, "ymax").text = str(int(bbox.coords[3]))

        self._indent_xml(annotation)
        return ET.tostring(annotation, encoding="unicode")

    def write(self, dataset: "UNDataset", ann_path: str, exist_ok: bool = True) -> None:
        if os.path.exists(ann_path):
            if not exist_ok:
                raise FileExistsError(f"{ann_path} already exists")
        else:
            os.makedirs(ann_path, exist_ok=exist_ok)

        for idx in tqdm(dataset.sample.keys(), desc="Exporting to VOC annotations"):
            sample = dataset.sample[idx]
            voc_str = self.write_sample(
                sample,
                labels_map=dataset.labels_map,
                rootdir=dataset.rootdir,
            )
            image_name, _ = os.path.splitext(os.path.basename(sample.image_path))
            with open(os.path.join(ann_path, image_name + ".xml"), "w") as fp:
                fp.write(voc_str)
