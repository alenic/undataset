from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from typing import TYPE_CHECKING, Dict, List, Tuple

from tqdm import tqdm

from undata.unbbox import UNBBox
from undata.datasets.od.odsample import ODSample

if TYPE_CHECKING:
    from undata.datasets.od.oddataset import ODDataset


class VOCReader:

    @classmethod
    def read_sample(
        cls,
        annotation_path: str,
        image_path: str,
        labels_to_id: Dict[str, int],
    ) -> "ODSample":
        try:
            root = ET.parse(annotation_path).getroot()
        except Exception as exc:
            raise ValueError(
                f"Failed to parse VOC xml: {annotation_path}: {exc}"
            ) from exc

        bboxes = []
        for obj in root.findall("object"):
            class_name = obj.findtext("name")
            bnd = obj.find("bndbox")
            if class_name is None or bnd is None:
                continue

            try:
                xmin = float(bnd.findtext("xmin"))
                ymin = float(bnd.findtext("ymin"))
                xmax = float(bnd.findtext("xmax"))
                ymax = float(bnd.findtext("ymax"))
            except Exception:
                continue

            if class_name not in labels_to_id:
                labels_to_id[class_name] = len(labels_to_id)

            bboxes.append(
                UNBBox(
                    coords=[xmin, ymin, xmax, ymax],
                    format="xyxy",
                    label_id=labels_to_id[class_name],
                )
            )
        return ODSample(image_path=image_path, bbox=bboxes or None)

    @classmethod
    def _list_files_by_stem(root: str) -> Dict[str, str]:
        files = {}
        for filename in os.listdir(root):
            fullpath = os.path.join(root, filename)
            if not os.path.isfile(fullpath):
                continue
            stem, _ = os.path.splitext(filename)
            files[stem] = filename
        return files

    @classmethod
    def _sample_order(items: Dict[str, str]) -> List[Tuple[str, str]]:
        return sorted(items.items(), key=lambda x: x[0])

    @classmethod
    def read(
        cls,
        annotations_dir: str,
        images_dir: str,
        dataset_cls: type[ODDataset],
        images_lead: bool = True,
    ) -> ODDataset:
        if not os.path.exists(annotations_dir):
            raise ValueError(f"Annotation path: {annotations_dir} does not exists")

        if not os.path.exists(images_dir):
            raise ValueError(f"Images path: {images_dir} does not exists")

        labels_to_id: Dict[str, int] = {}
        undataset = dataset_cls(rootdir=images_dir)

        image_names = cls._list_files_by_stem(images_dir)
        annotation_names = cls._list_files_by_stem(annotations_dir)

        if images_lead:
            iterator = cls._sample_order(image_names)
            for img_name, img_filename in tqdm(
                iterator, desc="Loading from voc images"
            ):
                sample = ODSample(image_path=img_filename)
                annotation_global_path = os.path.join(
                    annotations_dir, img_name + ".xml"
                )

                if os.path.exists(annotation_global_path):
                    parsed_sample = cls.read_sample(
                        annotation_path=annotation_global_path,
                        image_path=img_filename,
                        labels_to_id=labels_to_id,
                    )
                    sample.bbox = parsed_sample.bbox
                # No matching xml -> background image sample by design.

                sample.compute_image_wh(undataset.rootdir)
                undataset.append(sample)
        else:
            iterator = cls._sample_order(annotation_names)
            for ann_name, ann_filename in tqdm(
                iterator, desc="Loading from voc annotations"
            ):
                if ann_name not in image_names:
                    raise ValueError(
                        f"Image for annotation {ann_filename} does not exists in {images_dir}"
                    )

                sample = ODSample(image_path=image_names[ann_name])
                annotation_global_path = os.path.join(annotations_dir, ann_filename)
                parsed_sample = cls.read_sample(
                    annotation_path=annotation_global_path,
                    image_path=image_names[ann_name],
                    labels_to_id=labels_to_id,
                )
                sample.bbox = parsed_sample.bbox
                sample.compute_image_wh(undataset.rootdir)
                undataset.append(sample)

        undataset.set_labels_map({idx: name for name, idx in labels_to_id.items()})
        return undataset
