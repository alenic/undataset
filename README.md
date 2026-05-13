# UNDataset

UNDataset is a small Python toolkit for loading, inspecting, transforming, and exporting computer vision datasets.

The project currently focuses on image object detection datasets. Its internal model is intentionally simple:

- `UNDataset`: a collection of image samples plus label/tag maps.
- `UNSample`: one image and its annotations.
- `UNBBox`: one bounding box, with conversion between supported coordinate formats.

## Status

This package is early-stage. The core object model, basic dataset operations, JSON/DataFrame conversion, YOLO import/export, and VOC import/export are available. COCO support is planned but not implemented yet.

## Install

From the repository root:

```bash
pip install -e .
```

For development:

```bash
pip install -e ".[dev]"
pytest
```

## Supported Data

### Dataset Types

| Acronym | Description |
| --- | --- |
| IOD | Image object detection |
| IC | Image classification |

IOD is the current priority.

### Annotation Formats

| Format | Read | Write | Notes |
| --- | --- | --- | --- |
| UNDataset JSON | Yes | Yes | Native serialized representation |
| Pandas DataFrame | Yes | Yes | Useful for analysis and tabular workflows |
| YOLO | Yes | Yes | Uses class names from `.txt` or `.yaml` |
| VOC XML | Yes | Yes | Pascal VOC-style XML annotations |
| COCO | Planned | Planned | See `docs/formats/coco.md` |

### Bounding Box Formats

`UNBBox` supports conversion between:

- `xywh`
- `xyxy`
- `rel_xywh`
- `yolo`

Conversions that need absolute/relative scaling require `image_w` and `image_h` on the sample.

## Quick Start

```python
from undata import UNBBox, UNDataset, UNSample

dataset = UNDataset(rootdir="images", labels_map=["person", "car"])

sample = UNSample(
    image_path="frame_001.jpg",
    image_w=1280,
    image_h=720,
    bbox=[
        UNBBox(coords=[100, 80, 220, 300], format="xyxy", label_id=0),
    ],
)

dataset.append(sample)

converted = dataset.bbox_convert("yolo", inplace=False)
print(converted.get_stats())
```

## Reading And Writing Formats

### YOLO

```python
from undata import UNDataset

dataset = UNDataset.read_yolo(
    classes_path="dataset/classes.txt",
    annotations_dir="dataset/labels",
    images_dir="dataset/images",
)

dataset.to_yolo("out/labels")
```

`classes_path` may be a `.txt` file with one class name per line or a `.yaml` file with a `names` field.

### VOC

```python
from undata import UNDataset

dataset = UNDataset.read_voc(
    annotations_dir="dataset/Annotations",
    images_dir="dataset/JPEGImages",
)

dataset.to_voc("out/Annotations")
```

### JSON

```python
dataset.to_json("dataset.json")
dataset = UNDataset.read_json("dataset.json")
```

### DataFrame

```python
df = dataset.to_dataframe()
dataset = UNDataset.read_dataframe(df, rootdir="images")
```

## Common Operations

```python
# Return a deep copy by default.
sample = dataset.get_sample(0)

# Mutate the stored sample explicitly.
sample = dataset.get_sample(0, inplace=True)
sample.image_path = "renamed.jpg"

# Iterate over copies by default.
for idx, sample in dataset.items():
    ...

# Iterate over live samples when needed.
for idx, sample in dataset.items(inplace=True):
    sample.compute_image_wh(dataset.rootdir)
```

Other useful helpers:

```python
dataset.compute_image_wh()
dataset.get_image_paths()
dataset.get_label_counts()
dataset.get_stats()
dataset.find_duplicate_images()

filtered = dataset.filter_bbox_labels([0, 2], inplace=False)
filtered = dataset.filter_image_size(min_width=640, min_height=480)
filtered = dataset.filter_bbox_size(min_width=0.01, min_height=0.01)
```

## Design Notes

Converters are split into dedicated readers and writers:

- `YOLOReader` / `YOLOWriter`
- `VOCReader` / `VOCWriter`

`UNDataset` exposes convenience methods like `read_yolo()` and `to_voc()`, but the converter modules avoid importing `UNDataset` at module import time. That keeps package imports simple and avoids circular import issues.

## Development Tasks

### Near Term

- Add tests for YOLO read/write round trips.
- Add tests for VOC read/write round trips.
- Fix `reset_index()` so it preserves samples while rebuilding sequential ids.
- Add type hints for converter `read()` methods and writer methods.
- Replace broad `except` blocks in readers with targeted parse errors.
- Decide whether `__getitem__` should return a live sample or match `get_sample()` copy-by-default behavior.

### Format Support

- Implement COCO reader.
- Implement COCO writer.
- Add YOLO dataset YAML writer support.
- Add optional image copy/symlink export when writing YOLO or VOC datasets.
- Add support for background images with empty annotation files across formats.

### Quality And API

- Add a formal format registry for `read(format=...)` and `write(format=...)` if more formats are added.
- Add validation helpers for missing images, missing annotations, invalid boxes, and unknown labels.
- Add deterministic ordering for all readers and writers.
- Add configurable progress bars so library users can disable `tqdm`.
- Add documentation examples for label remapping, filtering, and statistics.
- Add CI with linting, formatting, and tests.

## Tests

Run:

```bash
pytest
```

The current test suite covers basic dataset operations and import behavior.
