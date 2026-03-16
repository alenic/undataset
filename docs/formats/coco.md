# Data Format

COCO supports several annotation types:

* Object detection
* Keypoint detection
* Stuff segmentation
* Panoptic segmentation
* DensePose
* Image captioning

Annotations are stored in **JSON**. The COCO API can be used to access and manipulate them. All annotation files share this base structure. ([GitHub][1])

## Common top-level structure

```json
{
  "info": info,
  "images": [image],
  "annotations": [annotation],
  "licenses": [license]
}
```

### `info`

```json
{
  "year": int,
  "version": str,
  "description": str,
  "contributor": str,
  "url": str,
  "date_created": datetime
}
```

### `image`

```json
{
  "id": int,
  "width": int,
  "height": int,
  "file_name": str,
  "license": int,
  "flickr_url": str,
  "coco_url": str,
  "date_captured": datetime
}
```

### `license`

```json
{
  "id": int,
  "name": str,
  "url": str
}
```

The annotation payload then changes depending on the task. ([GitHub][1])

## 1. Object Detection

Each object instance includes:

* `category_id`
* `segmentation`
* `area`
* `bbox`
* `iscrowd`

Segmentation depends on the annotation type:

* `iscrowd = 0`: polygons
* `iscrowd = 1`: RLE

A single object can have multiple polygons, for example when it is occluded. Bounding boxes are measured from the **top-left corner** and are **0-indexed**. Categories map category IDs to names and supercategories. ([GitHub][1])

### Detection annotation

```json
{
  "id": int,
  "image_id": int,
  "category_id": int,
  "segmentation": "RLE or [polygon]",
  "area": float,
  "bbox": [x, y, width, height],
  "iscrowd": 0
}
```

### Detection categories

```json
[
  {
    "id": int,
    "name": str,
    "supercategory": str
  }
]
```

## 2. Keypoint Detection

Keypoint annotations include all object-detection fields plus:

* `keypoints`: a flat array like `[x1, y1, v1, ...]`
* `num_keypoints`: number of labeled keypoints

Visibility flag `v` means:

* `0`: not labeled (`x = y = 0`)
* `1`: labeled but not visible
* `2`: labeled and visible

A keypoint is considered visible if it lies inside the object segment. COCO also defines per-category:

* `keypoints`: list of keypoint names
* `skeleton`: connectivity pairs used for visualization

Keypoints are currently annotated only for the **person** category. ([GitHub][1])

### Keypoint annotation

```json
{
  "keypoints": [x1, y1, v1, ...],
  "num_keypoints": int,
  "...": "inherits object-detection fields"
}
```

### Keypoint categories

```json
[
  {
    "keypoints": [str],
    "skeleton": [edge],
    "...": "inherits category fields"
  }
]
```

## 3. Stuff Segmentation

Stuff segmentation uses a format that is compatible with object detection, except `iscrowd` is not needed and is effectively `0` by default.

COCO provides stuff annotations in both:

* JSON
* PNG

In JSON, each category present in an image is represented by a single **RLE annotation**. The `category_id` identifies the stuff category. ([GitHub][1])

## 4. Panoptic Segmentation

Panoptic annotations are **per-image**, not per-object.

Each annotation contains:

1. A **PNG** file with class-agnostic segment IDs
2. A **JSON** structure with semantic information for each segment

Key points:

* Match annotations to images using `annotation.image_id == image.id`
* Segment IDs are stored in the PNG at `annotation.file_name`
* Void / unlabeled pixels have value `0`
* If the PNG is loaded as RGB, compute IDs as:

```text
id = R + G*256 + B*256^2
```

Each entry in `segments_info` describes one segment, including:

* `id`
* `category_id`
* `area`
* `bbox`
* `iscrowd`

Panoptic categories also include:

* `isthing`
* `color`

The thing categories match detection, while stuff categories differ from the separate stuff task. ([GitHub][1])

### Panoptic annotation

```json
{
  "image_id": int,
  "file_name": str,
  "segments_info": [segment_info]
}
```

### `segment_info`

```json
{
  "id": int,
  "category_id": int,
  "area": int,
  "bbox": [x, y, width, height],
  "iscrowd": 0
}
```

### Panoptic categories

```json
[
  {
    "id": int,
    "name": str,
    "supercategory": str,
    "isthing": 0,
    "color": [R, G, B]
  }
]
```

## 5. Image Captioning

Caption annotations store text descriptions for images.

Each image has at least **5 captions**. ([GitHub][1])

### Caption annotation

```json
{
  "id": int,
  "image_id": int,
  "caption": str
}
```

## 6. DensePose

DensePose annotations include:

* `category_id`
* `bbox`
* body part masks
* parametrization data for selected points

Crowd annotations are used for large groups, and bounding boxes are measured from the **top-left corner** and are **0-indexed**. ([GitHub][1])

DensePose uses `dp_*` fields:

### Annotated masks

* `dp_masks`: RLE-encoded dense masks
* masks are `256x256`
* they correspond to 14 semantic body parts, including torso, hands, feet, legs, arms, and head ([GitHub][1])

### Annotated points

* `dp_x`, `dp_y`: point coordinates, scaled so the bounding box is `256x256`
* `dp_I`: patch index indicating which surface patch the point belongs to
* `dp_U`, `dp_V`: coordinates in UV space

### DensePose annotation

```json
{
  "id": int,
  "image_id": int,
  "category_id": int,
  "is_crowd": 0,
  "area": int,
  "bbox": [x, y, width, height],
  "dp_I": [float],
  "dp_U": [float],
  "dp_V": [float],
  "dp_x": [float],
  "dp_y": [float],
  "dp_masks": ["RLE"]
}
```