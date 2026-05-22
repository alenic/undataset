from typing import Tuple, Dict, Optional, List
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import os
from undata.datasets.od.odsample import ODSample


class ODPlotter:
    def __init__(self, cmap="tab10"):
        # Get colormap from matplotlib
        try:
            self.cmap = plt.get_cmap(cmap)
        except Exception as e:
            raise e

    def get_color_from_idx(self, idx) -> Tuple[int, int, int]:
        rgb = self.cmap(idx % self.cmap.N)[:3]
        return (int(255 * rgb[0]), int(255 * rgb[1]), int(255 * rgb[2]))

    def crop_sample(
        self, rootdir: str, sample: ODSample, padding_perc: float = 0
    ) -> List[Image.Image]:

        image_path = os.path.join(rootdir, sample.image_path)
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            raise ValueError(f"Error to open image {image_path}")

        # Iterate trough bboxes and cropes the images
        cropped_images = []
        sample_c = sample.bbox_convert(to_format="xyxy", rounded=True, inplace=False)

        if sample_c.bbox is not None:
            width, height = image.size
            for bbox in sample_c.bbox:
                x1, y1, x2, y2 = bbox.coords

                # Apply padding
                pad_x = int((x2 - x1) * padding_perc)
                pad_y = int((y2 - y1) * padding_perc)
                x1_p = max(0, x1 - pad_x)
                y1_p = max(0, y1 - pad_y)
                x2_p = min(width, x2 + pad_x)
                y2_p = min(height, y2 + pad_y)

                cropped = image.crop((x1_p, y1_p, x2_p, y2_p))
                cropped_images.append(cropped)

        return cropped_images

    def draw_sample(
        self,
        rootdir: str,
        sample: ODSample,
        labels_map: Optional[Dict[int, str]],
        show_text: bool = True,
        font_size: int = 20,
        color: Optional[Tuple[int, int, int]] = None,
    ) -> Image.Image:

        image_path = os.path.join(rootdir, sample.image_path)
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            raise ValueError(f"Error to open image {image_path}")

        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default(size=font_size)

        sample_c = sample.bbox_convert(to_format="xyxy", rounded=True, inplace=False)

        if sample_c.bbox is not None:
            for bbox in sample_c.bbox:
                if color is None:
                    color = self.get_color_from_idx(bbox.label_id)

                x1, y1, x2, y2 = bbox.coords
                draw.rectangle([x1, y1, x2, y2], outline=color, width=2)

                bbox_text = ""
                if bbox.label_id:
                    bbox_text += f"{bbox.label_id}"

                if labels_map is not None and bbox.label_id is not None:
                    bbox_text += f":{labels_map[bbox.label_id]}"

                if bbox.score is not None and bbox.score < 1:
                    bbox_text += f" - {bbox.score:.4f}"

                # Use textbbox to get text size (left, top, right, bottom)
                text_bbox = draw.textbbox((0, 0), bbox_text, font=font)
                text_width = text_bbox[2] - text_bbox[0]
                text_height = text_bbox[3] - text_bbox[1]
                # Adjust y position if text would go above the image
                text_y = y1 - text_height if y1 - text_height > 0 else y1
                if show_text:
                    draw.rectangle(
                        [x1, text_y, x1 + text_width, text_y + text_height], fill=color
                    )
                    draw.text((x1, text_y), bbox_text, fill="white", font=font)

        return image
