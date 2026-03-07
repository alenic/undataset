class BBoxConverter:
    # rel_xywh (unbbox format)
    @staticmethod
    def rel_xywh_to_xywh(rel_xywh, img_width, img_height, rounded=False):
        x, y, w, h = rel_xywh
        xywh = [x * img_width, y * img_height, w * img_width, h * img_height]
        if rounded:
            xywh = list(map(round, xywh))
        return xywh

    @staticmethod
    def rel_xywh_to_xyxy(rel_xywh, img_width, img_height, rounded=False):
        if img_width <= 0 or img_height <= 0:
            raise ValueError("Image dimensions must be positive")
        xr, yr, wr, hr = rel_xywh
        x1 = xr * img_width
        y1 = yr * img_height
        x2 = x1 + wr * img_width
        y2 = y1 + hr * img_height
        xyxy = [x1, y1, x2, y2]
        if rounded:
            xyxy = list(map(round, xyxy))
        return xyxy

    @staticmethod
    def rel_xywh_to_yolo(rel_xywh):
        x, y, w, h = rel_xywh
        cx = x + w / 2
        cy = y + h / 2
        return [cx, cy, w, h]

    # xywh (unbbox format)
    @staticmethod
    def xywh_to_rel_xywh(xywh, img_width, img_height):
        x, y, w, h = xywh
        return [x / img_width, y / img_height, w / img_width, h / img_height]

    @staticmethod
    def xywh_to_xyxy(xywh, rounded=False):
        x, y, w, h = xywh
        xyxy = [x, y, x + w, y + h]
        if rounded:
            xyxy = list(map(round, xyxy))
        return xyxy

    @staticmethod
    def xywh_to_yolo(xywh, img_width, img_height):
        x, y, w, h = xywh
        cx = (x + w / 2) / img_width
        cy = (y + h / 2) / img_height
        return [cx, cy, w / img_width, h / img_height]

    # xyxy (unbbox format)
    @staticmethod
    def xyxy_to_xywh(xyxy, rounded=False):
        x1, y1, x2, y2 = xyxy
        xywh = [x1, y1, x2 - x1, y2 - y1]
        if rounded:
            xywh = list(map(round, xywh))
        return xywh

    @staticmethod
    def xyxy_to_yolo(xyxy, img_width, img_height):
        x1, y1, x2, y2 = xyxy
        w = x2 - x1
        h = y2 - y1
        cx = (x1 + w / 2) / img_width
        cy = (y1 + h / 2) / img_height
        return [cx, cy, w / img_width, h / img_height]

    @staticmethod
    def xyxy_to_rel_xywh(xyxy, img_width, img_height):
        xywh = BBoxConverter.xyxy_to_xywh(xyxy)
        return BBoxConverter.xywh_to_rel_xywh(xywh, img_width, img_height)

    # yolo
    @staticmethod
    def yolo_to_xywh(yolo, img_width, img_height, rounded=False):
        cx, cy, w, h = yolo
        abs_w = w * img_width
        abs_h = h * img_height
        x = (cx * img_width) - (abs_w / 2)
        y = (cy * img_height) - (abs_h / 2)
        xywh = [x, y, abs_w, abs_h]
        if rounded:
            xywh = list(map(round, xywh))
        return xywh

    @staticmethod
    def yolo_to_xyxy(yolo, img_width, img_height, rounded=False):
        cx, cy, w, h = yolo
        abs_w = w * img_width
        abs_h = h * img_height
        x = (cx * img_width) - (abs_w / 2)
        y = (cy * img_height) - (abs_h / 2)
        xyxy = [x, y, x + abs_w, y + abs_h]
        if rounded:
            xyxy = list(map(round, xyxy))
        return xyxy

    @staticmethod
    def yolo_to_rel_xywh(yolo, img_width, img_height):
        xywh = BBoxConverter.yolo_to_xywh(yolo, img_width, img_height)
        return BBoxConverter.xywh_to_rel_xywh(xywh, img_width, img_height)

