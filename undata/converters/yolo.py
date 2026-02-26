from undata.converters.base import UNDatasetConverter
from undata.converters.yolo_reader import YOLOReader
from undata.converters.yolo_writer import YOLOWriter


class YOLOConverter(UNDatasetConverter):
    reader = YOLOReader
    writer = YOLOWriter
