from undata.converters.base import UNDatasetConverter
from undata.converters.voc_reader import VOCReader
from undata.converters.voc_writer import VOCWriter


class VOCConverter(UNDatasetConverter):
    reader = VOCReader
    writer = VOCWriter
