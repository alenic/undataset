import os
import yaml
from tqdm import tqdm

from undata import UNDataset, UNSample
from undata.converters.base import UNDatasetWriter

class YOLOWriter(UNDatasetWriter):
    def write(self, dataset: UNDataset) -> None:
        pass