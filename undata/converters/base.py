from abc import ABC, abstractmethod
from typing import Optional
from undata import UNDataset

class UNDatasetReader(ABC):
    @abstractmethod
    def read(self, *args, **kwargs) -> UNDataset:
        pass

class UNDatasetWriter(ABC):
    @abstractmethod
    def write(self, dataset: UNDataset, *args, **kwargs) -> None:
        pass

class UNDatasetConverter(ABC):
    reader: Optional[UNDatasetReader] = None
    writer: Optional[UNDatasetWriter] = None

    @classmethod
    @abstractmethod
    def read(self, *args, **kwargs) -> UNDataset:
        return self.reader.read(*args, **kwargs)
    
    @classmethod
    @abstractmethod
    def write(self, dataset: UNDataset, *args, **kwargs) -> None:
        self.write(dataset, *args, **kwargs)