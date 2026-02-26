from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from undata.undataset import UNDataset


class UNDatasetReader(ABC):
    @abstractmethod
    def read(self, *args, **kwargs) -> "UNDataset":
        pass


class UNDatasetWriter(ABC):
    @abstractmethod
    def write(self, dataset: "UNDataset", *args, **kwargs) -> None:
        pass


class UNDatasetConverter(ABC):
    reader: Optional[UNDatasetReader] = None
    writer: Optional[UNDatasetWriter] = None

    @classmethod
    @abstractmethod
    def read(cls, *args, **kwargs) -> "UNDataset":
        if cls.reader is None:
            raise NotImplementedError("Reader is not configured for this converter")
        return cls.reader().read(*args, **kwargs)

    @classmethod
    @abstractmethod
    def write(cls, dataset: "UNDataset", *args, **kwargs) -> None:
        if cls.writer is None:
            raise NotImplementedError("Writer is not configured for this converter")
        cls.writer().write(dataset, *args, **kwargs)
