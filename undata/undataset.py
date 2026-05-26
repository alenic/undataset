from typing import Dict, Optional, Union, List
import copy

from pydantic import BaseModel, Field, PrivateAttr, field_validator

from undata.unsample import UNSample


class UNDataset(BaseModel):
    """
    The UNDataset base class

    Attributes:
        sample (Dict[int, ODSample]): A sample
        tags_map (Dict[int, str]): The {tags: "tag_name"} global map
    """

    sample: Dict[int, UNSample] = Field(default_factory=dict)

    tags_map: Optional[Dict[int, str]] = Field(
        default=None, exclude_if=lambda x: x is None
    )
    _next_sample_id: int = PrivateAttr(default=0)

    @field_validator("tags_map", mode="before")
    @classmethod
    def _normalize_map(cls, v):
        if v is None:
            return None
        if isinstance(v, list):
            return {i: name for i, name in enumerate(v)}
        if isinstance(v, dict):
            # normalize keys to int in case they come as "0", "1", ...
            return {int(k): str(val) for k, val in v.items()}
        raise TypeError("Expected list[str] or dict[int, str]")

    def _refresh_next_sample_id(self):
        if not self.sample:
            self._next_sample_id = 0
            return
        self._next_sample_id = max(self.sample.keys()) + 1

    def model_post_init(self, __context):
        self._refresh_next_sample_id()

    def append(self, sample: UNSample) -> "UNDataset":
        """
        Append a sample to the dataset. You have to create an UNSample first

        Parameters
        ----------
        sample : UNSample

        Returns
        -------
        The same UNDataset
        """
        if not isinstance(sample, UNSample):
            raise TypeError()

        if self._next_sample_id in self.sample:
            self._refresh_next_sample_id()

        new_id = self._next_sample_id
        self.sample[new_id] = sample
        self._next_sample_id = new_id + 1
        return self

    def delete(self, idx: int) -> "UNDataset":
        """
        Delete a specific sample, given its id

        Parameters
        ----------
        idx : int
        The id of the sample that has to be deleted

        Returns
        -------
        The same UNDataset
        """
        if idx not in self.sample:
            raise IndexError(f"Index {idx} does not exists")

        del self.sample[idx]
        return self

    def set_tags_map(self, tags_map: Union[Dict[int, str], List[str]]):
        self.tags_map = self._normalize_map(copy.deepcopy(tags_map))

    def reset_index(self) -> "UNDataset":
        sorted_index = sorted(self.sample.keys())
        self.sample = {i: v for i, v in enumerate(sorted_index)}
        self._refresh_next_sample_id()
        return self

    def check_tags(self):
        """
        Check if tags ids match the tags_map
        """
        tag_ids = set(self.tags_map.keys())
        for s_id, s_val in self.sample.items():
            if s_val.tags is not None:
                if not all([tag in tag_ids for tag in s_val.tags]):
                    raise ValueError(
                        f"example id {s_id} doesn't match the tags_map: {s_val.tags}"
                    )

    def get_sample(self, idx: int, inplace: bool = False) -> UNSample:
        if idx not in self.sample:
            raise IndexError(f"Index {idx} does not exists")

        sample = self.sample[idx]
        return sample.model_copy(deep=True) if not inplace else sample

    def items(self, inplace: bool = False):
        for idx in self.sample.keys():
            yield idx, self.get_sample(idx, inplace=inplace)

    def __iter__(self):
        for idx in self.sample.keys():
            yield self.get_sample(idx)

    def __getitem__(self, idx: int) -> UNSample:
        return self.get_sample(idx, inplace=True)

    def __len__(self) -> int:
        return len(self.sample)
