from typing import Optional, List

from pydantic import BaseModel, Field


class UNSample(BaseModel):
    """
    The UNSample base class

    Attributes:
        tags (List[int]): The {tags: "tag_name"} global map
    """

    tags: Optional[List[int]] = Field(
        default=None, exclude_if=lambda x: x is None
    )  # Tags
