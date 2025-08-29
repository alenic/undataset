from typing import List, Optional, Union

from pydantic import BaseModel

from undata.untypes import BBoxFormatType


class UNBBox(BaseModel):
    coords: List[Union[float, int]]
    format: BBoxFormatType
    label_id: Optional[int] = None
    score: Optional[float] = None
