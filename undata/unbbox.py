from typing import List, Optional, Union
from numbers import Real
from pydantic import BaseModel

from undata.untypes import BBoxFormatType

Number = int | float

class UNBBox(BaseModel):
    coords: tuple[Number, Number, Number, Number]
    format: BBoxFormatType
    label_id: Optional[int] = None
    score: Optional[float] = None
