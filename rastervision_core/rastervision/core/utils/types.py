from typing import Sequence
from typing_extensions import Annotated
from pydantic.types import StringConstraints
from pydantic import Field


NonEmptyStr = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1)
]
Proportion = Annotated[float, Field(ge=0, le=1)]
Vector = Sequence[float]
