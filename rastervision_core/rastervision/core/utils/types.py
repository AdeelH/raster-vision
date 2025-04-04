from collections.abc import Sequence
from typing import Annotated

from pydantic.types import StringConstraints

from rastervision.pipeline.config import Field

NonEmptyStr = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1)
]
Proportion = Annotated[float, Field(ge=0, le=1)]
Vector = Sequence[float]
