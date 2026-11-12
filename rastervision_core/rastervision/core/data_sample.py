from dataclasses import dataclass
from typing import Any, Literal

from numpy import ndarray

from rastervision.core.box import Box


@dataclass
class DataSample:
    """A chip and labels along with metadata."""

    chip: ndarray
    label: Any | None = None
    split: Literal['train', 'valid', 'test'] | None = None
    scene_id: str | None = None
    window: Box | None = None
