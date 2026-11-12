from abc import ABC, abstractmethod
from typing import Iterable

from rastervision.core.data import Scene


class Analyzer(ABC):
    """Analyzes scenes and writes some output while running the analyze command.

    This output can be used to normalize images, for example.
    """

    @abstractmethod
    def process(self, scenes: Iterable[Scene], tmp_dir: str):
        """Process scenes and save result."""
