from tensor import Tensor
import numpy as np
from typing import Iterator, NamedTuple

BATCH = NamedTuple("BATCH", [("inputs", Tensor), ("targets", Tensor)])


class DataIterator:
    def __call__(self, inputs: Tensor, targets: Tensor) -> Iterator[BATCH]:
        raise NotImplementedError