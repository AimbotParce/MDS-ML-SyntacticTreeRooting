import itertools as it
from typing import Any, Dict, Generic, Iterator, TypeVar

import numpy as np

T = TypeVar("T")


class Dimension(Generic[T]):
    def __init__(self, *values: T):
        self._values = values

    @property
    def options(self):
        return self._values

    def __len__(self):
        return len(self._values)


class GridSearch:
    """
    Perform a grid search over a set of dimensions. Dimensions will be iterated over in the revere order they were added.
    """

    def __init__(self, dimensions: Dict[str, Dimension] = None):
        if dimensions is None:
            dimensions = {}
        elif not isinstance(dimensions, dict):
            raise TypeError("Dimensions must be a dictionary of Dimension objects.")
        elif not all(isinstance(dim, Dimension) for dim in dimensions.values()):
            raise TypeError("All values in dimensions must be Dimension objects.")
        self._dimensions = dimensions

    def add_dimension(self, key: str, dimension: Dimension):
        self._dimensions[key] = dimension

    def __len__(self):
        return np.prod(list(map(lambda x: len(x), self._dimensions.values())))

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        for prod in it.product(*map(lambda x: x.options, self._dimensions.values())):
            yield {key: value for key, value in zip(self._dimensions.keys(), prod)}
