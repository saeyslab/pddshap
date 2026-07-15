from abc import abstractmethod
from numpy import typing as npt


class Model:
    """
    Interface that represents any model as a function that takes a numpy
    array as input and produces another numpy array.
    This class can be subclassed to produce a compatible model if necessary,
    but any function or callable that follows this interface should also work.
    """
    @abstractmethod
    def __call__(self, batch_x: npt.NDArray) -> npt.NDArray:
        raise NotImplementedError
