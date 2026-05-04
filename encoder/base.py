from typing import Any, List
from abc import ABC, abstractmethod


class SubtreeEncoder(ABC):
    """ base class for tree encoder """

    @abstractmethod
    def __init__(self, trees: "Trees" = None, *args, **kwargs) -> None:
        super().__init__()

    @abstractmethod
    def encode(self, ):
        pass

    @abstractmethod
    def batch_encoding(self) -> List[Any]:
        """批量编码树"""
        pass

    @abstractmethod
    def single_encoding(self, subtree: "ASTNode") -> Any: # type: ignore
        """单个编码树"""
        pass

    def __str__(self):
        return str(self.__class__.__name__)
