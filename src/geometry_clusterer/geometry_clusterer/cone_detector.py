from abc import abstractmethod, ABC
import numpy as np
from dataclasses import dataclass

@dataclass
class Cluster:
    position: np.ndarray
    confident: bool

class ConeDetector(ABC):

    @abstractmethod
    def detect_cones(self, cluster_centres: list[Cluster]) -> object:
        raise NotImplementedError