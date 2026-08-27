from abc import abstractmethod, ABC
import numpy as np
class ConeDetector(ABC):

    @abstractmethod
    def detect_cones(self, cluster_centres: np.ndarray):
        pass