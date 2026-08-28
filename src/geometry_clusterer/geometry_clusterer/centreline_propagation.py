import numpy as np
from .cone_detector import ConeDetector

class CentreLinePropagation(ConeDetector):
    def detect_cones(self, cluster_centres: np.ndarray):
        
        # high probability cones
        # generate stating centreline position
        # propagate forward 
            # Either via beam search or particles
        # Evaluate centreline probability
        # Follow top few hypothesis
        # repeat
        pass
