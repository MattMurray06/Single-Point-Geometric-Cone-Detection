import numpy as np
from pycpd import DeformableRegistration
from .cone_detector import ConeDetector, Cluster

class CPD(ConeDetector):

    def detect_cones(self, cluster_centres: list[Cluster]) -> np.ndarray:
        # run cpd on clusters
        template_track = self.generate_template_track(10)

        centres = np.stack([c.position for c in cluster_centres])

        reg = DeformableRegistration(X = centres, Y = template_track)
        # alpha = beta = 2 by default
        reg.alpha = 0.06
        reg.beta = 7
        reg.w = 0.01
        reg.max_iterations = 1000
        TY, _ = reg.register()
        return TY

    def generate_template_track(self, num_pairs_cones : int) -> np.ndarray:
        template = np.empty((num_pairs_cones * 2, 2))
        for i in range(num_pairs_cones):
            # x = forward coordinate
            template[i*2, 0] = template[i*2 + 1, 0] = (i+1)*3
            template[i*2, 1] = 1.5
            template[i*2+1, 1] = -1.5
        return template