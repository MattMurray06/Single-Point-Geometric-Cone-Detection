from .cone_detector import ConeDetector, Cluster
from scipy.spatial import cKDTree 
import numpy as np
from dataclasses import dataclass

@dataclass
class CentreLine:
    position: np.ndarray
    heading: float


@dataclass
class CentreLineVotingResult:
    cones: np.ndarray
    centre_lines: list[CentreLine]


class CentreLineVoting(ConeDetector):
    def __init__(self, track_width: float = 4.0):
        self.track_width = track_width

    def detect_cones(self, cluster_centres: list[Cluster]) -> CentreLineVotingResult:
        hypotheses = self.create_all_hypothesis(np.stack([c.position for c in cluster_centres]))
        return CentreLineVotingResult(
            cones=np.empty((0, 2), dtype=float),
            centre_lines=hypotheses,
        )

    def create_all_hypothesis(self, cluster_centres):
        hypotheses: list[CentreLine] = []
        tree = cKDTree(cluster_centres)
        for i, cluster in enumerate(cluster_centres):
            neighbours = tree.query_ball_point(cluster, r=5.0, p=2)
            for neighbour in neighbours:
                if neighbour == i:
                    continue
                h1, h2, h3 = self.generate_centreline_hypotheses(
                    cluster_centres[neighbour], cluster, self.track_width
                )
                hypotheses.append(h1)
                hypotheses.append(h2)
                hypotheses.append(h3)
        return hypotheses        

    def generate_centreline_hypotheses(
        self,
        p1: np.ndarray,
        p2: np.ndarray,
        track_width: float,
    ) -> tuple[CentreLine, CentreLine, CentreLine]:

        midpoint = (p1 + p2) / 2.0

        delta = p2 - p1
        distance = np.linalg.norm(delta)

        if distance == 0:
            raise ValueError("p1 and p2 must be distinct")

        tangent = delta / distance
        normal = np.array([-tangent[1], tangent[0]])

        heading = np.arctan2(delta[1], delta[0])

        perp_heading = heading + np.pi / 2
        perp_heading = (perp_heading + np.pi) % (2 * np.pi) - np.pi

        offset = (track_width / 2.0) * normal

        hypothesis_1 = CentreLine(
            position=midpoint + offset,
            heading=heading,
        )

        hypothesis_2 = CentreLine(
            position=midpoint - offset,
            heading=heading,
        )

        hypothesis_3 = CentreLine(
            position=midpoint,
            heading=perp_heading,
        )

        return hypothesis_1, hypothesis_2, hypothesis_3