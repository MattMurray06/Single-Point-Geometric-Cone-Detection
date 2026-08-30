import numpy as np
from .cone_detector import ConeDetector, Cluster
from dataclasses import dataclass
from scipy.spatial import cKDTree

@dataclass
class CentreLine:
    position: np.ndarray
    heading: float
    confidence: float

@dataclass
class CentreLineHypotheses:
    centrelines: list[CentreLine]

class CentreLinePropagation(ConeDetector):
    def __init__(
        self,
        propagation_distance: float = 3.0,
        track_width: float = 4.0,
        sigma_longitudinal: float = 2.0,
        sigma_lateral: float = 0.3,
        max_mahalanobis_distance: float = 2.0,
        score_threshold: float = 6.0,
    ):
        self.propagation_distance = propagation_distance
        self.track_width = track_width
        self.sigma_longitudinal = sigma_longitudinal
        self.sigma_lateral = sigma_lateral
        self.max_mahalanobis_distance = max_mahalanobis_distance
        self.score_threshold = score_threshold

    def detect_cones(self, cluster_centres: list[Cluster]) -> list[CentreLineHypotheses]:
        cluster_positions = np.array([c.position for c in cluster_centres])
        tree = cKDTree(cluster_positions)
        # high probability cones
        high_confidence_cones = np.array([c.position for c in cluster_centres if c.confident == True])

        active_hypotheses = [self.generate_starting_centreline(high_confidence_cones)]

        i =0
        propagated_hypothesis = []
        while i < 5:
            propagated_hypothesis = self.propagate_forward(active_hypotheses)
            hypotheses_with_confidence = [self.evaluate_centreline(p, tree, cluster_positions) for p in propagated_hypothesis]
            active_hypotheses = self.select_best_centrelines(hypotheses_with_confidence)
            i += 1

        return active_hypotheses 
        
    def generate_starting_centreline(self, high_confidence_cones: np.ndarray) -> CentreLineHypotheses:
        test_centreline = CentreLine(np.array([0, 0]), heading=0, confidence=1) 
        test_hypothesis = CentreLineHypotheses([test_centreline])
        return test_hypothesis

    def propagate_forward(self, active_hypotheses: list[CentreLineHypotheses]) -> list[CentreLineHypotheses]:
        new_hypotheses : list[CentreLineHypotheses] = []
        for hypothesis in active_hypotheses:
            new_hypotheses.extend(self.generate_hypotheses(hypothesis))

        return new_hypotheses
    
    def evaluate_centreline(self, centreline_candidate: CentreLineHypotheses, tree: cKDTree, cluster_positions: np.ndarray) -> CentreLineHypotheses:
        final_centreline = centreline_candidate.centrelines[-1]
        heading = final_centreline.heading
        unit_normal_vector = np.array([np.sin(heading), np.cos(heading)])
        scaled_normal_vector = unit_normal_vector * self.track_width / 2
        left_centre, right_centre = [final_centreline.position + scaled_normal_vector , final_centreline.position - scaled_normal_vector]

        left_indicies = tree.query_ball_point(left_centre, r=4, p=2)
        right_indicies = tree.query_ball_point(right_centre, r=4, p=2)

        score = 0
        for idx in left_indicies:
            d = self.mahalanobis_distance(left_centre, cluster_positions[idx], unit_normal_vector)
            if d < self.max_mahalanobis_distance:
                score += self.max_mahalanobis_distance - d

        for idx in right_indicies:
            d = self.mahalanobis_distance(right_centre, cluster_positions[idx], unit_normal_vector)
            if d < self.max_mahalanobis_distance:
                score += self.max_mahalanobis_distance - d

        centreline_candidate.centrelines[-1].confidence = score
        return centreline_candidate


    def mahalanobis_distance(self, centre: np.ndarray, point: np.ndarray, unit_normal_vector: np.ndarray) -> float:
        delta = point - centre 
        unit_tangent_vector = np.array([-1 * unit_normal_vector[1], unit_normal_vector[0]])
        lateral_error = np.dot(delta, unit_normal_vector)
        longitudinal_error = np.dot(delta, unit_tangent_vector)
        mahalanobis = np.sqrt((lateral_error / self.sigma_lateral)**2 + (longitudinal_error / self.sigma_longitudinal)**2)

        return mahalanobis 

    def select_best_centrelines(self, candidate_centrelines: list[CentreLineHypotheses]) -> list[CentreLineHypotheses]:
        best_centrelines: list[CentreLineHypotheses] = []
        for c in candidate_centrelines:
            if sum(cone.confidence for cone in c.centrelines[-1:]) > self.score_threshold:
                best_centrelines.append(c)
        return best_centrelines

    def generate_hypotheses(self, hypothesis: CentreLineHypotheses) -> list[CentreLineHypotheses]:
        generated_hypothesis: list[CentreLineHypotheses] = []

        for angle_deviance in np.deg2rad(np.linspace(-40, 40, 15)):
            new_angle = (hypothesis.centrelines[-1].heading + angle_deviance) % (2 * np.pi)
            v = self.propagation_distance * np.array([
                np.cos(new_angle),
                np.sin(new_angle),
            ])
            new_position = hypothesis.centrelines[-1].position + v 
            extended_centreline = CentreLine(position= new_position, heading=new_angle, confidence=0)

            new_hypothesis = CentreLineHypotheses(
                centrelines=[*hypothesis.centrelines, extended_centreline]
            )
            generated_hypothesis.append(new_hypothesis) #some logic here to create a new centreline hypothesis, like the old one, but with extended_centreline added
        return generated_hypothesis

