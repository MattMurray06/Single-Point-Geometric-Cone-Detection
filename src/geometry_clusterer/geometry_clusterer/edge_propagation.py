from .cone_detector import Cluster, ConeDetector
import numpy as np
from scipy.spatial import cKDTree
from dataclasses import dataclass

@dataclass
class FrameState:
    clusters: np.ndarray
    tree: cKDTree

@dataclass
class Edge:
    idxs: list[int]
    stopped_growing: bool = False
    family: int = 0

    @property
    def length(self):
        return len(self.idxs)

@dataclass
class Candidate:
    idx: int
    end: bool

@dataclass
class TrackSegment:
    pass


class EdgePropagation(ConeDetector):
    def __init__(
        self,
        max_neighbour_radius: float,
        min_neighbour_radius: float,
        max_angle_change: float = 30.0,
        plot_growth: bool = False,
    ):
        self.max_neighbour_radius = max_neighbour_radius
        self.min_neighbour_radius = min_neighbour_radius
        self.max_angle_change = max_angle_change
        self.plot_growth = plot_growth

    def detect_cones(self, cluster_centres: list[Cluster]) -> object:

        if not cluster_centres:
            return []
        cluster_xy = np.array([c.position for c in cluster_centres])
        tree = cKDTree(cluster_xy)
        state = FrameState(cluster_xy, tree)

        visited_idxs: list[int] = []
        edges: list[Edge] = []
        family = 0
        for i in range(len(state.clusters)):
            if i in visited_idxs:
                continue
            current_edges = self.detect_edges(i, state, family)

            useful_edges = [edge for edge in current_edges if edge.length > 2]
            visited_idxs.extend(list({idx for edge in edges for idx in edge.idxs}))
            edges.extend(useful_edges)
            family += 1

        # track_segments = self.retrieve_track_segments(edges, state)


        self.plot_edges(state, edges, 0, 0)

    def retrieve_track_segments(self, edges: list[Edge], state: FrameState):
        track_segments: list[[Edge, Edge]]
        
        for i in range(len(edges)):
            for j in range(i+1, len(edges)):
                if self.compatible(edges[i], edges[j]):
                    pass


    def detect_edges(self, start_idx: int, state: FrameState, family: int) -> list[Edge]:
        edge = Edge([start_idx], family=family)
        edge_hypotheses: list[Edge] = [edge]
        generation = 0
        if self.plot_growth:
            self.plot_edges(state, edge_hypotheses, start_idx, generation)

        while True:
            edge_hypotheses = self.grow_edges(edge_hypotheses, state)
            edge_hypotheses = self.remove_duplicates(edge_hypotheses)
            generation += 1
            if self.plot_growth:
                self.plot_edges(state, edge_hypotheses, start_idx, generation)
            if all(edge.stopped_growing for edge in edge_hypotheses):
                break

        for e in edge_hypotheses:
            e.family = family

        return edge_hypotheses

    def grow_edges(self, edges: list[Edge], state: FrameState) -> list[Edge]: 
        edge_hypotheses: list[Edge] = []
        for e in edges:
            if e.stopped_growing == True:
                edge_hypotheses.append(e)
                continue

            child_hypotheses = self.propose_extensions(e, state)
            if len(child_hypotheses) == 0:
                e.stopped_growing = True
                edge_hypotheses.append(e)
            else:
                edge_hypotheses.extend(child_hypotheses)
        return edge_hypotheses

    def remove_duplicates(self, edges: list[Edge]) -> list[Edge]:
        seen: set[tuple[int, ...]] = set()
        unique_edges: list[Edge] = []

        for edge in edges:
            key = tuple(edge.idxs)
            if key in seen:
                continue

            seen.add(key)
            unique_edges.append(edge)

        return unique_edges

    def propose_extensions(self, edge: Edge, state: FrameState) -> list[Edge]:
        child_edges = []
        if edge.length == 1:
            candidate_idxs = self.nearby_candidates(state.clusters[edge.idxs[0]], edge, state)
            child_edges.extend(self.append_candidates_to_edge(candidate_idxs, edge, True))
        else:
            start_candidate_idxs = self.nearby_candidates(state.clusters[edge.idxs[0]], edge, state)
            end_candidate_idxs = self.nearby_candidates(state.clusters[edge.idxs[-1]], edge, state)
            start_candidate_idxs = self.prune_invalid_angles(start_candidate_idxs, edge, False, state)
            end_candidate_idxs = self.prune_invalid_angles(end_candidate_idxs, edge, True, state)
            child_edges.extend(self.append_candidates_to_edge(start_candidate_idxs, edge, False))
            child_edges.extend(self.append_candidates_to_edge(end_candidate_idxs, edge, True))
        return child_edges

    def prune_invalid_angles(self, test_idxs: list[int], edge: Edge, end: bool, state: FrameState):
        if edge.length <= 1:
            return test_idxs

        valid_idxs: list[int] = []
        for idx in test_idxs:
            # end == 1 => we want indices -1 and -2, end == 0 => we want indices 0 and 1 - these formulas generate that
            if self.angle_between_points(
                state.clusters[idx],
                state.clusters[edge.idxs[-end]],
                state.clusters[edge.idxs[-3 * end + 1]],
            ) > np.deg2rad(180 - self.max_angle_change):
                valid_idxs.append(idx)
        return valid_idxs

    def angle_between_points(self, a, b, c):
        """
        Returns the angle ABC (in radians) between points a, b, c.
        Angle is always between 0 and pi.
        a, b, c are array-like (length 2 or 3).
        """
        v1 = a - b
        v2 = c - b

        # normalise
        v1_norm = v1 / np.linalg.norm(v1)
        v2_norm = v2 / np.linalg.norm(v2)

        # dot product
        dot = np.dot(v1_norm, v2_norm)

        # clamp for numerical stability
        dot = np.clip(dot, -1.0, 1.0)

        return np.arccos(dot)

    def append_candidates_to_edge(self, candidate_idxs: list[int], edge: Edge, end: bool):
        child_edges: list[Edge] = []
        for c in candidate_idxs:
            if end:
                child_edges.append(Edge([*edge.idxs, c], False, edge.family))            
            else:
                child_edges.append(Edge([c, *edge.idxs], False, edge.family))            
        return child_edges

    def plot_edges(
        self,
        state: FrameState,
        edges: list[Edge],
        starting_idx: int,
        generation: int,
    ) -> None:
        import matplotlib.pyplot as plt

        figure, axes = plt.subplots()
        axes.scatter(
            state.clusters[:, 1],
            state.clusters[:, 0],
            color="0.65",
            label="Detected cone",
            zorder=1,
        )

        for idx, point in enumerate(state.clusters):
            axes.annotate(
                str(idx),
                (point[1], point[0]),
                xytext=(3, 3),
                textcoords="offset points",
            )

        family_ids = sorted({edge.family for edge in edges})
        color_map = plt.get_cmap("tab10")
        family_colors = {
            family_id: color_map(index % 10)
            for index, family_id in enumerate(family_ids)
        }
        labelled_families: set[int] = set()

        for edge in edges:
            points = state.clusters[edge.idxs]
            color = family_colors[edge.family]
            label = (
                f"Family {edge.family + 1}"
                if edge.family not in labelled_families
                else None
            )
            labelled_families.add(edge.family)
            linestyle = "-" if edge.stopped_growing else "--"

            axes.plot(
                points[:, 1],
                points[:, 0],
                marker="o",
                color=color,
                linestyle=linestyle,
                alpha=0.55,
                label=label,
                zorder=2,
            )

        start = state.clusters[starting_idx]
        axes.scatter(
            start[1],
            start[0],
            marker="*",
            s=180,
            color="red",
            label="Starting cone",
            zorder=3,
        )
        axes.set_title(
            f"Edge generation {generation}: {len(edges)} hypotheses"
        )
        axes.set_xlabel("y - lateral / right (m)")
        axes.set_ylabel("x - forward (m)")
        axes.set_aspect("equal", adjustable="datalim")
        axes.grid(True, alpha=0.25)
        axes.legend(title="Plot key", loc="best")
        figure.tight_layout()

        plt.show(block=True)
        plt.close(figure)

    def nearby_candidates(self, centre_point, edge: Edge, state: FrameState) -> list[int]:
        """
        Given a centre point, find indices of all clusters nearby and that aren't already in the hypothesis
        Nearby definition: within the ring of where a cone could lie
        """
        nearby_idxs = state.tree.query_ball_point(centre_point, self.max_neighbour_radius, 2)
        valid_idxs: list[int] = []
        for idx in nearby_idxs:
            idx_already_in_hypothesis : bool = idx in edge.idxs or idx in edge.idxs
            if np.linalg.norm(state.clusters[idx] - centre_point) > self.min_neighbour_radius and not idx_already_in_hypothesis:
                valid_idxs.append(idx)
        return valid_idxs





