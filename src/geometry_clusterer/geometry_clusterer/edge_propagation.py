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

@dataclass
class EdgeFamily:
    edges: list[Edge]
    family_id: int
    max_length: int = 0

    @property
    def all_idxs(self):
        return set([idx for e in self.edges for idx in e.idxs])

    @property
    def stopped_growing(self) -> bool:
        return all(edge.stopped_growing for edge in self.edges)


class EdgePropagation(ConeDetector):
    def __init__(
        self,
        max_neighbour_radius: float,
        min_neighbour_radius: float,
        max_angle_change: float = 30.0,
        wall_density_threshold: float = 1.8,
        max_consecutive_angle_change: float = 20.0,
        plot_growth: bool = False,
    ):
        self.max_neighbour_radius = max_neighbour_radius
        self.min_neighbour_radius = min_neighbour_radius
        self.max_angle_change = max_angle_change
        self.wall_density_threshold = wall_density_threshold
        self.max_consecutive_angle_change = max_consecutive_angle_change
        self.plot_growth = plot_growth

    def detect_cones(self, cluster_centres: list[Cluster]) -> object:
        if not cluster_centres:
            return []
        cluster_xy = np.array([c.position for c in cluster_centres])
        tree = cKDTree(cluster_xy)
        state = FrameState(cluster_xy, tree)

        visited_idxs: set[int] = set()
        edge_families: list[EdgeFamily] = []
        family = 0
        for i in range(len(state.clusters)):
            if i in visited_idxs:
                continue

            edge_family, newly_visited = self.detect_edges(i, state, family)
            visited_idxs.update(newly_visited)

            # Many edges are dense, wall families. these should be removed
            if self.is_likely_wall_family(edge_family, state):
                continue

            if edge_family.edges:
                edge_families.append(edge_family)
                family += 1
                # print(f"Length of edge family {family}: {len(edge_family.edges)}")

        track_segments = self.retrieve_track_segments(edge_families, state)

        # track_edge_families = list({
        #     family.family_id: family
        #     for track_segment in track_segments
        #     for family in track_segment
        # }.values())

        self.plot_edges(state, edge_families, 0, 0)
        # self.plot_edges(state, track_edge_families, 0, 0)

    def retrieve_track_segments(self, edge_families: list[EdgeFamily], state: FrameState) -> list[tuple[EdgeFamily, EdgeFamily]]:
        track_segments: list[tuple[EdgeFamily, EdgeFamily]] = []

        for family_index, family_1 in enumerate(edge_families):
            for family_2 in edge_families[family_index + 1:]:
                if self.test_pair_compatability(family_1, family_2, state):
                    track_segments.append((family_1, family_2))

        return track_segments

    def edges_are_close(self, edge_1: Edge, edge_2: Edge, state: FrameState, max_gap: float = 4.0) -> bool:
        edge_1_start = state.clusters[edge_1.idxs[0]]
        edge_1_end = state.clusters[edge_1.idxs[-1]]
        edge_2_start = state.clusters[edge_2.idxs[0]]
        edge_2_end = state.clusters[edge_2.idxs[-1]]

        edge_1_min = np.minimum(edge_1_start, edge_1_end)
        edge_1_max = np.maximum(edge_1_start, edge_1_end)
        edge_2_min = np.minimum(edge_2_start, edge_2_end)
        edge_2_max = np.maximum(edge_2_start, edge_2_end)

        return bool(
            np.all(edge_1_max + max_gap >= edge_2_min)
            and np.all(edge_2_max + max_gap >= edge_1_min)
        )

    def test_pair_compatability(self, family_1: EdgeFamily, family_2: EdgeFamily, state: FrameState) -> bool:
        family_1_idxs = {idx 
                         for e in family_1.edges
                         for idx in e.idxs}

        family_2_idxs = {idx 
                         for e in family_2.edges
                         for idx in e.idxs}

        # check for overlap
        overlap = set(family_1_idxs) & set(family_2_idxs)
        overlap_count: int = len(overlap)
        ignore_idxs = None
        if overlap_count >= 2 or not self.edges_are_close(family_1.edges[0], family_2.edges[0], state):
            return False
        if overlap_count == 1:
            ignore_idxs = list(overlap)[0]

        return True

    def detect_edges(self, start_idx: int, state: FrameState, family: int) -> tuple[EdgeFamily, set[int]]:
        edge_family = EdgeFamily([Edge([start_idx])], family, 1)
        generation = 0

        while True:
            edge_family = self.grow_edges(edge_family, state)
            edge_family = self.remove_duplicates(edge_family)
            generation += 1
            if edge_family.stopped_growing:
                break
        
        edge_family.edges = [edge for edge in edge_family.edges if edge.length > 2]

        visited_idxs: set[int] = set(idx for edge in edge_family.edges for idx in edge.idxs)

        return (edge_family, visited_idxs)

    def grow_edges(self, edge_family: EdgeFamily, state: FrameState) -> EdgeFamily:
        grown_edge_family: EdgeFamily = EdgeFamily([], edge_family.family_id, edge_family.max_length)
        has_grown: bool = False
        for e in edge_family.edges:
            if e.stopped_growing == True:
                # If an edge is short, yet other edges in the family are long, it has very likely reached an outlier or started in the wrong direction
                stump: bool = e.length <= 3 and edge_family.max_length > 5
                if not stump:
                    grown_edge_family.edges.append(e)
                continue

            child_hypotheses = self.propose_extensions(e, state)
            if len(child_hypotheses) == 0:
                e.stopped_growing = True
                grown_edge_family.edges.append(e)
            else:
                grown_edge_family.edges.extend(child_hypotheses)
                has_grown = True

        if has_grown: 
            grown_edge_family.max_length += 1
        
        return grown_edge_family

    def remove_duplicates(self, edge_family: EdgeFamily) -> EdgeFamily:
        seen: set[tuple[int, ...]] = set()
        unique_edges: list[Edge] = []

        for edge in edge_family.edges:
            key = tuple(edge.idxs)
            if key in seen:
                continue

            seen.add(key)
            unique_edges.append(edge)

        return EdgeFamily(unique_edges, edge_family.family_id, edge_family.max_length)

    def is_likely_wall_family(self, edge_family: EdgeFamily, state: FrameState) -> bool:
        edge_densities: list[float] = []
        for e in edge_family.edges:
            if e.length < 5:
                continue

            prev_idx = e.idxs[0]
            total_dist = 0
            for idx in e.idxs[1:]:
                total_dist += np.linalg.norm(state.clusters[idx] - state.clusters[prev_idx])
            edge_densities.append(total_dist / e.length)

        # if 5 or more clusters are very close together, then this will not be a track family
        if edge_densities and min(edge_densities) < self.wall_density_threshold:
            return True
        return False
        
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
            # checks that angle is reasonable for track segment
            # end == 1 => we want indices -1 and -2, end == 0 => we want indices 0 and 1 - these formulas generate that
            final_angle = self.angle_between_points(
                state.clusters[idx],
                state.clusters[edge.idxs[-end]],
                state.clusters[edge.idxs[-3 * end + 1]],
            )
            if final_angle > np.deg2rad(180 - self.max_angle_change):
                if edge.length == 2:
                    valid_idxs.append(idx)
                else:
                    second_to_final_angle = self.angle_between_points(
                        state.clusters[edge.idxs[-end]],
                        state.clusters[edge.idxs[-3 * end + 1]],
                        state.clusters[edge.idxs[-5 * end + 2]]
                    )
                    # Checks that the change in angle isn't unnaturally big
                    if abs(second_to_final_angle - final_angle) < np.deg2rad(
                        self.max_consecutive_angle_change
                    ):
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
                child_edges.append(Edge([*edge.idxs, c]))
            else:
                child_edges.append(Edge([c, *edge.idxs]))
        return child_edges

    def plot_edges(
        self,
        state: FrameState,
        edge_families: list[EdgeFamily],
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

        family_ids = sorted(family.family_id for family in edge_families)
        color_map = plt.get_cmap("tab10")
        family_colors = {
            family_id: color_map(index % 10)
            for index, family_id in enumerate(family_ids)
        }
        labelled_families: set[int] = set()

        for family in edge_families:
            for edge in family.edges:
                points = state.clusters[edge.idxs]
                color = family_colors[family.family_id]
                label = (
                    f"Family {family.family_id + 1}"
                    if family.family_id not in labelled_families
                    else None
                )
                labelled_families.add(family.family_id)
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
            f"Edge generation {generation}: "
            f"{sum(len(family.edges) for family in edge_families)} hypotheses"
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





