from .cone_detector import Cluster, ConeDetector
import numpy as np
from scipy.spatial import cKDTree
from dataclasses import dataclass
from scipy.optimize import least_squares
from collections import deque

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
    starting_idx: int
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
        wall_density_threshold: float = 1.0,
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
        """
        From clusters (clusters may be cones OR noise), determine the set of cones within this. Hence, determining map of the track
        Steps:
        - Look for edges
        - These edges come in families. A family of edges are numerous different sets of clusters that could describe one true track boundary
        - Find track segments
            - These are found by finding pairs of edge families that are reasonably close and reasonably parallel 
        - Return cones that lie within these track segments (not implemented yet, but trivial)
        """
        if not cluster_centres:
            return []
        
        cluster_xy = np.array([c.position for c in cluster_centres])

        tree = cKDTree(cluster_xy)
        state = FrameState(cluster_xy, tree)

        edge_families: list[EdgeFamily] = self.find_edge_families(state)

        # best_edges: list[Edge] = self.extract_best_edges(edge_families, state)

        # Determine pairs of edges that form track segments
        track_segments: list[tuple[EdgeFamily, EdgeFamily]] = self.retrieve_track_segments(edge_families, state)

        # This will be a subset of edge_families - only the ones that form part of the track
        track_edge_families: list[EdgeFamily] = list({
            family.family_id: family
            for track_segment in track_segments
            for family in track_segment
        }.values())

        self.plot_edges(state, edge_families, 0)
        self.plot_edges(state, track_edge_families, 0)

    def find_edge_families(self, state: FrameState) -> list[EdgeFamily]:
        visited_idxs: set[int] = set()
        edge_families: list[EdgeFamily] = []
        family_idx = 0
        for i in range(len(state.clusters)):
            if i in visited_idxs:
                continue

            edge_family, newly_visited = self.find_edges_from_cluster(i, state, family_idx)
            visited_idxs.update(newly_visited)

            # Many edges are dense, wall families. these should be removed
            if self.is_likely_wall_family(edge_family, state):
                continue

            if edge_family.edges:
                edge_families.append(edge_family)
                family_idx += 1

        return edge_families

    # def extract_best_edges(self, edge_families: list[EdgeFamily], state: FrameState) -> list[Edge]:
    #     for familiy in edge_families:
    #         for edge in familiy:
    #             self.score(edge)
            
        
    #     return NotImplementedError

    def retrieve_track_segments(self, edge_families: list[EdgeFamily], state: FrameState) -> list[tuple[EdgeFamily, EdgeFamily]]:
        track_segments: list[tuple[EdgeFamily, EdgeFamily]] = []

        for family_index, family_1 in enumerate(edge_families):
            for family_2 in edge_families[family_index + 1:]:
                if self.test_pair_compatability(family_1, family_2, state):
                    track_segments.append((family_1, family_2))

        return track_segments

    def edges_are_close(self, edge_1: Edge, edge_2: Edge, state: FrameState, max_gap: float = 5.0) -> bool:
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
        # Temporary - annecdotally, families with over 100 hypothesis are always walls. However, an improved solution can be for sure found here
        if len(family_1.edges) > 100 or len(family_2.edges) > 100:
            return False

        family_1_idxs = {idx 
                         for e in family_1.edges
                         for idx in e.idxs}

        family_2_idxs = {idx 
                         for e in family_2.edges
                         for idx in e.idxs}

        overlap = set(family_1_idxs) & set(family_2_idxs)

        if len(overlap) > 0:
            return False

        edges_too_distant: bool = not self.edges_are_close(family_1.edges[0], family_2.edges[0], state) 
        if edges_too_distant:
            return False

        for i , edge_1 in enumerate(family_1.edges):
            for j, edge_2 in enumerate(family_2.edges):
                # check for overlap
                # overlap = set(edge_1.idxs) & set(edge_2.idxs)

                # if len(overlap) > 0:
                #     continue

                if self.test_edge_pair_too_close(edge_1, edge_2, state) == True:
                    continue

                if self.test_edge_pair_parallelity(edge_1, edge_2, state) == False:
                    continue

                # All tests past - return that a valid paid exists
                return True

        return False

    def test_edge_pair_too_close(self, edge_1: Edge, edge_2: Edge, state: FrameState):
        d2 = np.sum((state.clusters[edge_1.idxs][:, None, :] - state.clusters[edge_2.idxs][None, :, :])**2, axis=2)
        too_close = np.any(d2 < self.min_neighbour_radius**2)
        return too_close

    def test_edge_pair_parallelity(self, edge_1: Edge, edge_2: Edge, state: FrameState):
        parallel_pair: bool = False

        for k in range(len(edge_1.idxs) - 1):
            idx_nearest = self.find_nearest_opposing_cone(edge_1.idxs[k], edge_2, state)
            idx_next_nearest = self.find_nearest_opposing_cone(edge_1.idxs[k + 1], edge_2, state, idxs_to_avoid=[idx_nearest] if idx_nearest is not None else [])
            if idx_nearest == None or idx_next_nearest == None:
                continue
            edge_1_local_vector = state.clusters[edge_1.idxs[k+1]] - state.clusters[edge_1.idxs[k]]
            edge_2_local_vector = state.clusters[idx_next_nearest] - state.clusters[idx_nearest]

            if self.roughly_parallel(edge_1_local_vector, edge_2_local_vector) == False:
                parallel_pair = False
                break
            else:
                parallel_pair = True
        return parallel_pair

    def roughly_parallel(self, v1, v2):
        # normalise
        v1_norm = v1 / np.linalg.norm(v1)
        v2_norm = v2 / np.linalg.norm(v2)

        # dot product
        abs_dot = abs(np.dot(v1_norm, v2_norm))

        # clamp for numerical stability
        dot = np.clip(abs_dot, 0.0, 1.0)

        if np.arccos(dot) > np.deg2rad(35):
            return False
        return True

    def find_nearest_opposing_cone(self, idx: int, opposing_edge: Edge, state: FrameState, idxs_to_avoid: list[int] = []):
        p = state.clusters[idx]
        best_idx = None
        best_dist = self.max_neighbour_radius

        for j in opposing_edge.idxs:
            d = np.linalg.norm(state.clusters[j] - p)
            if d <= best_dist and d >= self.min_neighbour_radius and j not in idxs_to_avoid:
                best_dist = d
                best_idx = j

        return best_idx

    def find_edges_from_cluster(self, start_idx: int, state: FrameState, family: int) -> tuple[EdgeFamily, set[int]]:
        edge_family = EdgeFamily([Edge([start_idx])], family, start_idx, 1)
        generation = 0

        while True:
            edge_family = self.grow_edges(edge_family, state)
            edge_family = self.remove_duplicates(edge_family)
            generation += 1
            if edge_family.stopped_growing:
                break
        
        visited_idxs: set[int] = set(idx for edge in edge_family.edges for idx in edge.idxs)

        return (edge_family, visited_idxs)

    def grow_edges(self, edge_family: EdgeFamily, state: FrameState) -> EdgeFamily:
        """
        Based on a current set of edges, grow each of these by 1 cluster, if they can be
        Further, remove 'stumps'. These are edges that are significantly shorter than the 'best' (longest) edges
        """
        grown_edge_family: EdgeFamily = EdgeFamily(
            [],
            edge_family.family_id,
            edge_family.starting_idx,
            edge_family.max_length,
        )
        family_has_grown: bool = False
        for e in edge_family.edges:
            if e.stopped_growing == True:
                if self.check_for_stump(e, edge_family) == False:
                    grown_edge_family.edges.append(e)
                continue

            child_edge_hypotheses = self.propose_extensions(e, state)

            if len(child_edge_hypotheses) == 0:
                e.stopped_growing = True
                grown_edge_family.edges.append(e)
            else:
                grown_edge_family.edges.extend(child_edge_hypotheses)
                family_has_grown = True

        if family_has_grown: 
            grown_edge_family.max_length += 1
        
        return grown_edge_family

    def check_for_stump(self, edge: Edge, family: EdgeFamily):
        # If an edge is long, but still significantly shorter than the longest edge, then it's very likely a stump. Most often (annecdotally), it's jumped to the other side of the track, maybe with the help of an outlier. 
        stump_big: bool = edge.length > 3 and edge.length < family.max_length - 3
        # If an edge is short, yet other edges in the family are long, it has very likely reached an outlier or started in the wrong direction
        stump_3: bool = edge.length == 3 and family.max_length > 5
        # length 2 stumps are special - they form very easily initially (every direction is equally valid). As soon as a hypothesis grows finds a direction to start growing in, size 2 stumps should be removed
        stump_2: bool = edge.length == 2 and family.max_length > 2
        if stump_2 or stump_3 or stump_big:
            return True
        return False

    def remove_duplicates(self, edge_family: EdgeFamily) -> EdgeFamily:
        seen: set[tuple[int, ...]] = set()
        unique_edges: list[Edge] = []

        for edge in edge_family.edges:
            key = tuple(edge.idxs)
            if key in seen:
                continue

            seen.add(key)
            unique_edges.append(edge)

        return EdgeFamily(
            unique_edges,
            edge_family.family_id,
            edge_family.starting_idx,
            edge_family.max_length,
        )

    def is_likely_wall_family(self, edge_family: EdgeFamily, state: FrameState) -> bool:
        def point_distance(edge: Edge, i1: int, i2: int):
            return np.linalg.norm(state.clusters[i1] - state.clusters[i2])

        min_size = 5

        # Even for a human, it's hard to make inferences about short edges - could be a hairpin, oould be starting orange cones, or could be wall
        if edge_family.max_length < min_size:
            return False

        # We only check the maximal length families for efficiency - further, these will almost certainly include almost every wall cluster (edges are straight lines and there is no 'min distance' check - the max edge will just jump between every cluster)
        # In most cases, there will only be one maximal length family
        max_families = [e for e in edge_family.edges if e.length == edge_family.max_length]

        for edge in max_families:
            # We have sliding window of pair distances - if any set of 5 clusters are too close together, then that is sufficient evidence to assume wall
            # Definition of 'Too close together': in an ordered list of 5 edges, the average distance between a pair is less than a specified threshold (i.e. 1 meter)
            window_dists = deque([point_distance(edge, edge.idxs[i], edge.idxs[i-1]) for i in range(1, min_size)])
            prev_idx = edge.idxs[min_size-1]
            for idx in edge.idxs[min_size:]:
                window_dists.popleft()
                window_dists.append(point_distance(edge, idx, prev_idx))
                prev_idx = idx
                avg_dist: float = sum(window_dists)/min_size

                if avg_dist < self.wall_density_threshold:
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

            start_candidate_idxs = self.select_valid_candidates(start_candidate_idxs, edge, False, state)
            end_candidate_idxs = self.select_valid_candidates(end_candidate_idxs, edge, True, state)

            child_edges.extend(self.append_candidates_to_edge(start_candidate_idxs, edge, False))
            child_edges.extend(self.append_candidates_to_edge(end_candidate_idxs, edge, True))
        return child_edges

    def select_valid_candidates(self, test_idxs: list[int], edge: Edge, extend_at_end: bool, state: FrameState) -> list[int]:
        """
        Given an edge and a bunch of proposed extensions (i.e. adding a cluster to the end of the edge), test whether those extensions make sense geometrically. 
        To do this, we check whether each extension creates a sensible continution of the track, based on the change of angle it makes with previous segments.
        """
        if edge.length <= 1:
            return test_idxs

        max_consecutive_angle_change_rad = np.deg2rad(self.max_consecutive_angle_change)
        max_angle_change_rad = np.deg2rad(self.max_angle_change + 10)
        first, second, third = 0, 1, 2
        if extend_at_end:
            first, second, third = -1, -2, -3


        valid_idxs: list[int] = []

        for idx in test_idxs:
            idx_angle_off_straight = self.angle_off_straight(idx, edge.idxs[first], edge.idxs[second], state)

            # If test point is too squewed, immediately reject
            if abs(idx_angle_off_straight) > max_angle_change_rad:
                continue

            # # Consider curvature direction of edge
            if edge.length > 2:
                edge_end_angle = self.angle_off_straight(edge.idxs[first], edge.idxs[second], edge.idxs[third], state)
                # Checks that the change in angle isn't unnaturally big
                if abs(edge_end_angle - idx_angle_off_straight) > max_consecutive_angle_change_rad:
                    continue
            
            valid_idxs.append(idx)

        return valid_idxs

    def angle_off_straight(self, idx1, idx2, idx3, state: FrameState):
        """
        Returns the angle ABC (in radians) between clusters
        Angle is always between 0 and 2 pi.
        """
        a = state.clusters[idx1]
        b = state.clusters[idx2]
        c = state.clusters[idx3]
        v1 = b - a
        v2 = c - b
        cross = v1[0] * v2[1] - v1[1] * v2[0]
        dot = np.dot(v1, v2)
        return np.arctan2(cross, dot)
        # # normalise
        # v1_norm = v1 / np.linalg.norm(v1)
        # v2_norm = v2 / np.linalg.norm(v2)

        # dot product
        # dot = np.dot(v1_norm, v2_norm)

        # # clamp for numerical stability
        # dot = np.clip(dot, -1.0, 1.0)

        # return np.pi - np.arccos(dot)

    def fit_curve(self, edge: Edge, state: FrameState):
        x = state.clusters[edge.idxs][0]
        y = state.clusters[edge.idxs][1]

        assert len(x) > 2
        deg = len(x) - 2

        def residuals(coeffs):
            return np.polyval(coeffs, x) - y

        initial = np.polyfit(x, y, deg=deg)

        result = least_squares(
            residuals,
            initial,
            loss="soft_l1",
            f_scale=0.15
        )
        return result.x

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
        color_map = plt.get_cmap("tab20")
        family_colors = {
            family_id: color_map(index % 20)
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

        starting_idxs = [family.starting_idx for family in edge_families]
        if starting_idxs:
            starts = state.clusters[starting_idxs]
            axes.scatter(
                starts[:, 1],
                starts[:, 0],
                marker="*",
                s=180,
                color="red",
                label="Starting cones",
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
