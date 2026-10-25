from .cone_detector import Cluster, ConeDetector
import numpy as np
from scipy.spatial import cKDTree
from dataclasses import dataclass
from scipy.optimize import least_squares
from scipy.interpolate import BSpline, make_interp_spline
from collections import deque
import logging 

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

@dataclass
class FrameState:
    clusters: np.ndarray
    tree: cKDTree

@dataclass
class Edge:
    idxs: list[int]
    family_id: int
    stopped_growing: bool = False

    @property
    def length(self):
        return len(self.idxs)

@dataclass
class CandidateCluster:
    idx: int
    angle_off_straight_rad: float
    distance: float
    penalty: float


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
        max_track_width: float = 5.0,
        min_track_width: float = 2.0
    ):
        self.max_neighbour_radius = max_neighbour_radius
        self.min_neighbour_radius = min_neighbour_radius
        self.max_angle_change = max_angle_change
        self.wall_density_threshold = wall_density_threshold
        self.max_consecutive_angle_change = max_consecutive_angle_change
        self.plot_growth = plot_growth
        self.max_track_width = max_track_width
        self.min_track_width = min_track_width

    def detect_cones(self, cluster_centres: list[Cluster]) -> np.ndarray:
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
            return np.array([])
        
        cluster_xy = np.array([c.position for c in cluster_centres])

        tree = cKDTree(cluster_xy)
        state = FrameState(cluster_xy, tree)

        edge_candidates: list[Edge] = self.find_edges(state)

        # Determine pairs of edges that form track segments
        track_segments: list[tuple[Edge, Edge]] = self.retrieve_track_segments(edge_candidates, state)

        # This will be a subset of edge_families - only the ones that form part of the track
        track_edges = list({
            edge.family_id: edge
            for track_segment in track_segments
            for edge in track_segment
        }.values())

        cones = np.array([state.clusters[idx] for edge in track_edges for idx in edge.idxs])

        # self.plot_edges(state, track_edges)
        # self.plot_edges(state, edge_candidates)
        return cones

    def find_edges(self, state: FrameState) -> list[Edge]:
        """
        Returns a list of plausible track edges
        """
        visited_idxs: set[int] = set()
        edge_families: list[EdgeFamily] = []
        family_id = 0
        for i in range(len(state.clusters)):
            if i in visited_idxs:
                continue

            edge_family, newly_visited = self.find_edges_from_cluster(i, state, family_id)

            if not edge_family:
                logger.warning("Starting idx %i returned no family", edge_family.starting_idx)
                continue

            visited_idxs.update(newly_visited)

            if edge_family.max_length == 1:
                logger.info("Starting idx %i gave length 1 family", edge_family.family_id)
                continue

            # Many edges are dense, wall families. these should be removed
            if self.is_likely_wall_family(edge_family, state):
                logger.info("Family %i deemed likely wall family", edge_family.family_id)
                # continue

            # Edge families with few members ('stump' families) often are very multidirectional. If this is the case, then these should be broken into numerous families. They are, after all, representing truly different edge hypotheses
            # This also updates family_id to be ready for the next iteration
            new_edge_families, family_id = self.break_distinct_families(edge_family)

            edge_families.extend(new_edge_families)

        # extracting maximial edges can be done earlier in the pipeline - however, I'm keeping full family generation in for now, while experimenting
        maximal_edges: list[Edge] = []
        for family in edge_families:
            max_edges = [e for e in family.edges if e.length == family.max_length]
            if len(max_edges) > 1:
                logger.warning("More than 1 max edge detected on family %i", family.family_id + 1)

            maximal_edges.append(max_edges[0])
            
        return maximal_edges

    def retrieve_track_segments(self, max_edges: list[Edge], state: FrameState) -> list[tuple[Edge, Edge]]:
        track_segments: list[tuple[Edge, Edge]] = []

        for i, edge1 in enumerate(max_edges):
            for edge2 in max_edges[i+1:]:
                if not self.test_pair_compatability(edge1, edge2, state):
                    continue
                
                track_segments.append((edge1, edge2))

        return track_segments

    def break_distinct_families(self, edge_family: EdgeFamily) -> tuple[list[EdgeFamily], int]:
        """
        Breaks one family down into several distinct families. 
        Returns the new families, and the new family_id to work with
        Justification:
        Some families of edges should truly be numerous families. This is becuase they have edges that point in very different directions, representing different edge hypothese
        when a family of edges is only 2 or 3 clusters long, it hasn't had time to gain evidence for a certain edge / direction
        hence, we'll have possibly many different edges emanating from the start cluster, all in different directions
        - some of these will cross the track and land on the opposing side.
        - however, sometimes these form a genuine pair of clusters on an edge.
        we want these to be separte families - they are different edge hypothese, after all.

        hence, for families of few cones, we break them down into distinct families
        """
        # families with an edge bigger than 3 already have a defined 'most likely edge' (and will have performed stump removal)
        if edge_family.max_length > 3:
            return [edge_family], edge_family.family_id + 1

        expanded_edge_family = []
        cur_family_id = edge_family.family_id
        for edge in edge_family.edges:
            edge.family_id = cur_family_id
            expanded_edge_family.append(EdgeFamily([edge], cur_family_id, edge_family.starting_idx, edge.length))
            cur_family_id += 1

        return expanded_edge_family, cur_family_id


##########################################
# Track Finding

    def test_pair_compatability(self, edge0: Edge, edge1: Edge, state: FrameState) -> bool:
        """
        Given a pair of edges, test if they could make a valid track segment
        """
        logger.info("Testing compatability of edges %i and %i", edge0.family_id + 1, edge1.family_id+1)

        # We can't justify matching edges that are only a pair and a trio - this could be a wall segment
        if len(edge0.idxs) + len(edge1.idxs) <= 5:
            return False

        overlap = set(edge0.idxs) & set(edge1.idxs)

        if len(overlap) > 0:
            logger.debug("Removed due to overlap")
            return False

        edges_too_distant: bool = not self.edges_are_close(edge0, edge1, state, self.max_track_width) 
        if edges_too_distant:
            logger.debug("Removed since too distant")
            return False

        if not self.edge_pair_valid_distance(edge0, edge1, state):
            return False

        return True

    def edges_are_close(self, edge_1: Edge, edge_2: Edge, state: FrameState, max_gap: float) -> bool:
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

    def fit_spline(self, edge: Edge, state: FrameState) -> BSpline:
        edge_points = state.clusters[edge.idxs]
        logger.info("Edge x, y points: %s, %s", edge_points[:, 0], edge_points[:, 1])
        b: BSpline = make_interp_spline(edge_points[:, 0], edge_points[:, 1])

        return b

    def edge_pair_valid_distance(self, edge_0: Edge, edge_1: Edge, state: FrameState):
        def set_same_ordering(edge_0xy: np.ndarray, edge_1xy: np.ndarray) -> np.ndarray:
            # Assumptions: this assumes that the end points of two paired edges face in the same direction. In some cases, for example with one very long edge which curves around, this may be false
            # Set direction such that we're comparing in the right order
            ordering_0 = (edge_0xy[-1] - edge_0xy[0]) / np.linalg.norm(edge_0xy[-1] - edge_0xy[0])
            ordering_1 = (edge_1xy[-1] - edge_1xy[0]) / np.linalg.norm(edge_1xy[-1] - edge_1xy[0])
            edge_1xy = edge_1xy[::-1] if np.dot(ordering_0, ordering_1) < 0 else edge_1xy
            # closest_end = np.argmin([np.linalg.norm(edge_0xy[0] - edge_1xy[0]), np.linalg.norm(edge_0xy[0] - edge_1xy[-1])])
            # edge_1xy = edge_1xy[::-1] if closest_end == 1 else edge_1xy
            return edge_1xy

        edge_0xy: np.ndarray = state.clusters[edge_0.idxs]
        edge_1xy: np.ndarray = state.clusters[edge_1.idxs]

        edge_1xy_ordered = set_same_ordering(edge_0xy, edge_1xy)

        return self.compare_distances_and_angling(edge_0xy, edge_1xy_ordered, state)
    
    def compare_distances_and_angling(self, edge_0xy: np.ndarray, edge_1xy: np.ndarray, state:FrameState):
        def norm(v: np.ndarray):
            norm = np.linalg.norm(v)
            return np.array([-v[1], v[0]]) / norm
        
        def calculate_leading_edge(leading_at_start: bool) -> int:
            first, second = (0, 1) if leading_at_start else (-1, -2)

            # Outward-pointing tangent for each edge
            t0 = edge_0xy[first] - edge_0xy[second]
            t1 = edge_1xy[first] - edge_1xy[second]

            norm0 = np.linalg.norm(t0)
            norm1 = np.linalg.norm(t1)

            eps = 1e-10
            if norm0 < eps or norm1 < eps:
                logger.warning("Degenerate edge tangent")
                return 0

            t0 /= norm0
            t1 /= norm1

            # Average the unit tangents
            direction = t0 + t1
            norm_direction = np.linalg.norm(direction)

            # If tangents nearly cancel, fall back to edge 0
            if norm_direction < eps:
                logger.warning("Opposing tangents; falling back to edge 0")
                direction = t0
            else:
                direction /= norm_direction

            # Positive projection means edge 1 extends further outward
            dot_product = np.dot(
                direction,
                edge_1xy[first] - edge_0xy[first]
            )

            logger.debug("Leading direction: %s, projection: %.3f",
                        direction, dot_product)

            return 1 if dot_product > 0 else 0
        def calculate_leading_edge_old(leading_at_start: bool):
            # start of edge case
            first, second = 0, 1
            if leading_at_start is False:
                first, second = -1, -2
            
            start_direction = (edge_0xy[first] - edge_0xy[second]) / np.linalg.norm(edge_0xy[first] - edge_0xy[second])
            logger.info("Start direction: %s", start_direction)
            # In the coordinate frame of the start_direction, dot product gives us the coefficient in the start_direction direction. 
            # The minus edge_0xy[first] is to ensure the origin is at that point
            dot_product = np.dot(start_direction, edge_1xy[first] - edge_0xy[first])

            leading_edge: int = 0
            if (dot_product > 0):
                leading_edge = 1
            return leading_edge
    
        def densify(edge: np.ndarray, num_points = 5) -> np.ndarray:
            dense_edge = []

            for p0, p1 in zip(edge[:-1], edge[1:]):
                t = np.linspace(0, 1, num_points + 2)[:-1]  # exclude endpoint
                segment = p0 + t[:, None] * (p1 - p0)
                dense_edge.extend(segment)

            dense_edge.append(edge[-1])
            return np.asarray(dense_edge) 

        def valid_spacing(dists, leading_edge: int, start: bool):
            min_dist_idx = np.argmin(dists)

            first, second = (0, 1) if start else (-1, -2)

            # This is the perpencidular (unit) vector to the shorter edge
            perp = norm(edge_set[not leading_edge][first] - edge_set[not leading_edge][second])

            perp_separation = abs(np.dot(edge_set[leading_edge][min_dist_idx] - edge_set[not leading_edge][first], perp))

            logger.info("Dist was %f", min_dist_idx)
            if perp_separation > self.min_track_width and perp_separation < self.max_track_width:
                return True
            return False

        def valid_angling(dists, leading, test_at_start: bool):
            first, second = (0, 1) if test_at_start else (-1, -2)

            min_dist_idx = np.argmin(dists)

            edge = edge_set[leading]
            shorter = edge_set[1 - leading]

            neighbouring_idx = min_dist_idx - 1 if min_dist_idx > 0 else min_dist_idx + 1

            # Tangents to both edges at the closest points
            leading_edge_dir = edge[min_dist_idx] - edge[neighbouring_idx]
            shorter_edge_dir = shorter[first] - shorter[second]

            leading_edge_dir /= np.linalg.norm(leading_edge_dir)
            shorter_edge_dir /= np.linalg.norm(shorter_edge_dir)

            # Compare angles (dot product avoids arccos)
            dot = np.dot(leading_edge_dir, shorter_edge_dir)
            return abs(dot) > np.cos(np.deg2rad(50))

        logger.info("Edge 0 starts with %s, Edge 1 with %s", edge_0xy[0], edge_1xy[0])

        leading_start = calculate_leading_edge(leading_at_start = True)
        leading_end = calculate_leading_edge(leading_at_start = False)

        logger.info("Leading start edge was %i, end edge was %i", leading_start, leading_end)

        edge_0xy = densify(edge_0xy)
        edge_1xy = densify(edge_1xy)
        edge_set = [edge_0xy, edge_1xy]

        dists_start = np.linalg.norm(edge_set[leading_start] - edge_set[not leading_start][0], axis=1)
        logger.info("Distance taken between %s and %s", edge_set[leading_start][np.argmin(dists_start)], edge_set[not leading_start][0])
        dists_end = np.linalg.norm(edge_set[leading_end] - edge_set[not leading_end][-1], axis=1)
        logger.info("Distance taken between %s and %s", edge_set[leading_end][np.argmin(dists_end)], edge_set[not leading_end][-1])

        if not valid_spacing(dists_start, leading_start, start=True) or not valid_spacing(dists_end, leading_end, start=False):
            return False

        if not valid_angling(dists_start, leading_start, test_at_start=True) or not valid_angling(dists_end, leading_end, test_at_start=False):
            return False

        return True
    
    def edge_pair_roughly_parallel(self, edge_1: Edge, edge_2: Edge, state: FrameState):
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

    def find_edges_from_cluster(self, start_idx: int, state: FrameState, family_id: int) -> tuple[EdgeFamily, set[int]]:
        edge_family = EdgeFamily(
            [Edge([start_idx], family_id, stopped_growing=False)],
            family_id,
            start_idx, 
            max_length = 1
        )

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
                    logger.info("Adult edge %s kept in family %i", e, edge_family.family_id)
                else:
                    logger.info("Stump %s from family %i removed", e.idxs, edge_family.family_id + 1)
                continue

            child_edge_hypotheses = self.propose_extensions(e, state)

            if len(child_edge_hypotheses) == 0:
                e.stopped_growing = True
                grown_edge_family.edges.append(e)
            else:

                grown_edge_family.edges.extend(child_edge_hypotheses)
                family_has_grown = True

                if logger.isEnabledFor(logging.INFO):
                    new_idxs = [
                        idx
                        for h in child_edge_hypotheses
                        for idx in h.idxs
                        if idx not in e.idxs
                    ]

                    logger.info(
                        "Edge %s in family %i grown to idx's %s. Length now %i",
                        e.idxs,
                        edge_family.family_id + 1,
                        new_idxs,
                        e.length + 1,
                    )

        if family_has_grown: 
            grown_edge_family.max_length += 1

        logger.info("Family %i now finished growth with %i edges", edge_family.family_id + 1, len(grown_edge_family.edges))
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
            keyr = tuple(edge.idxs[::-1])
            if key in seen or keyr in seen:
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
        """
        Based on a (growing) edge, we find potential next cones in that edge
        These next cones must, obviously, lie roughly (+- 30 degrees ish) straight on from the previous edge - edges of the track must be relatively smooth, naturally
        Returns a list of proposed extended edges
        """

        if edge.length == 1:
            candidates = self.nearby_candidates(state.clusters[edge.idxs[0]], edge, state)

            # In the length = 1 case, this will simply select the relatively close candidates
            candidate_idxs = self.select_valid_candidates(candidates, edge, False, state)
            return self.append_candidates_to_edge(candidate_idxs, edge, True)

        child_edges = []
        start_candidates = self.nearby_candidates(state.clusters[edge.idxs[0]], edge, state)
        end_candidates = self.nearby_candidates(state.clusters[edge.idxs[-1]], edge, state)

        start_candidates = self.select_valid_candidates(start_candidates, edge, False, state)
        end_candidates = self.select_valid_candidates(end_candidates, edge, True, state)

        child_edges.extend(self.append_candidates_to_edge(start_candidates, edge, False))
        child_edges.extend(self.append_candidates_to_edge(end_candidates, edge, True))

        return child_edges

    def select_valid_candidates(self, candidates: list[CandidateCluster], edge: Edge, extend_at_end: bool, state: FrameState) -> list[int]:
        def penalty(candidate: CandidateCluster):
            # Prioritise closer candidates, unless angle is significantly nicer 
            ANGLE_WEIGHT = 0.0
            DISTANCE_WEIGHT = 1 - ANGLE_WEIGHT
            return ANGLE_WEIGHT * (abs(candidate.angle_off_straight_rad)/(np.pi/6)) + DISTANCE_WEIGHT * abs(1.2 - candidate.distance)

        """
        Given an edge and a bunch of proposed extensions (i.e. adding a cluster to the end of the edge), test whether those extensions make sense geometrically. 
        To do this, we check whether each extension creates a sensible continution of the track, based on the change of angle it makes with previous segments.
        """
        if not candidates:
            return []

        if edge.length <= 1:
            return [c.idx for c in candidates]
            # return [min(candidates, key=lambda c: c.distance).idx]

        max_consecutive_angle_change_rad = np.deg2rad(self.max_consecutive_angle_change)
        max_angle_change_rad = np.deg2rad(self.max_angle_change + 10)
        first, second, third = 0, 1, 2
        if extend_at_end:
            first, second, third = -1, -2, -3

        verified_candidates: list[CandidateCluster] = []
        
        for candidate in candidates:
            candidate.angle_off_straight_rad = self.angle_off_straight(candidate.idx, edge.idxs[first], edge.idxs[second], state)

            # If candidate is too squewed, immediately reject
            if abs(candidate.angle_off_straight_rad) > max_angle_change_rad:
                continue

            # Consider curvature direction of edge
            if edge.length > 2:
                edge_end_angle = self.angle_off_straight(edge.idxs[first], edge.idxs[second], edge.idxs[third], state)
                # Checks that the change in angle isn't unnaturally big
                if abs(edge_end_angle - candidate.angle_off_straight_rad) > max_consecutive_angle_change_rad:
                    continue

            # Calculates a penalty for choosing this candidate. Closer = better, smaller angle change = better
            candidate.penalty = penalty(candidate)
            verified_candidates.append(candidate)

        # Select the best scoring edge
        if not verified_candidates:
            return []

        # Finds the candidate with the lowest penalty - prioritises close and fairly straight candidates
        best_candidate = min(verified_candidates, key = lambda c: c.penalty)
        # Finds all candidates similar to the best candidate, including the best candidate. Often, this will only be the best. However, if there are more than 1 similarly scored candidates, it is important to keep both
        # good_candidates = [c for c in verified_candidates if (c.penalty <= best_candidate.penalty * 1.0)]
            
        # return [c.idx for c in good_candidates]
        return [best_candidate.idx]

    def angle_off_straight(self, idx1, idx2, idx3, state: FrameState):
        """
        Returns angle off straight (in radians) between three clusters
        Angle is between -pi and pi.
        """
        a = state.clusters[idx1]
        b = state.clusters[idx2]
        c = state.clusters[idx3]
        v1 = b - a
        v2 = c - b
        cross = v1[0] * v2[1] - v1[1] * v2[0]
        dot = np.dot(v1, v2)
        return np.arctan2(cross, dot)

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
                child_edges.append(Edge([*edge.idxs, c], edge.family_id))
            else:
                child_edges.append(Edge([c, *edge.idxs], edge.family_id))
        return child_edges

    def plot_families(
        self,
        state: FrameState,
        families: list[EdgeFamily],
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

        family_ids = sorted(family.family_id for family in families)
        color_map = plt.get_cmap("tab20")
        family_colors = {
            family_id: color_map(index % 20)
            for index, family_id in enumerate(family_ids)
        }
        labelled_families: set[int] = set()

        for family in families:
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

        starting_idxs = [family.starting_idx for family in families]
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
            f"{sum(len(family.edges) for family in families)} hypotheses"
        )
        axes.set_xlabel("y - lateral / right (m)")
        axes.set_ylabel("x - forward (m)")
        axes.set_ylim(-60, 60)  # lateral
        axes.set_xlim(-80, 80)  # forward
        axes.set_aspect("equal", adjustable="box")
        axes.grid(True, alpha=0.25)
        axes.legend(
            title="Plot key",
            loc="upper left",
            bbox_to_anchor=(1.02, 1),
            fontsize=8,
        )
        figure.tight_layout()

        plt.show(block=True)
        plt.close(figure)

    def plot_edges(
        self,
        state: FrameState,
        edges: list[Edge],
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

        family_ids = sorted(family.family_id for family in edges)
        color_map = plt.get_cmap("tab20")
        family_colors = {
            family_id: color_map(index % 20)
            for index, family_id in enumerate(family_ids)
        }
        labelled_families: set[int] = set()

        for edge in edges:
            points = state.clusters[edge.idxs]
            color = family_colors[edge.family_id]
            label = (
                f"Family {edge.family_id + 1}"
                if edge.family_id not in labelled_families
                else None
            )
            labelled_families.add(edge.family_id)
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

        axes.set_title(
            f"{len(edges)} hypotheses"
        )
        axes.set_xlabel("y - lateral / right (m)")
        axes.set_ylabel("x - forward (m)")
        axes.set_ylim(-60, 60)  # lateral
        axes.set_xlim(-80, 80)  # forward
        axes.set_aspect("equal", adjustable="box")
        axes.grid(True, alpha=0.25)
        axes.legend(
            title="Plot key",
            loc="upper left",
            bbox_to_anchor=(1.02, 1),
            fontsize=8,
        )
        figure.tight_layout()

        plt.show(block=True)
        plt.close(figure)

    def nearby_candidates(self, centre_point, edge: Edge, state: FrameState) -> list[CandidateCluster]:
        """
        Given a centre point, find indices of all clusters nearby and that aren't already in the hypothesis
        Nearby definition: within the ring of where a cone could lie
        """
        nearby_idxs = state.tree.query_ball_point(centre_point, self.max_neighbour_radius, 2)
        valid_idxs: list[CandidateCluster] = []
        for idx in nearby_idxs:
            idx_already_in_hypothesis : bool = idx in edge.idxs or idx in edge.idxs
            distance = np.linalg.norm(state.clusters[idx] - centre_point) 
            if distance > self.min_neighbour_radius and not idx_already_in_hypothesis:
                # 100.0 set arbritarily
                valid_idxs.append(CandidateCluster(idx, 100.0, distance, 100.0))
        return valid_idxs
