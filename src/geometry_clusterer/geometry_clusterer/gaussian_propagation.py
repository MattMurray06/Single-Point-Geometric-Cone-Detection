import numpy as np
from dataclasses import dataclass
from .cone_detector import ConeDetector, Cluster
from scipy.spatial import cKDTree


@dataclass
class TrackSegment:
    cones_idxs: list[int]

@dataclass
class FrameState:
    clusters: np.ndarray
    tree: cKDTree
    track_segments: list[TrackSegment]

@dataclass(frozen=True)
class Hypothesis:
    side_0_idxs: list[int]
    side_1_idxs: list[int]
    score: float = 0.0

    @property
    def total_cones(self) -> int:
        return len(self.side_0_idxs) + len(self.side_1_idxs)

    @property
    def side_0_size(self) -> int:
        return len(self.side_0_idxs)

    @property
    def side_1_size(self) -> int:
        return len(self.side_1_idxs)

@dataclass
class Cone:
    side: int
    cluster_idx: int
    end: bool

class GaussianPropagation(ConeDetector):
    def __init__(
        self,
        max_neighbour_radius: float,
        min_neighbour_radius: float,
        beam_search_angle_deg: float,
        plot_growth: bool = False,
    ):
        self.max_neighbour_radius = max_neighbour_radius
        self.min_neighbour_radius = min_neighbour_radius
        self.beam_search_angle_rad = np.deg2rad(beam_search_angle_deg)
        self.plot_growth = plot_growth

    def detect_cones(self, cluster_centres: list[Cluster]): # -> np.ndarray:
        if not cluster_centres:
            return []
        cluster_xy = np.array([c.position for c in cluster_centres])
        tree = cKDTree(cluster_xy)
        state = FrameState(cluster_xy, tree, [])

        cone_idxs = []
        for i, c in enumerate(state.clusters):
            if i in cone_idxs:
                continue
            cone_idxs.extend(self.determine_track(i, state))

        # pick a start point
        # find candidate cones



    def determine_track(self, starting_idx: int, state: FrameState): # -> list[int]:
        hypotheses: list[Hypothesis] = [Hypothesis(side_0_idxs = [starting_idx], side_1_idxs = [])]
        if self.plot_growth:
            self.plot_hypotheses(state, hypotheses, starting_idx)

        max_depth = 5

        for _ in range(max_depth):
            grown_hypotheses = self.expand(hypotheses, state)
            dedupe_hypotheses = self.remove_duplicates(grown_hypotheses)
            hypotheses = self.prune(dedupe_hypotheses, state)

            if self.plot_growth:
                self.plot_hypotheses(state, hypotheses, starting_idx)
        return []

    def expand(self, hypotheses, state):
        grown_hypotheses = []
        for h in hypotheses:
            children = self.grow_hypothesis(h, state)
            grown_hypotheses.extend(children)
        return grown_hypotheses

    def prune(self, hypotheses, state):
        pruned= []
        for h in hypotheses:
            if not self.is_invalid(h, state):
                pruned.append(h)
        return pruned


    def is_invalid(self, hypothesis: Hypothesis, state: FrameState):
        if not self.valid_angle(hypothesis, state):
            return True
        
        
    def valid_angle(self, hypothesis: Hypothesis, state: FrameState):
        for track_side in [hypothesis.side_0_idxs, hypothesis.side_1_idxs]:
            if len(track_side) > 2:
                # Only need to check end angles, as others will have already been checked when admitting the hypothesis previously
                end_angle = self.angle_between_points(state.clusters[track_side[-1]], state.clusters[track_side[-2]], state.clusters[track_side[-3]])
                start_angle = self.angle_between_points(state.clusters[track_side[0]], state.clusters[track_side[1]], state.clusters[track_side[2]])

                max_turn = np.deg2rad(45)
                end_turn = np.pi - end_angle
                start_turn = np.pi - start_angle

                if end_turn > max_turn or start_turn > max_turn:
                    return False

        return True


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


    def grow_hypothesis(self, hypothesis: Hypothesis, state: FrameState) -> list[Hypothesis]:
        extensions = self.propose_extensions(hypothesis, state)

        children = []
        for e in extensions:
            children.append(self.apply_extension(hypothesis, e))
        return children

    def propose_extensions(self, hypothesis: Hypothesis, state: FrameState) -> list[Cone]:
        extensions: list[Cone] = []
        # the indicies of cones that are on the edge of the hypothesis; ones to extend from
        leading_cones = self.leading_idxs(hypothesis)

        for cone in leading_cones:
            candidate_idxs = self.nearby_candidates(state.clusters[cone.cluster_idx], hypothesis, state)
            for idx in candidate_idxs:
                # Feasibility checks should be done here in future
                extensions.append(Cone(0, idx, cone.end))
                extensions.append(Cone(1, idx, cone.end))
        return extensions

    def leading_idxs(self, hypothesis: Hypothesis) -> list[Cone]:
        leading_cones = []

        if hypothesis.side_0_size == 1:
            leading_cones.append(Cone(0, hypothesis.side_0_idxs[0], True))
        elif hypothesis.side_0_size >= 2:
            leading_cones.extend([
                Cone(0, hypothesis.side_0_idxs[0], False),
                Cone(0, hypothesis.side_0_idxs[-1], True),
            ])

        if hypothesis.side_1_size == 1:
            leading_cones.append(Cone(1, hypothesis.side_1_idxs[0], True))
        elif hypothesis.side_1_size >= 2:
            leading_cones.extend([
                Cone(1, hypothesis.side_1_idxs[0], False),
                Cone(1, hypothesis.side_1_idxs[-1], True),
            ])

        return leading_cones

    def nearby_candidates(self, centre_point, hypothesis: Hypothesis, state: FrameState) -> list[int]:
        """
        Given a centre point, find indices of all clusters nearby and that aren't already in the hypothesis
        Nearby definition: within the ring of where a cone could lie
        """
        nearby_idxs = state.tree.query_ball_point(centre_point, self.max_neighbour_radius, 2)
        valid_idxs: list[int] = []
        for idx in nearby_idxs:
            idx_already_in_hypothesis : bool = idx in hypothesis.side_0_idxs or idx in hypothesis.side_1_idxs
            if np.linalg.norm(state.clusters[idx] - centre_point) > self.min_neighbour_radius and not idx_already_in_hypothesis:
                valid_idxs.append(idx)
        return valid_idxs

    def apply_extension(self, hypothesis: Hypothesis, extension: Cone) -> Hypothesis:
        side_idxs = (
            hypothesis.side_0_idxs
            if extension.side == 0
            else hypothesis.side_1_idxs
        )
        extended_side = (
            [extension.cluster_idx, *side_idxs]
            if extension.end == False
            else [*side_idxs, extension.cluster_idx]
        )

        if extension.side == 0:
            return Hypothesis(extended_side, hypothesis.side_1_idxs)
        return Hypothesis(hypothesis.side_0_idxs, extended_side)

    def grow_hypothesis_old(self, hypothesis: Hypothesis, state: FrameState) -> list[Hypothesis]:
        def find_next_candidates(centre_point) -> list[int]:
            """
            Given a centre point, find indices of all clusters nearby and that aren't already in the hypothesis
            Nearby definition: within the ring of where a cone could lie
            """
            nearby_idxs = state.tree.query_ball_point(centre_point, self.max_neighbour_radius, 2)
            valid_idxs: list[int] = []
            for idx in nearby_idxs:
                idx_already_in_hypothesis : bool = idx in hypothesis.side_0_idxs or idx in hypothesis.side_1_idxs
                if np.linalg.norm(state.clusters[idx] - centre_point) > self.min_neighbour_radius and not idx_already_in_hypothesis:
                    valid_idxs.append(idx)
            return valid_idxs

        def orient_tangent_toward_other_side(
            tangent: np.ndarray, side: int, pos: int | None
        ) -> np.ndarray:
            current_side_idxs = (
                hypothesis.side_0_idxs
                if side == 0
                else hypothesis.side_1_idxs
            )
            other_side_idxs = (
                hypothesis.side_1_idxs
                if side == 0
                else hypothesis.side_0_idxs
            )
            if not current_side_idxs or not other_side_idxs:
                return tangent

            if pos == 0:
                current_idx = current_side_idxs[0]
            else:
                current_idx = current_side_idxs[-1]

            current_point = state.clusters[current_idx]
            other_points = state.clusters[other_side_idxs]
            nearest_other_point = other_points[
                np.argmin(np.linalg.norm(other_points - current_point, axis=1))
            ]
            toward_other_side = nearest_other_point - current_point

            if np.dot(tangent, toward_other_side) < 0:
                return -tangent
            return tangent

        def parallel_and_tangent_vector(
            side_i: int, pos: int | None = None
        ) -> tuple[np.ndarray, np.ndarray]:
            """Return outward along-track and correctly oriented cross-track vectors."""
            assert len(hypothesis.side_0_idxs) + len(hypothesis.side_1_idxs) >= 2
            assert side_i in [0, 1]
            assert pos in [0, 1, None]

            track_side_idxs = (
                hypothesis.side_0_idxs
                if side_i == 0
                else hypothesis.side_1_idxs
            )

            if pos is None:
                if len(hypothesis.side_0_idxs) == 1:
                    tangent = (
                        state.clusters[hypothesis.side_1_idxs[0]]
                        - state.clusters[hypothesis.side_0_idxs[0]]
                    )
                    tangent = tangent / np.linalg.norm(tangent)
                    parallel = self.unit_normal_vector(tangent)
                else:
                    parallel = (
                        state.clusters[hypothesis.side_0_idxs[1]]
                        - state.clusters[hypothesis.side_0_idxs[0]]
                    )
                    parallel = parallel / np.linalg.norm(parallel)
                    tangent = self.unit_normal_vector(parallel)
            else:
                direction_side_idxs = track_side_idxs
                if len(direction_side_idxs) < 2:
                    direction_side_idxs = (
                        hypothesis.side_1_idxs
                        if side_i == 0
                        else hypothesis.side_0_idxs
                    )

                if pos == 1:
                    parallel = (
                        state.clusters[direction_side_idxs[-1]]
                        - state.clusters[direction_side_idxs[-2]]
                    )
                else:
                    parallel = (
                        state.clusters[direction_side_idxs[0]]
                        - state.clusters[direction_side_idxs[1]]
                    )
                parallel = parallel / np.linalg.norm(parallel)
                tangent = self.unit_normal_vector(parallel)

            tangent = orient_tangent_toward_other_side(
                tangent, side_i, pos
            )
            return parallel, tangent

        def extend_hypothesis_from_base(side: int, idx: int, pos: int = 1) -> Hypothesis:
            assert side in [0, 1]
            assert pos in [0, 1]

            side_idxs = (
                hypothesis.side_0_idxs
                if side == 0
                else hypothesis.side_1_idxs
            )
            extended_side = (
                [idx, *side_idxs]
                if pos == 0
                else [*side_idxs, idx]
            )

            if side == 0:
                return Hypothesis(extended_side, hypothesis.side_1_idxs)
            return Hypothesis(hypothesis.side_0_idxs, extended_side)

        def switch_side(side: int):
            assert side in [0, 1]
            return (side + 1) % 2

        def check_eligibility_against_other_side(candidate_idxs: list[int], side_to_check: int) -> list[int]:
            assert side_to_check in [0, 1]
            filtered_idxs = []
            for idx in candidate_idxs:
                ## currently, no checks done - TODO
                filtered_idxs.append(idx)

            return filtered_idxs
                
        hypothesis_size = len(hypothesis.side_1_idxs) + len(hypothesis.side_0_idxs) 
        generated_hypotheses : list[Hypothesis] = []

        ######## CASE 1: SINGULAR POINT
        if hypothesis_size == 1:
            starting_cone = state.clusters[hypothesis.side_0_idxs[0]]
            for i in find_next_candidates(starting_cone):
                # Add two hypothesis - one that both are cones on side 1, and one that they're opposing cones
                generated_hypotheses.append(Hypothesis([*hypothesis.side_0_idxs, i], []))
                generated_hypotheses.append(Hypothesis(hypothesis.side_0_idxs, [i]))


        ######## CASE 2: PAIR OF POINTS
        elif hypothesis_size == 2:
            # for cone in each of the sides
            for side_i, track_side in enumerate([hypothesis.side_0_idxs, hypothesis.side_1_idxs]):
                parallel, tangent = parallel_and_tangent_vector(side_i)
                for cone_idx in track_side: 
                    cone_pos = state.clusters[cone_idx]
                    candidate_idxs = find_next_candidates(cone_pos)

                    # for each valid candidate, create a new hypothesis 
                    for ci in candidate_idxs:
                        if self.beam_search(cone_pos, parallel, state.clusters[ci]):
                            # Add cone to current track side, make new hypothesis
                            new_hypothesis = extend_hypothesis_from_base(side_i, ci)
                            generated_hypotheses.append(new_hypothesis)

                        elif self.beam_search(cone_pos, tangent, state.clusters[ci]):
                            # add cone to opposing track side, make new hypothesis
                            new_hypothesis = extend_hypothesis_from_base(switch_side(side_i), ci)
                            generated_hypotheses.append(new_hypothesis)
        ######## CASE 3: 3 OR MORE POINTS        
        else:
            # form every valid quad, ensuring there is at least one point on both sides
            # for each of the 4 propagation cones

            # find valid next cones
            # check eligibility against other side
            # if eligible, then add
            for side_i, track_side in enumerate([hypothesis.side_0_idxs, hypothesis.side_1_idxs]):
                if not track_side:
                    continue
                for pos_i, cone_idx in enumerate([track_side[0], track_side[-1]]): 
                    parallel, tangent = parallel_and_tangent_vector(side_i, pos_i)
                    cone_pos = state.clusters[cone_idx]
                    candidate_idxs = find_next_candidates(cone_pos)
                    for ci in candidate_idxs:
                        if self.beam_search(cone_pos, parallel, state.clusters[ci]):
                            # Add cone to current track side, make new hypothesis
                            new_hypothesis = extend_hypothesis_from_base(side_i, ci, pos_i)
                            generated_hypotheses.append(new_hypothesis)

                        elif self.beam_search(cone_pos, tangent, state.clusters[ci]):
                            # add cone to opposing track side, make new hypothesis
                            new_hypothesis = extend_hypothesis_from_base(switch_side(side_i), ci, pos_i)
                            generated_hypotheses.append(new_hypothesis)

        return generated_hypotheses
    

    def remove_duplicates(self, hypotheses: list[Hypothesis]) -> list[Hypothesis]:
        seen: set[tuple[tuple[int, ...], tuple[int, ...]]] = set()
        output: list[Hypothesis] = []
        for h in hypotheses:
            sides = (tuple(h.side_0_idxs), tuple(h.side_1_idxs))
            flipped_sides = (sides[1], sides[0])
            key = min(sides, flipped_sides)
            if key not in seen:
                seen.add(key)
                output.append(h)

        return output


    def beam_search(
        self, cone: np.ndarray, direction: np.ndarray, test_point: np.ndarray
    ) -> bool:
        delta = test_point - cone
        delta_norm = np.linalg.norm(delta)
        direction_norm = np.linalg.norm(direction)
        if np.isclose(delta_norm, 0.0) or np.isclose(direction_norm, 0.0):
            return False

        cosine = np.dot(delta, direction) / (delta_norm * direction_norm)
        angle = np.arccos(np.clip(cosine, -1.0, 1.0))
        return bool(angle < self.beam_search_angle_rad)

    def unit_normal_vector(self, vector: np.ndarray):
        vector = vector.reshape(-1)
        if vector.shape != (2,):
            raise ValueError(f"Expected a 2D vector, got shape {vector.shape}")
        return np.array([-vector[1], vector[0]]) / np.linalg.norm(vector)

    def plot_hypotheses(
        self,
        state: FrameState,
        hypotheses: list[Hypothesis],
        starting_idx: int
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

        for hypothesis_idx, hypothesis in enumerate(hypotheses):
            for side_idxs, color, label in (
                (
                    hypothesis.side_0_idxs,
                    "tab:blue",
                    "Boundary candidate A (hypothesis paths)",
                ),
                (
                    hypothesis.side_1_idxs,
                    "tab:orange",
                    "Boundary candidate B (hypothesis paths)",
                ),
            ):
                if not side_idxs:
                    continue

                points = state.clusters[side_idxs]
                axes.plot(
                    points[:, 1],
                    points[:, 0],
                    marker="o",
                    color=color,
                    alpha=0.35,
                    label=label if hypothesis_idx == 0 else None,
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
            f"{len(hypotheses)} hypotheses"
        )
        # Use a vehicle-centric view: forward x points up, and lateral y is
        # shown on the horizontal axis.
        axes.set_xlabel("y - lateral / right (m)")
        axes.set_ylabel("x - forward (m)")
        axes.set_aspect("equal", adjustable="datalim")
        axes.grid(True, alpha=0.25)
        axes.legend(title="Plot key", loc="best")
        figure.tight_layout()

        plt.show(block=True)
        plt.close(figure)



    # def pair_direction_classification(self, left_idx: int, right_idx: int, tree: cKDTree, cluster_centres: np.ndarray):
    #     """
    #     Inputs:
    #     index of left and right starting cones
    #     - Left and right arbritrary - makes intuition nicer though
    #     """
    #     # Hypothesis 1: track segment
    #     opposite_hypothesis = Hypothesis([left_idx], [right_idx])
    #     # hpyothesis 2: track edge
    #     edge_hypothesis = Hypothesis([left_idx], [])
    #     # if both hypothesis found to be false, we'll discard

    #     self.grow_hypothesis(opposite_hypothesis, tree, cluster_centres)
    #     self.grow_hypothesis(edge_hypothesis, tree, cluster_centres)

    #     # delta points from left cone to right cone
    #     delta = cluster_centres[right_idx] - cluster_centres[left_idx]
    #     unit_delta = delta / np.linalg.norm(delta)
    #     unit_normal = self.unit_normal_vector(unit_delta)
    #     for c in candidate_idxs:
    #         if self.beam_search(cluster_centres[left_idx], unit_normal, cluster_centres[c]):
    #             opposite_hypothesis.side_1_idxs.append(c)
    #         elif self.beam_search(cluster_centres[right_idx], unit_normal, cluster_centres[c]):
    #             opposite_hypothesis.side_2_idxs.append(c)
    #         elif self.beam_search(cluster_centres[right_idx], unit_delta, cluster_centres[c]) or self.beam_search(cluster_centres[left_idx], unit_delta, cluster_centres[c]):
    #             edge_hypothesis.side_1_idxs.append(c)