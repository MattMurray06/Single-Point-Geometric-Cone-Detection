from .cone_detector import Cluster, Edge, Segment, ConeDetector
import math
from itertools import combinations
import numpy as np
import logging 

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

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
		min_track_width: float = 2.0,
		min_segment_clusters: int = 6,
	):
		self.max_neighbour_radius = max_neighbour_radius
		self.min_neighbour_radius = min_neighbour_radius
		self.max_angle_change = max_angle_change
		self.wall_density_threshold = wall_density_threshold
		self.max_consecutive_angle_change = max_consecutive_angle_change
		self.plot_growth = plot_growth
		self.max_track_width = max_track_width
		self.min_track_width = min_track_width
		self.min_segment_clusters = min_segment_clusters

	def detect_cones(self, clusters: list[Cluster]) -> list[Segment]:
		"""Main entry point. Find the cones in a frame by fitting track geometry to cluster centres.

		Clusters may be cones or noise. Cones are the clusters that lie on a
		plausible track: two roughly parallel edges a track-width apart.

		Steps:
			1. Discover edges: ordered chains of clusters forming smooth lines.
			   Each cluster belongs to at most one edge.
			2. Pair edges into segments. A pair is a segment when the edges are:
			   - close (within ``max_track_width``)
			   - a valid width apart (``min_track_width`` to ``max_track_width``)
			   - roughly parallel at both ends

		Args:
			cluster_centres: Cluster centres for one frame, in the LiDAR frame.

		Returns:
			Track segments found in the frame, each a pair of edges. Empty if
			there are no clusters or no valid pairs.
		"""
		if not clusters:
			return []

		best_edges: list[Edge] = self.find_edges(clusters)

		# Determine pairs of edges that form track segments
		track_segments: list[Segment] = self.compute_segments(best_edges)

		# This will be a subset of best_edges - only the ones that form part of the track
		# track_edges = list({
		# 	edge.family_id: edge
		# 	for track_segment in track_segments
		# 	for edge in track_segment
		# }.values())

		# cones = np.array([state.clusters[idx] for edge in track_edges for idx in edge.idxs])

		# if self.plot_growth:
		# 	self.plot_edges(state, track_edges, window="track")
		# 	self.plot_edges(state, edge_candidates, window="candidates")
			
		# return cones

	def find_edges(self, clusters: list[Cluster]) -> list[Edge]:
		"""
		Returns a list of plausible track edges
		"""
		remaining: list[Cluster] = list(clusters)  # unclaimed clusters; copy so the caller's list is untouched
		edges: list[Edge] = []

		while remaining:
			seed = remaining.pop()
			best = self.find_best_edge(remaining, seed)
			if best is None:
				continue
			edges.append(best)
			claimed = {c.idx for c in best.clusters}
			remaining = [c for c in remaining if c.idx not in claimed]   # remove the edge's points

		return edges

	@staticmethod
	def get_turn_angle(vector_a: tuple[float, float], vector_b: tuple[float, float]) -> float:
		"""Signed turn from direction vector_a to direction vector_b, in radians.

		Range is (-pi, pi]. 0 means straight on, positive means a left
		(counter-clockwise) turn and negative a right turn. Returns 0 if
		either vector has zero length.
		"""
		cross = vector_a[0] * vector_b[1] - vector_a[1] * vector_b[0]
		dot = vector_a[0] * vector_b[0] + vector_a[1] * vector_b[1]
		return math.atan2(cross, dot)

	def find_best_edge(self, clusters: list[Cluster], cluster: Cluster) -> Edge | None:
		#find all candidate edges for a single cluster centre, and return only the best, removing it from the clusters array
		
		edges: list[Edge] = []

		def get_next_candidates(edge_clusters: list[Cluster], at_end: bool) -> list[Cluster]:
			"""Clusters that could extend the edge at one end.

			Checks, as the edge gets long enough to support them:
			  1+ points: within the neighbour ring, not already in the edge
			  2+ points: turn from the edge's direction <= max_angle_change (+10)
			  3+ points: change from the previous turn <= max_consecutive_angle_change
			"""
			# Orient so the growing end is always last: [..., before, previous, frontier]
			ordered = edge_clusters if at_end else edge_clusters[::-1]
			frontier = ordered[-1]
			previous = ordered[-2] if len(ordered) >= 2 else None
			before = ordered[-3] if len(ordered) >= 3 else None

			follow_vector = previous.vector_to(frontier) if previous is not None else None
			previous_turn = (
				self.get_turn_angle(before.vector_to(previous), follow_vector)
				if before is not None and previous is not None and follow_vector is not None
				else None
			)

			in_edge = {c.idx for c in edge_clusters}
			candidates = []
			for clust in clusters:
				if clust.idx in in_edge:
					continue
				dist = frontier.distance_to(clust)
				if not (self.min_neighbour_radius < dist < self.max_neighbour_radius):
					continue
				if follow_vector is not None:
					turn = self.get_turn_angle(follow_vector, frontier.vector_to(clust))
					# Turn check: limits how sharp a single step can be
					if abs(turn) > math.radians(self.max_angle_change + 10):
						continue
					# Consecutive bend check: limits how fast the bend can change (no zig-zags)
					if previous_turn is not None and abs(turn - previous_turn) > math.radians(self.max_consecutive_angle_change):
						continue
				candidates.append(clust)
			return candidates

		# Step 1: grow an edge once in every possible direction
		for candidate in get_next_candidates([cluster], at_end=True):
			edges.append(Edge([cluster, candidate]))

		# Step 2: grow edges as far as possible
		final_edges = []
		while edges:
			edge = edges.pop() #Remove from array
			start_candidates = get_next_candidates(edge.clusters, at_end=False)
			end_candidates = get_next_candidates(edge.clusters, at_end=True)
			# If stopped growing, append to final edges
			if len(start_candidates) == 0 and len(end_candidates) == 0:
				final_edges.append(edge)
				continue
			# Else grow it
			for candidate in start_candidates:
				edges.append(Edge([candidate, *edge.clusters]))
			for candidate in end_candidates:
				edges.append(Edge([*edge.clusters, candidate]))

		def score_edge(edge: Edge) -> float:
			"""Score how likely an edge is a real track boundary. Higher is better.

			NOT USED YET: best edge is still chosen by length (matches Matt's
			behaviour). Swap this in later as a separate, measurable change.

			score = n
			      - W_SPACING * n * (std_gap / mean_gap)   # uneven spacing
			      - W_SMOOTH  * mean|turn change|          # wobble / zig-zag (radians)
			      + W_CONF    * n_confident                # trusted clusters
			Walls (mean gap < wall_density_threshold) score -inf.
			"""
			# TODO: move weights to params.yaml once in use, then tune on the dataset
			W_SPACING = 0.5
			W_SMOOTH = 0.5
			W_CONF = 0.5

			points = edge.clusters
			n = len(points)
			n_confident = sum(1 for p in points if p.confident)
			if n < 2:
				return float(n) + W_CONF * n_confident

			# Spacing consistency: coefficient of variation of the gaps (0 = perfectly even)
			gaps = [a.distance_to(b) for a, b in zip(points, points[1:])]
			mean_gap = sum(gaps) / len(gaps)
			if mean_gap < self.wall_density_threshold:
				return float("-inf")  # densely packed: likely a wall, not cones
			std_gap = math.sqrt(sum((g - mean_gap) ** 2 for g in gaps) / len(gaps))
			spacing_penalty = std_gap / mean_gap

			# Smoothness: mean absolute change between consecutive turns (needs 4+ points)
			vectors = [a.vector_to(b) for a, b in zip(points, points[1:])]
			turns = [self.get_turn_angle(u, v) for u, v in zip(vectors, vectors[1:])]
			turn_changes = [abs(t2 - t1) for t1, t2 in zip(turns, turns[1:])]
			smooth_penalty = sum(turn_changes) / len(turn_changes) if turn_changes else 0.0

			return (
				n
				- W_SPACING * n * spacing_penalty
				- W_SMOOTH * smooth_penalty
				+ W_CONF * n_confident
			)
		
		# Step 3: Choose a best edge, remove all included centres from clustres
		# FOR NOW WE CHOOSE LONGEST
		if not final_edges:
			return None
		return max(final_edges, key=lambda e: len(e.clusters))
		
	def compute_segments(self, edges: list[Edge]) -> list[Segment]:
		track_segments: list[Segment] = []

		# Each unordered pair once, never an edge with itself
		for edge_a, edge_b in combinations(edges, 2):
			# Step 1: enough evidence. Two short edges can line up by chance (walls, clutter)
			if len(edge_a.clusters) + len(edge_b.clusters) < self.min_segment_clusters:
				continue
			# Step 2: proximity check by bounding box (cheap rough filter)
			if not self.bounding_boxes_close(edge_a, edge_b, self.max_track_width):
				continue


		return track_segments

	@staticmethod
	def bounding_boxes_close(edge_a: Edge, edge_b: Edge, max_gap: float) -> bool:
		"""Cheap rough filter: are two edges' bounding boxes within max_gap of each other?

		Each box spans the edge's two endpoints. Passes when the gap between
		the boxes is at most max_gap on both x and y (touching counts). Loose
		by design; the precise width check happens later in pairing.
		"""
		a_ends = (edge_a.clusters[0], edge_a.clusters[-1])
		b_ends = (edge_b.clusters[0], edge_b.clusters[-1])
		a_min_x, a_max_x = min(c.x for c in a_ends), max(c.x for c in a_ends)
		a_min_y, a_max_y = min(c.y for c in a_ends), max(c.y for c in a_ends)
		b_min_x, b_max_x = min(c.x for c in b_ends), max(c.x for c in b_ends)
		b_min_y, b_max_y = min(c.y for c in b_ends), max(c.y for c in b_ends)
		return (
			a_max_x + max_gap >= b_min_x and b_max_x + max_gap >= a_min_x
			and a_max_y + max_gap >= b_min_y and b_max_y + max_gap >= a_min_y
		)

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
	
	def plot_edges(
		self,
		state: FrameState,
		edges: list[Edge],
		window: str = "edges",
	) -> None:
		"""
		Live top-down plot of edges. One persistent window per `window` name,
		redrawn each frame without blocking the node.
		"""
		import matplotlib.pyplot as plt

		# Create each named window once, then reuse it every frame
		if not hasattr(self, "_plot_windows"):
			plt.ion()
			self._plot_windows = {}
		if window not in self._plot_windows or not plt.fignum_exists(self._plot_windows[window][0].number):
			new_figure, new_axes = plt.subplots()
			new_figure.canvas.manager.set_window_title(window)
			self._plot_windows[window] = (new_figure, new_axes)
		figure, axes = self._plot_windows[window]
		axes.clear()

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
			f"{window}: {len(edges)} edges"
		)
		axes.set_xlabel("y - lateral / right (m)")
		axes.set_ylabel("x - forward (m)")
		axes.set_ylim(-50, 50)  # lateral
		axes.set_xlim(-50, 50)  # forward
		axes.set_aspect("equal", adjustable="box")
		axes.grid(True, alpha=0.25)
		axes.legend(
			title="Plot key",
			loc="upper left",
			bbox_to_anchor=(1.02, 1),
			fontsize=8,
		)
		figure.tight_layout()

		# Non-blocking redraw: refresh the window, then return to the node
		figure.canvas.draw_idle()
		plt.pause(0.001)
