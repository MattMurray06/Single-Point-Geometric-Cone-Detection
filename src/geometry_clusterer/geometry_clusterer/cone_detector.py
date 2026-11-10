from abc import abstractmethod, ABC
import math
import numpy as np
from dataclasses import dataclass

@dataclass
class Cluster:
	idx: int  # position in this frame's input; stable ID that survives copies
	x: float
	y: float
	confident: bool

	def __iter__(self):
		return iter((self.x, self.y, self.confident))

	def distance_to(self, other: "Cluster") -> float:
		"""Euclidean distance between this cluster's centre and another's, in metres."""
		return math.hypot(other.x - self.x, other.y - self.y)

	def vector_to(self, other: "Cluster") -> tuple[float, float]:
		"""Displacement (dx, dy) from this cluster's centre to another's, in metres."""
		return (other.x - self.x, other.y - self.y)

@dataclass
class Edge:
	clusters: list[Cluster]

@dataclass
class Segment:
	edge_a: Edge
	edge_b: Edge

class ConeDetector(ABC):
	@abstractmethod
	def detect_cones(self, cluster_centres: list[Cluster]) -> object:
		raise NotImplementedError