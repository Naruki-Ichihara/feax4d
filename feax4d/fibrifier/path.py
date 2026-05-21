"""Path class for representing and managing SVG path segments."""

from typing import List, Tuple, Optional, Set
from dataclasses import dataclass, field
import numpy as np


@dataclass
class Path:
    """
    Represents a single path segment with connectivity and geometric information.

    Attributes:
        path_id: Unique identifier for this path
        nodes: List of (x, y) coordinate tuples representing the path
        path_type: Type of path - either 'stripe' or 'contour' (density outline)
        is_closed: Whether the path forms a closed loop
        start_point: Starting coordinate (x, y)
        end_point: Ending coordinate (x, y)
        length: Total length of the path in physical units
        connected_paths: Set of path IDs that this path connects to
    """

    path_id: int
    nodes: List[Tuple[float, float]]
    path_type: str  # 'stripe' or 'contour'
    is_closed: bool = False
    start_point: Optional[Tuple[float, float]] = None
    end_point: Optional[Tuple[float, float]] = None
    length: float = 0.0
    connected_paths: Set[int] = field(default_factory=set)

    def __post_init__(self):
        """Initialize computed properties after dataclass initialization."""
        if len(self.nodes) < 2:
            raise ValueError(f"Path must have at least 2 nodes, got {len(self.nodes)}")

        # Validate path_type
        if self.path_type not in ['stripe', 'contour']:
            raise ValueError(f"path_type must be 'stripe' or 'contour', got '{self.path_type}'")

        # Set start and end points
        self.start_point = self.nodes[0]
        self.end_point = self.nodes[-1]

        # Check if path is closed
        if self.start_point == self.end_point:
            self.is_closed = True

        # Calculate total length
        self.length = self._calculate_length()

    def _calculate_length(self) -> float:
        """Calculate the total length of the path."""
        total_length = 0.0
        for i in range(len(self.nodes) - 1):
            x1, y1 = self.nodes[i]
            x2, y2 = self.nodes[i + 1]
            segment_length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            total_length += segment_length
        return total_length

    def add_connection(self, other_path_id: int):
        """
        Add a connection to another path.

        Args:
            other_path_id: ID of the path to connect to
        """
        self.connected_paths.add(other_path_id)

    def remove_connection(self, other_path_id: int):
        """
        Remove a connection to another path.

        Args:
            other_path_id: ID of the path to disconnect from
        """
        self.connected_paths.discard(other_path_id)

    def is_connected_to(self, other_path_id: int) -> bool:
        """
        Check if this path is connected to another path.

        Args:
            other_path_id: ID of the path to check

        Returns:
            True if connected, False otherwise
        """
        return other_path_id in self.connected_paths

    def reverse(self):
        """Reverse the direction of the path."""
        self.nodes = list(reversed(self.nodes))
        self.start_point, self.end_point = self.end_point, self.start_point

    def distance_to_point(self, point: Tuple[float, float]) -> float:
        """
        Calculate the minimum distance from a point to this path.

        Args:
            point: (x, y) coordinate

        Returns:
            Minimum distance to the path
        """
        px, py = point
        min_dist = float('inf')

        for i in range(len(self.nodes) - 1):
            x1, y1 = self.nodes[i]
            x2, y2 = self.nodes[i + 1]

            # Calculate distance from point to line segment
            dx = x2 - x1
            dy = y2 - y1

            if dx == 0 and dy == 0:
                # Degenerate segment
                dist = np.sqrt((px - x1)**2 + (py - y1)**2)
            else:
                # Parameter t for projection onto line
                t = max(0, min(1, ((px - x1) * dx + (py - y1) * dy) / (dx**2 + dy**2)))

                # Closest point on segment
                closest_x = x1 + t * dx
                closest_y = y1 + t * dy

                # Distance to closest point
                dist = np.sqrt((px - closest_x)**2 + (py - closest_y)**2)

            min_dist = min(min_dist, dist)

        return min_dist

    def to_svg_polyline(self, stroke_color: str = "#000000",
                       stroke_width: float = 0.01) -> str:
        """
        Convert path to SVG polyline string.

        Args:
            stroke_color: SVG stroke color
            stroke_width: SVG stroke width

        Returns:
            SVG polyline element as string
        """
        points_str = " ".join([f"{x:.6f},{y:.6f}" for x, y in self.nodes])
        return (f'<polyline points="{points_str}" '
                f'stroke="{stroke_color}" stroke-width="{stroke_width}" '
                f'fill="none" id="path_{self.path_id}"/>')

    def __repr__(self) -> str:
        """String representation of the path."""
        closed_str = "closed" if self.is_closed else "open"
        return (f"Path(id={self.path_id}, type={self.path_type}, "
                f"{closed_str}, nodes={len(self.nodes)}, "
                f"length={self.length:.3f}, connections={len(self.connected_paths)})")

    def __len__(self) -> int:
        """Return the number of nodes in the path."""
        return len(self.nodes)
