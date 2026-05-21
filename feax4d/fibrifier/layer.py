"""Layer class for managing collections of Path objects."""

from typing import List, Optional, Tuple, Iterator
import numpy as np
from feax4d.fibrifier.path import Path


class Layer:
    """
    Represents a layer containing multiple Path objects.

    A Layer is a collection of paths that can be managed together,
    typically representing a single printing layer or a group of related paths.

    Attributes:
        layer_id: Unique identifier for this layer
        paths: List of Path objects in this layer
        name: Optional name for the layer
        z_height: Optional z-height for 3D printing (mm)
    """

    def __init__(
        self,
        layer_id: int,
        paths: Optional[List[Path]] = None,
        name: Optional[str] = None,
        z_height: Optional[float] = None
    ):
        """
        Initialize a Layer.

        Args:
            layer_id: Unique identifier for this layer
            paths: List of Path objects (default: empty list)
            name: Optional name for the layer
            z_height: Optional z-height for 3D printing in mm
        """
        self.layer_id = layer_id
        self.paths = paths if paths is not None else []
        self.name = name if name is not None else f"Layer_{layer_id}"
        self.z_height = z_height

    def add_path(self, path: Path):
        """
        Add a path to this layer.

        Args:
            path: Path object to add
        """
        self.paths.append(path)

    def remove_path(self, path_id: int) -> bool:
        """
        Remove a path by its ID.

        Args:
            path_id: ID of the path to remove

        Returns:
            True if path was found and removed, False otherwise
        """
        for i, path in enumerate(self.paths):
            if path.path_id == path_id:
                self.paths.pop(i)
                return True
        return False

    def get_path(self, path_id: int) -> Optional[Path]:
        """
        Get a path by its ID.

        Args:
            path_id: ID of the path to retrieve

        Returns:
            Path object if found, None otherwise
        """
        for path in self.paths:
            if path.path_id == path_id:
                return path
        return None

    def get_paths_by_type(self, path_type: str) -> List[Path]:
        """
        Get all paths of a specific type.

        Args:
            path_type: Type of paths to retrieve ('stripe' or 'contour')

        Returns:
            List of Path objects matching the type
        """
        return [p for p in self.paths if p.path_type == path_type]

    def filter_by_length(self, min_length: float, max_length: Optional[float] = None) -> List[Path]:
        """
        Filter paths by length.

        Args:
            min_length: Minimum path length in mm
            max_length: Maximum path length in mm (optional)

        Returns:
            List of Path objects within the length range
        """
        if max_length is None:
            return [p for p in self.paths if p.length >= min_length]
        else:
            return [p for p in self.paths if min_length <= p.length <= max_length]

    def get_closed_paths(self) -> List[Path]:
        """
        Get all closed paths in this layer.

        Returns:
            List of closed Path objects
        """
        return [p for p in self.paths if p.is_closed]

    def get_open_paths(self) -> List[Path]:
        """
        Get all open paths in this layer.

        Returns:
            List of open Path objects
        """
        return [p for p in self.paths if not p.is_closed]

    def get_bounds(self) -> Tuple[float, float, float, float]:
        """
        Calculate the bounding box of all paths in this layer.

        Returns:
            Tuple of (x_min, y_min, x_max, y_max)
        """
        if not self.paths:
            return (0.0, 0.0, 0.0, 0.0)

        x_coords = []
        y_coords = []
        for path in self.paths:
            for x, y in path.nodes:
                x_coords.append(x)
                y_coords.append(y)

        return (min(x_coords), min(y_coords), max(x_coords), max(y_coords))

    def total_length(self) -> float:
        """
        Calculate total length of all paths in this layer.

        Returns:
            Total length in mm
        """
        return sum(p.length for p in self.paths)

    def total_nodes(self) -> int:
        """
        Count total number of nodes in all paths.

        Returns:
            Total number of nodes
        """
        return sum(len(p) for p in self.paths)

    def sort_paths(self, key='length', reverse=False):
        """
        Sort paths in this layer.

        Args:
            key: Sort key - 'length', 'nodes', or 'id' (default: 'length')
            reverse: If True, sort in descending order (default: False)
        """
        if key == 'length':
            self.paths.sort(key=lambda p: p.length, reverse=reverse)
        elif key == 'nodes':
            self.paths.sort(key=lambda p: len(p), reverse=reverse)
        elif key == 'id':
            self.paths.sort(key=lambda p: p.path_id, reverse=reverse)
        else:
            raise ValueError(f"Invalid sort key: {key}. Use 'length', 'nodes', or 'id'.")

    def find_nearby_paths(self, path: Path, distance_threshold: float) -> List[Path]:
        """
        Find paths whose endpoints are within distance_threshold of the given path's endpoints.

        Args:
            path: Reference path
            distance_threshold: Maximum distance in mm

        Returns:
            List of nearby Path objects
        """
        nearby = []
        for other_path in self.paths:
            if other_path.path_id == path.path_id:
                continue

            # Check distance between endpoints
            # path.end <-> other_path.start
            dx = path.end_point[0] - other_path.start_point[0]
            dy = path.end_point[1] - other_path.start_point[1]
            dist1 = (dx**2 + dy**2)**0.5

            # path.start <-> other_path.end
            dx = path.start_point[0] - other_path.end_point[0]
            dy = path.start_point[1] - other_path.end_point[1]
            dist2 = (dx**2 + dy**2)**0.5

            # path.end <-> other_path.end
            dx = path.end_point[0] - other_path.end_point[0]
            dy = path.end_point[1] - other_path.end_point[1]
            dist3 = (dx**2 + dy**2)**0.5

            # path.start <-> other_path.start
            dx = path.start_point[0] - other_path.start_point[0]
            dy = path.start_point[1] - other_path.start_point[1]
            dist4 = (dx**2 + dy**2)**0.5

            min_dist = min(dist1, dist2, dist3, dist4)
            if min_dist < distance_threshold:
                nearby.append(other_path)

        return nearby

    def calculate_travel_distance(self, closed_only: bool = False, open_only: bool = False) -> float:
        """
        Calculate the total travel distance between consecutive paths in the layer.

        Travel distance is the distance from the end point of one path to the start point
        of the next path (not including the path lengths themselves).

        Args:
            closed_only: If True, only calculate for closed paths
            open_only: If True, only calculate for open paths

        Returns:
            Total travel distance in mm
        """
        if closed_only and open_only:
            raise ValueError("Cannot specify both closed_only=True and open_only=True")

        # Select paths based on filter
        if closed_only:
            paths = self.get_closed_paths()
        elif open_only:
            paths = self.get_open_paths()
        else:
            paths = self.paths

        if len(paths) < 2:
            return 0.0

        total_travel = 0.0
        for i in range(len(paths) - 1):
            x1, y1 = paths[i].end_point
            x2, y2 = paths[i + 1].start_point
            travel_dist = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            total_travel += travel_dist

        return total_travel

    def optimize_open_path_order(self, start_point: Optional[Tuple[float, float]] = None) -> float:
        """
        Optimize the order and direction of open paths to minimize travel distance using TSP solver.

        This method creates a distance matrix including both start and end points of each path,
        allowing the TSP solver to determine both the optimal order and direction (forward/reverse)
        for each path.

        Closed paths are left in their original order and placed before open paths.

        Args:
            start_point: Optional starting point (x, y). If None, starts from the first path's optimal end.

        Returns:
            Total travel distance for the optimized order (mm)

        Example:
            >>> layer.optimize_open_path_order(start_point=(0, 0))
            >>> # Open paths are now ordered and oriented for minimum travel distance
        """
        # Separate closed and open paths
        closed_paths = self.get_closed_paths()
        open_paths = self.get_open_paths()

        if len(open_paths) == 0:
            print("No open paths to optimize")
            return 0.0

        if len(open_paths) == 1:
            # Only one open path, no optimization needed
            self.paths = closed_paths + open_paths
            return 0.0

        try:
            from tsp_solver.greedy import solve_tsp
        except ImportError:
            print("Warning: tsp-solver2 not available. Install with: pip install tsp-solver2")
            print("Skipping path optimization.")
            return 0.0

        # Build distance matrix including both endpoints of each path
        # For n paths, we have 2n nodes (start and end of each path)
        n_paths = len(open_paths)
        n_nodes = 2 * n_paths
        distance_matrix = np.zeros((n_nodes, n_nodes))

        # Create list of all endpoints
        # Even indices (0, 2, 4, ...) = start points
        # Odd indices (1, 3, 5, ...) = end points
        endpoints = []
        for path in open_paths:
            endpoints.append(path.start_point)  # Start
            endpoints.append(path.end_point)    # End

        # Fill distance matrix
        for i in range(n_nodes):
            for j in range(n_nodes):
                if i == j:
                    distance_matrix[i, j] = 0.0
                else:
                    # Check if i and j are endpoints of the same path
                    path_i = i // 2
                    path_j = j // 2

                    if path_i == path_j:
                        # Same path: connection distance is 0 (already connected)
                        distance_matrix[i, j] = 0.0
                    else:
                        # Different paths: calculate actual distance
                        x1, y1 = endpoints[i]
                        x2, y2 = endpoints[j]
                        distance_matrix[i, j] = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

        # Add virtual start node if start_point is provided
        if start_point is not None:
            # Calculate distances from start_point to all endpoints
            start_distances = np.zeros(n_nodes)
            for i in range(n_nodes):
                x, y = endpoints[i]
                start_distances[i] = np.sqrt((x - start_point[0])**2 + (y - start_point[1])**2)

            # Create augmented distance matrix with virtual start node
            aug_matrix = np.zeros((n_nodes + 1, n_nodes + 1))
            aug_matrix[0, 1:] = start_distances
            aug_matrix[1:, 0] = start_distances
            aug_matrix[1:, 1:] = distance_matrix

            # Solve TSP on augmented matrix
            node_sequence = solve_tsp(aug_matrix)

            # Remove the virtual start node (index 0) and adjust indices
            node_sequence = [idx - 1 for idx in node_sequence if idx > 0]
        else:
            # Solve TSP without specific start point
            node_sequence = solve_tsp(distance_matrix)

        # Convert node sequence to path sequence with orientation
        optimized_paths = []
        used_paths = set()

        for node_idx in node_sequence:
            path_idx = node_idx // 2

            if path_idx in used_paths:
                continue  # Already processed this path

            used_paths.add(path_idx)
            path = open_paths[path_idx]

            # Determine if we need to reverse the path
            # If we're entering through the end point (odd index), reverse the path
            is_end_point = (node_idx % 2 == 1)

            if is_end_point:
                # Create a copy and reverse it
                path_copy = Path(
                    path_id=path.path_id,
                    nodes=list(reversed(path.nodes)),
                    path_type=path.path_type
                )
                optimized_paths.append(path_copy)
            else:
                # Use path as-is
                optimized_paths.append(path)

        # Calculate total travel distance
        total_travel = 0.0

        if start_point is not None and len(optimized_paths) > 0:
            # Distance from start to first path
            x, y = optimized_paths[0].start_point
            total_travel += np.sqrt((x - start_point[0])**2 + (y - start_point[1])**2)

        # Distance between consecutive paths
        for i in range(len(optimized_paths) - 1):
            x1, y1 = optimized_paths[i].end_point
            x2, y2 = optimized_paths[i + 1].start_point
            total_travel += np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

        # Update paths: closed paths first (original order), then open paths (optimized)
        self.paths = closed_paths + optimized_paths

        return total_travel

    def optimize_closed_path_order(self, start_point: Optional[Tuple[float, float]] = None) -> float:
        """
        Optimize the order of closed paths (contours) to minimize travel distance using TSP solver.

        Open paths are left in their original order and placed after closed paths.

        Args:
            start_point: Optional starting point (x, y). If None, uses the first closed path's start point.

        Returns:
            Total travel distance for the optimized order (mm)

        Example:
            >>> layer.optimize_closed_path_order(start_point=(0, 0))
            >>> # Closed paths are now ordered for minimum travel distance
        """
        # Separate closed and open paths
        closed_paths = self.get_closed_paths()
        open_paths = self.get_open_paths()

        if len(closed_paths) == 0:
            print("No closed paths to optimize")
            return 0.0

        if len(closed_paths) == 1:
            # Only one closed path, no optimization needed
            self.paths = closed_paths + open_paths
            return 0.0

        try:
            from tsp_solver.greedy import solve_tsp
        except ImportError:
            print("Warning: tsp-solver2 not available. Install with: pip install tsp-solver2")
            print("Skipping path optimization.")
            return 0.0

        # Build distance matrix between closed paths (using start points)
        n = len(closed_paths)
        distance_matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                if i == j:
                    distance_matrix[i, j] = 0.0
                else:
                    # Distance from path i's start to path j's start
                    x1, y1 = closed_paths[i].start_point
                    x2, y2 = closed_paths[j].start_point
                    distance_matrix[i, j] = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

        # If start_point is provided, find the closest closed path to start from
        if start_point is not None:
            # Add virtual start node
            start_distances = np.zeros(n)
            for i in range(n):
                x, y = closed_paths[i].start_point
                start_distances[i] = np.sqrt((x - start_point[0])**2 + (y - start_point[1])**2)

            # Find closest path to start
            start_idx = int(np.argmin(start_distances))

            # Create augmented distance matrix with virtual start node
            aug_matrix = np.zeros((n + 1, n + 1))
            aug_matrix[0, 1:] = start_distances
            aug_matrix[1:, 0] = start_distances
            aug_matrix[1:, 1:] = distance_matrix

            # Solve TSP on augmented matrix
            path_indices = solve_tsp(aug_matrix)

            # Remove the virtual start node (index 0) and adjust indices
            path_indices = [idx - 1 for idx in path_indices if idx > 0]
        else:
            # Solve TSP without specific start point
            path_indices = solve_tsp(distance_matrix)

        # Reorder closed paths according to TSP solution
        optimized_closed = [closed_paths[i] for i in path_indices]

        # Calculate total travel distance
        total_travel = 0.0
        if start_point is not None:
            # Distance from start to first path
            x, y = optimized_closed[0].start_point
            total_travel += np.sqrt((x - start_point[0])**2 + (y - start_point[1])**2)

        # Distance between consecutive paths
        for i in range(len(optimized_closed) - 1):
            x1, y1 = optimized_closed[i].start_point
            x2, y2 = optimized_closed[i + 1].start_point
            total_travel += np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

        # Update paths: closed paths first (optimized), then open paths (original order)
        self.paths = optimized_closed + open_paths

        return total_travel

    def statistics(self) -> dict:
        """
        Get statistics about this layer.

        Returns:
            Dictionary with layer statistics
        """
        stripe_paths = self.get_paths_by_type('stripe')
        contour_paths = self.get_paths_by_type('contour')
        closed_paths = self.get_closed_paths()

        return {
            'layer_id': self.layer_id,
            'name': self.name,
            'z_height': self.z_height,
            'total_paths': len(self.paths),
            'stripe_paths': len(stripe_paths),
            'contour_paths': len(contour_paths),
            'closed_paths': len(closed_paths),
            'open_paths': len(self.paths) - len(closed_paths),
            'total_length': self.total_length(),
            'total_nodes': self.total_nodes(),
            'bounds': self.get_bounds()
        }

    def clear(self):
        """Remove all paths from this layer."""
        self.paths.clear()

    def __len__(self) -> int:
        """Return the number of paths in this layer."""
        return len(self.paths)

    def __iter__(self) -> Iterator[Path]:
        """Iterate over paths in this layer."""
        return iter(self.paths)

    def __getitem__(self, index: int) -> Path:
        """Get path by index."""
        return self.paths[index]

    def to_points_csv(
        self,
        output_path: str,
        z_height: Optional[float] = None,
        include_travel: bool = True,
        extrusion_on_value: float = 1.0,
        extrusion_off_value: float = 0.0
    ) -> int:
        """
        Export layer as CSV point list for FullControl.

        Creates a CSV file with columns: x, y, z, extrusion
        Each row represents a point that FullControl can read.

        Args:
            output_path: Output CSV file path
            z_height: Z height for all points (uses layer.z_height if not specified)
            include_travel: Include travel moves between paths
            extrusion_on_value: Value for extrusion ON (default: 1.0)
            extrusion_off_value: Value for extrusion OFF (default: 0.0)

        Returns:
            Number of points written

        Example:
            >>> layer.to_points_csv("output.csv", z_height=0.2)
            >>> # CSV format:
            >>> # x,y,z,extrusion
            >>> # 10.5,20.3,0.2,0.0
            >>> # 10.5,20.3,0.2,1.0
            >>> # 11.2,21.1,0.2,1.0
            >>> # ...
        """
        import csv

        z = z_height if z_height is not None else self.z_height
        if z is None:
            z = 0.0

        points = []

        for path_idx, path in enumerate(self.paths):
            # Add travel move to start of path (if not first path and include_travel)
            if path_idx > 0 and include_travel:
                x_start, y_start = path.start_point
                points.append({
                    'x': x_start,
                    'y': y_start,
                    'z': z,
                    'extrusion': extrusion_off_value
                })

            # Add all points in the path with extrusion ON
            for x, y in path.nodes:
                points.append({
                    'x': x,
                    'y': y,
                    'z': z,
                    'extrusion': extrusion_on_value
                })

        # Write to CSV
        with open(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['x', 'y', 'z', 'extrusion'])
            writer.writeheader()
            writer.writerows(points)

        return len(points)

    def __repr__(self) -> str:
        """String representation of the layer."""
        z_str = f", z={self.z_height:.3f}mm" if self.z_height is not None else ""
        return (f"Layer(id={self.layer_id}, name='{self.name}', "
                f"paths={len(self.paths)}, "
                f"length={self.total_length():.2f}mm{z_str})")
