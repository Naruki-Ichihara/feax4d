"""SVG parser for converting SVG files to Path objects.

Also provides utilities for creating SVGs from geometry:
    create_fiber_svg — Generate SVG with fiber stripes and contour boundary
"""

from typing import List, Tuple
import xml.etree.ElementTree as ET
from pathlib import Path as FilePath
from feax4d.fibrifier.path import Path


def parse_svg_to_paths(svg_file: str, segment_length: float = 0.1) -> List[Path]:
    """
    Parse an SVG file and convert all paths/polylines to Path objects.

    This function reads an SVG file containing paths and polylines, and converts
    them to a list of Path objects. Polylines are already line segments, so they
    are directly converted. For SVG path elements with curves, svgpathtools is
    used to discretize them into line segments at the specified resolution.

    Args:
        svg_file: Path to the SVG file
        segment_length: Maximum length of line segments when discretizing curves (default: 0.1).
                       Smaller values give smoother approximation but more points.

    Returns:
        List of Path objects representing all paths in the SVG

    Example:
        >>> paths = parse_svg_to_paths("output.svg", segment_length=0.05)
        >>> print(f"Loaded {len(paths)} paths")
        >>> for path in paths[:5]:
        ...     print(path)
    """
    # Parse SVG file
    tree = ET.parse(svg_file)
    root = tree.getroot()

    # Handle namespace
    ns = {'svg': 'http://www.w3.org/2000/svg'}

    path_objects = []
    path_id_counter = 0

    # Process all groups in the SVG
    for group in root.findall('.//svg:g', ns):
        group_id = group.get('id', '')

        # Determine path type based on group ID
        if 'stripe' in group_id.lower():
            path_type = 'stripe'
        elif 'density' in group_id.lower() or 'contour' in group_id.lower():
            path_type = 'contour'
        else:
            path_type = 'stripe'  # Default to stripe

        # Process polylines in this group
        for polyline in group.findall('svg:polyline', ns):
            points_str = polyline.get('points')
            if not points_str:
                continue

            # Parse polyline points
            nodes = []
            for point in points_str.strip().split():
                if ',' in point:
                    x, y = map(float, point.split(','))
                    nodes.append((x, y))

            if len(nodes) >= 2:
                path_obj = Path(
                    path_id=path_id_counter,
                    nodes=nodes,
                    path_type=path_type
                )
                path_objects.append(path_obj)
                path_id_counter += 1

        # Process path elements (curves, arcs, etc.) using svgpathtools
        for path_elem in group.findall('svg:path', ns):
            d_attr = path_elem.get('d')
            if not d_attr:
                continue

            try:
                from svgpathtools import parse_path

                # Parse the path
                svg_path = parse_path(d_attr)

                # Discretize the path into line segments
                nodes = discretize_svg_path(svg_path, segment_length)

                if len(nodes) >= 2:
                    path_obj = Path(
                        path_id=path_id_counter,
                        nodes=nodes,
                        path_type=path_type
                    )
                    path_objects.append(path_obj)
                    path_id_counter += 1

            except ImportError:
                print("Warning: svgpathtools not available. Skipping path elements.")
                print("Install with: pip install svgpathtools")
                continue
            except Exception as e:
                print(f"Warning: Failed to parse path element: {e}")
                continue

    return path_objects


def discretize_svg_path(svg_path, segment_length: float = 0.1) -> List[Tuple[float, float]]:
    """
    Discretize an SVG path into line segments.

    Uses svgpathtools to sample points along the path at regular intervals,
    converting curves and arcs into polyline approximations.

    Args:
        svg_path: svgpathtools Path object
        segment_length: Maximum length between consecutive points

    Returns:
        List of (x, y) coordinate tuples
    """
    if len(svg_path) == 0:
        return []

    # Calculate total path length
    total_length = svg_path.length()

    if total_length == 0:
        # Degenerate path
        start = svg_path.point(0)
        return [(start.real, start.imag)]

    # Number of segments needed
    num_segments = max(2, int(total_length / segment_length) + 1)

    # Sample points along the path
    nodes = []
    for i in range(num_segments + 1):
        t = i / num_segments
        point = svg_path.point(t)
        nodes.append((point.real, point.imag))

    return nodes


def save_paths_to_svg(paths: List[Path], output_file: str,
                     width: float, height: float,
                     viewbox: Tuple[float, float, float, float] = None):
    """
    Save a list of Path objects back to an SVG file.

    Args:
        paths: List of Path objects to save
        output_file: Output SVG file path
        width: SVG width
        height: SVG height
        viewbox: Optional viewBox (x_min, y_min, width, height).
                If None, uses (0, 0, width, height)
    """
    if viewbox is None:
        viewbox = (0, 0, width, height)

    # Group paths by type
    stripe_paths = [p for p in paths if p.path_type == 'stripe']
    contour_paths = [p for p in paths if p.path_type == 'contour']

    # Build SVG
    svg_lines = []
    svg_lines.append('<?xml version="1.0" encoding="UTF-8" standalone="no"?>')
    svg_lines.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{width:.6f}mm" height="{height:.6f}mm" '
        f'viewBox="{viewbox[0]:.6f} {viewbox[1]:.6f} {viewbox[2]:.6f} {viewbox[3]:.6f}">'
    )

    # Add stripe paths group
    if stripe_paths:
        svg_lines.append('  <g id="stripe_contours" fill="none" stroke="#000000" stroke-width="0.01">')
        for path in stripe_paths:
            svg_lines.append(f'    {path.to_svg_polyline()}')
        svg_lines.append('  </g>')

    # Add contour paths group
    if contour_paths:
        svg_lines.append('  <g id="density_contours" fill="none" stroke="#FF0000" stroke-width="0.02">')
        for path in contour_paths:
            svg_lines.append(f'    {path.to_svg_polyline(stroke_color="#FF0000", stroke_width=0.02)}')
        svg_lines.append('  </g>')

    svg_lines.append('</svg>')

    # Write to file
    svg_content = '\n'.join(svg_lines)
    with open(output_file, 'w') as f:
        f.write(svg_content)


def create_fiber_svg(
    output_file: str,
    contour: list,
    fibers: list,
    width: float = None,
    height: float = None,
):
    """
    Create an SVG file with fiber stripe paths and contour boundary.

    Accepts coordinates as simple lists — no need to construct Path objects.

    Args:
        output_file: Output SVG file path.
        contour: Contour boundary as list of (x, y) tuples forming a closed polygon.
                 Or list of such polygons for multiple contours.
        fibers: List of fiber lines. Each fiber is a list of (x, y) tuples
                (polyline with 2+ points).
        width: SVG width (mm). Auto-computed from coordinates if None.
        height: SVG height (mm). Auto-computed from coordinates if None.

    Returns:
        dict with: output_file, n_fibers, n_contours, width, height

    Example:
        >>> # Rectangle contour with vertical fiber lines
        >>> contour = [(0,0), (20,0), (20,50), (0,50), (0,0)]
        >>> fibers = [[(x, 0), (x, 50)] for x in range(1, 20)]
        >>> create_fiber_svg("rect.svg", contour, fibers)

        >>> # Using svgpathtools geometry
        >>> from svgpathtools import Line, Path as SvgPath
        >>> rect = SvgPath(Line(0+0j, 20+0j), Line(20+0j, 20+50j), ...)
        >>> create_fiber_svg("rect.svg", rect, [fiber1, fiber2])
    """
    # Normalize contour input
    contours = _normalize_path_input(contour, close=True)
    fiber_lines = _normalize_path_input(fibers, close=False)

    # Auto-compute dimensions
    all_pts = [pt for path in contours + fiber_lines for pt in path]
    if not all_pts:
        raise ValueError("No coordinates provided")

    all_x = [p[0] for p in all_pts]
    all_y = [p[1] for p in all_pts]
    x_min, x_max = min(all_x), max(all_x)
    y_min, y_max = min(all_y), max(all_y)

    if width is None:
        width = x_max - x_min
    if height is None:
        height = y_max - y_min

    # Build libertas Path objects
    path_id = 0
    stripe_paths = []
    for nodes in fiber_lines:
        if len(nodes) >= 2:
            stripe_paths.append(Path(path_id=path_id, nodes=nodes, path_type='stripe'))
            path_id += 1

    contour_paths = []
    for nodes in contours:
        if len(nodes) >= 2:
            contour_paths.append(Path(path_id=path_id, nodes=nodes, path_type='contour'))
            path_id += 1

    # Use existing save function
    all_paths = stripe_paths + contour_paths
    save_paths_to_svg(all_paths, output_file, width, height,
                      viewbox=(x_min, y_min, width, height))

    return {
        "output_file": output_file,
        "n_fibers": len(stripe_paths),
        "n_contours": len(contour_paths),
        "width": width,
        "height": height,
    }


def _normalize_path_input(paths_input, close=False):
    """
    Normalize various input formats to list of list-of-(x,y) tuples.

    Accepts:
        - list of (x,y) tuples (single path)
        - list of list of (x,y) tuples (multiple paths)
        - svgpathtools Path object (single path)
        - list of svgpathtools Path objects (multiple paths)
    """
    if not paths_input:
        return []

    # Single svgpathtools Path
    if _is_svgpathtools_path(paths_input):
        return [_svgpath_to_nodes(paths_input, close)]

    # Single list of tuples: [(x,y), (x,y), ...]
    if isinstance(paths_input, list) and len(paths_input) > 0:
        first = paths_input[0]

        # List of svgpathtools Paths
        if _is_svgpathtools_path(first):
            return [_svgpath_to_nodes(p, close) for p in paths_input]

        # Single polyline: [(x,y), (x,y), ...]
        if isinstance(first, (tuple, list)) and len(first) == 2 and isinstance(first[0], (int, float)):
            # Check if this is a list of points or list of paths
            # If all elements are 2-tuples of numbers, it's a single path
            if all(isinstance(p, (tuple, list)) and len(p) == 2 for p in paths_input):
                nodes = [(float(x), float(y)) for x, y in paths_input]
                if close and nodes[0] != nodes[-1]:
                    nodes.append(nodes[0])
                return [nodes]

        # List of polylines: [[(x,y),...], [(x,y),...], ...]
        if isinstance(first, list) and len(first) > 0:
            result = []
            for path in paths_input:
                if _is_svgpathtools_path(path):
                    result.append(_svgpath_to_nodes(path, close))
                else:
                    nodes = [(float(x), float(y)) for x, y in path]
                    if close and nodes[0] != nodes[-1]:
                        nodes.append(nodes[0])
                    result.append(nodes)
            return result

    return []


def _is_svgpathtools_path(obj):
    """Check if object is a svgpathtools Path (without importing it)."""
    return hasattr(obj, '_segments') and hasattr(obj, 'point')


def _svgpath_to_nodes(svgpath, close=False):
    """Convert svgpathtools Path to list of (x, y) tuples."""
    nodes = []
    for seg in svgpath:
        nodes.append((seg.start.real, seg.start.imag))
    if svgpath:
        nodes.append((svgpath[-1].end.real, svgpath[-1].end.imag))
    if close and nodes and nodes[0] != nodes[-1]:
        nodes.append(nodes[0])
    return nodes
