"""
CFRTP Gcode Generator for curved fiber paths from SVG.

Output format matches Fibrifier (9T Labs) gcode exactly.

Public API:
    FibrifierLayer — Define a single layer (polymer / fiber / infill)
    FibrifierModel — Stack layers and generate gcode
    FibrifierParams — Printing parameters
    svg_to_fibrifier_gcode — Convenience function (legacy)
    FibrifierGcodeGenerator — Low-level generator class
"""

import math
import datetime
import argparse
import numpy as np
import matplotlib.pyplot as plt
from feax4d.fibrifier.svg_parser import parse_svg_to_paths
from feax4d.fibrifier.layer import Layer
from feax4d.fibrifier.path import Path


# ══════════════════════════════════════════════════════════════════
# Path preparation
# ══════════════════════════════════════════════════════════════════

def load_and_prepare_paths(
    svg_path: str,
    connection_threshold: float = 5.0,
    min_fiber_length: float = 23.73,
    start_point: tuple = None,
    flip_y: bool = True,
    smooth_sigma: float = 3.0,
    decimate_epsilon: float = 0.05,
):
    all_paths = parse_svg_to_paths(svg_path)
    print(f"Loaded {len(all_paths)} paths from SVG")

    import xml.etree.ElementTree as ET
    tree = ET.parse(svg_path)
    root = tree.getroot()
    viewbox = root.attrib.get("viewBox", "0 0 0 0").split()
    svg_height = float(viewbox[3])

    stripe_paths = [p for p in all_paths if p.path_type == "stripe"]
    contour_paths = [p for p in all_paths if p.path_type == "contour"]
    print(f"  Stripe (fiber): {len(stripe_paths)}, Contour (perimeter): {len(contour_paths)}")

    if flip_y:
        stripe_paths = _flip_paths_y(stripe_paths, svg_height)
        contour_paths = _flip_paths_y(contour_paths, svg_height)

    layer = Layer(layer_id=0, paths=stripe_paths)
    travel = layer.optimize_open_path_order(start_point=start_point)
    # Keep open paths as-is; instead of *discarding* closed stripe contours
    # (loops that don't touch the mask boundary — e.g. curved/concentric
    # stripes), cut them open at the first node so they still become fibre
    # paths.  Dropping them left whole fibre regions with no toolpath.
    stripe_paths = []
    n_opened = 0
    for p in layer.paths:
        if not p.is_closed:
            stripe_paths.append(p)
        elif len(p.nodes) >= 4:
            stripe_paths.append(Path(path_id=p.path_id, nodes=list(p.nodes[:-1]),
                                     path_type=p.path_type))
            n_opened += 1
    print(f"  TSP travel distance: {travel:.2f} mm  "
          f"({len(stripe_paths)} paths, {n_opened} closed loops opened)")

    connected = _connect_adjacent_paths(stripe_paths, connection_threshold)
    print(f"  After connection (threshold={connection_threshold} mm): {len(connected)} paths")

    total_before = sum(len(p.nodes) for p in connected)
    simplified = []
    for p in connected:
        new_nodes = simplify_path_nodes(p.nodes, sigma=smooth_sigma, epsilon=decimate_epsilon)
        simplified.append(Path(path_id=p.path_id, nodes=new_nodes, path_type=p.path_type))
    total_after = sum(len(p.nodes) for p in simplified)
    _pct = (100 * total_after / total_before) if total_before else 0.0
    print(f"  Smoothing (σ={smooth_sigma}) + decimation (ε={decimate_epsilon}mm): "
          f"{total_before} → {total_after} points ({_pct:.0f}%)")

    filtered = [p for p in simplified if p.length >= min_fiber_length]
    dropped = len(simplified) - len(filtered)
    print(f"  Dropped {dropped} paths shorter than {min_fiber_length} mm → {len(filtered)} remain")

    return filtered, contour_paths, svg_height


def _flip_paths_y(paths, svg_height):
    flipped = []
    for p in paths:
        new_nodes = [(x, svg_height - y) for x, y in p.nodes]
        fp = Path(path_id=p.path_id, nodes=new_nodes, path_type=p.path_type)
        flipped.append(fp)
    return flipped


def _segments_cross(seg_a, seg_b):
    """Check if two line segments (p0,p1) and (p2,p3) properly cross (not just touch)."""
    (x1, y1), (x2, y2) = seg_a
    (x3, y3), (x4, y4) = seg_b
    d = (x2 - x1) * (y4 - y3) - (y2 - y1) * (x4 - x3)
    if abs(d) < 1e-12:
        return False  # parallel
    t = ((x3 - x1) * (y4 - y3) - (y3 - y1) * (x4 - x3)) / d
    u = ((x3 - x1) * (y2 - y1) - (y3 - y1) * (x2 - x1)) / d
    # Proper crossing: both parameters strictly inside (0, 1)
    return 0.01 < t < 0.99 and 0.01 < u < 0.99


def _seg_crosses_polyline(seg, nodes, lo=0, hi=None):
    """True if ``seg`` properly crosses any segment ``nodes[i]→nodes[i+1]``
    for ``lo <= i < hi`` (``hi`` defaults to the last segment)."""
    if hi is None:
        hi = len(nodes) - 1
    for i in range(lo, hi):
        if _segments_cross(seg, (nodes[i], nodes[i + 1])):
            return True
    return False


def _polylines_intersect(A, B):
    """True if any segment of polyline ``A`` properly crosses one of ``B``.

    Vectorised (numpy) over ``B`` for each segment of the shorter polyline,
    using the same strict interior test (0.01, 0.99) as ``_segments_cross`` so
    shared endpoints are not flagged.
    """
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    if len(A) < 2 or len(B) < 2:
        return False
    if len(A) > len(B):            # loop over the shorter one
        A, B = B, A
    q = B[:-1]
    s = B[1:] - B[:-1]
    sx, sy, qx, qy = s[:, 0], s[:, 1], q[:, 0], q[:, 1]
    for i in range(len(A) - 1):
        p = A[i]
        r = A[i + 1] - A[i]
        denom = r[0] * sy - r[1] * sx
        wx = qx - p[0]
        wy = qy - p[1]
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (wx * sy - wy * sx) / denom
            u = (wx * r[1] - wy * r[0]) / denom
        hit = (np.abs(denom) > 1e-12) & (t > 0.01) & (t < 0.99) & (u > 0.01) & (u < 0.99)
        if bool(hit.any()):
            return True
    return False


def _connect_adjacent_paths(paths, threshold):
    """Greedily chain paths whose endpoints are within ``threshold``.

    A candidate is attached at whichever of its two ends is nearer the current
    tail (reversing it if needed, which shortens the jump and avoids many
    crossings).  A join is accepted only if the connecting segment crosses
    **neither** the already-built path **nor** the candidate's own body — this
    prevents self-intersections within a single fibre path.  Adjacent segments
    that merely share the join endpoint are not flagged (the crossing test is
    strict interior), so we only exclude the one segment touching each end.
    """
    if not paths:
        return []
    merged = []
    current = list(paths[0].nodes)

    for i in range(1, len(paths)):
        cand = list(paths[i].nodes)
        tail = current[-1]
        d_start = math.hypot(cand[0][0] - tail[0], cand[0][1] - tail[1])
        d_end = math.hypot(cand[-1][0] - tail[0], cand[-1][1] - tail[1])
        if d_end < d_start:            # attach the nearer end → reverse candidate
            cand = cand[::-1]
            dist = d_end
        else:
            dist = d_start

        conn = (tail, cand[0])
        crosses = (
            _seg_crosses_polyline(conn, current, lo=0, hi=len(current) - 2)
            or _seg_crosses_polyline(conn, cand, lo=1)
        )
        if dist <= threshold and not crosses:
            current.extend(cand)
        else:
            merged.append(Path(path_id=len(merged), nodes=current, path_type="stripe"))
            current = cand

    merged.append(Path(path_id=len(merged), nodes=current, path_type="stripe"))
    return merged


# ══════════════════════════════════════════════════════════════════
# Smoothing & decimation
# ══════════════════════════════════════════════════════════════════

def _smooth_nodes(nodes, sigma=3):
    pts = np.array(nodes)
    from scipy.ndimage import gaussian_filter1d
    sx = gaussian_filter1d(pts[:, 0], sigma=sigma, mode="nearest")
    sy = gaussian_filter1d(pts[:, 1], sigma=sigma, mode="nearest")
    sx[0], sy[0] = pts[0, 0], pts[0, 1]
    sx[-1], sy[-1] = pts[-1, 0], pts[-1, 1]
    return list(zip(sx.tolist(), sy.tolist()))


def _rdp_decimate(nodes, epsilon=0.05):
    if len(nodes) <= 2:
        return nodes
    start = np.array(nodes[0])
    end = np.array(nodes[-1])
    line_vec = end - start
    line_len = np.linalg.norm(line_vec)
    if line_len < 1e-12:
        dists = [np.linalg.norm(np.array(p) - start) for p in nodes]
        idx = int(np.argmax(dists))
        if dists[idx] < epsilon:
            return [nodes[0], nodes[-1]]
    else:
        line_unit = line_vec / line_len
        dists = []
        for p in nodes:
            v = np.array(p) - start
            proj = np.dot(v, line_unit)
            perp = np.linalg.norm(v - proj * line_unit)
            dists.append(perp)
        idx = int(np.argmax(dists))
    if dists[idx] > epsilon:
        left = _rdp_decimate(nodes[: idx + 1], epsilon)
        right = _rdp_decimate(nodes[idx:], epsilon)
        return left[:-1] + right
    else:
        return [nodes[0], nodes[-1]]


def simplify_path_nodes(nodes, sigma=3, epsilon=0.05):
    if sigma > 0:
        smoothed = _smooth_nodes(nodes, sigma=sigma)
    else:
        smoothed = nodes
    decimated = _rdp_decimate(smoothed, epsilon=epsilon)
    return decimated


# ══════════════════════════════════════════════════════════════════
# Geometry helpers
# ══════════════════════════════════════════════════════════════════

def _segment_angle_deg(x0, y0, x1, y1):
    return math.degrees(math.atan2(y1 - y0, x1 - x0))


def _compute_smooth_angles(nodes, sigma=2):
    """
    Compute direction angle at each node with optional Gaussian smoothing.

    At sharp turns (>30° deviation from raw), the raw angle is preserved.
    """
    n = len(nodes)
    raw = []
    for i in range(n - 1):
        raw.append(_segment_angle_deg(
            nodes[i][0], nodes[i][1], nodes[i + 1][0], nodes[i + 1][1]))
    raw.append(raw[-1])

    if sigma <= 0:
        return raw

    from scipy.ndimage import gaussian_filter1d
    rads = np.deg2rad(raw)
    s_sin = gaussian_filter1d(np.sin(rads), sigma=sigma, mode="nearest")
    s_cos = gaussian_filter1d(np.cos(rads), sigma=sigma, mode="nearest")
    smooth_rads = np.arctan2(s_sin, s_cos)
    smoothed = np.rad2deg(smooth_rads).tolist()

    # Identify sharp turns in raw angles (large change between consecutive segments)
    SHARP_TURN = 30.0
    is_near_turn = [False] * n
    for i in range(1, n):
        if abs(_wrap_angle(raw[i] - raw[i - 1])) > SHARP_TURN:
            # Mark nodes near this turn to revert to raw
            for j in range(max(0, i - int(sigma) - 1), min(n, i + int(sigma) + 2)):
                is_near_turn[j] = True

    for i in range(n):
        if is_near_turn[i]:
            smoothed[i] = raw[i]

    return smoothed


def _cumulative_distances_from_end(nodes):
    n = len(nodes)
    dist = np.zeros(n)
    for i in range(n - 2, -1, -1):
        dx = nodes[i + 1][0] - nodes[i][0]
        dy = nodes[i + 1][1] - nodes[i][1]
        dist[i] = dist[i + 1] + math.sqrt(dx * dx + dy * dy)
    return dist


def _find_cut_index(nodes, nozzle_dead_length):
    """Find node index where remaining arc-length to end = nozzle_dead_length."""
    dist_from_end = _cumulative_distances_from_end(nodes)
    for i in range(len(nodes) - 1, 0, -1):
        if dist_from_end[i] >= nozzle_dead_length:
            if i < len(nodes) - 1:
                overshoot = dist_from_end[i] - nozzle_dead_length
                seg_len = dist_from_end[i] - dist_from_end[i + 1]
                if seg_len > 1e-9:
                    t = overshoot / seg_len
                    x = nodes[i][0] + t * (nodes[i + 1][0] - nodes[i][0])
                    y = nodes[i][1] + t * (nodes[i + 1][1] - nodes[i][1])
                    return i, (x, y)
            return i, nodes[i]
    return 0, nodes[0]


def _anchor_point(nodes, anchor_distance):
    x0, y0 = nodes[0]
    x1, y1 = nodes[1]
    dx, dy = x1 - x0, y1 - y0
    seg_len = math.sqrt(dx * dx + dy * dy)
    if seg_len < 1e-9:
        return (x0, y0 - anchor_distance)
    ux, uy = dx / seg_len, dy / seg_len
    return (x0 - ux * anchor_distance, y0 - uy * anchor_distance)


def _ironing_point(nodes, ironing_distance):
    x0, y0 = nodes[-2]
    x1, y1 = nodes[-1]
    dx, dy = x1 - x0, y1 - y0
    seg_len = math.sqrt(dx * dx + dy * dy)
    if seg_len < 1e-9:
        return (x1, y1)
    ux, uy = dx / seg_len, dy / seg_len
    return (x1 + ux * ironing_distance, y1 + uy * ironing_distance)


# ══════════════════════════════════════════════════════════════════
# Infill generation
# ══════════════════════════════════════════════════════════════════

def _build_infill_region(contour_paths, inset=0.3):
    """
    Build a Shapely geometry representing the solid region for infill.

    Uses contour containment to determine outer boundaries vs holes:
    the largest contour is the outer shell, contours contained within it
    are holes, etc. The result is a single (Multi)Polygon with holes cut out.

    Args:
        contour_paths: list of Path objects (closed contours).
        inset: shrink boundary inward (mm).

    Returns:
        shapely Polygon/MultiPolygon, or None if empty.
    """
    from shapely.geometry import Polygon
    from shapely.ops import unary_union

    # Collect closed contours as Shapely polygons
    polys = []
    for cp in contour_paths:
        if not cp.is_closed or len(cp.nodes) < 4:
            continue
        p = Polygon(cp.nodes)
        if not p.is_valid:
            p = p.buffer(0)
        if not p.is_empty:
            polys.append(p)

    if not polys:
        return None

    # Sort by area descending — largest is outer shell
    polys.sort(key=lambda p: p.area, reverse=True)

    # Build region: start with largest, subtract contained smaller ones,
    # but smaller ones inside holes are solid again (nested shells).
    # Simple approach: use even-odd rule via XOR-like union.
    # Shapely's approach: build polygon with holes explicitly.

    # Classify each polygon as shell or hole by counting how many
    # larger polygons contain its representative point.
    classified = []  # (poly, is_hole)
    for i, poly in enumerate(polys):
        pt = poly.representative_point()
        depth = 0
        for j in range(i):
            if polys[j].contains(pt):
                depth += 1
        # Even depth = shell (solid), odd depth = hole
        classified.append((poly, depth % 2 == 1))

    # Build final geometry: union of shells, then subtract holes
    shells = [p for p, is_hole in classified if not is_hole]
    holes = [p for p, is_hole in classified if is_hole]

    region = unary_union(shells)
    if holes:
        hole_union = unary_union(holes)
        region = region.difference(hole_union)

    # Apply inset
    if inset > 0 and not region.is_empty:
        region = region.buffer(-inset)

    if region.is_empty:
        return None

    return region


def _generate_infill_scanlines(region, angle_deg, pitch):
    """
    Generate infill scan lines within a Shapely geometry at given angle.

    Args:
        region: shapely Polygon/MultiPolygon (with holes already cut).
        angle_deg: scan line angle in degrees.
        pitch: spacing between scan lines (mm).

    Returns:
        list of ((x0,y0),(x1,y1)) line segments, alternating direction.
    """
    from shapely.geometry import LineString, MultiLineString
    from shapely.affinity import rotate

    angle_rad = math.radians(angle_deg)
    cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)

    # Rotate region to align scan lines with X axis
    region_rot = rotate(region, -angle_deg, origin=(0, 0))
    rminx, rminy, rmaxx, rmaxy = region_rot.bounds

    lines = []
    y = rminy + pitch / 2.0
    direction = 1

    while y <= rmaxy:
        scan = LineString([(rminx - 1, y), (rmaxx + 1, y)])
        intersection = region_rot.intersection(scan)

        segments = []
        if intersection.is_empty:
            pass
        elif isinstance(intersection, LineString):
            segments = [intersection]
        elif isinstance(intersection, MultiLineString):
            segments = list(intersection.geoms)
        else:
            for geom in intersection.geoms:
                if isinstance(geom, LineString):
                    segments.append(geom)

        for seg in segments:
            coords = list(seg.coords)
            if len(coords) >= 2:
                # Rotate back to original space
                p0 = (coords[0][0] * cos_a - coords[0][1] * sin_a,
                       coords[0][0] * sin_a + coords[0][1] * cos_a)
                p1 = (coords[-1][0] * cos_a - coords[-1][1] * sin_a,
                       coords[-1][0] * sin_a + coords[-1][1] * cos_a)
                if direction == 1:
                    lines.append((p0, p1))
                else:
                    lines.append((p1, p0))

        y += pitch
        direction *= -1

    return lines


def generate_infill_paths(contour_paths, angle_deg, pitch, inset=0.3):
    """
    Generate infill Path objects respecting contour topology (holes excluded).

    Args:
        contour_paths: list of Path objects (closed contours).
        angle_deg: infill angle in degrees.
        pitch: line spacing (mm).
        inset: boundary inset (mm).

    Returns:
        list of Path objects representing infill zig-zag lines.
    """
    region = _build_infill_region(contour_paths, inset=inset)
    if region is None:
        return []

    all_lines = _generate_infill_scanlines(region, angle_deg, pitch)
    if not all_lines:
        return []

    # Step 1: Group consecutive scan segments that can be connected
    # without crossing a hole (stay inside the solid region).
    from shapely.geometry import LineString

    raw_paths = []
    current_nodes = [all_lines[0][0], all_lines[0][1]]

    for i in range(1, len(all_lines)):
        prev_end = current_nodes[-1]
        next_start = all_lines[i][0]

        travel = LineString([prev_end, next_start])
        if region.contains(travel):
            current_nodes.append(next_start)
            current_nodes.append(all_lines[i][1])
        else:
            if len(current_nodes) >= 2:
                raw_paths.append(Path(
                    path_id=len(raw_paths), nodes=current_nodes, path_type="contour"))
            current_nodes = [all_lines[i][0], all_lines[i][1]]

    if len(current_nodes) >= 2:
        raw_paths.append(Path(
            path_id=len(raw_paths), nodes=current_nodes, path_type="contour"))

    if not raw_paths:
        return []

    # Step 2: Optimize order via TSP + reconnect nearby paths
    layer = Layer(layer_id=0, paths=raw_paths)
    layer.optimize_open_path_order()
    optimized = layer.get_open_paths()

    connected = _connect_adjacent_paths(optimized, threshold=pitch * 10)

    return connected


# ══════════════════════════════════════════════════════════════════
# Parameter dataclasses
# ══════════════════════════════════════════════════════════════════

from dataclasses import dataclass, field


@dataclass
class FibrifierTemperatureParams:
    """Temperature settings for Fibrifier printing."""
    cf_print_temp: int = 220            # Fiber extruder printing temperature (°C)
    cf_first_layer_temp: int = 220      # Fiber extruder first layer temperature (°C)
    pl_print_temp: int = 230            # Polymer nozzle printing temperature (°C)
    pl_first_layer_temp: int = 230      # Polymer nozzle first layer temperature (°C)
    bed_temperature: int = 90           # Bed temperature (°C)
    build_chamber_temperature: int = 90 # Build chamber temperature (°C)
    material_storage_temperature: int = 70  # Material chamber temperature (°C)
    nozzle_cooldown_temperature: int = 50   # Nozzle cooldown temp during tool switch (°C)
    preheat_target_temperature: int = 230   # Preheat target for polymer nozzle (°C)
    preheat_time: int = 60              # Preheat time (s)


@dataclass
class FibrifierSpeedParams:
    """Speed settings for Fibrifier printing."""
    cf_print_feedrate: int = 600        # Fiber printing feedrate (mm/min)
    perimeter_speed: int = 40           # Perimeter speed factor
    after_cut_speed: int = 75           # Post-cut speed as % of cf_print_feedrate
    rapid_move_feedrate: int = 5500     # Rapid travel feedrate (mm/min)
    pl_first_layer_speed: int = 30      # Polymer first layer speed factor
    # Polymer feedrates (mm/min) — used directly per layer.
    pl_perimeter_feedrate: int = 2400   # polymer perimeter (non-first layers)
    pl_infill_feedrate: int = 3600      # polymer infill (non-first layers)
    pl_first_layer_feedrate: int = 1800 # polymer perimeter & infill, slow layers
    pl_first_layers: int = 1            # number of initial layers printed slow


@dataclass
class FibrifierExtrusionParams:
    """Extrusion settings for Fibrifier printing.

    Polymer extrusion is **volumetric**: the relative filament advance for a
    move of length L is ``E = L * h * w / (pi/4 * filament_diameter**2) *
    pl_extrusion_multiplier`` (h = layer height, w = polymer line width), so it
    scales correctly with layer thickness and line width.  ``pl_extrusion_-
    multiplier`` is therefore a dimensionless **flow** factor (≈ 1.0).

    Fibre extrusion stays length-based (``E = L * cf_extrusion_multiplier``):
    the continuous tow is fed ~1:1 with the path length, not extruded by volume.
    """
    cf_extrusion_multiplier: float = 0.99   # fibre tow feed factor (E = dist * this)
    pl_extrusion_multiplier: float = 1.0    # polymer flow factor (volumetric)
    filament_diameter: float = 1.75         # polymer filament diameter [mm]
    polymer_line_width: float = 0.5         # polymer extruded bead width [mm]
    first_layer_extrusion_multiplier: float = 1.0  # (reserved) first-layer flow


@dataclass
class FibrifierFiberParams:
    """Fiber-specific parameters for Fibrifier printing."""
    nozzle_dead_length: float = 21.5        # Distance from cutter to nozzle tip (mm)
    minimal_printable_length: float = 23.73 # Min fiber length to print (mm)
    no_speed_up_length: float = 5.0         # Ironing distance after path end (mm)
    anchoring_dwell: int = 600              # Anchoring dwell time (ms)
    anchoring_height: int = 3               # Z-lift for anchoring approach (mm)
    anchoring_length: int = 2               # Anchoring contact length (mm)
    fiber_width: float = 1.5                # Fiber width (mm)
    fiber_cut: bool = True                  # insert a fibre cut at the end of each
                                            # stretch; False = print continuously
                                            # (no cutting, full-length extrusion)
    # Manual-cut mode (only when fiber_cut is False): at the end of each fibre
    # layer, raise the nozzle in place while dispensing fibre, then pause so the
    # operator can cut the tow by hand and resume.
    manual_cut: bool = False
    manual_cut_lift: float = 20.0           # Z lift during the pause [mm]
    manual_cut_extrude: float = 20.0        # fibre dispensed during the lift [mm]

    # Continuous-fibre stretch re-anchoring (when ``fiber_cut`` is False).
    # At each in-layer stretch boundary (e.g. pass-to-pass in ``kind="lines"``
    # test paths) the tow stays uncut, so the nozzle:
    #   1. continues forward in the *last segment's direction* for
    #      ``uturn_escape`` mm while lifting Z by ``uturn_lift`` mm and
    #      dispensing ``uturn_extrude`` mm of tow (the dispense keeps the
    #      tow flowing while the nozzle escapes upward off the part);
    #   2. rapids to the next stretch's anchor position with W absolute set
    #      to the new heading (handles the 180° rotation);
    #   3. descends onto the new start, dragging the tow;
    #   4. dwells ``anchoring_dwell`` ms (standard anchor dwell);
    #   5. continues printing.
    # Same-Z stretch transitions only — layer-to-layer climbs (different Z)
    # use the simpler "climb in place" path.  Snake (single connected path)
    # has no stretch transitions inside a layer, so this feature does
    # nothing in snake mode.
    # ``uturn_escape = 0`` disables the feature and falls back to a simple
    # climb everywhere (the original behaviour).
    uturn_escape: float = 0.0               # mm forward in last segment's direction
                                            # during the escape (0 = disabled)
    uturn_lift: float = 5.0                 # mm of Z lift during the escape
    uturn_extrude: float = 5.0              # mm of fibre dispensed during the escape


@dataclass
class FibrifierRetractionParams:
    """Retraction settings for polymer extrusion."""
    retract_length: float = 3.0     # Retraction length (mm)
    retract_speed: int = 60         # Retraction speed (mm/s)
    retract_lift: float = 0.0       # Z-lift on retraction (mm)


@dataclass
class FibrifierParams:
    """
    Complete parameter set for Fibrifier gcode generation.

    Groups all printing parameters into logical sub-dataclasses.
    Default values match the 9T Labs reference configuration.

    Example:
        >>> params = FibrifierParams()
        >>> params.temperature.bed_temperature = 100
        >>> params.speed.cf_print_feedrate = 800
        >>> gen = FibrifierGcodeGenerator(params=params)
    """
    # Geometry
    offset_x: float = 175.0            # Machine X offset (mm)
    offset_y: float = 135.0            # Machine Y offset (mm)
    layer_height: float = 0.15         # Layer height (mm)

    # Start-up
    skip_mesh_leveling: bool = False   # True = omit the G29 mesh-bed-leveling probe
                                       # at the start of the print

    # Layer structure
    polymer_perimeter: bool = True      # Include polymer perimeter layer (P) before each fiber layer
    infill_angle: float = None          # Infill angle for perimeter layers (deg). None=no infill
    infill_pitch: float = 0.8           # Infill line spacing (mm)
    infill_inset: float = 0.3           # Inset from contour boundary for infill region (mm)
    polymer_infill_layers: list = None  # Insert pure polymer infill layers after these fiber layer
                                        # indices (0-based). e.g. [0,2] = after 1st and 3rd fiber.
                                        # None = no extra infill layers.
    polymer_infill_angle: float = 45    # Infill angle for extra polymer infill layers (deg)

    # Grouped parameters
    temperature: FibrifierTemperatureParams = field(default_factory=FibrifierTemperatureParams)
    speed: FibrifierSpeedParams = field(default_factory=FibrifierSpeedParams)
    extrusion: FibrifierExtrusionParams = field(default_factory=FibrifierExtrusionParams)
    fiber: FibrifierFiberParams = field(default_factory=FibrifierFiberParams)
    retraction: FibrifierRetractionParams = field(default_factory=FibrifierRetractionParams)


# ══════════════════════════════════════════════════════════════════
# FibrifierLayer / FibrifierModel — High-level stacking API
# ══════════════════════════════════════════════════════════════════

class FibrifierLayer:
    """
    Defines a single layer in a Fibrifier print stack.

    Use factory methods to create layers:
        FibrifierLayer.polymer(svg, infill_angle=45)
        FibrifierLayer.fiber(svg, threshold=8.0)
        FibrifierLayer.infill(svg, angle=-45)

    Args for all factories:
        svg_path: Path to SVG file containing stripe + contour paths.

    The SVG is loaded lazily on first access (or eagerly via prepare()).
    """

    def __init__(self, svg_path: str, layer_type: str, layer_height: float = None, **kwargs):
        """
        Don't call directly — use factory methods instead.

        layer_type: 'P' (polymer), 'F' (fiber), 'PI' (infill-only polymer)
        layer_height: Layer thickness (mm). None = use FibrifierParams default.
        """
        self.svg_path = svg_path
        self.layer_type = layer_type
        self.layer_height = layer_height  # None means use params default
        self.kwargs = kwargs

        # Will be populated by prepare()
        self._fiber_paths = None
        self._contour_paths = None
        self._prepared = False

    @classmethod
    def polymer(cls, svg_path: str, infill_angle: float = None,
                infill_pitch: float = 0.8, infill_inset: float = 0.3,
                layer_height: float = None):
        """
        Polymer layer (contour outlines + optional infill).

        Args:
            svg_path: SVG file with contour paths.
            infill_angle: Angle for infill (deg). None = contour only, no infill.
            infill_pitch: Infill line spacing (mm).
            infill_inset: Inset from contour boundary (mm).
            layer_height: Layer thickness (mm). None = use params default.
        """
        return cls(svg_path, "P", layer_height=layer_height,
                   infill_angle=infill_angle,
                   infill_pitch=infill_pitch,
                   infill_inset=infill_inset)

    @classmethod
    def fiber(cls, svg_path: str, threshold: float = 8.0,
              min_length: float = 23.73, smooth_sigma: float = 3.0,
              decimate_epsilon: float = 0.05, layer_height: float = None):
        """
        Fiber layer (stripe paths with TSP ordering, smoothing, decimation).

        Args:
            svg_path: SVG file with stripe paths.
            threshold: Connection threshold for merging adjacent paths (mm).
            min_length: Minimum fiber length (mm). Shorter paths are dropped.
            smooth_sigma: Gaussian smoothing sigma.
            decimate_epsilon: RDP decimation tolerance (mm).
            layer_height: Layer thickness (mm). None = use params default.
        """
        return cls(svg_path, "F", layer_height=layer_height,
                   threshold=threshold,
                   min_length=min_length,
                   smooth_sigma=smooth_sigma,
                   decimate_epsilon=decimate_epsilon)

    @classmethod
    def infill(cls, svg_path: str, angle: float = 45,
               pitch: float = 0.8, inset: float = 0.3,
               layer_height: float = None):
        """
        Polymer infill-only layer (contour outlines + full infill).

        Args:
            svg_path: SVG file with contour paths.
            angle: Infill angle (deg).
            pitch: Infill line spacing (mm).
            inset: Inset from contour boundary (mm).
            layer_height: Layer thickness (mm). None = use params default.
        """
        return cls(svg_path, "PI", layer_height=layer_height,
                   infill_angle=angle,
                   infill_pitch=pitch,
                   infill_inset=inset)

    def prepare(self, flip_y: bool = True):
        """Load and process SVG paths. Called automatically by FibrifierModel."""
        if self._prepared:
            return

        fiber, contour, _ = load_and_prepare_paths(
            svg_path=self.svg_path,
            connection_threshold=self.kwargs.get("threshold", 8.0),
            min_fiber_length=self.kwargs.get("min_length", 23.73),
            flip_y=flip_y,
            smooth_sigma=self.kwargs.get("smooth_sigma", 3.0),
            decimate_epsilon=self.kwargs.get("decimate_epsilon", 0.05),
        )
        self._fiber_paths = fiber
        self._contour_paths = contour
        self._prepared = True

    def to_layer_dict(self):
        """Convert to the dict format expected by FibrifierGcodeGenerator."""
        if not self._prepared:
            raise RuntimeError("Call prepare() before to_layer_dict()")

        result = {"layer_height": self.layer_height}

        if self.layer_type == "F":
            result.update({"fiber": self._fiber_paths, "contour": [], "type": "F"})

        elif self.layer_type in ("P", "PI"):
            paths = list(self._contour_paths)
            infill_angle = self.kwargs.get("infill_angle")
            if infill_angle is not None:
                infill = generate_infill_paths(
                    self._contour_paths,
                    infill_angle,
                    self.kwargs.get("infill_pitch", 0.8),
                    self.kwargs.get("infill_inset", 0.3),
                )
                paths.extend(infill)
            result.update({"fiber": [], "contour": paths, "type": "P"})

        else:
            raise ValueError(f"Unknown layer_type: {self.layer_type}")

        return result

    @property
    def fiber_paths(self):
        return self._fiber_paths or []

    @property
    def contour_paths(self):
        return self._contour_paths or []

    def __repr__(self):
        label = {"P": "Polymer", "F": "Fiber", "PI": "Infill"}[self.layer_type]
        extra = ""
        if self.layer_type in ("P", "PI"):
            a = self.kwargs.get("infill_angle")
            if a is not None:
                extra = f", infill={a}°"
        elif self.layer_type == "F":
            extra = f", threshold={self.kwargs.get('threshold', 8.0)}mm"
        return f"FibrifierLayer({label}{extra}, svg={self.svg_path})"


class FibrifierModel:
    """
    Stack of FibrifierLayers for generating Fibrifier-format gcode.

    Example:
        >>> model = lb.FibrifierModel(params=params)
        >>> model.add(lb.FibrifierLayer.polymer(svg_10, infill_angle=45))
        >>> model.add(lb.FibrifierLayer.fiber(svg_10, threshold=8))
        >>> model.add(lb.FibrifierLayer.infill(svg_10, angle=-45))
        >>> model.add(lb.FibrifierLayer.polymer(svg_m10, infill_angle=45))
        >>> model.add(lb.FibrifierLayer.fiber(svg_m10, threshold=8))
        >>> result = model.generate("output.gcode")
    """

    def __init__(self, params: FibrifierParams = None):
        self.params = params or FibrifierParams()
        self.layers: list = []

    def add(self, layer: FibrifierLayer):
        """Add a layer to the stack (bottom-up order)."""
        self.layers.append(layer)
        return self  # allow chaining

    def generate(self, output_gcode: str, flip_y: bool = True,
                 preview_path: str = None):
        """
        Prepare all layers and generate gcode.

        Args:
            output_gcode: Output gcode file path.
            flip_y: Flip SVG Y-axis.
            preview_path: Preview image path. Auto-generated if None.

        Returns:
            dict with generation results.
        """
        # Prepare all layers, sharing results for same SVG + same params
        cache = {}  # (svg_path, threshold, min_len, sigma, eps) -> (fiber, contour)
        for layer in self.layers:
            key = (layer.svg_path,
                   layer.kwargs.get("threshold", 8.0),
                   layer.kwargs.get("min_length", 23.73),
                   layer.kwargs.get("smooth_sigma", 3.0),
                   layer.kwargs.get("decimate_epsilon", 0.05))
            if key in cache:
                layer._fiber_paths, layer._contour_paths = cache[key]
                layer._prepared = True
            else:
                layer.prepare(flip_y=flip_y)
                cache[key] = (layer._fiber_paths, layer._contour_paths)

        # Convert to generator format
        layer_dicts = [layer.to_layer_dict() for layer in self.layers]

        # Collect paths for preview
        all_fiber = []
        all_contour = []
        for layer in self.layers:
            if layer.layer_type == "F":
                all_fiber.extend(layer.fiber_paths)
            else:
                all_contour.extend(layer.contour_paths)

        # Preview
        if preview_path is None:
            preview_path = output_gcode.replace(".gcode", "_preview.png")
        fig, _ = plot_paths(all_fiber, all_contour,
                            title=f"{len(self.layers)}-layer stack "
                                  f"({len(all_fiber)} fiber, {len(all_contour)} contour)")
        fig.savefig(preview_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"Plot saved to {preview_path}")

        # Generate gcode
        gen = FibrifierGcodeGenerator(params=self.params)
        gen.generate_multilayer(layer_dicts, filename=output_gcode)

        total_fiber = sum(p.length for p in all_fiber)
        return {
            "gcode_path": output_gcode,
            "preview_path": preview_path,
            "n_layers": len(self.layers),
            "n_fiber_paths": len(all_fiber),
            "n_contour_paths": len(all_contour),
            "total_fiber_mm": total_fiber,
            "total_stretches": gen._stretch_counter,
        }

    def summary(self):
        """Print a summary of the layer stack."""
        print(f"FibrifierModel: {len(self.layers)} layers")
        z = 0.0
        for i, layer in enumerate(self.layers):
            lh = layer.layer_height or self.params.layer_height
            z = round(z + lh, 4)
            h_str = f" (h={lh})" if layer.layer_height else ""
            print(f"  [{i}] z={z:.2f}mm{h_str}  {layer}")

    def __len__(self):
        return len(self.layers)

    def __repr__(self):
        return f"FibrifierModel({len(self.layers)} layers)"


# ══════════════════════════════════════════════════════════════════
# Gcode generation — Fibrifier-compatible format
# ══════════════════════════════════════════════════════════════════

class FibrifierGcodeGenerator:
    """Generate gcode in Fibrifier (9T Labs) format."""

    def __init__(self, params: FibrifierParams = None, **kwargs):
        """
        Initialize generator.

        Args:
            params: FibrifierParams dataclass. If None, uses defaults.
            **kwargs: Override individual params fields
                      (offset_x, offset_y, layer_height, etc.)
        """
        if params is None:
            params = FibrifierParams()

        # Apply any top-level overrides
        for key in ("offset_x", "offset_y", "layer_height"):
            if key in kwargs:
                setattr(params, key, kwargs.pop(key))

        self.params = params
        p = params

        # Shortcuts for internal use
        self.offset_x = p.offset_x
        self.offset_y = p.offset_y
        self.layer_height = p.layer_height
        self.skip_mesh_leveling = p.skip_mesh_leveling

        self.cf_em = p.extrusion.cf_extrusion_multiplier
        self.pl_flow = p.extrusion.pl_extrusion_multiplier
        self.filament_d = p.extrusion.filament_diameter
        self.polymer_line_width = p.extrusion.polymer_line_width
        self.cf_feed = p.speed.cf_print_feedrate
        self.pl_em = p.extrusion.pl_extrusion_multiplier
        self.pl_temp = p.temperature.pl_print_temp
        self.cf_temp = p.temperature.cf_print_temp
        self.nozzle_dead_length = p.fiber.nozzle_dead_length
        self.nozzle_cooldown_temp = p.temperature.nozzle_cooldown_temperature
        self.anchor_dwell = p.fiber.anchoring_dwell
        self.anchor_height = p.fiber.anchoring_height
        self.after_cut_speed = p.speed.after_cut_speed
        self.no_speed_up_length = p.fiber.no_speed_up_length
        self.rapid_feed = p.speed.rapid_move_feedrate
        self.perimeter_speed = p.speed.perimeter_speed
        self.pl_perimeter_feed = p.speed.pl_perimeter_feedrate
        self.pl_infill_feed = p.speed.pl_infill_feedrate
        self.pl_first_layer_feed = p.speed.pl_first_layer_feedrate
        self.pl_first_layers = p.speed.pl_first_layers
        self.bed_temp = p.temperature.bed_temperature
        self.chamber_temp = p.temperature.build_chamber_temperature
        self.material_temp = p.temperature.material_storage_temperature
        self.retract_length = p.retraction.retract_length
        self.retract_speed = p.retraction.retract_speed
        self.min_length = p.fiber.minimal_printable_length
        self.fiber_cut = p.fiber.fiber_cut
        self.manual_cut = p.fiber.manual_cut
        self.manual_cut_lift = p.fiber.manual_cut_lift
        self.manual_cut_extrude = p.fiber.manual_cut_extrude
        self.uturn_escape = p.fiber.uturn_escape
        self.uturn_lift = p.fiber.uturn_lift
        self.uturn_extrude = p.fiber.uturn_extrude
        # State carried between stretches for re-anchoring.
        self._last_end_xy = None        # (x, y) world coords of previous stretch end
        self._last_end_angle = None     # heading at the end of the previous stretch
        self._last_z = None             # Z of the previous stretch (for same-Z check)

        # Derived
        self.after_cut_feed = int(self.after_cut_speed / 100.0 * self.cf_feed)

        # Global stretch counter
        self._stretch_counter = 0
        # Whether any fibre stretch has been laid yet (continuous-fibre mode).
        self._fiber_started = False

    def _tx(self, x):
        return x + self.offset_x

    def _ty(self, y):
        return y + self.offset_y

    def _polymer_e(self, dist, layer_h):
        """Volumetric polymer extrusion E for a move of length ``dist``.

        ``E = dist * h * w / (pi/4 * d_filament**2) * flow`` — the filament
        length whose volume equals the deposited bead (height h, width w).
        """
        area = (math.pi / 4.0) * self.filament_d ** 2
        return dist * layer_h * self.polymer_line_width / area * self.pl_flow

    def _next_stretch(self):
        idx = self._stretch_counter
        self._stretch_counter += 1
        return idx

    def generate_multilayer(self, layers, filename="output.gcode"):
        """
        layers: list of dicts with keys:
            'fiber': list of Path (stripe)
            'contour': list of Path (contour)
            'type': 'P' (polymer) or 'F' (fiber) — auto-detected if missing
        Alternating P/F layers like the reference: P, F, P, F, ...
        """
        self._stretch_counter = 0
        self._fiber_started = False
        self._last_end_xy = None
        self._last_end_angle = None
        self._last_z = None

        with open(filename, "w") as f:
            # Compute bounding box in machine coords
            all_x, all_y = [], []
            for ld in layers:
                for plist in [ld.get("fiber", []), ld.get("contour", [])]:
                    for p in plist:
                        for x, y in p.nodes:
                            all_x.append(self._tx(x))
                            all_y.append(self._ty(y))
            if all_x:
                bbox = (min(all_x), min(all_y), max(all_x), max(all_y))
            else:
                bbox = (0, 0, 0, 0)

            n_layers = len(layers)
            self._write_header(f, bbox=bbox, n_layers=n_layers)
            z = 0.0
            for layer_idx, layer_data in enumerate(layers):
                # Honor an explicit per-layer height, including 0.0 (a layer
                # printed coplanar with the previous one, e.g. fibre laid in
                # the same Z as its polymer matrix).
                lh = layer_data.get("layer_height")
                lh = self.layer_height if lh is None else lh
                z = round(z + lh, 4)
                fiber_paths = layer_data.get("fiber", [])
                contour_paths = layer_data.get("contour", [])
                layer_type = layer_data.get("type", None)

                # Auto-detect type if not specified
                if layer_type is None:
                    layer_type = "F" if fiber_paths else "P"

                f.write(f";========================\n")
                f.write(f"; - START OF ZCHUNK #{layer_idx + 1}, "
                        f"range=[{z:.2f}, {z:.2f}] -\n")
                f.write(f";========================\n")

                if layer_type == "P":
                    self._write_polymer_layer(f, z, layer_idx, contour_paths, layers, lh)
                elif layer_type == "F":
                    self._write_fiber_layer(f, z, layer_idx, fiber_paths, layers)

                # After first layer
                if layer_idx == 0:
                    f.write(f";- after first layer -\n")
                    f.write(f"M104 S{self.cf_temp} T0\n")
                    f.write(f"M140 S{self.bed_temp} ; wait for bed temp\n")
                    f.write(f";- after first layer -\n")

                f.write(f";========================\n")
                f.write(f"; - END OF ZCHUNK #{layer_idx + 1}, "
                        f"range=[{z:.2f}, {z:.2f}] -\n")
                f.write(f";========================\n")

            self._write_footer(f)

        total_fiber = sum(
            sum(p.length for p in ld.get("fiber", []))
            for ld in layers
        )
        print(f"[{filename}] generated: {n_layers} layers, "
              f"{self._stretch_counter} stretches, "
              f"total fiber={total_fiber:.1f}mm")

    # ── Header ──────────────────────────────────────────────

    def _write_header(self, f, bbox=None, n_layers=1):
        now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
        zmax = round(n_layers * self.layer_height, 3)

        # Bounding box (machine coords)
        if bbox:
            xmin, ymin, xmax, ymax = bbox
        else:
            xmin = ymin = xmax = ymax = 0.0

        f.write(f"; 9Code-Version = 1.0\n")
        f.write(f"; Creation-Date = {now}\n")
        f.write(f"; Creator = fiber_gcode_tool.py\n")
        f.write(f"; Slicing-Engine = Custom Python Script\n")
        f.write(f"\n")

        # Parts block
        f.write(f"; -- Parts --\n")
        f.write(f"; generated_part\n")
        f.write(f"; xmin = {xmin:.3f}\n")
        f.write(f"; xmax = {xmax:.3f}\n")
        f.write(f"; ymin = {ymin:.3f}\n")
        f.write(f"; ymax = {ymax:.3f}\n")
        f.write(f"; zmin = 0.000\n")
        f.write(f"; zmax = {zmax:.3f}\n")
        f.write(f"; -- End Parts --\n")
        f.write(f";\n")

        # Full Fibrifier config (all keys from reference, alphabetical order within groups)
        f.write(f";-- Fibrifier Config --\n")
        f.write(f"; bed_shape = 0x0,350x0,350x280,0x280\n")
        f.write(f"; fiber_helix = 0\n")
        f.write(f"; first_layers_at_once = 0\n")
        f.write(f"; gcode_stubs_dir = stubs\n")
        f.write(f"; after_cut_speed = {self.after_cut_speed}\n")
        f.write(f"; anchoring_angle = 9\n")
        f.write(f"; anchoring_dwell = {self.anchor_dwell}\n")
        f.write(f"; anchoring_height = {self.anchor_height}\n")
        f.write(f"; anchoring_length = 2\n")
        f.write(f"; bed_temperature = {self.bed_temp}\n")
        f.write(f"; build_chamber_temperature = {self.chamber_temp}\n")
        f.write(f"; cf_extrusion_multiplier = {self.cf_em}\n")
        f.write(f"; cf_first_layer_temperature = {self.cf_temp}\n")
        f.write(f"; cf_print_feedrate = {self.cf_feed}\n")
        f.write(f"; cf_print_temp = {self.cf_temp}\n")
        f.write(f"; corner_direction_correction_angle_treshold = 30\n")
        f.write(f"; corner_dwell = 0\n")
        f.write(f"; corner_nozzle_lift_slack = 0\n")
        f.write(f"; corner_nozzle_z_lift = 0\n")
        f.write(f"; corner_overshoot_length = 0\n")
        f.write(f"; corner_slack_ratio = 0\n")
        f.write(f"; deposition_offset = 0\n")
        f.write(f"; fiber_height = {self.layer_height}\n")
        f.write(f"; fiber_helix_z_transition = 10\n")
        f.write(f"; fiber_max_extrusion_radius = 100\n")
        f.write(f"; fiber_min_extrusion_radius = 8\n")
        f.write(f"; fiber_min_radius_extrusion_multiplier = {self.cf_em}\n")
        f.write(f"; fiber_width = 1.5\n")
        f.write(f"; first_layer_bed_temperature = {self.bed_temp}\n")
        f.write(f"; first_layer_extrusion_multiplier = 1\n")
        f.write(f"; layer_height = {self.layer_height}\n")
        f.write(f"; material_storage_temperature = {self.material_temp}\n")
        f.write(f"; minimal_extrusion_plastic_stretch = 0.02\n")
        f.write(f"; minimal_printable_length = {self.min_length}\n")
        f.write(f"; no_speed_up_length = {int(self.no_speed_up_length)}\n")
        f.write(f"; nozzle_cooldown_temperature = {self.nozzle_cooldown_temp}\n")
        f.write(f"; nozzle_dead_length = {self.nozzle_dead_length}\n")
        f.write(f"; perimeter_speed = {self.perimeter_speed}\n")
        f.write(f"; pl_extrusion_multiplier = {self.pl_em}\n")
        f.write(f"; pl_first_layer_speed = 30\n")
        f.write(f"; pl_first_layer_temperature = {self.pl_temp}\n")
        f.write(f"; pl_print_temp = {self.pl_temp}\n")
        f.write(f"; plastic_cover_extrusion_multipier = 0.85\n")
        f.write(f"; plastic_cover_feedrate = 60\n")
        f.write(f"; preheat_target_temperature = {self.pl_temp}\n")
        f.write(f"; preheat_time = 60\n")
        f.write(f"; rapid_move_feedrate = {self.rapid_feed}\n")
        f.write(f"; retract_length = {int(self.retract_length)}\n")
        f.write(f"; retract_lift = 0\n")
        f.write(f"; retract_speed = {self.retract_speed}\n")
        f.write(f"; straight_line_feedrate = {self.cf_feed}\n")
        f.write(f"; stretch_beginning_length = 0\n")
        f.write(f"; stretch_beginning_multiplier = 1.2\n")
        f.write(f";-- End Fibrifier Config --\n")
        f.write(f"\n")

        # Start code — matches reference exactly
        f.write(f";*-- START CODE --\n")
        f.write(f"CLEAR_PAUSE ; clear any saved pause states\n")
        f.write(f"\n")
        f.write(f"G21 ; set units to millimeters\n")
        f.write(f"G90 ; use absolute coordinates\n")
        f.write(f"M83 ; extruder relative mode\n")
        f.write(f"\n")
        f.write(f"SET_PIN PIN=lamps VALUE=0.7\n")
        f.write(f"\n")
        f.write(f"M104 S130 T0 ; set fiber extruder temperature\n")
        f.write(f"M140 S{self.bed_temp} ; set bed temp\n")
        f.write(f"\n")
        f.write(f"G28 ; Home all axes\n")
        f.write(f"\n")
        f.write(f"M109 S130 T0 ; set fiber extruder temp\n")
        f.write(f"G28 W ; re-home FG after it has heated up\n")
        f.write(f"\n")
        f.write(f"M190 S{self.bed_temp} ; wait for bed temp\n")
        f.write(f"M104 S{self.cf_temp} T0 ; set fiber extruder temp\n")
        f.write(f"\n")
        pl_75 = int(self.pl_temp * 0.75)
        f.write(f"M104 S{pl_75} T1 ; set 75% polymer nozzle temp\n")
        if not self.skip_mesh_leveling:
            f.write(f"G29 ; Meshbed leveling\n")
        f.write(f"\n")
        f.write(f"G0 Z3.0000 F{self.rapid_feed}\n")
        f.write(f"\n")
        f.write(f"M109 S{self.pl_temp} T1 ; wait for first layer polymer temeprature\n")
        f.write(f"M109 S{self.cf_temp} T0 ; wait for FG printing temperature\n")
        f.write(f"\n")
        f.write(f"T1 ; change to polymer extruder\n")
        f.write(f"\n")
        f.write(f"G0 X100.0000 Y3.0000 Z0.5000 F{self.rapid_feed} ; go outside print area\n")
        f.write(f"G1 X60.0000 E9.0000 F1000 ; intro line\n")
        f.write(f"G1 X20.0000 E12.5000 F1000 ; intro line\n")
        f.write(f"G92 E0.0\n")
        f.write(f"\n")
        f.write(f"SET_RETRACTION RETRACT_LENGTH={self.retract_length:.1f} "
                f"RETRACT_SPEED={float(self.retract_speed):.1f}\n")
        f.write(f"SET_HEATER_TEMPERATURE HEATER=print_chamber "
                f"target={float(self.chamber_temp):.1f}\n")
        f.write(f"SET_HEATER_TEMPERATURE HEATER=material_chamber "
                f"target={float(self.material_temp):.1f}\n")
        f.write(f";*-- END OF START CODE --\n")

    # ── Polymer layer ───────────────────────────────────────

    def _write_polymer_layer(self, f, z, layer_idx, contour_paths, all_layers,
                             layer_h=None):
        """Write a full polymer layer (perimeters + infill)."""
        if layer_h is None:
            layer_h = self.layer_height

        # If previous layer was fiber, do Fiber→Polymer transition
        if layer_idx > 0:
            prev_type = all_layers[layer_idx - 1].get("type", "F")
            if prev_type == "F":
                self._write_fiber_to_polymer_init(f, z)

        is_first = (layer_idx == 0)
        # "Slow" feedrate for the first ``pl_first_layers`` layers (e.g. set to
        # 2 so the 2nd layer is also slowed); faster feedrates afterwards.
        slow = layer_idx < self.pl_first_layers
        perimeter_f = self.pl_first_layer_feed if slow else self.pl_perimeter_feed
        infill_f = self.pl_first_layer_feed if slow else self.pl_infill_feed
        z_lift = z + 0.35 if is_first else z

        # Separate perimeters (closed contours) from infill paths (open zig-zag)
        perimeter_paths = [p for p in contour_paths if p.is_closed]
        infill_paths = [p for p in contour_paths if not p.is_closed]

        # --- Perimeters ---
        for cidx, cp in enumerate(perimeter_paths):
            stretch_id = self._next_stretch()
            nodes = cp.nodes
            sx, sy = self._tx(nodes[0][0]), self._ty(nodes[0][1])

            f.write(f";- START OF POLYMER STRETCH INITIALIZATION -\n")
            if cidx == 0 and is_first:
                f.write(f"G10\n")
            f.write(f"G0 Z{z_lift:.4f} F{self.rapid_feed} ; lift Z\n")
            f.write(f"G0 X{sx:.4f} Y{sy:.4f}\n")
            f.write(f"G0 Z{z:.4f} ; restore layer Z\n")
            f.write(f"G1 F{perimeter_f}\n")
            if cidx == 0:
                f.write(f"G11\n")
            f.write(f";- END OF POLYMER STRETCH INITIALIZATION -\n")

            f.write(f";------------------------\n")
            f.write(f"; - START OF POLYMER STRETCH #{stretch_id} -\n")
            f.write(f";------------------------\n")

            prev = nodes[0]
            for nx, ny in nodes[1:]:
                dx = nx - prev[0]
                dy = ny - prev[1]
                dist = math.sqrt(dx * dx + dy * dy)
                if dist > 1e-5:
                    e_val = self._polymer_e(dist, layer_h)
                    f.write(f"G1 X{self._tx(nx):.4f} Y{self._ty(ny):.4f} "
                            f"E{e_val:.4f} ; perimeter\n")
                prev = (nx, ny)

            f.write(f";------------------------\n")
            f.write(f"; - END OF POLYMER STRETCH #{stretch_id} -\n")
            f.write(f";------------------------\n")

        # --- Infill paths ---
        for iidx, ip in enumerate(infill_paths):
            stretch_id = self._next_stretch()
            nodes = ip.nodes
            sx, sy = self._tx(nodes[0][0]), self._ty(nodes[0][1])

            # Infill initialization: retract, Z-lift, move to start, unretract
            f.write(f";- START OF POLYMER STRETCH INITIALIZATION -\n")
            f.write(f"G10\n")
            f.write(f"G0 Z{z_lift:.4f} F{self.rapid_feed}\n")
            f.write(f"G0 X{sx:.4f} Y{sy:.4f}\n")
            f.write(f"G0 Z{z:.4f}\n")
            f.write(f"G1 F{infill_f}\n")
            f.write(f"G11\n")
            f.write(f";- END OF POLYMER STRETCH INITIALIZATION -\n")

            f.write(f";------------------------\n")
            f.write(f"; - START OF POLYMER STRETCH #{stretch_id} (Infill) -\n")
            f.write(f";------------------------\n")

            # Continuous extrusion: scan lines + step moves, all with E
            prev = nodes[0]
            is_scan = True  # alternates: scan line, step move, scan line, ...
            for nx, ny in nodes[1:]:
                dx = nx - prev[0]
                dy = ny - prev[1]
                dist = math.sqrt(dx * dx + dy * dy)
                if dist > 1e-5:
                    e_val = self._polymer_e(dist, layer_h)
                    label = "infill" if is_scan else "infill step"
                    f.write(f"G1 X{self._tx(nx):.4f} Y{self._ty(ny):.4f} "
                            f"E{e_val:.4f} ; {label}\n")
                is_scan = not is_scan
                prev = (nx, ny)

            f.write(f";------------------------\n")
            f.write(f"; - END OF POLYMER STRETCH #{stretch_id} -\n")
            f.write(f";------------------------\n")

    def _write_fiber_to_polymer_init(self, f, z):
        """Fiber→Polymer transition (matches Fibrifier format exactly)."""
        f.write(f";- START OF POLYMER STRETCH INITIALIZATION -\n")
        f.write(f"G10 S1\n")
        f.write(f"; nozzle change gcode for switching from the fiber guide (T0) "
                f"to the polymer nozzle(T1)\n")
        f.write(f"M104 S{self.pl_temp} T1 ; set polymer nozzle temperature\n")
        f.write(f"SET_STATE_TARGET STATE=buffer TARGET=0\n")
        f.write(f"\n")
        f.write(f"G91 ; use relative coordinates\n")
        f.write(f"G0 Z3.0000 F{self.rapid_feed} ; sits currently at heigth +3 over layer\n")
        f.write(f"G90 ; use absolute coordinates\n")
        f.write(f"M83 ; extruder relative mode\n")
        f.write(f"\n")
        f.write(f"M109 S{self.pl_temp} T1 ; wait for polymer nozzle temperature\n")
        f.write(f"M104 S{self.pl_temp} T1 ; set polymer nozzle temperature\n")
        f.write(f"T1 ; change nozzle\n")
        f.write(f"; end of T1 for Shoed-Red-Series\n")

    # ── Fiber layer ─────────────────────────────────────────

    def _write_fiber_layer(self, f, z, layer_idx, fiber_paths, all_layers):
        """Write a full fiber layer."""
        f.write(f"G0 Z{z:.4f} ; to next layer transition\n")

        # Polymer→Fiber transition (tool change T1→T0) only when coming from a
        # non-fibre layer.  Consecutive fibre layers stay on the fibre guide
        # (T0) — no redundant tool change / polymer-nozzle cooldown.
        if layer_idx > 0:
            prev = all_layers[layer_idx - 1]
            prev_type = prev.get("type") or ("F" if prev.get("fiber") else "P")
        else:
            prev_type = None
        if prev_type != "F":
            self._write_polymer_to_fiber_init(f, z)

        n_passes = len(fiber_paths)
        for pidx, fp in enumerate(fiber_paths):
            is_last_pass_of_last_fiber_before_polymer = False
            # Check if polymer follows: preheat polymer nozzle during last passes
            next_layer_type = None
            if layer_idx + 1 < len(all_layers):
                next_layer_type = all_layers[layer_idx + 1].get("type", "P")

            self._write_fiber_stretch(
                f, fp, z, pidx, n_passes,
                preheat_polymer=(next_layer_type == "P"),
                is_last_fiber_layer_pass=(pidx >= n_passes - 2),
            )

        # Manual-cut mode: end the fibre layer by lifting the nozzle in place
        # (dispensing fibre so a tail hangs out), then pause for the operator to
        # cut the tow by hand and resume.  The severed tow means the next fibre
        # layer must re-anchor, so clear the continuous-fibre flag.
        if self.manual_cut and not self.fiber_cut:
            self._write_manual_cut_pause(f, z)
            self._fiber_started = False
            # Manual cut severs the tow → the next stretch is a fresh anchor.
            self._last_end_xy = None
            self._last_end_angle = None
            self._last_z = None

    def _write_manual_cut_pause(self, f, z):
        """Lift the nozzle while extruding fibre, then pause for a hand cut."""
        e = self.manual_cut_extrude * self.cf_em   # apply fibre feed coefficient
        f.write(";------------------------\n")
        f.write("; - MANUAL FIBRE CUT (lift + pause) -\n")
        f.write(";------------------------\n")
        f.write("G91 ; relative coordinates\n")
        f.write(f"G1 Z{self.manual_cut_lift:.4f} E{e:.4f} "
                f"F{self.cf_feed} ; raise nozzle while dispensing fibre\n")
        f.write("G90 ; absolute coordinates\n")
        f.write("M0 ; PAUSE — cut the fibre by hand, then press resume\n")

    def _write_polymer_to_fiber_init(self, f, z):
        """Polymer→Fiber transition (matches Fibrifier format exactly)."""
        f.write(f";- START OF Fiber STRETCH INITIALIZATION -\n")
        f.write(f"G10 S1\n")
        f.write(f"; nozzle change gcode for switching from the polymer nozzle(T1) "
                f"to the fiber guide (T0)\n")
        f.write(f"M104 S{self.nozzle_cooldown_temp} T1 ; cool down polymer nozzle\n")
        f.write(f"G10\n")
        f.write(f"G91 ; use relative coordinates\n")
        f.write(f"G0 Z3.0000 F{self.rapid_feed}\n")
        f.write(f"G90 ; use absolute coordinates\n")
        f.write(f"M83 ; extruder relative mode\n")
        f.write(f"\n")
        f.write(f"T0 ; change to fiber guide\n")
        f.write(f"SET_STATE_TARGET STATE=buffer TARGET={z:.1f}\n")
        f.write(f"; end of T0 for Shoed-Red-Series\n")
        f.write(f";- END OF Fiber STRETCH INITIALIZATION -\n")

    def _write_fiber_stretch(self, f, path, z, pass_idx, total_passes,
                              preheat_polymer=False, is_last_fiber_layer_pass=False):
        stretch_id = self._next_stretch()
        nodes = path.nodes
        n = len(nodes)

        f.write(f";------------------------\n")
        f.write(f"; - START OF Fiber STRETCH #{stretch_id} -\n")
        f.write(f";------------------------\n")

        # Compute cut position.  With fibre cutting disabled, set the cut index
        # past the end so the loop never cuts and the whole path is extruded
        # (continuous fibre, no severing between stretches).
        if self.fiber_cut:
            cut_idx, cut_point = _find_cut_index(nodes, self.nozzle_dead_length)
        else:
            cut_idx, cut_point = n, None

        # Compute smoothed direction angles
        angles = _compute_smooth_angles(nodes)

        # Anchor point — use raw first segment angle for approach direction
        raw_start_angle = _segment_angle_deg(
            nodes[0][0], nodes[0][1], nodes[1][0], nodes[1][1])
        anchor = _anchor_point(nodes, self.nozzle_dead_length)

        if (not self.fiber_cut) and self._fiber_started:
            # Continuous fibre — tow uncut from previous stretch.
            #
            # Same-Z (in-layer pass-to-pass) and uturn_escape > 0:
            #   ESCAPE in last segment's direction (forward + lift, dispensing
            #   tow), then APPROACH new anchor (rapid, W rotates absolute to the
            #   new heading — this absorbs the 180°), DESCEND onto new start
            #   dragging tow, DWELL.  Mirrors the initial-anchor sequence but
            #   skips the standalone "G0 E23.5 load" since we dispensed during
            #   the escape.
            #
            # Different Z (Z-chunk / layer climb) or uturn_escape = 0:
            #   The original "climb to next layer at start point" — single G1
            #   to the new (X, Y, Z) with W rotated absolute to the new heading.
            same_z = (self._last_z is not None
                       and abs(z - self._last_z) < 1e-5)
            do_reanchor = (self.uturn_escape > 0.0
                            and same_z
                            and self._last_end_xy is not None
                            and self._last_end_angle is not None)
            if do_reanchor:
                ang = math.radians(self._last_end_angle)
                esc_x = self._last_end_xy[0] + self.uturn_escape * math.cos(ang)
                esc_y = self._last_end_xy[1] + self.uturn_escape * math.sin(ang)
                esc_z = self._last_z + self.uturn_lift
                e_esc = self.uturn_extrude * self.cf_em
                f.write(";------------------------\n")
                f.write(f"; - U-TURN RE-ANCHOR "
                        f"(escape {self.uturn_escape:.2f}mm fwd, "
                        f"lift {self.uturn_lift:.2f}mm, "
                        f"dispense {self.uturn_extrude:.2f}mm fibre) -\n")
                f.write(";------------------------\n")
                # ── 1. escape: lift + forward + dispense in last direction ──
                f.write(f"G1 X{self._tx(esc_x):.4f} Y{self._ty(esc_y):.4f} "
                        f"Z{esc_z:.4f} E{e_esc:.4f} F{self.cf_feed} "
                        f";escape in last direction while lifting & dispensing\n")
                # ── 2. approach new anchor — W absolute = new heading ──
                f.write(f"USE_ABSOLUTE_ROTARY_POSITION\n")
                f.write(f"G0 X{self._tx(anchor[0]):.4f} Y{self._ty(anchor[1]):.4f} "
                        f"Z{z + self.anchor_height:.4f} W{raw_start_angle:.4f} "
                        f"F{self.rapid_feed} ;rapid to new anchor (W rotates absolute)\n")
                f.write(f"USE_RELATIVE_ROTARY_POSITION\n")
                # ── 3. descend onto start, drag fibre, dwell ──
                f.write(f"G1 X{self._tx(nodes[0][0]):.4f} Y{self._ty(nodes[0][1]):.4f} "
                        f"Z{z:.4f} F{self.cf_feed} ;descend onto start (drag fibre in)\n")
                f.write(f"G4 P{self.anchor_dwell} ;re-anchor dwell\n")
            else:
                # Simple climb (Z-chunk transition, or feature disabled).
                f.write(f"USE_ABSOLUTE_ROTARY_POSITION\n")
                f.write(f"G1 X{self._tx(nodes[0][0]):.4f} Y{self._ty(nodes[0][1]):.4f} "
                        f"Z{z:.4f} W{raw_start_angle:.4f} F{self.cf_feed} "
                        f";continuous fibre — climb to next layer\n")
                f.write(f"USE_RELATIVE_ROTARY_POSITION\n")
        else:
            # Move to anchor position with correct nozzle orientation
            f.write(f"USE_ABSOLUTE_ROTARY_POSITION\n")
            f.write(f"G0 X{self._tx(anchor[0]):.4f} Y{self._ty(anchor[1]):.4f} "
                    f"Z{z + self.anchor_height:.4f} W{raw_start_angle:.4f} "
                    f"F{self.rapid_feed} ;move to anchoring start\n")
            f.write(f"USE_RELATIVE_ROTARY_POSITION\n")
            f.write(f"G0 E23.5000 F{self.cf_feed}\n")

            # Move to start
            f.write(f"G1 X{self._tx(nodes[0][0]):.4f} Y{self._ty(nodes[0][1]):.4f} "
                    f"Z{z:.4f} F{self.cf_feed}\n")
            f.write(f"G4 P{self.anchor_dwell} ;anchoring dwell\n")
        self._fiber_started = True

        # Lay fiber with rotation and extrusion
        # Track absolute nozzle angle to correctly handle turns
        current_abs_angle = raw_start_angle
        prev = nodes[0]
        cut_emitted = False

        # Threshold for in-place turn (degrees)
        IN_PLACE_TURN_THRESHOLD = 30.0

        # Compute cumulative distance from end for preheat scheduling
        dist_from_end = _cumulative_distances_from_end(nodes)

        for i in range(1, n):
            x, y = nodes[i]
            # angles[i-1] = direction of segment (i-1) → i (the segment we're about to travel)
            target_angle = angles[i - 1]

            # Insert cut point
            if not cut_emitted and i == cut_idx + 1 and cut_point != nodes[cut_idx]:
                cx, cy = cut_point
                dist = math.sqrt((cx - prev[0])**2 + (cy - prev[1])**2)
                if dist > 1e-5:
                    e_val = dist * self.cf_em
                    w_delta = _wrap_angle(target_angle - current_abs_angle)
                    w_cmd = f" W{w_delta:.4f}" if abs(w_delta) > 0.01 else ""
                    f.write(f"G1 X{self._tx(cx):.4f} Y{self._ty(cy):.4f} "
                            f"Z{z:.4f} E{e_val:.4f}{w_cmd} F{self.cf_feed}\n")
                    prev = (cx, cy)
                    current_abs_angle = target_angle
                f.write(f"cut_filament\n")
                cut_emitted = True

            # Check if we need to emit cut right at this node
            if not cut_emitted and i > cut_idx:
                f.write(f"cut_filament\n")
                cut_emitted = True

            dx = x - prev[0]
            dy = y - prev[1]
            dist = math.sqrt(dx * dx + dy * dy)
            if dist < 1e-5:
                prev = (x, y)
                continue

            w_delta = _wrap_angle(target_angle - current_abs_angle)
            past_cut = cut_emitted

            if past_cut:
                # After cut: no extrusion, follow path for pull-through
                # Use in-place turn if needed, then move with W=0
                if abs(w_delta) > IN_PLACE_TURN_THRESHOLD:
                    f.write(f"G1 W{w_delta:.4f} ;in place turn\n")
                    current_abs_angle = target_angle
                f.write(f"G1 X{self._tx(x):.4f} Y{self._ty(y):.4f} "
                        f"Z{z:.4f} W0.0000 "
                        f"F{self.after_cut_feed}\n")
            else:
                # Before cut: extrude with rotation
                if abs(w_delta) > IN_PLACE_TURN_THRESHOLD:
                    # Large direction change — in-place turn first (ref.py style)
                    f.write(f"G1 W{w_delta:.4f} ;in place turn\n")
                    current_abs_angle = target_angle
                    # Then move with W=0
                    e_val = dist * self.cf_em
                    f.write(f"G1 X{self._tx(x):.4f} Y{self._ty(y):.4f} "
                            f"Z{z:.4f} W0.0000 E{e_val:.4f} F{self.cf_feed}\n")
                else:
                    # Small/gradual direction change — rotate while moving
                    e_val = dist * self.cf_em
                    w_cmd = f" W{w_delta:.4f}" if abs(w_delta) > 0.01 else ""
                    f.write(f"G1 X{self._tx(x):.4f} Y{self._ty(y):.4f} "
                            f"Z{z:.4f} E{e_val:.4f}{w_cmd} F{self.cf_feed}\n")
                    current_abs_angle = target_angle

            prev = (x, y)

        if self.fiber_cut:
            if not cut_emitted:
                f.write(f"cut_filament\n")

            # Sum of skipped extrusion comment (matches reference)
            f.write(f"; sum of skipped extrusion after the cut: {self.nozzle_dead_length}\n")

            # Ironing out move
            iron = _ironing_point(nodes, self.no_speed_up_length)
            f.write(f"G0 X{self._tx(iron[0]):.4f} Y{self._ty(iron[1]):.4f} "
                    f"Z{z:.4f} W0.0000 F{self.after_cut_feed} ;Ironing out move\n")

        f.write(f";------------------------\n")
        f.write(f"; - END OF Fiber STRETCH #{stretch_id} -\n")
        f.write(f";------------------------\n")

        # Record state so the NEXT stretch's continuous-fibre re-anchor knows
        # where the tow currently is and which direction it was heading.  Use
        # the raw last-segment angle (not the smoothed one) so the escape goes
        # along the physical direction of the final move.
        if n >= 2:
            self._last_end_xy = (nodes[-1][0], nodes[-1][1])
            self._last_end_angle = _segment_angle_deg(
                nodes[-2][0], nodes[-2][1], nodes[-1][0], nodes[-1][1])
            self._last_z = z

    # ── Footer ──────────────────────────────────────────────

    def _write_footer(self, f):
        f.write(f";*-- END CODE --\n")
        f.write(f"G90\n")
        f.write(f"M83\n")
        f.write(f"USE_ABSOLUTE_ROTARY_POSITION\n")
        f.write(f"G1 Z100.0000 W0.0000 F6000 ; move to anchoring start\n")
        f.write(f"USE_RELATIVE_ROTARY_POSITION\n")
        f.write(f"M104 S0 T0 ; turn off fiber guide\n")
        f.write(f"M104 S0 T1 ; turn off polymer nozzle\n")
        f.write(f"M140 S0 ; turn off bed\n")
        f.write(f"set_state_target state=buffer target=0 ; turn off buffer\n")
        f.write(f"set_heater_temperature heater=print_chamber target=0 "
                f"; turn off chamber heater\n")
        f.write(f";*-- END OF END CODE --\n")
        f.write(f"\n")


def _wrap_angle(a):
    """Wrap angle to [-180, 180]."""
    return (a + 180) % 360 - 180


# ══════════════════════════════════════════════════════════════════
# Visualisation
# ══════════════════════════════════════════════════════════════════

def plot_paths(fiber_paths, contour_paths, title="Connected Fiber Paths",
               arrow_interval=0, arrow_length=None):
    """
    Plot fiber and contour paths with nozzle direction arrows.

    Args:
        arrow_interval: Arc-length interval between arrows (mm). 0 to disable.
        arrow_length: Arrow length in mm. Auto-scaled if None.
    """
    fig, ax = plt.subplots(figsize=(16, 5))
    cmap = plt.cm.tab20

    # Auto-scale arrow length based on plot extent
    all_x = [n[0] for fp in fiber_paths for n in fp.nodes]
    all_y = [n[1] for fp in fiber_paths for n in fp.nodes]
    if all_x:
        extent = max(max(all_x) - min(all_x), max(all_y) - min(all_y))
        if arrow_length is None:
            arrow_length = max(0.5, extent * 0.015)

    # Contour paths (red)
    for cp in contour_paths:
        xs = [n[0] for n in cp.nodes]
        ys = [n[1] for n in cp.nodes]
        ax.plot(xs, ys, color="red", linewidth=0.8, alpha=0.6)

    # Fiber paths with nozzle direction arrows
    for idx, fp in enumerate(fiber_paths):
        colour = cmap(idx % 20)
        nodes = fp.nodes
        xs = [n[0] for n in nodes]
        ys = [n[1] for n in nodes]
        ax.plot(xs, ys, color=colour, linewidth=1.0,
                label=f"F{idx} ({fp.length:.1f}mm, {len(nodes)}pts)")
        ax.plot(xs[0], ys[0], "o", color=colour, markersize=4)
        ax.plot(xs[-1], ys[-1], "s", color=colour, markersize=4)

        # Nozzle direction arrows
        if arrow_interval > 0 and len(nodes) >= 2:
            angles = _compute_smooth_angles(nodes, sigma=0)
            cum_dist = 0.0
            next_at = 0.0
            for i in range(len(nodes)):
                if i > 0:
                    dx = nodes[i][0] - nodes[i-1][0]
                    dy = nodes[i][1] - nodes[i-1][1]
                    cum_dist += math.sqrt(dx*dx + dy*dy)
                if cum_dist >= next_at:
                    x, y = nodes[i]
                    a = math.radians(angles[i])
                    adx = arrow_length * math.cos(a)
                    ady = arrow_length * math.sin(a)
                    ax.annotate('', xy=(x+adx, y+ady), xytext=(x, y),
                                arrowprops=dict(arrowstyle='->', color='green',
                                                lw=0.8, shrinkA=0, shrinkB=0))
                    next_at = cum_dist + arrow_interval

    # Travel moves (dashed grey)
    for i in range(len(fiber_paths) - 1):
        x0, y0 = fiber_paths[i].end_point
        x1, y1 = fiber_paths[i + 1].start_point
        ax.plot([x0, x1], [y0, y1], "--", color="grey", linewidth=0.5, alpha=0.5)

    ax.set_aspect("equal")
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_title(title)
    if len(fiber_paths) <= 30:
        ax.legend(fontsize=5, ncol=3, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    fig.tight_layout()
    return fig, ax


# ══════════════════════════════════════════════════════════════════
# Public API
# ══════════════════════════════════════════════════════════════════

def svg_to_fibrifier_gcode(
    svg_paths,
    output_gcode: str = "output_fibrifier.gcode",
    params: FibrifierParams = None,
    connection_threshold: float = 5.0,
    min_fiber_length: float = 23.73,
    smooth_sigma: float = 3.0,
    decimate_epsilon: float = 0.05,
    flip_y: bool = True,
    preview_path: str = None,
    # Legacy direct overrides (applied to params)
    offset_x: float = None,
    offset_y: float = None,
    layer_height: float = None,
):
    """
    Convert one or more SVG stripe-pattern files to Fibrifier-format gcode.

    Each SVG produces two layers: P (polymer perimeter from contours) + F (fiber from stripes).

    Args:
        svg_paths: Single SVG path (str) or list of SVG paths for multi-layer stacking.
        output_gcode: Output gcode file path.
        params: FibrifierParams dataclass with all printing parameters.
                If None, uses defaults.
        connection_threshold: Max distance to merge adjacent paths (mm).
        min_fiber_length: Minimum fiber length; shorter paths are dropped (mm).
        smooth_sigma: Gaussian smoothing sigma for path coordinates.
        decimate_epsilon: RDP decimation tolerance (mm).
        flip_y: Flip SVG Y-axis (SVG Y-down → machine Y-up).
        preview_path: Path to save preview plot. If None, auto-generated next to gcode.
        offset_x: Override params.offset_x.
        offset_y: Override params.offset_y.
        layer_height: Override params.layer_height.

    Returns:
        dict with keys: gcode_path, preview_path, n_layers, n_fiber_paths,
                        n_contour_paths, total_fiber_mm, total_stretches

    Example:
        >>> import feax4d
        >>> params = feax4d.FibrifierParams()
        >>> params.temperature.bed_temperature = 100
        >>> result = feax4d.svg_to_gcode(
        ...     ["fibre_paths_layer0.svg", "fibre_paths_layer1.svg"],
        ...     output_gcode="cantilever_2layer.gcode",
        ...     params=params,
        ...     connection_threshold=10.0,
        ... )
    """
    if params is None:
        params = FibrifierParams()

    # Apply direct overrides
    if offset_x is not None:
        params.offset_x = offset_x
    if offset_y is not None:
        params.offset_y = offset_y
    if layer_height is not None:
        params.layer_height = layer_height

    if isinstance(svg_paths, str):
        svg_paths = [svg_paths]

    layers = []
    all_fiber = []
    all_contour = []

    for i, svg in enumerate(svg_paths):
        print(f"\n=== SVG {i + 1}: {svg} ===")
        fiber, contour, svg_h = load_and_prepare_paths(
            svg_path=svg,
            connection_threshold=connection_threshold,
            min_fiber_length=min_fiber_length,
            flip_y=flip_y,
            smooth_sigma=smooth_sigma,
            decimate_epsilon=decimate_epsilon,
        )
        if params.polymer_perimeter and contour:
            # Perimeter layer: contour outlines + optional infill
            perimeter_paths = list(contour)
            if params.infill_angle is not None:
                infill = generate_infill_paths(
                    contour, params.infill_angle, params.infill_pitch, params.infill_inset)
                perimeter_paths.extend(infill)
            layers.append({"fiber": [], "contour": perimeter_paths, "type": "P"})
            all_contour.extend(perimeter_paths)

        layers.append({"fiber": fiber, "contour": [], "type": "F"})
        all_fiber.extend(fiber)

        # Insert extra polymer infill layer after this fiber layer if requested
        infill_positions = params.polymer_infill_layers or []
        if i in infill_positions and contour:
            infill_only = generate_infill_paths(
                contour, params.polymer_infill_angle, params.infill_pitch, params.infill_inset)
            # Perimeters + infill for the extra polymer layer
            extra_paths = list(contour) + infill_only
            layers.append({"fiber": [], "contour": extra_paths, "type": "P"})
            all_contour.extend(extra_paths)

    # Preview
    if preview_path is None:
        preview_path = output_gcode.replace(".gcode", "_preview.png")
    fig, _ = plot_paths(all_fiber, all_contour,
                        title=f"{len(svg_paths)}-SVG stack ({len(all_fiber)} fiber paths)")
    fig.savefig(preview_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved to {preview_path}")

    # Generate
    gen = FibrifierGcodeGenerator(params=params)
    gen.generate_multilayer(layers, filename=output_gcode)

    total_fiber = sum(p.length for p in all_fiber)
    return {
        "gcode_path": output_gcode,
        "preview_path": preview_path,
        "n_layers": len(layers),
        "n_fiber_paths": len(all_fiber),
        "n_contour_paths": len(all_contour),
        "total_fiber_mm": total_fiber,
        "total_stretches": gen._stretch_counter,
    }


# ══════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="CFRTP curved-fiber gcode generator (Fibrifier format)")
    parser.add_argument("svg", nargs="+",
                        help="Input SVG file(s), one per layer (bottom-up)")
    parser.add_argument("-o", "--output", type=str, default=None,
                        help="Output gcode file path")
    parser.add_argument("--threshold", type=float, default=5.0,
                        help="Connection threshold in mm (default: 5.0)")
    parser.add_argument("--min-length", type=float, default=23.73,
                        help="Minimum fiber length in mm (default: 23.73)")
    parser.add_argument("--offset-x", type=float, default=175.0,
                        help="Bed offset X (default: 175.0)")
    parser.add_argument("--offset-y", type=float, default=135.0,
                        help="Bed offset Y (default: 135.0)")
    parser.add_argument("--layer-height", type=float, default=0.15,
                        help="Layer height (default: 0.15)")
    parser.add_argument("--smooth-sigma", type=float, default=3.0,
                        help="Gaussian smoothing sigma (default: 3.0)")
    parser.add_argument("--decimate-epsilon", type=float, default=0.05,
                        help="RDP decimation tolerance mm (default: 0.05)")
    parser.add_argument("--no-flip-y", action="store_true",
                        help="Do not flip Y axis")
    parser.add_argument("--plot", type=str, default=None,
                        help="Save path preview plot to file")
    args = parser.parse_args()

    # Each SVG produces 2 layers: P (contour perimeters) + F (fiber stripes)
    layers = []
    all_fiber = []
    all_contour = []
    for i, svg in enumerate(args.svg):
        print(f"\n=== SVG {i + 1}: {svg} ===")
        fiber, contour, svg_h = load_and_prepare_paths(
            svg_path=svg,
            connection_threshold=args.threshold,
            min_fiber_length=args.min_length,
            flip_y=not args.no_flip_y,
            smooth_sigma=args.smooth_sigma,
            decimate_epsilon=args.decimate_epsilon,
        )
        # P layer: contour perimeters only
        layers.append({"fiber": [], "contour": contour, "type": "P"})
        # F layer: fiber stripes only
        layers.append({"fiber": fiber, "contour": [], "type": "F"})
        all_fiber.extend(fiber)
        all_contour.extend(contour)

    output = args.output or "output_fibrifier.gcode"

    # Always save preview plot alongside gcode
    plot_path = args.plot or output.replace(".gcode", "_preview.png")
    fig, ax = plot_paths(all_fiber, all_contour,
                         title=f"{len(args.svg)}-SVG stack ({len(all_fiber)} fiber paths)")
    fig.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved to {plot_path}")
    gen = FibrifierGcodeGenerator(
        offset_x=args.offset_x,
        offset_y=args.offset_y,
        layer_height=args.layer_height,
    )
    gen.generate_multilayer(layers, filename=output)


if __name__ == "__main__":
    main()
