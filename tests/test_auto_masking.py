from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "src" / "pose_trainer_blender" / "auto_masking.py"
SPEC = importlib.util.spec_from_file_location("pose_trainer_blender_auto_masking", MODULE_PATH)
auto_masking = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = auto_masking
SPEC.loader.exec_module(auto_masking)


def _grid_mesh(width: int, height: int):
    positions = [(float(x), float(y), 0.0) for y in range(height) for x in range(width)]
    faces = []
    for y in range(height - 1):
        for x in range(width - 1):
            v = y * width + x
            faces.append([v, v + 1, v + width + 1, v + width])
    return faces, positions


def _adjacency(vertex_count: int, faces):
    neighbors = [set() for _ in range(vertex_count)]
    for face in faces:
        for index, vertex in enumerate(face):
            other = face[(index + 1) % len(face)]
            neighbors[vertex].add(other)
            neighbors[other].add(vertex)
    return neighbors


def _is_connected(vertices, neighbors) -> bool:
    vertices = set(vertices)
    if not vertices:
        return False
    seen = {next(iter(vertices))}
    stack = list(seen)
    while stack:
        vertex = stack.pop()
        for neighbor in neighbors[vertex]:
            if neighbor in vertices and neighbor not in seen:
                seen.add(neighbor)
                stack.append(neighbor)
    return seen == vertices


def test_auto_mask_is_deterministic_and_creates_requested_local_regions():
    faces, positions = _grid_mesh(6, 6)

    first = auto_masking.generate_auto_mask_areas(faces, positions, area_count=6, softness_iterations=0)
    second = auto_masking.generate_auto_mask_areas(faces, positions, area_count=6, softness_iterations=0)

    assert first == second
    assert len(first) == 6
    assert all(area.weights for area in first)

    neighbors = _adjacency(len(positions), faces)
    owner_vertices = [[] for _area in first]
    for vertex in range(len(positions)):
        owners = [area_index for area_index, area in enumerate(first) if vertex in area.weights]
        assert len(owners) == 1
        owner_vertices[owners[0]].append(vertex)

    assert all(_is_connected(vertices, neighbors) for vertices in owner_vertices)


def test_auto_mask_softening_keeps_weights_normalized_and_compact():
    faces, positions = _grid_mesh(5, 5)

    areas = auto_masking.generate_auto_mask_areas(
        faces,
        positions,
        area_count=4,
        softness_iterations=3,
        max_influences=3,
    )

    sums = [0.0] * len(positions)
    counts = [0] * len(positions)
    for area in areas:
        for vertex, weight in area.weights.items():
            assert weight > 0.0
            sums[vertex] += weight
            counts[vertex] += 1

    assert all(abs(total - 1.0) < 1.0e-6 for total in sums)
    assert all(1 <= count <= 3 for count in counts)


def test_auto_mask_covers_disconnected_mesh_components():
    faces = [[0, 1, 2, 3], [4, 5, 6, 7]]
    positions = [
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (1.0, 1.0, 0.0),
        (0.0, 1.0, 0.0),
        (10.0, 0.0, 0.0),
        (11.0, 0.0, 0.0),
        (11.0, 1.0, 0.0),
        (10.0, 1.0, 0.0),
    ]

    areas = auto_masking.generate_auto_mask_areas(faces, positions, area_count=1, softness_iterations=0)

    assert len(areas) == 2
    covered = {vertex for area in areas for vertex in area.weights}
    assert covered == set(range(len(positions)))
