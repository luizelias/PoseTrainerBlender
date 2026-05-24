from __future__ import annotations

from dataclasses import dataclass
from heapq import heappop, heappush
from math import sqrt
from typing import Iterable, Sequence


@dataclass(frozen=True)
class AutoMaskArea:
    seed_vertex: int
    weights: dict[int, float]


def _distance(a: Sequence[float], b: Sequence[float]) -> float:
    dx = float(a[0]) - float(b[0])
    dy = float(a[1]) - float(b[1])
    dz = float(a[2]) - float(b[2])
    return sqrt(dx * dx + dy * dy + dz * dz)


def _build_adjacency(
    vertex_count: int,
    faces: Iterable[Iterable[int]],
    positions: Sequence[Sequence[float]],
) -> list[list[tuple[int, float]]]:
    edge_sets: list[set[int]] = [set() for _ in range(vertex_count)]
    for face in faces:
        vertices = [int(vertex) for vertex in face]
        if len(vertices) < 2:
            continue
        for index, vertex in enumerate(vertices):
            next_vertex = vertices[(index + 1) % len(vertices)]
            if vertex == next_vertex:
                continue
            if not (0 <= vertex < vertex_count and 0 <= next_vertex < vertex_count):
                raise ValueError("face contains a vertex index outside the positions array")
            edge_sets[vertex].add(next_vertex)
            edge_sets[next_vertex].add(vertex)

    adjacency: list[list[tuple[int, float]]] = []
    for vertex, neighbors in enumerate(edge_sets):
        weighted = []
        for neighbor in sorted(neighbors):
            length = _distance(positions[vertex], positions[neighbor])
            weighted.append((neighbor, max(length, 1.0e-6)))
        adjacency.append(weighted)
    return adjacency


def _connected_components(adjacency: Sequence[Sequence[tuple[int, float]]]) -> list[list[int]]:
    visited = [False] * len(adjacency)
    components = []
    for start in range(len(adjacency)):
        if visited[start]:
            continue
        stack = [start]
        visited[start] = True
        component = []
        while stack:
            vertex = stack.pop()
            component.append(vertex)
            for neighbor, _weight in adjacency[vertex]:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    stack.append(neighbor)
        components.append(sorted(component))
    return components


def _allocate_seed_counts(components: Sequence[Sequence[int]], requested_count: int) -> list[int]:
    if not components:
        return []

    vertex_count = sum(len(component) for component in components)
    requested_count = max(1, min(int(requested_count), vertex_count))
    counts = [1] * len(components)
    if requested_count <= len(components):
        return counts

    remaining = requested_count - len(components)
    weighted = []
    for index, component in enumerate(components):
        exact = requested_count * (len(component) / vertex_count)
        extra = max(0, int(exact) - 1)
        extra = min(extra, len(component) - 1, remaining)
        counts[index] += extra
        remaining -= extra
        weighted.append((exact - int(exact), len(component), -component[0], index))

    for _fraction, _size, _negative_min_vertex, index in sorted(weighted, reverse=True):
        if remaining <= 0:
            break
        if counts[index] < len(components[index]):
            counts[index] += 1
            remaining -= 1

    index = 0
    while remaining > 0 and any(counts[i] < len(components[i]) for i in range(len(components))):
        if counts[index] < len(components[index]):
            counts[index] += 1
            remaining -= 1
        index = (index + 1) % len(components)
    return counts


def _dijkstra(
    adjacency: Sequence[Sequence[tuple[int, float]]],
    sources: Sequence[tuple[int, int]],
) -> tuple[list[float], list[int]]:
    distances = [float("inf")] * len(adjacency)
    labels = [-1] * len(adjacency)
    heap = []
    for label, vertex in sources:
        if distances[vertex] > 0.0 or labels[vertex] == -1 or label < labels[vertex]:
            distances[vertex] = 0.0
            labels[vertex] = label
            heappush(heap, (0.0, label, vertex))

    while heap:
        distance, label, vertex = heappop(heap)
        if distance != distances[vertex] or label != labels[vertex]:
            continue
        for neighbor, edge_weight in adjacency[vertex]:
            next_distance = distance + edge_weight
            if next_distance < distances[neighbor] or (
                next_distance == distances[neighbor] and label < labels[neighbor]
            ):
                distances[neighbor] = next_distance
                labels[neighbor] = label
                heappush(heap, (next_distance, label, neighbor))
    return distances, labels


def _centroid_seed(component: Sequence[int], positions: Sequence[Sequence[float]]) -> int:
    inv_count = 1.0 / len(component)
    centroid = [
        sum(float(positions[vertex][axis]) for vertex in component) * inv_count
        for axis in range(3)
    ]
    return min(
        component,
        key=lambda vertex: (
            (float(positions[vertex][0]) - centroid[0]) ** 2
            + (float(positions[vertex][1]) - centroid[1]) ** 2
            + (float(positions[vertex][2]) - centroid[2]) ** 2,
            vertex,
        ),
    )


def _sample_component_seeds(
    component: Sequence[int],
    seed_count: int,
    positions: Sequence[Sequence[float]],
    adjacency: Sequence[Sequence[tuple[int, float]]],
) -> list[int]:
    seeds = [_centroid_seed(component, positions)]
    nearest_distances, _labels = _dijkstra(adjacency, [(0, seeds[0])])

    for _index in range(1, min(seed_count, len(component))):
        next_seed = max(component, key=lambda vertex: (nearest_distances[vertex], -vertex))
        seeds.append(next_seed)
        distances, _labels = _dijkstra(adjacency, [(0, next_seed)])
        for vertex in component:
            if distances[vertex] < nearest_distances[vertex]:
                nearest_distances[vertex] = distances[vertex]
    return seeds


def _compact_weights(
    weights: dict[int, float],
    min_weight: float,
    max_influences: int,
) -> dict[int, float]:
    if not weights:
        return {}
    ordered = sorted(weights.items(), key=lambda item: (-item[1], item[0]))
    kept = [(area, weight) for area, weight in ordered if weight >= min_weight]
    if max_influences > 0:
        kept = kept[:max_influences]
    if not kept:
        kept = ordered[:1]

    total = sum(weight for _area, weight in kept)
    if total <= 0.0:
        area, _weight = kept[0]
        return {area: 1.0}
    return {area: weight / total for area, weight in kept}


def _smooth_vertex_weights(
    hard_labels: Sequence[int],
    adjacency: Sequence[Sequence[tuple[int, float]]],
    iterations: int,
    min_weight: float,
    max_influences: int,
) -> list[dict[int, float]]:
    vertex_weights = [{label: 1.0} for label in hard_labels]
    for _iteration in range(max(0, iterations)):
        next_weights = []
        for vertex, own in enumerate(vertex_weights):
            if not adjacency[vertex]:
                next_weights.append(dict(own))
                continue

            neighbor_average: dict[int, float] = {}
            for neighbor, _edge_weight in adjacency[vertex]:
                for area, weight in vertex_weights[neighbor].items():
                    neighbor_average[area] = neighbor_average.get(area, 0.0) + weight

            inv_degree = 1.0 / len(adjacency[vertex])
            blended: dict[int, float] = {}
            for area, weight in own.items():
                blended[area] = blended.get(area, 0.0) + 0.55 * weight
            for area, weight in neighbor_average.items():
                blended[area] = blended.get(area, 0.0) + 0.45 * weight * inv_degree
            next_weights.append(_compact_weights(blended, min_weight, max_influences))
        vertex_weights = next_weights
    return vertex_weights


def generate_auto_mask_areas(
    faces: Iterable[Iterable[int]],
    positions: Sequence[Sequence[float]],
    area_count: int = 32,
    softness_iterations: int = 2,
    min_weight: float = 1.0e-4,
    max_influences: int = 4,
) -> list[AutoMaskArea]:
    """Create deterministic, local, topology-connected deformation areas.

    The output is sparse per-area vertex weights. Hard ownership is a geodesic
    Voronoi partition seeded by farthest-point sampling; smoothing only blends
    across local graph rings and keeps each vertex normalized to a compact set
    of area influences.
    """

    vertex_count = len(positions)
    if vertex_count == 0:
        return []
    if any(len(position) < 3 for position in positions):
        raise ValueError("positions must contain xyz coordinates")

    adjacency = _build_adjacency(vertex_count, faces, positions)
    components = _connected_components(adjacency)
    seed_counts = _allocate_seed_counts(components, area_count)

    seeds = []
    for component, seed_count in zip(components, seed_counts):
        seeds.extend(_sample_component_seeds(component, seed_count, positions, adjacency))

    if not seeds:
        return []

    _distances, labels = _dijkstra(adjacency, list(enumerate(seeds)))
    clusters: list[list[int]] = [[] for _seed in seeds]
    for vertex, label in enumerate(labels):
        if label < 0:
            label = 0
            labels[vertex] = label
        clusters[label].append(vertex)

    sorted_labels = sorted(
        range(len(clusters)),
        key=lambda label: (min(clusters[label]) if clusters[label] else vertex_count, seeds[label]),
    )
    label_remap = {old_label: new_label for new_label, old_label in enumerate(sorted_labels)}
    remapped_labels = [label_remap[label] for label in labels]
    remapped_seeds = [seeds[old_label] for old_label in sorted_labels]

    vertex_weights = _smooth_vertex_weights(
        remapped_labels,
        adjacency,
        softness_iterations,
        min_weight,
        max_influences,
    )

    area_weights: list[dict[int, float]] = [dict() for _seed in remapped_seeds]
    for vertex, weights in enumerate(vertex_weights):
        for area, weight in weights.items():
            area_weights[area][vertex] = weight

    return [
        AutoMaskArea(seed_vertex=seed, weights=weights)
        for seed, weights in zip(remapped_seeds, area_weights)
        if weights
    ]
