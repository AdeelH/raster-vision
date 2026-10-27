from typing import Sequence

import numpy as np
import shapely
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union


class AoiSampler:
    """Allows efficiently sampling points inside an AOI uniformly at random.

    To achieve this, each polygon in the AOI is first partitioned into
    triangles (triangulation). Then, to sample a single point, we first sample
    a triangle at random with probability proportional to its area and then
    sample a point within that triangle uniformly at random.
    """

    def __init__(self, polygons: Sequence[Polygon]) -> None:
        # merge overlapping polygons, if any
        merged_polygons = unary_union(polygons)
        if isinstance(merged_polygons, Polygon):
            merged_polygons = [merged_polygons]
        elif isinstance(merged_polygons, MultiPolygon):
            merged_polygons = list(merged_polygons.geoms)
        self.polygons = merged_polygons
        self.triangulate(self.polygons)

    def triangulate(self, polygons) -> dict:
        triangulations = [self.triangulate_polygon(p) for p in polygons]
        self.triangulations = triangulations
        self.origins = np.vstack([t['origins'] for t in triangulations])
        self.vec_AB = np.vstack([t['bases'][0] for t in triangulations])
        self.vec_AC = np.vstack([t['bases'][1] for t in triangulations])
        areas = np.concatenate([t['areas'] for t in triangulations])
        self.weights = areas / areas.sum()
        self.ntriangles = len(self.origins)

    def sample(self, n: int = 1) -> np.ndarray:
        """Sample a random point within the AOI, using the following algorithm:
            - Randomly sample one triangle (ABC) with probability proportional
            to its area.
            - Starting at A, travel a random distance along vectors AB and AC.
            - Return the final position.

        Args:
            n (int): Number of points to sample. Defaults to 1.

        Returns:
            np.ndarray: (n, 2) 2D coordinates of the sampled points.
        """
        tri_idx = np.random.choice(self.ntriangles, p=self.weights, size=n)
        origin = self.origins[tri_idx]
        vec_AB = self.vec_AB[tri_idx]
        vec_AC = self.vec_AC[tri_idx]
        # the fractions to travel along each of the two vectors
        r, s = np.random.uniform(size=(2, n, 1))
        # If the fractions will land us in the wrong half of the parallelogram
        # defined by vec AB and vec AC, reflect them into the correct half.
        mask = (r + s) > 1
        r[mask] = 1 - r[mask]
        s[mask] = 1 - s[mask]
        loc = origin + (r * vec_AB + s * vec_AC)
        return loc

    def triangulate_polygon(self, polygon: Polygon) -> dict:
        """Triangulate polygon.

        Uses a constrained Delaunay triangulation (via shapely/GEOS), which
        respects the polygon's boundary and holes, if any.
        """
        triangles = shapely.constrained_delaunay_triangles(polygon)
        # Each triangle's exterior ring has 4 coords, the last of which is a
        # duplicate of the first. Drop it to get a (T, 3, 2) array of the 3
        # vertices of each of the T triangles.
        coords = shapely.get_coordinates(triangles).reshape(-1, 4, 2)[:, :3]
        vertices = coords.reshape(-1, 2)
        simplices = np.arange(len(vertices)).reshape(-1, 3)
        origins, bases = self.triangle_origin_and_basis(vertices, simplices)

        out = {
            'vertices': vertices,
            'simplices': simplices,
            'origins': origins,
            'bases': bases,
            'areas': self.triangle_area(vertices, simplices)
        }
        return out

    def triangle_side_lengths(self, vertices: np.ndarray, simplices: np.ndarray
                              ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate lengths of all 3 sides of each triangle specified by the
        simplices array.

        Args:
            vertices: (N, 2) array of vertex coords in 2D.
            simplices: (N, 3) array of indexes to entries in the
                vertices array. Each row represents one triangle.

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray]: ||AB||, ||BC||, ||AC||
        """
        A = vertices[simplices[:, 0]]
        B = vertices[simplices[:, 1]]
        C = vertices[simplices[:, 2]]
        AB, AC, BC = B - A, C - A, C - B
        ab = np.linalg.norm(AB, axis=1)
        bc = np.linalg.norm(BC, axis=1)
        ac = np.linalg.norm(AC, axis=1)
        return ab, bc, ac

    def triangle_origin_and_basis(
            self, vertices: np.ndarray, simplices: np.ndarray
    ) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]:
        """For each triangle ABC, return point A, vector AB, and vector AC.

        Args:
            vertices: (N, 2) array of vertex coords in 2D.
            simplices: (N, 3) array of indexes to entries in the
                vertices array. Each row represents one triangle.

        Returns:
            3 arrays of shape (N, 2), organized into tuples like so:
            (point A, (vector AB, vector AC)).
        """
        A = vertices[simplices[:, 0]]
        B = vertices[simplices[:, 1]]
        C = vertices[simplices[:, 2]]
        AB = B - A
        AC = C - A
        return A, (AB, AC)

    def triangle_area(self, vertices: np.ndarray,
                      simplices: np.ndarray) -> np.ndarray:
        """Calculate area of each triangle specified by the simplices array
        using Heron's formula.

        Args:
            vertices: (N, 2) array of vertex coords in 2D.
            simplices: (N, 3) array of indexes to entries in the
                vertices array. Each row represents one triangle.

        Returns:
            np.ndarray: (N,) array of areas
        """
        a, b, c = self.triangle_side_lengths(vertices, simplices)
        p = (a + b + c) * 0.5
        area = p * (p - a) * (p - b) * (p - c)
        area[area < 0] = 0
        area = np.sqrt(area)
        return area
