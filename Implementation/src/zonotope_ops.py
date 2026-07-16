"""
Zonotope operations using established computational geometry libraries.

This module provides a clean API for zonotope operations using pypoman and PuLP.
Zonotopes are represented in two forms:
1. Generator representation: Z = {c + Σ ξᵢgᵢ | ξᵢ ∈ [-1,1]}
2. Constraint representation: Z = {x | Ax ≤ b}

References:
- Althoff, M. (2010). "Reachability analysis of nonlinear systems"
- Scott, J. K., et al. (2016). "Constrained zonotopes"
"""

import numpy as np
from typing import List, Tuple, Optional, Union
from dataclasses import dataclass
import pulp
from scipy.spatial import ConvexHull
import warnings


@dataclass
class Zonotope:
    """
    N-dimensional zonotope in generator representation.
    
    A zonotope is defined as: Z = {c + G @ ξ | ξ ∈ [-1,1]^p}
    where:
        - c ∈ R^n is the center point
        - G ∈ R^{n×p} is the generator matrix (p generators)
        - n is the dimension, p is the number of generators (typically p >= n)
    
    Attributes:
        center: Center point (n-dimensional vector)
        generators: Generator matrix (n × p matrix)
        dimension: Dimension of the zonotope (automatically inferred)
    
    Properties:
        - Closed under Minkowski sum and linear transformation
        - Efficient to compute for reachability analysis
        - Can represent non-axis-aligned regions
    """
    
    center: np.ndarray      # Shape: (n,)
    generators: np.ndarray  # Shape: (n, p)
    
    def __post_init__(self):
        """Validate and normalize the zonotope representation."""
        self.center = np.asarray(self.center, dtype=float)
        self.generators = np.asarray(self.generators, dtype=float)
        
        # Ensure proper shapes
        if self.center.ndim != 1:
            raise ValueError(f"Center must be 1D array, got shape {self.center.shape}")
        
        if self.generators.ndim == 1:
            # Single generator - reshape to column vector
            self.generators = self.generators.reshape(-1, 1)
        elif self.generators.ndim != 2:
            raise ValueError(f"Generators must be 2D array, got shape {self.generators.shape}")
        
        # Check dimension consistency
        n_center = self.center.shape[0]
        n_gen = self.generators.shape[0]
        
        if n_center != n_gen:
            raise ValueError(
                f"Dimension mismatch: center is {n_center}D but generators are {n_gen}D"
            )
        
        # Remove zero generators (they don't contribute to the set)
        self.generators = self._remove_zero_generators(self.generators)
    
    @property
    def dimension(self) -> int:
        """Return the dimension of the zonotope."""
        return self.center.shape[0]
    
    @property
    def num_generators(self) -> int:
        """Return the number of generators."""
        return self.generators.shape[1]
    
    @staticmethod
    def _remove_zero_generators(generators: np.ndarray, tol: float = 1e-12) -> np.ndarray:
        """Remove generator columns that are effectively zero."""
        if generators.shape[1] == 0:
            # Keep at least one generator (even if zero)
            return generators
        
        non_zero_mask = np.linalg.norm(generators, axis=0) > tol
        
        if not np.any(non_zero_mask):
            # All generators are zero - return first one
            return generators[:, :1]
        
        return generators[:, non_zero_mask]
    
    @classmethod
    def from_box(cls, bounds: List[Tuple[float, float]]) -> 'Zonotope':
        """
        Create axis-aligned box zonotope from bounds.
        
        Args:
            bounds: List of (min, max) tuples for each dimension
            
        Returns:
            Zonotope representing the axis-aligned box
            
        Example:
            >>> z = Zonotope.from_box([(0, 10), (5, 15)])  # 2D box
            >>> z.center
            array([5., 10.])
        """
        bounds = np.array(bounds, dtype=float)
        n = len(bounds)
        
        # Center at midpoint
        center = (bounds[:, 0] + bounds[:, 1]) / 2.0
        
        # Generators: diagonal matrix with half-widths
        generators = np.diag((bounds[:, 1] - bounds[:, 0]) / 2.0)
        
        return cls(center, generators)
    
    def to_box_bounds(self) -> np.ndarray:
        """
        Compute axis-aligned bounding box (AABB).
        
        For Z = {c + G @ ξ | ξ ∈ [-1,1]^p}, the AABB is:
            [c - Σ|gᵢ|, c + Σ|gᵢ|]
        
        Returns:
            Array of shape (n, 2) with (min, max) for each dimension
        """
        # Half-widths in each dimension
        half_widths = np.sum(np.abs(self.generators), axis=1)
        
        bounds = np.column_stack([
            self.center - half_widths,
            self.center + half_widths
        ])
        
        return bounds

    def support(self, direction: np.ndarray) -> float:
        """Support function h_Z(a) = max_{x in Z} a^T x.

        For Z = {c + G @ xi | xi in [-1,1]^p}:
            h_Z(a) = a^T c + sum_i |a^T g_i|.

        Used by CRR's Stage-2 Type-II certificate to test whether a tightened
        linear constraint (e.g. a battery-energy budget) still contains the
        region's peak, i.e. h_Z(a) <= b.
        """
        a = np.asarray(direction, dtype=float)
        return float(a @ self.center + np.sum(np.abs(a @ self.generators)))

    def is_empty(self, tol: float = 1e-12) -> bool:
        """
        Check if zonotope is empty (degenerate).
        
        A zonotope is empty if all generators are zero or if the
        bounding box has zero volume.
        """
        if self.num_generators == 0:
            return True
        
        bounds = self.to_box_bounds()
        volumes = bounds[:, 1] - bounds[:, 0]
        
        return np.any(volumes < tol)
    
    def volume(self) -> float:
        """
        Compute zonotope volume (uses AABB approximation for efficiency).
        
        Exact volume computation for zonotopes is expensive (requires vertex
        enumeration). For cache management, AABB volume is sufficient and
        provides a conservative upper bound.
        
        For exact volume, use volume_exact() which is more expensive.
        """
        bounds = self.to_box_bounds()
        volumes = bounds[:, 1] - bounds[:, 0]
        return np.prod(np.maximum(volumes, 0))
    
    def volume_exact(self) -> float:
        """
        Compute exact zonotope volume using vertex enumeration.
        
        WARNING: Expensive operation! Use sparingly.
        Complexity: O(2^p) where p is number of generators
        
        For cache management, prefer volume() which uses AABB.
        """
        if self.dimension > 3 or self.num_generators > 10:
            warnings.warn(
                f"Exact volume computation is expensive for {self.dimension}D "
                f"zonotope with {self.num_generators} generators. "
                "Consider using volume() instead."
            )
        
        try:
            vertices = self.vertices()
            if len(vertices) < self.dimension + 1:
                return 0.0
            
            hull = ConvexHull(vertices)
            return hull.volume
        except Exception as e:
            warnings.warn(f"Exact volume computation failed: {e}. Using AABB volume.")
            return self.volume()
    
    def contains(self, point: np.ndarray, method: str = 'box', tol: float = 1e-9) -> bool:
        """
        Check if point is contained in zonotope.
        
        Args:
            point: Point to check (n-dimensional)
            method: 'box' (fast, conservative) or 'exact' (slow, precise)
            tol: Numerical tolerance
            
        Returns:
            True if point is in zonotope
            
        Methods:
            - 'box': Check if point is in AABB (fast, may have false positives)
            - 'exact': Solve LP to check exact containment (slower, precise)
        """
        point = np.asarray(point, dtype=float)
        
        if point.shape != (self.dimension,):
            raise ValueError(
                f"Point dimension {point.shape[0]} doesn't match zonotope dimension {self.dimension}"
            )
        
        if method == 'box':
            # Fast check using AABB
            bounds = self.to_box_bounds()
            return np.all(point >= bounds[:, 0] - tol) and np.all(point <= bounds[:, 1] + tol)
        
        elif method == 'exact':
            # Exact check: solve G @ ξ = (point - center) with ξ ∈ [-1, 1]
            return self._contains_exact(point, tol)
        
        else:
            raise ValueError(f"Unknown method: {method}. Use 'box' or 'exact'.")
    
    def _contains_exact(self, point: np.ndarray, tol: float = 1e-9) -> bool:
        """
        Exact containment check using linear programming.
        
        Solves: minimize sum of absolute residuals |G @ ξ - (point - center)|
                subject to: -1 <= ξ <= 1
        
        Point is in zonotope if optimal objective is ~0.
        """
        p = self.num_generators
        n = self.dimension
        
        # Create LP problem
        prob = pulp.LpProblem("Zonotope_Containment", pulp.LpMinimize)
        
        # Variables: ξ ∈ [-1, 1]^p
        xi = [pulp.LpVariable(f"xi_{i}", lowBound=-1, upBound=1) for i in range(p)]
        
        # Auxiliary variables for absolute values: r_i = |residual_i|
        r = [pulp.LpVariable(f"r_{i}", lowBound=0) for i in range(n)]
        
        # Objective: minimize sum of absolute residuals
        prob += pulp.lpSum(r)
        
        # Constraints: r_i >= |residual_i| where residual_i = (G @ ξ - (point - center))_i
        target = point - self.center
        for i in range(n):
            residual_i = pulp.lpSum(self.generators[i, j] * xi[j] for j in range(p)) - target[i]
            prob += r[i] >= residual_i    # r_i >= residual_i
            prob += r[i] >= -residual_i   # r_i >= -residual_i
        
        # Solve
        try:
            prob.solve(pulp.PULP_CBC_CMD(msg=0))  # Silent mode
            
            if prob.status == pulp.LpStatusOptimal:
                objective_value = pulp.value(prob.objective)
                return objective_value < tol
            else:
                # Solver failed - fall back to box check
                warnings.warn(f"LP solver failed with status {pulp.LpStatus[prob.status]}. Using box check.")
                return self.contains(point, method='box', tol=tol)
        
        except Exception as e:
            warnings.warn(f"LP containment check failed: {e}. Using box check.")
            return self.contains(point, method='box', tol=tol)
    
    def vertices(self, max_vertices: int = 10000) -> np.ndarray:
        """
        Compute vertices of the zonotope.
        
        WARNING: Number of vertices can be exponential in number of generators!
        For p generators: up to 2^p vertices (though typically fewer due to degeneracy)
        
        Args:
            max_vertices: Maximum number of vertices to compute (safety limit)
            
        Returns:
            Array of shape (num_vertices, n) containing vertex coordinates
        """
        p = self.num_generators
        
        if 2**p > max_vertices:
            raise ValueError(
                f"Too many potential vertices: 2^{p} = {2**p} > {max_vertices}. "
                "Zonotope is too complex for vertex enumeration."
            )
        
        # Generate all combinations of ξ ∈ {-1, +1}^p
        vertices = []
        for i in range(2**p):
            # Convert integer to binary, map 0->-1, 1->+1
            xi = np.array([1 if (i >> j) & 1 else -1 for j in range(p)])
            vertex = self.center + self.generators @ xi
            vertices.append(vertex)
        
        vertices = np.array(vertices)
        
        # Remove duplicate vertices (can occur due to zero generators or alignment)
        vertices = np.unique(vertices, axis=0)
        
        return vertices
    
    def reduce_generators(self, target_generators: Optional[int] = None, method: str = 'pca') -> 'Zonotope':
        """
        Reduce number of generators while preserving approximation quality.
        
        Useful for keeping zonotopes tractable in long computations.
        
        Args:
            target_generators: Target number of generators (default: 2*dimension)
            method: 'pca' (preserve main directions) or 'box' (use AABB)
            
        Returns:
            New zonotope with fewer generators
        """
        if target_generators is None:
            target_generators = 2 * self.dimension
        
        if self.num_generators <= target_generators:
            return self
        
        if method == 'pca':
            # Use PCA to find principal directions
            # Weight generators by their norm
            weights = np.linalg.norm(self.generators, axis=0)
            
            # Perform weighted PCA
            from sklearn.decomposition import PCA
            pca = PCA(n_components=min(target_generators, self.dimension))
            
            # Transform generators
            transformed = pca.fit_transform(self.generators.T)
            
            # Create new generators from principal components
            new_generators = pca.components_.T * np.max(weights)
            
            return Zonotope(self.center, new_generators)
        
        elif method == 'box':
            # Fall back to bounding box
            return Zonotope.from_box(self.to_box_bounds())
        
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def __repr__(self) -> str:
        """String representation."""
        bounds = self.to_box_bounds()
        bounds_str = ", ".join([f"[{b[0]:.2f}, {b[1]:.2f}]" for b in bounds])
        return f"Zonotope({self.dimension}D, {self.num_generators} generators, bounds: {bounds_str})"


def zonotope_intersection(z1: Zonotope, z2: Zonotope) -> Zonotope:
    """
    Compute intersection of two zonotopes (conservative over-approximation).
    
    Exact zonotope intersection is not a zonotope in general. We use AABB
    intersection which is conservative (over-approximates) but fast and correct.
    
    For exact intersection, use constrained zonotopes (more complex).
    
    Args:
        z1, z2: Input zonotopes
        
    Returns:
        Zonotope over-approximating the intersection
    """
    if z1.dimension != z2.dimension:
        raise ValueError(f"Dimension mismatch: {z1.dimension}D vs {z2.dimension}D")
    
    # Get AABBs
    bounds1 = z1.to_box_bounds()
    bounds2 = z2.to_box_bounds()
    
    # Intersect AABBs
    intersected_bounds = np.column_stack([
        np.maximum(bounds1[:, 0], bounds2[:, 0]),
        np.minimum(bounds1[:, 1], bounds2[:, 1])
    ])
    
    # Check if intersection is empty
    if np.any(intersected_bounds[:, 0] >= intersected_bounds[:, 1]):
        # Return empty zonotope (degenerate)
        return Zonotope.from_box([(0, 0)] * z1.dimension)
    
    # Create zonotope from intersected box
    return Zonotope.from_box(intersected_bounds)


def zonotope_union(z1: Zonotope, z2: Zonotope) -> Zonotope:
    """
    Compute union of two zonotopes (over-approximation using AABB).
    
    Exact zonotope union is not a zonotope in general. We use AABB hull
    which is conservative but fast.
    
    Args:
        z1, z2: Input zonotopes
        
    Returns:
        Zonotope over-approximating the union
    """
    if z1.dimension != z2.dimension:
        raise ValueError(f"Dimension mismatch: {z1.dimension}D vs {z2.dimension}D")
    
    # Get AABBs
    bounds1 = z1.to_box_bounds()
    bounds2 = z2.to_box_bounds()
    
    # Compute hull
    union_bounds = np.column_stack([
        np.minimum(bounds1[:, 0], bounds2[:, 0]),
        np.maximum(bounds1[:, 1], bounds2[:, 1])
    ])
    
    return Zonotope.from_box(union_bounds)


def zonotope_minkowski_sum(z1: Zonotope, z2: Zonotope) -> Zonotope:
    """
    Compute Minkowski sum of two zonotopes (EXACT operation).
    
    For Z₁ = {c₁ + G₁ξ₁} and Z₂ = {c₂ + G₂ξ₂}:
        Z₁ ⊕ Z₂ = {c₁ + c₂ + [G₁ G₂][ξ₁; ξ₂]}
    
    This is one of the key advantages of zonotopes: Minkowski sum is closed!
    
    Args:
        z1, z2: Input zonotopes
        
    Returns:
        Zonotope representing the Minkowski sum (exact)
    """
    if z1.dimension != z2.dimension:
        raise ValueError(f"Dimension mismatch: {z1.dimension}D vs {z2.dimension}D")
    
    # New center: sum of centers
    new_center = z1.center + z2.center
    
    # New generators: concatenation of generator matrices
    new_generators = np.hstack([z1.generators, z2.generators])
    
    return Zonotope(new_center, new_generators)


def zonotope_linear_map(z: Zonotope, matrix: np.ndarray) -> Zonotope:
    """
    Apply linear transformation to zonotope (EXACT operation).
    
    For Z = {c + Gξ} and matrix A:
        A @ Z = {Ac + (AG)ξ}
    
    Another key advantage: linear transformations are closed!
    
    Args:
        z: Input zonotope
        matrix: Transformation matrix (m × n)
        
    Returns:
        Transformed zonotope (exact)
    """
    matrix = np.asarray(matrix, dtype=float)
    
    if matrix.ndim != 2:
        raise ValueError(f"Matrix must be 2D, got shape {matrix.shape}")
    
    if matrix.shape[1] != z.dimension:
        raise ValueError(
            f"Matrix columns ({matrix.shape[1]}) must match zonotope dimension ({z.dimension})"
        )
    
    # Apply transformation
    new_center = matrix @ z.center
    new_generators = matrix @ z.generators
    
    return Zonotope(new_center, new_generators)


def zonotope_subtract(z1: Zonotope, z2: Zonotope) -> List[Zonotope]:
    r"""
    Compute set difference z1 / z2 (returns multiple zonotopes).
    
    Unlike intersection and Minkowski sum, set difference is NOT closed
    under zonotopes. Result may require multiple zonotopes to represent.
    
    We use AABB difference which returns axis-aligned box zonotopes.
    
    Args:
        z1: Zonotope to subtract from
        z2: Zonotope to subtract
        
    Returns:
        List of zonotopes representing z1 \ z2
    """
    if z1.dimension != z2.dimension:
        raise ValueError(f"Dimension mismatch: {z1.dimension}D vs {z2.dimension}D")
    
    # Use AABB-based subtraction (conservative but tractable)
    bounds1 = z1.to_box_bounds()
    bounds2 = z2.to_box_bounds()
    
    # Compute intersection bounds
    intersection_bounds = np.column_stack([
        np.maximum(bounds1[:, 0], bounds2[:, 0]),
        np.minimum(bounds1[:, 1], bounds2[:, 1])
    ])
    
    # Check if no intersection
    if np.any(intersection_bounds[:, 0] >= intersection_bounds[:, 1]):
        return [z1]  # No overlap, return original
    
    # Check if z2 fully contains z1
    if np.all(bounds2[:, 0] <= bounds1[:, 0]) and np.all(bounds2[:, 1] >= bounds1[:, 1]):
        return []  # Fully contained, return empty list
    
    # Carve out intersection dimension by dimension
    result = []
    
    def carve(dim: int, current_bounds: np.ndarray):
        """Recursively carve out pieces."""
        if dim == z1.dimension:
            # Base case: create zonotope if valid
            if not np.any(current_bounds[:, 0] >= current_bounds[:, 1]):
                # Check this isn't the intersection itself
                is_intersection = np.allclose(current_bounds, intersection_bounds, atol=1e-12)
                if not is_intersection:
                    result.append(Zonotope.from_box(current_bounds))
            return
        
        b1_lo, b1_hi = bounds1[dim]
        int_lo, int_hi = intersection_bounds[dim]
        
        # Piece before intersection
        if b1_lo < int_lo - 1e-12:
            new_bounds = current_bounds.copy()
            new_bounds[dim] = [b1_lo, int_lo]
            carve(dim + 1, new_bounds)
        
        # Middle piece (restricted to intersection)
        mid_lo = max(b1_lo, int_lo)
        mid_hi = min(b1_hi, int_hi)
        if mid_lo < mid_hi - 1e-12:
            new_bounds = current_bounds.copy()
            new_bounds[dim] = [mid_lo, mid_hi]
            carve(dim + 1, new_bounds)
        
        # Piece after intersection
        if int_hi < b1_hi - 1e-12:
            new_bounds = current_bounds.copy()
            new_bounds[dim] = [int_hi, b1_hi]
            carve(dim + 1, new_bounds)
    
    carve(0, bounds1.copy())
    
    return result
