"""
Tile Information Data Structures

Defines the core data structures for representing tiles in matrix computations.
Supports multi-matmul scenarios with proper identification and grouping.
"""

from dataclasses import dataclass
from typing import Optional, Tuple, List
from enum import Enum


class TileType(Enum):
    """Type of tile operation"""
    COMPUTE = "compute"
    NOP = "nop"
    PADDING = "padding"


@dataclass
class TileInfo:
    """
    Information about a single tile computation.

    This is the primary data structure used throughout the optimization pipeline.
    """

    # Unique identifiers
    tile_id: int
    matrix_id: int  # Which matmul this tile belongs to

    # Position in the computation
    m_offset: int  # Row offset in output matrix
    n_offset: int  # Column offset in output matrix
    k_iteration: int  # K-dimension iteration (for reduction)

    # Actual dimensions (may be less than tile size for edge tiles)
    actual_m: int
    actual_n: int
    actual_k: int

    # Slice information for source matrices
    a_slice: Tuple[int, int, int, int]  # [m_start, m_end, k_start, k_end]
    b_slice: Tuple[int, int, int, int]  # [k_start, k_end, n_start, n_end]
    c_slice: Tuple[int, int, int, int]  # [m_start, m_end, n_start, n_end]

    # Computation properties
    accumulate: bool = False  # Whether to accumulate to existing C value
    tile_type: TileType = TileType.COMPUTE

    # Optimization hints
    priority: int = 0  # Higher priority tiles should be scheduled first
    dependencies: Optional[List[int]] = None  # Tile IDs this depends on

    def __post_init__(self):
        """Validate and compute derived properties"""
        if self.dependencies is None:
            self.dependencies = []

        # Validate dimensions
        assert 1 <= self.actual_m <= 16, f"Invalid M dimension: {self.actual_m}"
        assert 1 <= self.actual_n <= 16, f"Invalid N dimension: {self.actual_n}"
        assert 1 <= self.actual_k <= 16, f"Invalid K dimension: {self.actual_k}"

    @property
    def compute_volume(self) -> int:
        """Calculate the computational volume (number of MACs)"""
        if self.tile_type == TileType.NOP:
            return 0
        return self.actual_m * self.actual_n * self.actual_k

    @property
    def padding_ratio(self) -> float:
        """Calculate padding ratio for tiles smaller than 16x16x16"""
        if self.tile_type == TileType.NOP:
            return 1.0
        max_volume = 16 * 16 * 16
        actual_volume = self.compute_volume
        return 1.0 - (actual_volume / max_volume)

    @property
    def group_key(self) -> Tuple[int, int]:
        """
        Group key for constraint enforcement.
        Tiles with the same (matrix_id, k_iteration) represent the same
        K-reduction step and may need to be scheduled together.
        """
        return (self.matrix_id, self.k_iteration)

    @property
    def is_edge_tile(self) -> bool:
        """Check if this is an edge tile (not full 16x16x16)"""
        return (self.actual_m < 16 or
                self.actual_n < 16 or
                self.actual_k < 16)

    def create_nop(self) -> 'TileInfo':
        """Create a NOP tile for padding"""
        return TileInfo(
            tile_id=-1,
            matrix_id=-1,
            m_offset=0,
            n_offset=0,
            k_iteration=0,
            actual_m=0,
            actual_n=0,
            actual_k=0,
            a_slice=(0, 0, 0, 0),
            b_slice=(0, 0, 0, 0),
            c_slice=(0, 0, 0, 0),
            accumulate=False,
            tile_type=TileType.NOP
        )


@dataclass
class TileDescriptor:
    """
    Legacy compatibility wrapper for existing code.
    Maps to TileInfo for backward compatibility.
    """

    tile_id: int
    matrix_id: int
    k_iteration: int
    m_offset: int
    n_offset: int
    actual_m: int
    actual_n: int
    actual_k: int

    def to_tile_info(self) -> TileInfo:
        """Convert to TileInfo"""
        return TileInfo(
            tile_id=self.tile_id,
            matrix_id=self.matrix_id,
            m_offset=self.m_offset,
            n_offset=self.n_offset,
            k_iteration=self.k_iteration,
            actual_m=self.actual_m,
            actual_n=self.actual_n,
            actual_k=self.actual_k,
            a_slice=(self.m_offset, self.m_offset + self.actual_m,
                    self.k_iteration * 16, self.k_iteration * 16 + self.actual_k),
            b_slice=(self.k_iteration * 16, self.k_iteration * 16 + self.actual_k,
                    self.n_offset, self.n_offset + self.actual_n),
            c_slice=(self.m_offset, self.m_offset + self.actual_m,
                    self.n_offset, self.n_offset + self.actual_n),
            accumulate=(self.k_iteration > 0)
        )

    @property
    def num_elements(self) -> int:
        return self.actual_m * self.actual_n

    @property
    def padding_ratio(self) -> float:
        return 1.0 - (self.num_elements / 256.0)

    @property
    def group_key(self) -> Tuple[int, int]:
        return (self.matrix_id, self.k_iteration)


def generate_tiles_for_matmul(
    m: int, n: int, k: int,
    matrix_id: int = 0,
    tile_size: Tuple[int, int, int] = (16, 16, 16)
) -> List[TileInfo]:
    """
    Generate tiles for a single matrix multiplication.

    Args:
        m, n, k: Matrix dimensions (M×K @ K×N)
        matrix_id: Unique identifier for this matmul
        tile_size: Tile dimensions (tile_m, tile_n, tile_k)

    Returns:
        List of TileInfo objects representing all tiles
    """
    tile_m, tile_n, tile_k = tile_size
    tiles = []
    tile_id = 0

    for m_start in range(0, m, tile_m):
        m_end = min(m_start + tile_m, m)
        actual_m = m_end - m_start

        for n_start in range(0, n, tile_n):
            n_end = min(n_start + tile_n, n)
            actual_n = n_end - n_start

            # K dimension requires accumulation
            for k_iter, k_start in enumerate(range(0, k, tile_k)):
                k_end = min(k_start + tile_k, k)
                actual_k = k_end - k_start

                tile = TileInfo(
                    tile_id=tile_id,
                    matrix_id=matrix_id,
                    m_offset=m_start,
                    n_offset=n_start,
                    k_iteration=k_iter,
                    actual_m=actual_m,
                    actual_n=actual_n,
                    actual_k=actual_k,
                    a_slice=(m_start, m_end, k_start, k_end),
                    b_slice=(k_start, k_end, n_start, n_end),
                    c_slice=(m_start, m_end, n_start, n_end),
                    accumulate=(k_iter > 0),  # First K tile initializes, others accumulate
                    tile_type=TileType.COMPUTE
                )
                tiles.append(tile)
                tile_id += 1

    return tiles


def generate_tiles_for_multiple_matmuls(
    matmul_specs: List[dict]
) -> List[TileInfo]:
    """
    Generate tiles for multiple matrix multiplications.

    Args:
        matmul_specs: List of dicts with 'm', 'n', 'k' keys

    Returns:
        Combined list of TileInfo objects with unique tile_ids and matrix_ids
    """
    all_tiles = []
    global_tile_id = 0

    for matrix_id, spec in enumerate(matmul_specs):
        m, n, k = spec['m'], spec['n'], spec['k']
        tiles = generate_tiles_for_matmul(m, n, k, matrix_id)

        # Update tile IDs to be globally unique
        for tile in tiles:
            tile.tile_id = global_tile_id
            global_tile_id += 1
            all_tiles.append(tile)

    return all_tiles