from __future__ import annotations

from pathlib import Path
import numpy as np


class EmbeddingTable:
    """
    bf16 memory map embedding lookup.
    Lookup(ids) returns float32 (len(ids), dim).
    Exact conversion: (memmap[ids].astype(np.uint32) << 16).view(np.float32)
    """
    def __init__(self, path: str | Path, rows: int, dim: int) -> None:
        self.path = Path(path)
        self.rows = int(rows)
        self.dim = int(dim)
        # raw little-endian uint16
        self.mmap = np.memmap(self.path, dtype="<u2", mode="r", shape=(self.rows, self.dim))

    def lookup(self, ids: np.ndarray | list[int]) -> np.ndarray:
        ids_arr = np.asarray(ids, dtype=np.int64)
        bf16_rows = self.mmap[ids_arr]  # (..., dim) uint16
        u32 = bf16_rows.astype(np.uint32) << 16
        fp32 = u32.view(np.float32)
        return np.ascontiguousarray(fp32)
