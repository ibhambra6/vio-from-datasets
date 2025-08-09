"""Dataset loader scaffolds for EuRoC and TUM-VI.

These provide minimal structure and types to later add real-world dataset support
without changing the core project goal. They avoid extra heavy dependencies.
"""

from .base import DatasetSequence, TrajectorySample

__all__ = [
    "DatasetSequence",
    "TrajectorySample",
]
