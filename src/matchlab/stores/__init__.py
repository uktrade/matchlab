"""Stores that cache data for matchlab.

A store persists collected DAG-step artifacts keyed by fingerprint. `DuckDBStore` is t
he reference implementation.
"""

from matchlab.stores.base import (
    Fingerprint,
    PruneResult,
    Store,
    StoreStats,
    Transaction,
    TransactionStats,
    format_bytes,
)
from matchlab.stores.duckdb import DuckDBStore, DuckDBStoreStats, DuckDBTransaction

__all__ = (
    "Store",
    "DuckDBStore",
    "DuckDBStoreStats",
    "Fingerprint",
    "PruneResult",
    "StoreStats",
    "Transaction",
    "TransactionStats",
    "DuckDBTransaction",
    "format_bytes",
)
