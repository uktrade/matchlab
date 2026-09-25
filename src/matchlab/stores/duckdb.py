"""The DuckDB store for matchlab.

A single DuckDB database (a file, or `:memory:`) holds every collected artifact,
keyed by step fingerprint. There is no engine here that resolves on demand. Resolvers
arrive already materialised (merge-forward), so reads are plain table scans. Analysts
can point their own SQL at the `resolver_output` table. That is the whole point.
"""

import os
from collections.abc import Iterable, Mapping
from pathlib import Path

import duckdb
import polars as pl

from matchlab.core.kinds import StepKind
from matchlab.core.resolver_output import root_id_of
from matchlab.core.schemas import (
    SCHEMA_CLUSTER_EXPANSION,
    SCHEMA_JUDGEMENTS,
    SCHEMA_MODEL_EDGES,
    SCHEMA_RESOLVER_OUTPUT,
    check_schema_subset,
)
from matchlab.eval.judgements import Judgement
from matchlab.stores.base import Fingerprint, PruneResult, Store, StoreStats

# Bumped whenever the stored shape changes, or stored IDs stop meaning what they did.
# A store written by an older matchlab is recreated rather than half-read. That's the
# honest failure mode for a cache.
_SCHEMA_VERSION = 6

_SCHEMA_DDL = """
CREATE TABLE IF NOT EXISTS meta (
    schema_version INTEGER
);
-- `fp` is BLOB in every table below. A fingerprint is a 32-byte SHA-256 digest, and
-- DuckDB's widest integer holds only 16 bytes, so an integer key would mean truncating
-- the hash.
--
-- Truncating would buy nothing. Every `fp` predicate here is scalar equality. A store
-- holds few fingerprints against many rows, so the column dictionary-compresses to a
-- code per row. The zonemaps prune whole row groups regardless (see `source_leaves`
-- below). Measured against a UBIGINT key over 20M rows, the BLOB is no slower laid out
-- that way, and it is faster interleaved, where random 64-bit keys do not compress.
--
-- Truncation is a bad trade too. A leaf ID collision shows up as a wrong merge in the
-- data. A fingerprint collision is worse: it is silent. It makes `_ensure` skip the
-- step and read back a different artifact.
CREATE TABLE IF NOT EXISTS artifacts (
    fp BLOB PRIMARY KEY, kind VARCHAR
);
-- A label: a pointer from a string someone chose to the resolver output they want to
-- find again. It is kept separate from `artifacts` because labelling is an act, not a
-- property. Most artifacts carry no label, and moving a label to a newer fingerprint
-- never disturbs the artifact it used to point at.
CREATE TABLE IF NOT EXISTS labels (
    label VARCHAR PRIMARY KEY, fp BLOB, published_at TIMESTAMP
);
-- A stored source knows its own key column, so its extract can be read back and
-- joined to a resolver output without the plan that built it.
CREATE TABLE IF NOT EXISTS source_meta (
    fp BLOB PRIMARY KEY, key_field VARCHAR
);
-- Which source artifacts a resolver's output was built from. A resolver's output
-- records source *names*, and one store can hold several generations of the same name.
CREATE TABLE IF NOT EXISTS resolver_output_sources (
    fp BLOB, source_name VARCHAR, source_fp BLOB
);
-- The three tables below hold every artifact of their kind side by side. They are the
-- only ones here that grow with the data rather than with the plan.
--
-- Every read of them is `WHERE fp = ?` for one artifact, and none carries an index.
-- That works because the write path already sorts them: each artifact arrives as
-- exactly one INSERT, so its rows are appended as one contiguous run, and no row group
-- (~122k rows) straddles two artifacts of any size. DuckDB keeps min/max statistics
-- per row group, so a scan for one fingerprint skips the others' groups without
-- decompressing them. On a digest, whose leading bytes are already random, the 8-byte
-- prefix those statistics truncate to discriminates as well as the whole value would.
--
-- This is a property of how we write, not one the storage layer enforces. Writing an
-- artifact in several statements, or interleaving two artifacts' writes, would scatter
-- each across row groups whose min/max then span both fingerprints. That prunes 
-- nothing. Nothing else disturbs the layout: `_purge` marks rows deleted in place, a 
-- re-collect appends afresh, and `_rewrite` copies in scan order. Artifacts small 
-- enough to share a row group prune poorly, which costs no more than scanning them 
-- would anyway.
CREATE TABLE IF NOT EXISTS source_leaves (
    fp BLOB, key VARCHAR, leaf UBIGINT
);
CREATE TABLE IF NOT EXISTS model_edges (
    fp BLOB, left_id UBIGINT, right_id UBIGINT, score FLOAT
);
CREATE TABLE IF NOT EXISTS resolver_output (
    fp BLOB, root UBIGINT, leaf UBIGINT, key VARCHAR, source VARCHAR
);
CREATE TABLE IF NOT EXISTS judgements (
    tag VARCHAR, user_name VARCHAR, endorsed UBIGINT, shown UBIGINT
);
CREATE TABLE IF NOT EXISTS expansion (
    root UBIGINT PRIMARY KEY, leaves UBIGINT[]
);
"""


class DuckDBStoreStats(StoreStats):
    """What a DuckDB store holds, plus what only a file of blocks can report.

    Attributes:
        path: The database file, or `None` for `:memory:`. `location` already names the
            store. This is the file itself, for code that wants to `stat` or delete it.
        free_bytes: Space already freed inside the file. DuckDB reuses those blocks for
            later writes, but never returns them to the OS. This is the gap between
            what the store weighs and what it holds. It is the only figure that says
            what a reclaim could recover, without first deciding what to delete. It has
            no meaning for a backend that is not a file of reusable blocks. That is
            why it lives here rather than on `StoreStats`.
    """

    path: Path | None = None
    free_bytes: int = 0

    @property
    def size(self) -> str:
        """Say when the bytes are resident rather than written.

        `4.6 MB` reads as a disk size, but for `:memory:` it isn't one. The store
        vanishes with the process. The distinction matters most exactly when a user is
        least likely to be thinking about it.
        """
        return super().size if self.path is not None else f"{super().size} in memory"


def _mint_cluster_id(leaves: list[int]) -> int:
    """Deterministic, content-addressed cluster ID for a group of leaves.

    A singleton maps to the leaf itself, so evaluation's singleton-expansion fallback
    (`process_judgements`) works without an explicit expansion row.

    A group uses `root_id`, the same function a resolver mints its roots with. That
    match is load-bearing, not just tidy. Scoring compares a judged group against the
    resolver output's clusters by ID. If the two ever disagree, every comparison
    misses, and precision and recall get computed over an empty set.
    """
    if len(leaves) == 1:
        return int(leaves[0])
    return root_id_of(leaves)


class DuckDBStore(Store):
    """Store collected artifacts in a DuckDB database, keyed by fingerprint."""

    def __init__(self, path: str | Path = ":memory:") -> None:
        """Open (or create) the store at `path`. Use `:memory:` for ephemeral stores."""
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = duckdb.connect(self.path)
        self._open_schema()

    def _open_schema(self) -> None:
        """Create the schema, recreating the store if it predates this version."""
        existing = self.conn.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_name = 'meta'"
        ).fetchone()
        if existing:
            row = self.conn.execute("SELECT schema_version FROM meta").fetchone()
            if row and row[0] == _SCHEMA_VERSION:
                return

        # Either a pre-versioned store or an older version, so start clean. Artifacts
        # are a cache, and everything in here can be recomputed from the plan that
        # made it.
        for (table,) in self.conn.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = 'main'"
        ).fetchall():
            self.conn.execute(f'DROP TABLE IF EXISTS "{table}" CASCADE')

        self.conn.execute(_SCHEMA_DDL)
        self.conn.execute("INSERT INTO meta VALUES (?)", [_SCHEMA_VERSION])

    # -- helpers ----------------------------------------------------------------------

    @staticmethod
    def _extract_table(fp: Fingerprint) -> str:
        """Name of the per-source extract table (arbitrary-schema, so its own table)."""
        return f"extract_{fp.hex()}"

    @staticmethod
    def _transform_table(fp: Fingerprint) -> str:
        """Name of a per-transform materialised table (arbitrary-schema, so its own)."""
        return f"transform_{fp.hex()}"

    def _register(self, name: str, df: pl.DataFrame) -> None:
        self.conn.register(name, df.to_arrow())

    def _kind(self, fp: Fingerprint) -> StepKind | None:
        row = self.conn.execute(
            "SELECT kind FROM artifacts WHERE fp = ?", [fp]
        ).fetchone()
        return StepKind(row[0]) if row else None

    def _register_artifact(self, fp: Fingerprint, kind: StepKind) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO artifacts VALUES (?, ?)", [fp, kind.value]
        )

    def _purge(self, fp: Fingerprint) -> None:
        """Remove any existing artifact for `fp`, so stores are idempotent.

        Only ever called immediately before storing that same fingerprint again. A
        fingerprint addresses content, so the replacement is the same data by
        construction. That is why any **label** pointing at `fp` is left alone. It
        still resolves, to bytes indistinguishable from the ones it resolved to before.
        A re-collect should not quietly revoke a publication.
        """
        kind = self._kind(fp)
        if kind is None:
            return
        if kind is StepKind.SOURCE:
            self.conn.execute(f'DROP TABLE IF EXISTS "{self._extract_table(fp)}"')
            self.conn.execute("DELETE FROM source_leaves WHERE fp = ?", [fp])
            self.conn.execute("DELETE FROM source_meta WHERE fp = ?", [fp])
        elif kind is StepKind.TRANSFORM:
            self.conn.execute(f'DROP TABLE IF EXISTS "{self._transform_table(fp)}"')
        elif kind is StepKind.MODEL:
            self.conn.execute("DELETE FROM model_edges WHERE fp = ?", [fp])
        elif kind is StepKind.RESOLVER:
            self.conn.execute("DELETE FROM resolver_output WHERE fp = ?", [fp])
            self.conn.execute("DELETE FROM resolver_output_sources WHERE fp = ?", [fp])
        self.conn.execute("DELETE FROM artifacts WHERE fp = ?", [fp])

    # -- existence --------------------------------------------------------------------

    def has(self, fp: Fingerprint) -> bool:
        return self._kind(fp) is not None

    # -- introspection ----------------------------------------------------------------

    def stats(self) -> DuckDBStoreStats:
        """Report the store's size and contents.

        **Checkpoints a file store before measuring it.** That makes this the one
        method here that writes without being asked to. Without it, the figure is not
        just imprecise. It is the wrong order of magnitude. Recent writes sit in the
        write-ahead log as a compact journal, and settling them into 256 KB blocks can
        turn 21 KB of log into 5.5 MB of file. Measured on `examples/companies`: 33 KB
        reported against a store that became 5.5 MB the moment it was closed. Reporting
        a size that a user's next `du` contradicts by 170x is worse than reporting none.

        DuckDB's own block count is no help before that point. It reads zero until a
        checkpoint has happened. Afterwards the two agree to within the file header, so
        `stat` is used instead. It counts the `.wal` sibling too, and it is the number
        a user can actually check.

        The checkpoint is cheap, because DuckDB has usually already done most of it:
        1.4 ms after writing 10M rows, 0.08 ms when there is nothing pending.

        `free_blocks` still comes from DuckDB. Nothing outside the file can see how
        much of it is reusable, and an in-memory store's size is no different: it
        allocates no blocks and has no file to measure.
        """
        if self.path == ":memory:":
            # `sum` over an empty table is NULL, so an untouched store needs the `or 0`
            # as much as a missing row does.
            resident = self.conn.execute(
                "SELECT sum(memory_usage_bytes) FROM duckdb_memory()"
            ).fetchone()
            size = (resident[0] if resident else 0) or 0
            free, path = 0, None
        else:
            self.conn.execute("CHECKPOINT")
            blocks = self.conn.execute(
                "SELECT block_size, free_blocks FROM pragma_database_size()"
            ).fetchone()
            free = blocks[0] * blocks[1] if blocks else 0
            # `self.path` is the constructor's argument verbatim, so it's resolved
            # here. Otherwise a store opened as "./run.duckdb" would report itself
            # against a working directory that has since moved on.
            path = Path(self.path).resolve()
            size = sum(
                candidate.stat().st_size
                for candidate in (path, path.with_name(f"{path.name}.wal"))
                if candidate.exists()
            )

        labelled = self.conn.execute("SELECT count(*) FROM labels").fetchone()
        return DuckDBStoreStats(
            location=self.path,
            path=path,
            bytes=int(size),
            artifacts=dict(
                self.conn.execute(
                    "SELECT kind, count(*) FROM artifacts GROUP BY kind"
                ).fetchall()
            ),
            labels=labelled[0] if labelled else 0,
            free_bytes=free,
        )

    # -- sources ----------------------------------------------------------------------

    def store_source(
        self,
        fp: Fingerprint,
        key_field: str,
        extract: pl.DataFrame,
        leaves: pl.DataFrame,
    ) -> None:
        missing = {"key", "leaf"} - set(leaves.columns)
        if missing:
            raise ValueError(f"leaves is missing columns: {sorted(missing)}")

        self._purge(fp)

        self._register("_reg_extract", extract)
        self.conn.execute(
            f'CREATE OR REPLACE TABLE "{self._extract_table(fp)}" '
            "AS SELECT * FROM _reg_extract"
        )
        self.conn.unregister("_reg_extract")

        leaves = leaves.select(
            pl.col("key").cast(pl.Utf8), pl.col("leaf").cast(pl.UInt64)
        )
        self._register("_reg_leaves", leaves)
        self.conn.execute(
            "INSERT INTO source_leaves SELECT ? AS fp, key, leaf FROM _reg_leaves", [fp]
        )
        self.conn.unregister("_reg_leaves")

        self.conn.execute("INSERT INTO source_meta VALUES (?, ?)", [fp, key_field])
        self._register_artifact(fp, StepKind.SOURCE)

    def read_source_extract(self, fp: Fingerprint) -> pl.DataFrame:
        if self._kind(fp) is not StepKind.SOURCE:
            raise KeyError(f"No stored source for fingerprint {fp.hex()}")
        return self.conn.execute(f'SELECT * FROM "{self._extract_table(fp)}"').pl()

    def read_source_leaves(self, fp: Fingerprint) -> pl.DataFrame:
        if self._kind(fp) is not StepKind.SOURCE:
            raise KeyError(f"No stored source for fingerprint {fp.hex()}")
        return self.conn.execute(
            "SELECT key, leaf FROM source_leaves WHERE fp = ?", [fp]
        ).pl()

    # -- identifiers ------------------------------------------------------------------

    def read_identifiers(
        self,
        source_fp: Fingerprint,
        source_name: str,
        resolver_fp: Fingerprint | None = None,
        ids: set[int] | None = None,
    ) -> pl.DataFrame:
        if self._kind(source_fp) is not StepKind.SOURCE:
            raise KeyError(f"No stored source for fingerprint {source_fp.hex()}")

        if resolver_fp is None:
            if ids is not None:
                return self.conn.execute(
                    "SELECT leaf AS id, CAST(? AS VARCHAR) AS source, key, leaf "
                    "FROM source_leaves WHERE fp = ? "
                    "AND leaf IN (SELECT UNNEST(?::UBIGINT[]))",
                    [source_name, source_fp, list(ids)],
                ).pl()
            return self.conn.execute(
                "SELECT leaf AS id, CAST(? AS VARCHAR) AS source, key, leaf "
                "FROM source_leaves WHERE fp = ?",
                [source_name, source_fp],
            ).pl()

        if self._kind(resolver_fp) is not StepKind.RESOLVER:
            raise KeyError(f"No stored resolver for fingerprint {resolver_fp.hex()}")
        # Both predicates matter. `source` keeps this from scanning every source's
        # rows, and `resolver_output` holds one generation per collect of the plan.
        if ids is not None:
            return self.conn.execute(
                "SELECT root AS id, source, key, leaf "
                "FROM resolver_output WHERE fp = ? AND source = ? "
                "AND root IN (SELECT UNNEST(?::UBIGINT[]))",
                [resolver_fp, source_name, list(ids)],
            ).pl()
        return self.conn.execute(
            "SELECT root AS id, source, key, leaf "
            "FROM resolver_output WHERE fp = ? AND source = ?",
            [resolver_fp, source_name],
        ).pl()

    # -- maintenance ------------------------------------------------------------------

    def _keep_set(self, keep: Iterable[Fingerprint]) -> set[Fingerprint]:
        """Every fingerprint that must survive a prune.

        Published labels always go in. Each one also drags in the sources its
        resolver output needs. Reading a published resolver output without a plan goes
        through `resolver_output_sources` to each source's extract. Without them, a
        kept label resolves to a fingerprint whose data is gone.
        """
        kept: set[Fingerprint] = set()
        for item in keep:
            if not isinstance(item, bytes):
                raise TypeError("`prune(keep=...)` expects fingerprints.")
            kept.add(item)

        for label in self.labels():
            kept |= self._label_closure(label)

        return kept

    def _label_closure(self, label: str) -> set[Fingerprint]:
        """A label's resolver-output fingerprint, plus the sources it reads through."""
        fp = self.find(label)
        if fp is None:
            raise ValueError(
                f"No resolver output is labelled '{label}' in this store. "
                f"Known labels: {', '.join(self.labels()) or 'none'}."
            )
        return {fp, *self.resolver_output_sources(fp).values()}

    def prune(self, keep: Iterable[Fingerprint] = ()) -> PruneResult:
        """Delete every artifact except the ones named, and reclaim what that frees.

        Deleting is only half of it. DuckDB marks freed blocks for reuse but never
        hands them back to the OS, so purging alone does not shrink the file at all.
        Measured on a real 575 MB store, deleting 77% of its artifacts freed 0 bytes.
        The space comes back only by rewriting the database, copying what is left into a
        fresh file and swapping it in. Purge and rewrite together recovered 437 MB of
        that store, in half a second.

        The swap is a rename over the original. A failure anywhere leaves the store
        exactly as it was, with the half-written copy orphaned beside it.

        **This reopens the connection.** That is the one internal detail anything
        outside this class needs to know about. Session settings applied through
        `store.conn` (`memory_limit` and `temp_directory`, as the guide suggests) do
        not survive. The store itself stays valid, so anything holding *it*, rather
        than its connection, is unaffected.

        An in-memory store is purged but not rewritten. It has no file, and reopening
        one would hand back an empty database rather than a smaller one. Its freed
        blocks return to the allocator anyway, so the reclaim is real regardless.
        """
        kept = self._keep_set(keep)
        stored = {
            fp for (fp,) in self.conn.execute("SELECT fp FROM artifacts").fetchall()
        }
        doomed = stored - kept

        if not kept and stored:
            raise ValueError(
                "Pruning with nothing to keep would empty the store. Name a plan to "
                "keep, or delete the file instead if that is what you meant."
            )

        before = self.stats().bytes
        for fp in doomed:
            self._purge(fp)

        if self.path != ":memory:":
            self._rewrite()

        return PruneResult(
            removed=len(doomed),
            kept=len(stored) - len(doomed),
            reclaimed=max(before - self.stats().bytes, 0),
        )

    def _rewrite(self) -> None:
        """Copy the store into a fresh file and swap it in, returning freed space.

        The temp file sits beside the store rather than in a temp directory, so the
        swap is a rename within one filesystem and therefore atomic.
        """
        path = Path(self.path).resolve()
        temporary = path.with_name(f"{path.name}.prune-tmp")
        temporary.unlink(missing_ok=True)

        database = self.conn.execute(
            "SELECT database_name FROM duckdb_databases() WHERE NOT internal LIMIT 1"
        ).fetchone()
        if database is None:  # a connected store always has one
            return

        self.conn.execute("CHECKPOINT")
        self.conn.execute(f"ATTACH '{temporary}' AS pruned")
        self.conn.execute(f'COPY FROM DATABASE "{database[0]}" TO pruned')
        self.conn.execute("DETACH pruned")
        self.conn.close()

        os.replace(temporary, path)
        self.conn = duckdb.connect(self.path)
        self._open_schema()

    # -- models -----------------------------------------------------------------------

    def store_model(self, fp: Fingerprint, edges: pl.DataFrame) -> None:
        check_schema_subset(SCHEMA_MODEL_EDGES, edges.to_arrow().schema)
        self._purge(fp)
        edges = edges.select(
            pl.col("left_id").cast(pl.UInt64),
            pl.col("right_id").cast(pl.UInt64),
            pl.col("score").cast(pl.Float32),
        )
        self._register("_reg_edges", edges)
        self.conn.execute(
            "INSERT INTO model_edges "
            "SELECT ? AS fp, left_id, right_id, score FROM _reg_edges",
            [fp],
        )
        self.conn.unregister("_reg_edges")
        self._register_artifact(fp, StepKind.MODEL)

    def read_model(self, fp: Fingerprint) -> pl.DataFrame:
        if self._kind(fp) is not StepKind.MODEL:
            raise KeyError(f"No stored model for fingerprint {fp.hex()}")
        return self.conn.execute(
            "SELECT left_id, right_id, score FROM model_edges WHERE fp = ?", [fp]
        ).pl()

    # -- transforms -------------------------------------------------------------------

    def store_transform(self, fp: Fingerprint, table: pl.DataFrame) -> None:
        self._purge(fp)
        self._register("_reg_transform", table)
        self.conn.execute(
            f'CREATE OR REPLACE TABLE "{self._transform_table(fp)}" '
            "AS SELECT * FROM _reg_transform"
        )
        self.conn.unregister("_reg_transform")
        self._register_artifact(fp, StepKind.TRANSFORM)

    def read_transform(self, fp: Fingerprint) -> pl.DataFrame:
        if self._kind(fp) is not StepKind.TRANSFORM:
            raise KeyError(f"No stored transform for fingerprint {fp.hex()}")
        return self.conn.execute(f'SELECT * FROM "{self._transform_table(fp)}"').pl()

    # -- resolvers --------------------------------------------------------------------

    def store_resolver(
        self,
        fp: Fingerprint,
        resolver_output: pl.DataFrame,
        sources: Mapping[str, Fingerprint] | None = None,
    ) -> None:
        check_schema_subset(SCHEMA_RESOLVER_OUTPUT, resolver_output.to_arrow().schema)
        self._purge(fp)
        resolver_output = resolver_output.select(
            pl.col("root").cast(pl.UInt64),
            pl.col("leaf").cast(pl.UInt64),
            pl.col("key").cast(pl.Utf8),
            pl.col("source").cast(pl.Utf8),
        )
        self._register("_reg_res", resolver_output)
        self.conn.execute(
            "INSERT INTO resolver_output "
            "SELECT ? AS fp, root, leaf, key, source FROM _reg_res",
            [fp],
        )
        self.conn.unregister("_reg_res")
        for source_name, source_fp in (sources or {}).items():
            self.conn.execute(
                "INSERT INTO resolver_output_sources VALUES (?, ?, ?)",
                [fp, source_name, source_fp],
            )
        self._register_artifact(fp, StepKind.RESOLVER)

    def read_resolver(self, fp: Fingerprint) -> pl.DataFrame:
        if self._kind(fp) is not StepKind.RESOLVER:
            raise KeyError(f"No stored resolver for fingerprint {fp.hex()}")
        return self.conn.execute(
            "SELECT root, leaf, key, source FROM resolver_output WHERE fp = ?", [fp]
        ).pl()

    # -- evaluation -------------------------------------------------------------------

    def store_judgement(self, judgement: Judgement, user_name: str = "local") -> None:
        shown_id = _mint_cluster_id(judgement.shown)
        self._upsert_expansion(shown_id, judgement.shown)

        for group in judgement.endorsed:
            endorsed_id = _mint_cluster_id(group)
            self._upsert_expansion(endorsed_id, group)
            self.conn.execute(
                "INSERT INTO judgements VALUES (?, ?, ?, ?)",
                [judgement.tag, user_name, endorsed_id, shown_id],
            )

    def _upsert_expansion(self, root: int, leaves: list[int]) -> None:
        self.conn.execute(
            "INSERT INTO expansion (root, leaves) "
            "SELECT ?, CAST(? AS UBIGINT[]) ON CONFLICT (root) DO NOTHING",
            [root, sorted(int(leaf) for leaf in leaves)],
        )

    def read_eval_data(
        self, tag: str | None = None
    ) -> tuple[pl.DataFrame, pl.DataFrame]:
        if tag is None:
            judgements = self.conn.execute(
                "SELECT user_name, endorsed, shown FROM judgements"
            ).pl()
        else:
            judgements = self.conn.execute(
                "SELECT user_name, endorsed, shown FROM judgements WHERE tag = ?",
                [tag],
            ).pl()
        expansion = self.conn.execute("SELECT root, leaves FROM expansion").pl()

        # Present empty results with the right columns/dtypes. We deliberately do not
        # re-validate against the arrow transport schemas here. Those pin `leaves` to a
        # small `list`, while polars naturally emits `large_list`. That's a
        # serialisation detail that is meaningless locally. Types are guaranteed by the
        # table DDL, and inputs are validated on write.
        if judgements.height == 0:
            judgements = pl.DataFrame(schema=pl.Schema(SCHEMA_JUDGEMENTS))
        if expansion.height == 0:
            expansion = pl.DataFrame(schema=pl.Schema(SCHEMA_CLUSTER_EXPANSION))

        return judgements, expansion

    def sample(
        self, resolver_fp: Fingerprint, n: int, seed: int | None = None
    ) -> pl.DataFrame:
        resolver_output = self.read_resolver(resolver_fp)
        roots = resolver_output["root"].unique()
        if n < roots.len():
            roots = roots.sample(n=n, seed=seed, shuffle=True)
        return resolver_output.filter(pl.col("root").is_in(roots.to_list())).select(
            "root", "leaf", "key", "source"
        )

    # -- lookups ----------------------------------------------------------------------

    def publish(self, label: str, fp: Fingerprint) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO labels VALUES (?, ?, now())", [label, fp]
        )

    def find(self, label: str) -> Fingerprint | None:
        row = self.conn.execute(
            "SELECT fp FROM labels WHERE label = ?", [label]
        ).fetchone()
        return row[0] if row else None

    def labels(self) -> list[str]:
        return [
            row[0]
            for row in self.conn.execute(
                "SELECT label FROM labels ORDER BY label"
            ).fetchall()
        ]

    def source_key_field(self, fp: Fingerprint) -> str:
        row = self.conn.execute(
            "SELECT key_field FROM source_meta WHERE fp = ?", [fp]
        ).fetchone()
        if row is None:
            raise KeyError(f"No stored source for fingerprint {fp.hex()}")
        return row[0]

    def resolver_output_sources(self, fp: Fingerprint) -> dict[str, Fingerprint]:
        return {
            name: source_fp
            for name, source_fp in self.conn.execute(
                "SELECT source_name, source_fp FROM resolver_output_sources "
                "WHERE fp = ?",
                [fp],
            ).fetchall()
        }

    # -- lifecycle --------------------------------------------------------------------

    def close(self) -> None:
        self.conn.close()
