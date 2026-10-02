# Stores

Entity resolution is iterative, long-running, and can easily explode in compute and memory. matchlab uses caching for stability and sharing. Stores are how you configure it.

## Provisional collection

`Transaction` is the abstract contract for a provisional artifact view.
`DuckDBStore.transaction()` creates a `DuckDBTransaction`. It reads existing artifacts
from its parent Store and stages new writes in a private schema of the same database.
An in-memory parent keeps staging in memory. A file-backed parent also writes staging
to its file. Ordinary plans can collect against either without changing their steps.

```python
store = DuckDBStore("matching.duckdb")
tx = store.transaction()
try:
    result = plan.collect(store=tx)
    candidates = result.entities()
    tx.commit()  # Or tx.rollback() to discard the staged artifacts.
finally:
    tx.close()
```

Commit copies all staged artifacts into the parent's committed schema and removes
the staging schema in one native DuckDB transaction. Results collected
through the transaction remain readable after commit. Rollback discards staging.
Reads through a rolled-back transaction raise an error. Collect the plan again into
a live Store to use it.

This is an experiment, not online matching or incremental ingestion. Keep the parent
open and unchanged while a transaction is active. Transactions do not provide snapshot
isolation or concurrency guarantees. They do not support publication, evaluation,
pruning, or nesting. Collection summaries count staged artifacts, not durable growth.
Closing a transaction never closes its parent.

Staging uses its own connection cursor and schema setting. It does not change the
parent's default schema. Rollback drops staging but does not shrink a database file.
Staging is disposable even when stored on disk. A process crash can leave an abandoned
schema. There is no resume or recovery API.

::: matchlab.stores.default
    options:
        show_root_heading: true
        show_root_full_path: true
        members_order: source
        show_if_no_docstring: true
        docstring_style: google
        show_signature_annotations: true
        separate_signature: true
        filters:
            - "!^[A-Z]$"
            - "!^_"

::: matchlab.stores.base
    options:
        show_root_heading: true
        show_root_full_path: true
        members_order: source
        show_if_no_docstring: true
        docstring_style: google
        show_signature_annotations: true
        separate_signature: true
        filters:
            - "!^[A-Z]$"
            - "!^_"
