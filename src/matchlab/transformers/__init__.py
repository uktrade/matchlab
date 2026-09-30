"""Pluggable, serialisable reshapings of a record step.

`Select` projects, `Clean` derives, `Group` collapses each `id` to one row, and
`Explode` cross-joins each `id`'s columns back out to every combination, each one job.
Register a custom one with `add_transformer_class`, as `add_model_class` does a
deduper.
"""

from matchlab.transformers.base import PreparedTransformer, Transformer
from matchlab.transformers.clean import Clean
from matchlab.transformers.explode import Explode
from matchlab.transformers.group import Group
from matchlab.transformers.select import Select
from matchlab.transformers.transform import Transform, add_transformer_class

__all__ = (
    "Clean",
    "Explode",
    "Group",
    "PreparedTransformer",
    "Select",
    "Transform",
    "Transformer",
    "add_transformer_class",
)
