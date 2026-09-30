"""Select projects. Keep the named columns, drop the rest."""

from typing import ClassVar

import polars as pl
from pydantic import Field

from matchlab.transformers.base import Transformer


class Select(Transformer):
    """Keep only the named columns, plus `id`.

    Projection is `Select`'s one job. Naming a column keeps it, and every column you do
    not name is dropped. `id` survives whether or not you name it, since it is the
    grouping a model matches on. Columns are the source-qualified names (`crn_company`).
    """

    version: ClassVar[int] = 1

    columns: tuple[str, ...] = Field(
        default=(),
        description="The columns to keep, by their current (qualified) names.",
    )

    def __init__(self, *columns: str, **data: object) -> None:
        """Take the columns to keep positionally, e.g. `Select('crn_company')`."""
        if columns:
            data["columns"] = columns
        super().__init__(**data)

    def apply(
        self, prepared_state: object, data: pl.DataFrame, *, baseline: pl.DataFrame
    ) -> pl.DataFrame:
        """Project to `id` and the named columns, in that order."""
        kept = [column for column in self.columns if column != "id"]
        return data.select("id", *kept)
