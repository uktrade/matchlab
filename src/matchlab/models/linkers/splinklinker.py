"""A linking methodology leveraging Splink."""

import inspect
import json
from copy import deepcopy
from typing import Any, ClassVar, Literal, Protocol

import duckdb
import polars as pl
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_serializer,
    field_validator,
    model_validator,
)
from pydantic.functional_serializers import SerializationInfo
from splink import DuckDBAPI, SettingsCreator
from splink import Linker as SplinkLibLinkerClass
from splink.internals.linker_components.training import LinkerTraining

from matchlab.models.linkers.base import Linker, PreparedLinker

DEFAULT_TRAINING_SEED = 0
# Splink's search API requires a strict weight cutoff, even when no probability
# threshold is configured. Filter probabilities after searching instead.
MIN_SEARCH_WEIGHT = -1e308


class _SplinkResult(Protocol):
    def as_duckdbpyrelation(self) -> duckdb.DuckDBPyRelation: ...


class SplinkLinkerFunction(BaseModel):
    """A method of splink.Linker.training used to train the linker."""

    function: str
    arguments: dict[str, Any]

    @model_validator(mode="after")
    def validate_function_and_arguments(self) -> "SplinkLinkerFunction":
        """Ensure the function and arguments are valid."""
        if not hasattr(LinkerTraining, self.function):
            raise ValueError(
                f"Function {self.function} not found as method of Splink Linker class"
            )

        splink_linker_func = getattr(LinkerTraining, self.function)
        splink_linker_func_param_set = set(
            inspect.signature(splink_linker_func).parameters.keys()
        )
        current_func_param_set = set(self.arguments.keys())

        if not current_func_param_set <= splink_linker_func_param_set:
            raise ValueError(
                f"Function {self.function} given incorrect arguments: "
                f"{current_func_param_set.difference(splink_linker_func_param_set)}. "
                "Consider referring back to the Splink documentation: "
                "https://moj-analytical-services.github.io/splink/linker.html"
            )

        return self


class SplinkLinker(Linker):
    """A linker that leverages Bayesian record linkage using Splink.

    Sampling training functions are seeded automatically with `0` when they
    accept a `seed` argument and none was provided, so repeated collections with
    the same settings stay deterministic and cache-safe.
    """

    version: ClassVar[int] = 2

    model_config = ConfigDict(arbitrary_types_allowed=True)

    linker_training_functions: list[SplinkLinkerFunction] = Field(
        description="""
            A list of dictionaries where keys are the names of methods for
            splink.Linker.training and values are dictionaries encoding the arguments of
            those methods. Each function will be run in the order supplied.

            Example:
            
                >>> linker_training_functions=[
                ...     {
                ...         "function": "estimate_probability_two_random_records_match",
                ...         "arguments": {
                ...             "deterministic_matching_rules": \"""
                ...                 l.company_name = r.company_name
                ...             \""",
                ...             "recall": 0.7,
                ...         },
                ...     },
                ...     {
                ...         "function": "estimate_u_using_random_sampling",
                ...         "arguments": {"max_pairs": 1e6},
                ...     }
                ... ]
            
        """
    )
    linker_settings: SettingsCreator = Field(
        description="""
            A valid Splink SettingsCreator.

            See Splink's documentation for a full description of available settings.
            https://moj-analytical-services.github.io/splink/api_docs/settings_dict_guide.html

            * link_type must be set to "link_only"
            * unique_id_name is overridden to the value of left_id and right_id,
                which must match

            Example:

                >>> from splink import SettingsCreator, block_on
                ... import splink.comparison_library as cl
                ... import splink.comparison_template_library as ctl
                ... 
                ... splink_settings = SettingsCreator(
                ...     retain_matching_columns=False,
                ...     retain_intermediate_calculation_columns=False,
                ...     blocking_rules_to_generate_predictions=[
                ...         block_on("company_name"),
                ...         block_on("postcode"),
                ...     ],
                ...     comparisons=[
                ...         cl.jaro_winkler_at_thresholds(
                ...             "company_name", 
                ...             [0.9, 0.6], 
                ...             term_frequency_adjustments=True
                ...         ),
                ...         ctl.postcode_comparison("postcode"), 
                ...     ]
                ... )         
        """
    )
    threshold: float | None = Field(
        default=None,
        description="""
            Keep predictions with a match probability at or above this value.
            None keeps all predictions. A value of 1 keeps only predictions
            whose match probability is 1.
        """,
        gt=0,
        le=1,
    )

    @model_validator(mode="after")
    def check_link_only(self) -> "SplinkLinker":
        """Ensure link_type is set to "link_only"."""
        if self.linker_settings.link_type != "link_only":
            raise ValueError('link_type must be set to "link_only"')
        return self

    @model_validator(mode="after")
    def add_enforced_settings(self) -> "SplinkLinker":
        """Ensure ID is the only field we link on."""
        self.linker_settings.unique_id_column_name = self.left_id
        return self

    @field_validator("linker_settings", mode="before")
    @classmethod
    def load_linker_settings(cls, value: str | SettingsCreator) -> SettingsCreator:
        """Load serialised settings into SettingsCreator."""
        if isinstance(value, str):
            value = SettingsCreator.from_path_or_dict(json.loads(value))
        return value

    @field_serializer("linker_settings")
    def serialise_settings(
        self, value: SettingsCreator, info: SerializationInfo
    ) -> str:
        """Convert Splink settings to string."""
        return json.dumps(value.create_settings_dict("duckdb"))

    @staticmethod
    def _schema_summary(data: pl.DataFrame) -> str:
        """Render a compact column-to-dtype summary for diagnostics."""
        return (
            ", ".join(f"{column}={dtype}" for column, dtype in data.schema.items())
            or "<empty>"
        )

    @classmethod
    def _conformancy_error(cls, left: pl.DataFrame, right: pl.DataFrame) -> ValueError:
        """Build a detailed error describing how the two inputs differ."""
        left_only = [column for column in left.columns if column not in right.columns]
        right_only = [column for column in right.columns if column not in left.columns]
        shared = [column for column in left.columns if column in right.columns]
        differing_dtypes = [
            f"{column}: left={left.schema[column]}, right={right.schema[column]}"
            for column in shared
            if left.schema[column] != right.schema[column]
        ]

        details = [
            "SplinkLinker requires input data to be conformant, meaning they "
            "share the same column names and data formats.",
            f"Left schema: {cls._schema_summary(left)}",
            f"Right schema: {cls._schema_summary(right)}",
        ]
        if left_only:
            details.append(f"Columns only on left: {left_only}")
        if right_only:
            details.append(f"Columns only on right: {right_only}")
        if differing_dtypes:
            details.append("Differing dtypes: " + ", ".join(differing_dtypes))
        return ValueError("\n".join(details))

    def prepare(self, left: pl.DataFrame, right: pl.DataFrame) -> PreparedLinker:
        """Return a fresh backend trained on both complete inputs."""
        if self.left_id not in left.columns or self.right_id not in right.columns:
            raise ValueError(f"Both inputs must contain the ID column {self.left_id!r}")
        if left.schema != right.schema:
            raise self._conformancy_error(left, right)

        # Convert to pandas for Splink compatibility
        left_pd = left.with_columns(pl.col(self.left_id).cast(pl.String)).to_pandas()
        right_pd = right.with_columns(pl.col(self.right_id).cast(pl.String)).to_pandas()

        # Splink adds a random linker_uid to its settings. Copy them so the
        # model's fingerprint does not change when prepare() runs.
        linker = SplinkLibLinkerClass(
            input_table_or_tables=[left_pd, right_pd],
            input_table_aliases=["l", "r"],
            settings=deepcopy(self.linker_settings),
            db_api=DuckDBAPI(),
        )

        for func in self.linker_training_functions:
            proc_func = getattr(linker.training, func.function)
            arguments = dict(func.arguments)
            if (
                "seed" in inspect.signature(proc_func).parameters
                and "seed" not in arguments
            ):
                arguments["seed"] = DEFAULT_TRAINING_SEED
            proc_func(**arguments)

        return PreparedLinker(methodology=self, state=linker)

    def link(
        self,
        prepared_state: SplinkLibLinkerClass,
        left: pl.DataFrame | None = None,
        right: pl.DataFrame | None = None,
        *,
        baseline_left: pl.DataFrame,
        baseline_right: pl.DataFrame,
    ) -> pl.DataFrame:
        """Score supplied additions against prepared and supplied opposite sides.

        Only pairs touching a supplied row are returned. Training and term
        frequencies always use the complete inputs passed to `prepare()`.
        """
        if left is None and right is None:
            raise ValueError("Provide at least one affected side to link()")

        if left is not None and left.schema != baseline_left.schema:
            raise self._conformancy_error(left, baseline_left)
        if right is not None and right.schema != baseline_right.schema:
            raise self._conformancy_error(right, baseline_right)

        # Full collection can predict directly from Splink's prepared inputs.
        # Compare rows, not IDs, because an ID can have changed content.
        if (
            left is not None
            and right is not None
            and not left.is_empty()
            and not right.is_empty()
            and left.equals(baseline_left)
            and right.equals(baseline_right)
        ):
            return self._select_scores(
                baseline_left[self.left_id].dtype,
                baseline_right[self.right_id].dtype,
                prepared_state.inference.predict(
                    threshold_match_probability=(
                        None if self.threshold == 1 else self.threshold
                    )
                ),
            )

        predictions = []

        # Splink searches both prepared datasets and puts each new row on the
        # right. Keep prepared-right matches, then restore left-to-right order.
        if left is not None and not left.is_empty() and not baseline_right.is_empty():
            found = prepared_state.inference.find_matches_to_new_records(
                left.with_columns(pl.col(self.left_id).cast(pl.String)).to_pandas(),
                blocking_rules=self.linker_settings.blocking_rules_to_generate_predictions,
                match_weight_threshold=MIN_SEARCH_WEIGHT,
            )
            predictions.append(
                self._select_scores(
                    baseline_left[self.left_id].dtype,
                    baseline_right[self.right_id].dtype,
                    found,
                    source="r",
                    reverse=True,
                )
            )

        # The right-side search also sees both prepared datasets. Keep only
        # prepared-left matches.
        if right is not None and not right.is_empty() and not baseline_left.is_empty():
            found = prepared_state.inference.find_matches_to_new_records(
                right.with_columns(pl.col(self.right_id).cast(pl.String)).to_pandas(),
                blocking_rules=self.linker_settings.blocking_rules_to_generate_predictions,
                match_weight_threshold=MIN_SEARCH_WEIGHT,
            )
            predictions.append(
                self._select_scores(
                    baseline_left[self.left_id].dtype,
                    baseline_right[self.right_id].dtype,
                    found,
                    source="l",
                )
            )

        # Splink scores every new-left/new-right pair here. Apply the configured
        # blocking rules before returning the scores.
        if (
            left is not None
            and right is not None
            and not left.is_empty()
            and not right.is_empty()
        ):
            compared = prepared_state.inference.compare_two_records(
                left.with_columns(pl.col(self.left_id).cast(pl.String)).to_pandas(),
                right.with_columns(pl.col(self.right_id).cast(pl.String)).to_pandas(),
                include_found_by_blocking_rules=True,
            )
            predictions.append(
                self._select_scores(
                    baseline_left[self.left_id].dtype,
                    baseline_right[self.right_id].dtype,
                    compared,
                    blocked=True,
                )
            )

        if not predictions:
            return pl.DataFrame(
                schema={
                    "left_id": baseline_left[self.left_id].dtype,
                    "right_id": baseline_right[self.right_id].dtype,
                    "score": pl.Float32,
                }
            )
        return (
            pl.concat(predictions)
            .group_by("left_id", "right_id")
            .agg(pl.col("score").max())
        )

    def _select_scores(
        self,
        left_id_dtype: pl.DataType,
        right_id_dtype: pl.DataType,
        result: _SplinkResult,
        source: Literal["l", "r"] | None = None,
        reverse: bool = False,
        blocked: bool = False,
    ) -> pl.DataFrame:
        """Orient Splink predictions and keep only eligible pairs."""
        predictions = result.as_duckdbpyrelation().pl().lazy()

        if source is not None:
            predictions = predictions.filter(
                pl.col(f"{self.linker_settings.source_dataset_column_name}_l") == source
            )

        if blocked:
            predictions = predictions.filter(pl.col("found_by_blocking_rules"))

        if self.threshold is not None:
            predictions = predictions.filter(
                pl.col("match_probability") >= self.threshold
            )

        return (
            predictions.select(
                pl.col(f"{self.left_id}_{'r' if reverse else 'l'}")
                .cast(left_id_dtype)
                .alias("left_id"),
                pl.col(f"{self.right_id}_{'l' if reverse else 'r'}")
                .cast(right_id_dtype)
                .alias("right_id"),
                pl.col("match_probability").cast(pl.Float32).alias("score"),
            )
            # Multiple blocking rules can lead to multiple matches
            .group_by(["left_id", "right_id"])
            .agg(pl.col("score").max())
            .collect()
        )
