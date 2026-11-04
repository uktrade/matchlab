"""A linking methodology leveraging Splink."""

import inspect
import json
from copy import deepcopy
from typing import Any, ClassVar

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

from matchlab.core.logging import logger
from matchlab.models.linkers.base import Linker

DEFAULT_TRAINING_SEED = 0


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

    version: ClassVar[int] = 1

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
                ... 
                ... splink_settings = SettingsCreator(
                ...     link_type="link_only",
                ...     retain_matching_columns=False,
                ...     retain_intermediate_calculation_columns=False,
                ...     blocking_rules_to_generate_predictions=[
                ...         block_on("company_name"),
                ...         block_on("postcode"),
                ...     ],
                ...     comparisons=[
                ...         cl.JaroWinklerAtThresholds(
                ...             "company_name", 
                ...             [0.9, 0.6], 
                ...         ).configure(
                ...             term_frequency_adjustments=True
                ...         ),
                ...         cl.PostcodeComparison("postcode"),
                ...     ]
                ... )         
        """
    )
    threshold: float | None = Field(
        default=None,
        description="""
            The score above which matches will be kept.

            None is used to indicate no threshold.
            
            Inclusive, so a value of 1 will keep only exact matches across all 
            comparisons.
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

    _linker: SplinkLibLinkerClass
    _id_dtype_l: pl.DataType
    _id_dtype_r: pl.DataType

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
            details.append("Differing dtypes: " + "; ".join(differing_dtypes))
        return ValueError("\n".join(details))

    def prepare(self, left: pl.DataFrame, right: pl.DataFrame) -> None:
        """Build the Splink linker over left and right, and run its training functions.

        Runs each function in `settings.linker_training_functions`, in order, against
        the built linker. `link()` then just predicts against the trained state.
        """
        self._id_dtype_l = left[self.left_id].dtype
        self._id_dtype_r = right[self.right_id].dtype

        if (set(left.columns) != set(right.columns)) or not left.dtypes == right.dtypes:
            raise self._conformancy_error(left, right)

        # Convert to pandas for Splink compatibility
        left_pd = left.with_columns(pl.col(self.left_id).cast(pl.String)).to_pandas()
        right_pd = right.with_columns(pl.col(self.right_id).cast(pl.String)).to_pandas()
        db_api = DuckDBAPI()
        left_table = db_api.register(left_pd, dataset_display_name="l")
        right_table = db_api.register(right_pd, dataset_display_name="r")

        # A copy, because building a Splink linker writes into the settings it is
        # given: it stamps a random `linker_uid` onto them. Handed our own object,
        # that mutation lands in `SplinkSettings`, and so in this model's `spec` and
        # its fingerprint, which would then be different after running than it was
        # before. A step has to address the same artifact whether or not it has been
        # run, and a random value could never be part of that address anyway.
        self._linker = SplinkLibLinkerClass(
            splink_dataframe_or_dataframes=[left_table, right_table],
            settings=deepcopy(self.linker_settings),
        )

        for func in self.linker_training_functions:
            proc_func = getattr(self._linker.training, func.function)
            arguments = dict(func.arguments)
            if (
                "seed" in inspect.signature(proc_func).parameters
                and "seed" not in arguments
            ):
                arguments["seed"] = DEFAULT_TRAINING_SEED
            proc_func(**arguments)

    def link(
        self, left: pl.DataFrame = None, right: pl.DataFrame = None
    ) -> pl.DataFrame:
        """Predict match scores using the linker trained in `prepare()`.

        `left`/`right` are accepted only to satisfy the `Linker` contract. The data
        was already fixed when `prepare()` built the underlying Splink linker, so
        passing values here logs a warning and has no effect.
        """
        if left is not None or right is not None:
            logger.warning(
                "Left and right data are declared in .prepare() for SplinkLinker. "
                "These values will be ignored"
            )

        res = self._linker.inference.predict(threshold_match_probability=self.threshold)

        return (
            res.as_duckdbpyrelation()
            .pl()
            .lazy()
            .select(
                [
                    f"{self.left_id}_l",
                    f"{self.right_id}_r",
                    "match_probability",
                ]
            )
            .with_columns(
                [
                    pl.col(f"{self.left_id}_l").cast(self._id_dtype_l).alias("left_id"),
                    pl.col(f"{self.right_id}_r")
                    .cast(self._id_dtype_r)
                    .alias("right_id"),
                    pl.col("match_probability").cast(pl.Float32).alias("score"),
                ]
            )
            .select(["left_id", "right_id", "score"])
            # Multiple blocking rules can lead to multiple matches
            .group_by(["left_id", "right_id"])
            .agg(pl.col("score").max())
            .collect()
        )
