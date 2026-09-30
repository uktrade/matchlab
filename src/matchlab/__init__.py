"""The matchlab plan API.

Build a plan from a source, chain verbs, then `collect()`:

```python
import matchlab as mb

crn = mb.read_database("crn", sql=..., client=engine, key_field="pk")
deduped = crn.clean(...).dedupe(model_class=mb.NaiveDeduper, ...).resolve()
lookup = deduped.collect().get_lookup()
```

Nothing runs until `collect()`, and collecting again only does the work whose plan
or inputs changed.

Everything needed to write a plan is re-exported here, including built-in methodologies.
Evaluation stays behind `matchlab.eval`, since a plan runs without it.
"""

from importlib.metadata import version

from matchlab.document import PlanDocument, dump, load
from matchlab.models import Model, add_model_class
from matchlab.models.dedupers import NaiveDeduper
from matchlab.models.dedupers.base import Deduper, PreparedDeduper
from matchlab.models.linkers import (
    DeterministicLinker,
    SplinkLinker,
    WeightedDeterministicLinker,
)
from matchlab.models.linkers.base import Linker, PreparedLinker
from matchlab.recordstep import RecordStep
from matchlab.resolvers import Components, Resolver
from matchlab.resolvers.base import PreparedResolverMethod, ResolverMethod
from matchlab.resolvers.resolvers import add_resolver_class
from matchlab.resources import FromResources, Resource
from matchlab.sources import (
    DataFrame,
    Location,
    RelationalDB,
    Source,
    add_location_class,
    read_database,
    read_dataframe,
)
from matchlab.steps import Step
from matchlab.stores import DuckDBStore
from matchlab.stores.default import default_store, set_default_store
from matchlab.transformers import (
    Clean,
    Explode,
    Group,
    PreparedTransformer,
    Select,
    Transform,
    Transformer,
    add_transformer_class,
)

__version__ = version("matchlab")

__all__ = (
    "Clean",
    "Components",
    "DataFrame",
    "Deduper",
    "DeterministicLinker",
    "DuckDBStore",
    "Explode",
    "FromResources",
    "Group",
    "Linker",
    "Location",
    "Model",
    "NaiveDeduper",
    "PlanDocument",
    "PreparedDeduper",
    "PreparedLinker",
    "PreparedResolverMethod",
    "PreparedTransformer",
    "RecordStep",
    "RelationalDB",
    "Resolver",
    "ResolverMethod",
    "Resource",
    "Select",
    "Source",
    "SplinkLinker",
    "Step",
    "Transform",
    "Transformer",
    "WeightedDeterministicLinker",
    "__version__",
    "add_location_class",
    "add_model_class",
    "add_resolver_class",
    "add_transformer_class",
    "default_store",
    "dump",
    "load",
    "read_database",
    "read_dataframe",
    "set_default_store",
)
