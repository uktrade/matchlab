# Custom methodologies

A methodology is the pluggable class a step runs. matchlab ships one for every kind of step. To write one, subclass a base, declare settings as fields, implement its methods, and register the class.

```python
from typing import ClassVar

import matchlab as mb
import polars as pl


class Initials(mb.Transformer):
    """Reduce a name to its initials."""

    version: ClassVar[int] = 1

    column: str

    def prepare(self, data: pl.DataFrame) -> None:
        """This transform needs no baseline state."""

    def apply(self, data: pl.DataFrame) -> pl.DataFrame:
        initials = pl.col(self.column).str.extract_all(r"\b\w").list.join("")
        return data.with_columns(initials.alias("initials"))


mb.add_transformer_class(Initials)
```

The base you subclass depends on the step. `Transformer` reshapes a record step. `Deduper` and `Linker` score candidate matches. `ResolverMethod` turns edges into clusters, and `Location` reads rows into a source.

Transformers, dedupers, linkers, and resolver methods must implement `prepare()`. It receives the complete collected input. Their action methods then handle supplied work using that prepared state. A transformer or resolver method with no setup still implements a no-op `prepare()`. A deduper scores supplied rows against its prepared input.

A linker treats supplied rows as additions to the prepared left or right input. When both sides are supplied, it scores new left rows against the prepared right, prepared left rows against the new right, and the new rows against each other. It does not return pairs made only from prepared rows or change either prepared input. During collection, both sides contain the full input.

After a cache hit, a step prepares from the stored baseline when its methodology is first needed. It does not train merely to read a cached result.

A model step scores affected input without storing the new edges. Its `Record` output contains no current data. It keeps the edges under that model and adds each touched ID to the affected set. An empty edge result still carries the affected IDs. The model passes both affected linker sides to `link()` in one call, so the methodology can apply its own matching rules.

Each registry is keyed by class name, which is how a [plan document](./serialise.md) names your class.

## Settings and resources

Every field you declare is a setting. Settings are serialised into the step's [fingerprint](../glossary.md#fingerprint), so editing one re-runs the step and everything below it.

Anything that cannot be serialised is a resource instead. Mark the field and pass it in `*_resources`:

```python
class Lookup(mb.Transformer):
    column: str  # a setting
    engine: mb.FromResources[Engine]  # a resource
```

Each field is checked against its own declaration. Passing one in the wrong argument raises, naming the field:

```
ResourceError: 'engine' on Lookup is a setting, not a resource, so must be passed in
`transformer_settings`. Only a field marked `FromResources` may go in
`transformer_resources`.
```

## Declaring a version

A fingerprint covers your settings. It does not cover the code those settings run. Edit `prepare` or `apply` and nothing in the key moves, so `collect()` hands back whatever the old code produced.

`version` is how you close that gap, and it is a promise:

> This class computes a deterministic function of its settings.

Make that promise and matchlab caches what you produce. Count the version up by one whenever you change what the class computes. That retires every artifact the old code wrote.

Leave `version` unset and you promise nothing. matchlab knows nothing about your code, so it refuses to trust a stored artifact. The step re-runs on every collect, and so does every step below it. Edit the class, run again, and you see the new result.

That is why an unset version is the default. It is also what you want while a methodology is still moving:

```
↻ [2] model(NaiveDeduper) refreshed 1.4s
    └── ↻ [1] transform(Initials) refreshed 0.2s
        └── ◍ [0] source 'crn' cached
```

`refreshed` says the step ran because it cannot be cached. It does not mean your edit invalidated it. Step 2 is refreshed because step 1 is. A step that may produce something new leaves nothing below it worth trusting.

Declare `version` as a class attribute, never as a field:

```python
version: ClassVar[int] = 1  # correct
version: int = 1  # a setting called "version", which is not the same thing
```

The second form is a setting. matchlab raises rather than letting it pass for a promise.

## Pitfalls

Each of these makes a methodology depend on something its settings do not describe. Leave `version` unset until you have fixed the cause.

!!! warning "A resource that changes the output"

    A resource never reaches a fingerprint. Pass a lookup table or a fitted model as one, and two runs against different data share a key. The second reads back the first's result. Move the input into settings, read it through a [source](../glossary.md#source) so its rows are hashed, or declare no version.

!!! warning "Sampling without a seed"

    A methodology that samples is not a function of its settings. `SplinkLinker` training functions such as `estimate_u_using_random_sampling` need a `seed` in their arguments. Without one, the first result is cached and every later run reads it back.

Three more are easy to miss.

* **Ambient state.** A library version, a locale, a clock. Upgrade a dependency your methodology calls and the same settings give a different answer under the same key. Bump your version when you upgrade.
* **Mutating your own settings.** A methodology's settings are hashed when the step is built. Writing to them afterwards changes what runs and not what was hashed. Copy at the boundary if a library you wrap mutates what you hand it.
* **Two classes with one name.** A step is identified by its methodology's class name. Two classes called `Normalise` therefore share a fingerprint while computing different things. Names have to be unique across a codebase, not just a module.

Dict settings are ordered, and that order is part of the key. `Clean` and `Group` project their entries in the order you wrote them, so reordering them is a real edit and re-runs the step.

A transformer must also pass `id` through untouched. Every model matches on it, and matchlab derives it from record content. A transformer that writes to it changes which records count as the same:

```
ValueError: `id` is not a column you can write. It is the grouping every model matches
on, derived from record content, so replacing it would silently change which records
count as the same. Give the expression another name.
```

## What an unset version costs

Everything below an unversioned step re-runs too. The expensive steps are usually the ones below. A cheap transformer near the top of a plan can mean re-running a linker on every collect.

Each run stores its own artifact rather than replacing the last. That is what keeps a published label pointing at the result you published. It also means the store grows by one artifact per refreshed step per run. Prune it as you go:

```python
store.prune(keep=plan.fingerprints())
```

Set a version once your methodology settles, then bump it when the code changes. That gets the caching back without asking matchlab to take your code on trust.
