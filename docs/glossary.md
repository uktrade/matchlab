# Glossary

A list of repository-specific term definitions, used consistently across matchlab's code and documentation.

## Key terms

### Artifact

The stored output of one [step](#step), such as a source's [extract](#extract) and leaf assignment, a [transform](#transform)'s materialised record step, a model's edge list, or a [resolver](#resolver)'s complete, merge-forward output. A [store](#store) keeps every artifact it is given until something explicitly [prunes](#prune) it.

### Collect

Run a plan. `collect()` walks a plan upstream-first and runs only the steps that aren't already in the store, so re-collecting an unchanged plan does no work and adding a step to a collected plan runs only the new one. See [content-addressed](#content-addressed).

### Content-addressed

Identified by what it is, not by when or how it was built. A step's [fingerprint](#fingerprint) comes from its own configuration and its inputs' fingerprints, so the same plan keys the same artifact. Collecting it twice reads the cached one back rather than recomputing it.

This holds for a step whose [methodology](#methodology) declares a [version](#version). One that declares none is keyed afresh every time, so it is never read back from the store.

### Deduper

A [methodology](#methodology) that finds candidate duplicate pairs within one [record step](#record-step), run by `.dedupe()`. `NaiveDeduper` is the built-in. A custom one subclasses `Deduper` and registers with `add_model_class`, the same pattern [transformers](#transform) plug in with.

### Entity

The real-world thing several records refer to. A [resolver](#resolver) groups records under one [root](#root) ID when it decides they describe the same entity.

### Extract

The rows a [source](#source)'s query returns, cached exactly as read so they can be read back without another trip to the warehouse. A source's [fingerprint](#fingerprint) folds in a hash of its extract, so a changed warehouse produces a new fingerprint rather than silently reusing stale data.

### Fingerprint

The 32-byte SHA-256 digest that keys a step's stored [artifact](#artifact). Two steps with identical configuration and identical input fingerprints hash to the same fingerprint, which is what makes a store [content-addressed](#content-addressed).

A fingerprint covers configuration, never the code that configuration runs. A [methodology](#methodology) closes that gap by declaring a [version](#version). Declaring none puts fresh random bytes in the fingerprint instead, so the step re-runs on every [collect](#collect), and so does every step below it.

### Judgement

A person's decision that one reviewed cluster's records do, or do not, describe the same [entity](#entity), recorded by matchlab's evaluation tools and scored against a [resolver](#resolver)'s clusters. A judgement is anchored to the content a reviewer was shown, not to any record's key, which is why a [leaf](#leaf) ID, a hash of content, rather than a key, is what a judgement endorses or rejects.

### Key

A record's identifier at the origin, in the [source](#source)'s own data. Its column is named by `key_field`, which defaults to `id`.

A key is not the `id` a [record step](#record-step) carries. That one is a [leaf](#leaf) or a [root](#root), derived by hashing content when the plan is collected. The two never collide, because every column an [extract](#extract) returns is qualified with the source name first, so a key named `id` reaches a record step as `crn_id`. `leaves()` returns both, under the names `key` and `leaf`.

A key is read as a string whatever the location stores it as, and a null key is refused. `get_lookup()` returns keys, which is how a plan's result joins back to your own tables.

Distinct from an [answer key](#answer-key), which is a [testkit](#testkits) term that shares only the word.

### Label

A pointer from a name someone chose to a [resolver](#resolver)'s [fingerprint](#fingerprint), created by [publishing](#publish) a resolver's output. A label belongs to the store, not the plan. It can be re-aimed at a different resolver, whereas a source's `name` is part of that source's own output and never moves.

### Leaf

The stable ID of one record, a hash of its content rather than its [key](#key). Identity coming from content rather than key is why a leaf ID changes only when the underlying data does, which is the basis matchlab anchors evaluation judgements to. Reading a source or record step directly, without going through a resolver, exposes the leaf as the `id` column.

### Linker

A [methodology](#methodology) that finds candidate matches between two [record steps](#record-step), run by `.link()`. `DeterministicLinker`, `WeightedDeterministicLinker`, and `SplinkLinker` are the built-ins. A custom one subclasses `Linker` and registers with `add_model_class`.

### Location

A generic way of getting a [source](#source)'s rows, and a [methodology](#methodology) like any other. `RelationalDB` runs a query. `DataFrame` reads a frame already in memory. A custom one subclasses `Location` and registers with `add_location_class`.

A location is where a source's rows come from. Do not confuse it with the [store](#store), which keeps a plan's [artifacts](#artifact) once you collect it. A location only reads.

Its [settings](#settings) are hashed into the source's [fingerprint](#fingerprint), so editing a query re-runs the source. Its [resources](#resource) are not hashed, so renaming a warehouse changes nothing.

### Merge-forward

The guarantee that a [resolver](#resolver)'s stored output carries forward every record reachable from its inputs, not just the ones its own models formed edges over. A record no model touches keeps its upstream grouping, or becomes a singleton, rather than vanishing. matchlab's resolvers materialise this complete table once, at collect time, rather than resolving on demand.

### Methodology

The pluggable class a [step](#step) runs. Every kind of step has one. A [source](#source) runs a [location](#location), a [transform](#transform) runs a transformer, a [model](#model) runs a [deduper](#deduper) or a [linker](#linker), and a [resolver](#resolver) runs a resolver method.

Every step is built the same way. It names its methodology in a `*_class` argument and configures it with `*_settings`. Anything that cannot be serialised goes in `*_resources`. Swapping a methodology means changing `*_class`, never restructuring the plan around it.

Each kind keeps its own registry, so a custom methodology can be named in a [plan document](#plan-document). Register with `add_location_class`, `add_transformer_class`, `add_model_class`, or `add_resolver_class`.

A methodology also says whether matchlab may cache what it computes, by declaring a [version](#version). See [Custom methodologies](guide/custom-methodologies.md).

Transformers, dedupers, linkers, and resolver methods prepare on complete collected input before they run their action. A cached step prepares on demand from stored input if an action later needs it. Preparation is private runtime state. It does not change the step's settings or fingerprint.

### Model

A step that scores candidate matches by running one [methodology](#methodology). A [deduper](#deduper) matches within one [record step](#record-step) via `.dedupe()`, and a [linker](#linker) matches between two via `.link()`. A model produces edges, not clusters. Turning edges into entities is a [resolver's](#resolver) job.

### Plan

A tree of [steps](#step). Each step holds a reference to its own inputs, so the step you are holding is the pipeline. There is no separate object to register steps with. Nothing runs until you [collect](#collect) it.

### Plan document

A serialisable description of a [plan](#plan), dumped with `dump()` and loaded back with `load()`. Called a **document** for short, which is what the rest of these docs call it. It is JSON, derived from a plan rather than hand-authored, which is why its nodes refer to each other by [position](#position) rather than by any name.

A document carries each step's [settings](#settings) and the edges between steps. It carries no code, no [label](#label), no data, and no [resource](#resource), only the name each resource was given. A plan rebuilt from a document [fingerprints](#fingerprint) identically to the plan it came from, given the same data, so a [store](#store) holding the original's [artifacts](#artifact) serves them rather than redoing the work. `PlanDocument` is the class.

### Position

Where something falls in an ordered sequence, used instead of a name. The repository uses the word for two different sequences, and the numbers do not agree with each other.

A step's place in `collect()`'s run order is one sense. `draw()` shows it in brackets, and logs quote it (`[step 5]`). Steps have no names, so position is how a log line, a drawn tree, and `plan.lineage()` all refer to the same step. Positions are relative to the step a plan or drawing starts from, so a sub-plan numbers its steps differently from the full plan it came from.

A setting that must point at one of a step's own inputs uses the other sense. It names that input by its index among the step's own inputs, not by that input's place in the whole plan. `Components.thresholds` keys a per-model threshold this way. A model can sit at plan position 5 while still being resolver-input position 0.

### Prune

Delete every artifact except the fingerprints you name and the sources needed by a [published](#publish) resolver output, and reclaim the space that frees. A file-backed store's prune rewrites the file itself, because deleting rows inside it does not return space to the operating system.

### Publish

Point a [label](#label) at a [resolver](#resolver)'s output, so it can be found without the plan that built it. Publishing is an act, not a property of the plan. Nothing exists to point at until the resolver has been [collected](#collect), so publishing always happens after collection, never before or instead of it.

Re-publishing the same label at the same resolver output is a no-op. Aiming an existing label at a different resolver output needs `overwrite=True`, since that is how you lose track of what a label used to mean.

### Record

An immutable in-memory value for one incoming row as it moves through a collected plan. It holds current data, affected IDs, a reference to its [source](#source), the original one-row extract and model edges. Public access to its dataframes returns copies. The source reference identifies the branch. It does not read the incoming row from the source location. A [transform](#transform) returns a new Record with reshaped data and unchanged source evidence. A [model](#model) returns a new Record with no current data, its scored edges and the IDs they touch. It keeps the affected IDs even if it finds no edges. A Record is not a [record step](#record-step) or a stored artifact.

### Record step

A record step is a kind of step whose job is to hold data that a [model](#model) can match over. It is not a table. Reading one gives a table with an id column holding the identity a model reads.

A [source](#source), a [transform](#transform), and a [resolver](#resolver) are the three kinds of record step. On a source or reshaped record step, id is the content-derived leaf, not the source key field. On a resolver, id is the entity root, which is what makes layering possible. Every record step chains the same verbs, `select`, `clean`, `group`, `transform`, `dedupe` and `link`.

A [model](#model) is the one step that is not a record step. It yields edges rather than records.

### Resolver

A step that collapses a model's scored edges into clusters, one per [entity](#entity). `.resolve()` defaults to connected components. A resolver's stored output is always complete and [merge-forward](#merge-forward).

Call this a **Resolver**, or its **merge-forwarded Resolver output** if you mean the stored table specifically. Don't call it "a resolution". That noun has been retired.

### Resource

A named object a node needs to run that can't be serialised. Pass one in a step's `*_resources` argument, never among its [settings](#settings). Wrap it as `Resource("warehouse", engine)` to give it the name a [plan document](#plan-document) records. Supply it again when that document is loaded.

For sources, resources are the data-generating mechanism, which affects the [fingerprint](#fingerprint). For all other types of node, resources cannot influence the output of the node because they're ignored by the fingerprint.

A location or methodology marks its resource fields `FromResources`. Every other field is a setting. Passing a field in the wrong argument raises an error that names the field.

Sharing a resource name means sharing a resource. Two steps of different kinds can't share a resource.

### Root

The ID of a cluster a resolver produces, a hash of the sorted set of [leaf](#leaf) IDs it contains. Two runs that produce the same clustering produce the same root ID, whatever order the underlying algorithm found its clusters in. Reading a [record step](#record-step) through a resolver exposes the root as the `id` column, so several records can share one `id`.

### Settings

A step's serialisable configuration. Passed in a step's `*_settings` argument, carried in a plan document, and hashed into the step's [fingerprint](#fingerprint).

Hashing is what makes settings the opposite of a [resource](#resource): **editing settings re-runs the step and everything below it**, whether or not the output would differ.

Every field of a location or methodology is a setting unless the class marks it a resource.

### Source

The leaf a plan starts from: a [location](#location) to read, plus the column that [keys](#key) it. Every column the location returns is part of a record's identity, so two rows are the same record exactly when it returns identical values for both. Build one with `read_database` or `read_dataframe`.

### Step

One node in a [plan](#plan). Steps are sources, transforms, models, or resolvers. A step's kind decides what it produces, and what [artifact](#artifact) a store keeps for it.

### Store

Where a plan's artifacts are stored and read back, keyed by [fingerprint](#fingerprint). Stores are not engines and do not resolve anything themselves. `DuckDBStore` is the reference implementation, backing a single DuckDB database file (or `:memory:`). A store keeps everything given to it until something explicitly [prunes](#prune) it or the storage itself is deleted.

### Transform

A step that reshapes a [record step](#record-step) with a pluggable, serialisable **transformer**. `select` keeps only the named columns, `clean` derives new ones with DuckDB SQL while keeping the rest, `group` collapses each `id` to one row, and `Explode` gives one row per combination of each column's non-null values. `transform()` is the general verb they desugar to (`.clean(...)` is `.transform(Clean(...))`). `Explode` has no verb of its own, so it is reached only that way. `Transformer` is the base class. `Select`, `Clean`, `Group`, and `Explode` are the built-ins, and a custom one registers with `add_transformer_class`, the same pattern [dedupers](#deduper) and [linkers](#linker) plug in with. Its configuration is part of the plan and its [fingerprint](#fingerprint), like any [methodology](#methodology)'s settings.

### Version

What a [methodology](#methodology) promises about its own code. Declared as a class attribute, `version: ClassVar[int] = 1`, never as a [setting](#settings).

Declaring one says the methodology computes a deterministic function of its settings, so matchlab may cache what it produces and read it back on a later [collect](#collect). Count it up by one whenever a change to the code changes what the class computes. That retires every [artifact](#artifact) the previous code wrote.

Declaring none is the default, and it says the opposite. matchlab knows nothing about the code, so it refuses to trust a stored artifact. The step re-runs on every collect, and so does every step below it. This is what makes editing a methodology and re-running work as you would expect.

A [location](#location) declares no version and needs none. A [source](#source) hashes the rows it read into its own [fingerprint](#fingerprint), so a change in what a location returns already moves it.

## Testkits

Testkits are resources that matchlab's deverlopers use to simplify automated testing.

A testkit is an object that generates test data with a known answer, and exposes what it knows about that data. `matchlab.testkit` builds these. `GeneratedSource`, `GeneratedModel`, and `LinkedSources` each wrap a real plan node, but expose only the fixture, meaning its rows, its features, and its planted answer. Anything you do to the plan itself goes through the node a testkit wraps, never through a forwarding shortcut on the testkit.

A testkit is what makes the oracle testing pattern possible here. Assert against a known, planted answer rather than an answer you have to work out by hand or guess at.

The package checks a plan's answer two ways, because matchlab derives a record's real ID by content-hashing at collect time, so no ID is knowable when a fixture is built. One check scores a [methodology](#methodology) directly, against the testkit's own synthetic IDs, without collecting anything. The other scores a collected plan, keyed by row values instead, through an [answer key](#answer-key). See `matchlab.testkit`'s module docstring for which to use.

### Answer key

A [testkit's](#testkits) map from a generated row's own values to the [true entity](#true-entity) that produced it. It exists because matchlab derives a row's real ID by content-hashing at collect time, so no ID is knowable when the answer key is built. A [matcher](#matcher) built from one reads a row's values back out of whatever record step it is handed, and looks up the entity that row belongs to, which is what lets it score a plan it has never seen. `AnswerKey` is the class.

### Cluster

In a [testkit](#testkits), a set of records claimed to be one entity, an answer rather than the answer. It carries only membership, which is what an actual model or resolver result can be compared against. Contrast [true entity](#true-entity), which is the planted answer a cluster is checked against. `Cluster` is the class.

### Matcher

A model in a [testkit](#testkits) that already knows the answer, built from an [answer key](#answer-key) instead of scoring anything. `PerfectDeduper` and `PerfectLinker` are the two matchers. A matcher makes no mistakes, so pointing one at generated data leaves the plan as the only variable under test. Build one through `LinkedSources.dedupe()` or `.link()` rather than directly.

### True entity

The planted answer a [testkit](#testkits) generated data from. It is one real-world thing, identified by the values it was generated from, spanning every source it landed in. Project it onto the sources under test to get a [cluster](#cluster), the only form that is comparable with an actual result. `TrueEntity` is the class.