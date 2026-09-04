---
hide:
  - toc
---

<figure markdown="span">

![The Matchbox logo in light mode](./assets/matchlab-logo-light.svg#only-light){ width="500" } ![The Matchbox logo in dark mode](./assets/matchlab-logo-dark.svg#only-dark){ width="500" }

</figure>

# Introducing matchlab

**A local-first library for building, running and evaluating entity resolution pipelines.**

Record matching is a chore. You match with Splink and pandas, measure with `er-evaluation`, and hand-roll the operational pipeline. Every library in the chain fights the others. There is no common data format, no common way to compare methodologies, and no common way to run the result.

matchlab is one library for all of it, and it runs on your machine.

```python
import matchlab as mb

crn = mb.read_database(
    "crn",
    sql="select pk, company, town from crn",
    client=warehouse,
    key_field="pk",
)

companies = crn.dedupe(
    model_class=mb.NaiveDeduper,
    model_settings={"unique_fields": [crn.f("company")]},
).resolve()

lookup = companies.collect().get_lookup()
```

<div class="grid cards" markdown>

- :material-run-fast:{ .lg .middle } **Get started**

    ---

    Install matchlab and build your first pipeline.

    [:octicons-zap-16: Install](./guide/install.md){ .md-button .md-button--primary } [:octicons-book-16: Build a plan](./guide/build-a-plan.md){ .md-button }

- :material-swap-horizontal:{ .lg .middle } **Coming from Matchbox?**

    ---

    matchlab is more than Matchbox without the server.

    [:octicons-arrow-right-16: Migration guide](./guide/matchbox-to-matchlab.md){ .md-button }

</div>

## What it does

**A language for pipelines.** Bring cleaning, deduplication and linking steps together into a plan that is serialisable and cheap to iterate on. 

**Lazy, like Polars.** Nothing runs until you `collect()`. Each step knows the whole plan behind it, and results are content-addressed. Re-collecting only redoes the work whose inputs actually changed.

**Evaluation as a first-class concern.** Entity resolution has no single right answer. The same data supports many methodologies, each with many configurations. matchlab ships the tools to compare them: cluster sampling, judgements, and precision/recall against ground truth.

**A path from analysis to production.** The plan you iterate on in a notebook is the plan you run operationally. Its output has a stable interface for the analysts and services that consume it.

## What it doesn't do

matchlab does not store your raw data. It reads from your warehouse, indexes what it needs to match on, and keeps its own artifacts in a local store (DuckDB by default).

There is no server, no accounts, and nothing to deploy.
