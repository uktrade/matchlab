<p align="center">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/uktrade/matchbox/refs/heads/matchlab/docs/assets/matchlab-logo-dark.svg">
      <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/uktrade/matchbox/refs/heads/matchlab/docs/assets/matchlab-logo-light.svg">
      <img alt="Shows the matchlab logo in light or dark color mode." src="https://raw.githubusercontent.com/uktrade/matchbox/refs/heads/matchlab/docs/assets/matchlab-logo-light.svg">
    </picture>
</p>

**A local-first library for building, running and evaluating entity resolution pipelines.** TEST_PR

Record matching is a chore. matchlab makes it a pipeline you can build, run, query and measure — on your machine, against your warehouse, with nothing to deploy.

```python
import matchlab as mb

companies = mb.read_database(
    name="crn",
    sql="select pk, company, town from companies",
    client=warehouse,
    key_field="pk",
)

entities = (
    companies.clean({"name": "lower(crn_company)"})
    .dedupe(
        model_class=mb.NaiveDeduper,
        model_settings={"unique_fields": ["name"]},
    )
    .resolve()
    .collect()
)

entities.lookup_key(from_source="crn", to_sources=["dh"], key="a1")
```

Read the [full documentation](https://uktrade.github.io/matchlab/).

## What it does

* **A lazy plan.** `Source(...).dedupe(...).resolve()` builds a tree of steps. Nothing runs until you `collect()`.
* **Content-addressed caching.** Re-collecting an unchanged plan does no work. Adding a step runs only that step.
* **Materialised resolver output.** A collected resolver writes a complete `(root, leaf, key, source)` table, so lookups are reads, not re-derivations.
* **Measurement as a first-class job.** Sample clusters, record judgements, score precision and recall, and compare methodologies on equal terms.

## What it doesn't do

No server, no accounts, no permissions, nothing to deploy. If you need a shared, governed matching service, matchlab is not that.

## Installation

```shell
pip install matchlab
```

## Coming from Matchbox?

matchlab is the successor to `matchbox-db`, with the server removed and the client API rebuilt. It's a hard break — see the [migration guide](https://uktrade.github.io/matchlab/guide/matchbox-to-matchlab/).

## Development

See our full development guide and coding standards on our [contribution guide](https://uktrade.github.io/matchlab/contributing/).
