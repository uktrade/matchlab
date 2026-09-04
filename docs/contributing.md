This document describes how to get started developing matchlab.

## Dependencies

* [Python 3.11+](https://www.python.org)
* [uv](https://docs.astral.sh/uv/)
* [just](https://just.systems/man/en/)
* [Docker](https://www.docker.com), for the TruffleHog secret scan that pre-commit runs on every commit.

## Setup

This project is managed by [uv](https://docs.astral.sh/uv/), linted and formatted with [ruff](https://docs.astral.sh/ruff/), type checked with [ty](https://docs.astral.sh/ty/), and tested with [pytest](https://docs.pytest.org/en/stable/). Documentation is built with [Zensical](https://zensical.org/docs/get-started/).

Install all dependencies:

```shell
uv sync
```

There is no `.env` to configure. matchlab reads no environment variables. Warehouse connections are passed to `Location`. Storage is passed to `collect()`, or set with `set_default_store()`.

Secret scanning is done with [TruffleHog](https://github.com/trufflesecurity/trufflehog).

For security, we expect you to install [pre-commit](https://pre-commit.com). We mandate [git trailers](https://git-scm.com/docs/git-interpret-trailers) to confirm your local hooks ran. Ensure your hooks are installed:

```shell
pre-commit install --install-hooks --overwrite -t commit-msg -t pre-commit
```

Task running is done with [just](https://just.systems/man/en/). To see all available commands:

```shell
just -l
```

## Run tests

The whole suite runs on Python alone. Storage uses DuckDB in memory, and warehouses use SQLite in a temp file. Run it with:

```shell
just test
```

No container is needed: every warehouse test runs against SQLite, over a SQLAlchemy engine or an ADBC connection.
If you want a real warehouse to point matchlab at while developing, bring up whatever database you like and pass its client to a `Location`. matchlab has no opinion about where it runs, and ships nothing to manage one.

## Documentation

```shell
just docs
```

Serves the site with live reload. CI builds the site in strict mode, so broken cross-references fail the build. Check this before you push:

```shell
uv run zensical build --strict
```

## Releasing

We release matchlab by creating and publishing a GitHub release from `main`. Tags must follow [semantic versioning](https://semver.org) in the form `vX.X.X` (for example, `v1.2.3` for a patch release or `v2.0.0` for a major release with breaking changes).

Publishing the release triggers the CD workflow, which builds and publishes the Python package to PyPI and deploys the documentation to GitHub Pages.

## Standards

### Code

When contributing to matchlab and its associated repos, we try to follow consistent standards. Python code should be:

* Unit tested, and pass new and existing tests
* Documented via docstrings, in the [Google style](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html)
* Linted and auto-formatted (`just format`)
* Type hinted and checked (`just check`)
* Structured as a Python package with `pyproject.toml`
* Using dependencies managed automatically by uv
* Integrated with the justfile when relevant

### Steps

New plan steps subclass [`Step`](api/plan/steps). A step must:

* Hold references to its inputs, and to nothing downstream of it
* Return a stable `_spec_key()` covering everything that changes its output, so caching is correct
* Do all its work in `_execute()`, reading inputs from the store by fingerprint
* Settle in `__init__`, listing everything its spec is built from in `_READ_ONLY` so a later assignment raises

That last rule is what keeps a step's configuration, the thing it runs, and its cached artifact from disagreeing. A step builds its methodology from its settings at construction, so an assignment afterwards moves one attribute and nothing derived from it: set `model_class` and the instance still runs the old class, leaving `spec` naming one configuration while `_execute` runs another. The cache leans on the same guarantee from the other side — `_ensure` short-circuits on a stored `_fp` *because* a settled step cannot have changed, so it would hand back the old artifact without consulting the spec at all. Validate first, then write through `_set()`, which is the only thing that gets past the guard.

Derive anything you can rather than storing it. `Transform.transformer_class` and `Model.model_type` are properties over the attributes they describe, so they cannot drift from them.

A methodology must not write into its own settings either, for the same reason: those settings are hashed into the owning step's fingerprint. Copy at the boundary if a library you wrap mutates what you hand it, as `SplinkLinker.prepare` does.

### Stores

New storage backends subclass [`Store`](api/stores). Beyond reading and writing, `stats()` must report the store's size and contents. Every collect calls it, and a store nobody can measure is one that quietly fills a disk.

`prune()` is the other half. Be careful with it, since it deletes. It must hold to three rules:

* **Keep what the caller named, and never work out the rest for yourself.**
* **Keep every published label, listed or not**, along with whatever its resolver output needs to stay readable. Never touch stored judgements. They are the one thing in a store that cannot be recomputed.
* **Report what you actually reclaimed**, measured. Deleting and reclaiming are not the same number in every backend.

### Git

We commit as frequently as possible. We keep our commits as atomic as possible. We never push straight to main, instead we merge feature branches. Before merging to main, branches are peer reviewed.

!!! warning
    Pre-commit **must** be turned on. Any secrets you commit to the repo are your own responsibility.

### AI

To help reviewers prioritise their time, declare any use of AI in your PR comment.

### Actions

To avoid supply chain attacks, we [pin all actions in workflows](https://codeql.github.com/codeql-query-help/actions/actions-unpinned-tag/).

When upgrading actions, we expect PR comments to confirm that the new commit is safe. You need to cover:

* That the commit's `action.yml` only uses pinned child actions, if it has children
* That there are no critical security concerns raised in the issues

We suggest using tools like [`wayneashleyberry/gh-act`](https://github.com/wayneashleyberry/gh-act) to help. It performs the upgrade in a single line:

```shell
gh act update --pin
```

You will still need to independently verify that the new pins are safe.
