"""The lazy plan node.

A `Step` holds references to its **inputs only** (`parents`). There is no registry,
no parent pointer, and no downstream list. "The DAG" is whatever is reachable
upstream from the node you hold, and lineage operations are pure functions of a root
node (`matchlab.lineage`).

Nothing is computed until `collect()`. Collection walks the plan upstream-first and
runs only the steps whose artifact is not already stored.

**Fingerprints identify artifacts.** A step's fingerprint combines its kind, its
spec, and its inputs' fingerprints. Everything downstream of a source can therefore
derive it from the *plan alone*, before any work happens, and `collect`
can skip a cached step without running it.

Two kinds of step are keyed by more than their plan. A **source** is where raw data
enters, so its spec key includes a content hash of the data it read. Constructing a
fresh `Source` therefore re-reads the warehouse (the documented way to refresh), while
an existing `Source` object memoises its read. A step whose **methodology declares no
version** is keyed by a fresh nonce on every collect, because matchlab has been told
nothing about that code and will not cache what it cannot identify. See
`matchlab.core.versioning`.

**A step is settled once built.** Everything its spec is derived from refuses
assignment, so a plan is changed by rebuilding it, not by editing a node in place.
"""

import inspect
from abc import ABC, abstractmethod
from typing import Any, ClassVar, Self

from pydantic import BaseModel

from matchlab import lineage
from matchlab.core import logging as mlog
from matchlab.core.exceptions import ResourceError
from matchlab.core.hash import HASH_FUNC
from matchlab.core.kinds import StepKind
from matchlab.core.versioning import declared_version, methodology_key
from matchlab.lineage import StepStatus
from matchlab.progress import report
from matchlab.resources import (
    Resource,
    as_resources,
    resource_fields,
    values_of,
)
from matchlab.stores import Fingerprint, Store
from matchlab.stores.default import default_store


class Step(ABC):
    """A node in a lazy plan."""

    kind: ClassVar[StepKind]

    # The prefix on this kind's `*_settings` and `*_resources` arguments. Empty for a
    # step that runs no methodology, which is `RecordStep` and any test double.
    # `_build_methodology` quotes it when a field was passed in the wrong one, so it
    # has to name arguments the step really takes
    _METHODOLOGY: ClassVar[str] = ""

    def __init_subclass__(cls, **kwargs: object) -> None:
        """Derive `_METHODOLOGY` from the `*_resources` parameter `__init__` declares.

        A step that takes no such parameter builds no methodology, so it keeps whatever
        it inherits. That covers `RecordStep` and any test double subclassing `Step`
        directly.

        Raises:
            TypeError: If `__init__` declares more than one `*_resources` parameter,
                which leaves no single prefix to quote back at a caller.
        """
        super().__init_subclass__(**kwargs)

        prefixes = [
            parameter.removesuffix("_resources")
            for parameter in inspect.signature(cls.__init__).parameters
            if parameter.endswith("_resources")
        ]
        if len(prefixes) > 1:
            raise TypeError(
                f"{cls.__name__}.__init__ declares more than one resources argument "
                f"({', '.join(f'{prefix}_resources' for prefix in prefixes)}), so "
                "there is no single prefix to name in an error. A step runs one "
                "methodology."
            )
        if prefixes:
            cls._METHODOLOGY = prefixes[0]

    @classmethod
    def _build_methodology(
        cls,
        methodology_class: type[BaseModel],
        settings: dict[str, Any] | None,
        resources: dict[str, Any] | None,
    ) -> tuple[BaseModel, dict[str, Any], dict[str, Resource]]:
        """Build the one methodology this step runs, and split its configuration.

        The settings come back as the methodology *resolved* them, read off the
        instance. A defaulted field left out at the call site is still part of the
        configuration, so it belongs in the spec. The resources are excluded, so nothing
        injected reaches a fingerprint.

        Args:
            methodology_class: The location or methodology to build.
            settings: The `*_settings` argument, as passed.
            resources: The `*_resources` argument, as passed.

        Returns:
            The built instance, its resolved settings, and its wrapped resources.

        Raises:
            ResourceError: If a marked field appears in the settings, or an unmarked
                field appears in the resources.
        """
        settings = settings or {}
        wrapped = as_resources(resources)

        name = methodology_class.__name__
        prefix = cls._METHODOLOGY
        declared = resource_fields(methodology_class)

        if misplaced := sorted(set(settings) & declared):
            fields = ", ".join(f"'{field}'" for field in misplaced)
            raise ResourceError(
                f"{fields} on {name} {'are' if len(misplaced) > 1 else 'is'} marked "
                f"`FromResources`, so must be passed in `{prefix}_resources` rather "
                f"than `{prefix}_settings`. A document then records the name rather "
                "than the value, and no credential is serialised."
            )

        for field in sorted(set(wrapped) - declared):
            if field not in methodology_class.model_fields:
                raise ResourceError(f"{name} has no field '{field}'.")
            raise ResourceError(
                f"'{field}' on {name} is a setting, not a resource, so must be passed "
                f"in `{prefix}_settings`. Only a field marked `FromResources` may go "
                f"in `{prefix}_resources`: a step excludes its resources from the "
                "settings it hashes, so this one would change with no re-run."
            )

        instance = methodology_class(**settings, **values_of(wrapped))
        return (
            instance,
            instance.model_dump(mode="json", exclude=set(wrapped)),
            wrapped,
        )

    @property
    @abstractmethod
    def parents(self) -> "tuple[Step, ...]":
        """This step's direct inputs, in the order they fold into its fingerprint.

        The kind-agnostic view of a step's edges.
        """
        ...

    def __init__(self) -> None:
        """Initialise a plan node.

        Steps have no names. They are identified by **position**, where they fall in
        `lineage.walk`, which is the order `collect` runs them in and the order
        `PlanDocument` lists them in. `step 7` in a log, `[7]` in `draw()`, and
        `steps[7]` in a document are therefore the same node.

        A position is not stored here, because it is not a property of the step. It
        belongs to the walk it came from, and the same step numbers differently in
        `walk(deduped)` and `walk(companies)`. Whoever does the walking passes it to
        whoever needs it, `collect` to its reporter, `draw` to its own renderer.

        Finding a result later is a separate matter, and a separate act.
        `Resolver.publish` points a **label** at a resolver's output. `Source` is the
        one step with a name, and it means something else again. A source's name is
        part of its output, prefixing every column it contributes and tagging its rows.
        """
        self._fp: Fingerprint | None = None
        self._store: Store | None = None

    # -- immutability -----------------------------------------------------------------
    #
    # A step builds what it runs from its settings at construction, so an assignment
    # afterwards would move one attribute and nothing derived from it: setting
    # `model_class` leaves the built methodology instance running the old class, with
    # `spec` naming one configuration while `_execute` runs another. The cache leans on
    # the same guarantee from the other side. `_ensure` short-circuits on a stored
    # fingerprint *because* a settled step cannot have changed.

    # Attributes only `__init__` may write. A kind names its own and unions this, so a
    # `Step` subclass keeps whatever attributes it likes. The guard protects what the
    # plan machinery reads, not every attribute on the object. `parents` is not
    # here: it is a property, which refuses assignment without any help.
    _READ_ONLY: ClassVar[frozenset[str]] = frozenset()

    def __setattr__(self, name: str, value: object) -> None:
        """Refuse writes to what a spec, or the plan's shape, is built from."""
        if name in self._READ_ONLY:
            raise AttributeError(
                f"{type(self).__name__}.{name} is read-only. A step is settled once "
                "built, so rebuild the plan to change it."
            )
        object.__setattr__(self, name, value)

    def _set(self, **attributes: object) -> None:
        """Write attributes past the read-only guard. `__init__` is the only caller."""
        for name, value in attributes.items():
            object.__setattr__(self, name, value)

    def _resources(self) -> dict[str, Resource]:
        """The resources this step was given, whatever kind it is."""
        if not self._METHODOLOGY:
            return {}
        return getattr(self, f"{self._METHODOLOGY}_resources", {})

    @property
    def _claimed_name(self) -> str | None:
        """The name this step claims across its plan, or `None`, as steps have none."""
        return None

    def _check_names(self) -> None:
        """Reject a plan where one name means two objects.

        Raises:
            ResourceError: If one resource name covers two different objects.
            ValueError: If one step name covers two different steps.
        """
        resources: dict[str, Any] = {}
        names: dict[str, Step] = {}

        # Resource names
        for step in lineage.walk(self):
            for resource in step._resources().values():
                if resource.name is None:  # anonymous; only a document needs a name
                    continue
                seen = resources.setdefault(resource.name, resource.value)
                if seen is not resource.value:
                    raise ResourceError(
                        f"Resource '{resource.name}' covers two different objects in "
                        f"this plan ({type(seen).__name__} and "
                        f"{type(resource.value).__name__}). A document records only "
                        "the name, so both would be loaded as one. Give them different "
                        "names, or pass the same object to both."
                    )

            # Other claims made by upstream steps
            name = step._claimed_name
            if name is not None and names.setdefault(name, step) is not step:
                raise ValueError(
                    f"Name '{name}' covers two different steps in this plan."
                )

    def __repr__(self) -> str:
        """Return a short representation showing kind and collection state."""
        return f"<{type(self).__name__} {'collected' if self.is_collected else 'lazy'}>"

    def __str__(self) -> str:
        """How this step appears in a drawing."""
        return self.kind

    @property
    def is_collected(self) -> bool:
        """Whether this step has been materialised."""
        return self._fp is not None

    # -- plan identity ----------------------------------------------------------------

    @property
    @abstractmethod
    def spec(self) -> BaseModel:
        """This step's settings, as a serialisable model.

        One model per step kind, in `matchlab.specs`. It must carry everything this
        step's output depends on and nothing else. That is the invariant `_spec_key`
        rests on, and the one to check when adding a setting. Omit something that
        changes the output and collect will hand back a stale artifact without
        re-running (see `_fingerprint`).

        Specs describe a step's own settings, not its inputs'. Edges live on
        `parents`, and `_fingerprint` already folds in their fingerprints.
        """
        ...

    def _spec_key(self) -> bytes:
        """Bytes identifying this step's spec.

        `Source` extends this with a content hash of the data it read, so that the
        fingerprint tracks the warehouse. For every other kind the spec is the
        whole story.

        Serialised through pydantic, which preserves both field order and a settings
        dict's insertion order.

        The methodology adds what it promises about its own code: a version where it
        declares one, a fresh nonce where it does not. See `matchlab.core.versioning`.
        """
        return self.spec.model_dump_json().encode() + methodology_key(
            self._methodology_class()
        )

    @abstractmethod
    def _methodology_class(self) -> type | None:
        """The class whose code this step runs, where there is one.

        `None` for a step keyed by something other than code. `Source` is the real
        case, hashing the rows it read, so its location needs no version.
        """
        ...

    @property
    def _cacheable(self) -> bool:
        """Whether a later collect may reuse this step's stored artifact.

        False as soon as this step's methodology declares no version, or any step above
        it declares none. An unversioned step re-runs and may produce something else, so
        nothing below it can trust what it stored last time either.

        Fingerprints mostly take care of this on their own. `_spec_key` re-keys an
        unversioned step through its nonce, and a child folds its parents' fingerprints
        into its own. This method covers the one place they do not reach: `_ensure`
        short-circuits on a fingerprint this object already holds, without recomputing
        it, which would let a versioned step below an unversioned one report a cache hit
        against a parent that had just moved.
        """
        methodology = self._methodology_class()
        stable = methodology is None or declared_version(methodology) is not None
        return stable and all(parent._cacheable for parent in self.parents)

    def _fingerprint(self) -> Fingerprint:
        """Address this step by kind, spec and inputs.

        The key is derived from the *plan*, not from the step's output. That is what
        makes it computable before the step runs, which is what lets `_ensure` skip
        work. An output digest would only be knowable once the work was already done.

        **A step's output must be a deterministic function of its inputs and its
        settings.** `_ensure` skips a step whose fingerprint is stored, and a
        fingerprint is exactly kind + spec + parent fingerprints. Parent fingerprints
        stand in for the inputs, the spec for the settings.
        Same fingerprint must therefore mean same output.

        That is what a methodology promises by declaring a `version`, and what one
        declaring none refuses to promise. An unversioned methodology takes a fresh
        nonce here instead, so it re-runs and takes everything below it with it. Three
        things break the rule where it is claimed, and they are one defect in different
        clothes:

        * **Resources.** Anything passed in a step's `*_resources` reaches no spec, so
            it must be something the step reads *through* and never data it reads. See
            `matchlab.resources`.
        * **Randomness.** A methodology that samples is not a function of its settings
            unless it is seeded or otherwise made deterministic. `SplinkLinker` fills
            in `seed=0` for any training function that accepts a seed and omits it.
        * **Ambient state.** A library version, a locale, a clock. Upgrade a matcher and
            the same settings produce different edges under the same key.

        Each is a reason to leave `version` unset, or to bump it once the cause is
        fixed. Code matchlab ships declares one, so a change here is a bump.

        A **source** is a special case. It has no parents, so its
        inputs live outside matchlab; `_spec_key` reads the rows and hashes them to make
        them an input. That is why computing a source's key always costs a read.

        Break the rule and the key disagrees with the bytes, in one of two directions.

        * the spec omits something that changes the output, a **stale hit**, because
            `_ensure` never runs the step and reads the old artifact. This is the
            dangerous direction: it is silent, and it hands back wrong results.
            All three breakers above land here.
        * the spec includes something that doesn't change the output, a **spurious
            miss**, re-running the step and everything below it for nothing. Wasteful
            but safe. For example, in `SourceSpec.location_settings` the query is hashed
            even though the data hash already covers anything it could change.

        There is also no early cutoff. A `Transform` whose SQL is reformatted but
        semantically unchanged invalidates the whole subtree beneath it.

        TODO(fingerprints): split this into an action key (plan-derived, as now)
        mapping to an output digest (content-derived, recorded by the store on
        write), and build a child's key from its parents' output digests rather than
        their action keys. That buys early cutoff, storage dedup and rename tolerance.
        It does **not** fix stale hits. An output digest governs propagation, never
        admission, so a step whose action key is a hit still never runs and its digest
        is never consulted. Costs an indirection table in the store contract, a hash
        of every artifact on write, and the loss of a work-set knowable before running.
        See PLAN.md, "Known limitations", for the full ledger.
        """
        parts: list[bytes] = [self.kind.encode(), self._spec_key()]
        for parent in self.parents:
            if parent._fp is None:  # collect orders upstream first
                raise RuntimeError(
                    f"A {parent.kind} input of this {self.kind} has no fingerprint yet."
                )
            parts.append(parent._fp)
        return HASH_FUNC(b"|".join(parts)).digest()

    # -- execution --------------------------------------------------------------------

    @abstractmethod
    def _execute(self, store: Store, fp: Fingerprint) -> None:
        """Compute this step and persist its artifact under `fp`."""
        ...

    def _ensure(self, store: Store) -> StepStatus:
        """Materialise this step unless its artifact is already stored.

        A step this object already collected short-circuits on its stored `_fp`, without
        recomputing the fingerprint. A settled step cannot have changed, so recomputing
        would only confirm the fingerprint already held. That saves a spec hash per step
        per collect, which for a `Source` means re-hashing every row hash it memoised.

        That holds only while the step is `_cacheable`. An unversioned methodology
        re-keys on every collect, and so does everything below it, so a step in that
        subtree recomputes its fingerprint and runs again. It reports `REFRESHED` rather
        than `DONE`, since it did the work for want of a promise about the code rather
        than because anything moved.

        This classifies the outcome but reports none of it. `collect` holds the walk,
        and therefore each step's position, the thing a record has to quote to be
        findable in the plan. Reporting belongs there, with the single
        `matchlab.progress.Progress` that owns what is drawn and what is logged. That is
        also what keeps the two from ever disagreeing. This keeps to the work.

        Returns:
            What it took. `DONE` if this call computed the step, `CACHED` if the
            artifact was already stored, `REFRESHED` if the step cannot be cached and
            was computed again.
        """
        self._store = store

        if self._cacheable and self._fp is not None and store.has(self._fp):
            return StepStatus.CACHED

        # Once per collect, and it has to stay that way. An unversioned methodology
        # keys off a fresh nonce, so asking twice would address two artifacts.
        fp = self._fingerprint()

        if store.has(fp):  # cache hit, skip the work entirely
            if fp != self._fp and hasattr(self, "_prepared"):
                self._prepared = None
            self._fp = fp
            return StepStatus.CACHED

        self._execute(store, fp)
        self._fp = fp
        return StepStatus.DONE if self._cacheable else StepStatus.REFRESHED

    # -- public API -------------------------------------------------------------------

    def collect(
        self, store: Store | None = None, interactive: bool | None = None
    ) -> Self:
        """Materialise this step and everything it depends on.

        Steps whose artifact is already stored are skipped without being run, so
        re-collecting after adding a downstream step only does the new work.

        Reports as it goes: the plan, a record per step, and a closing summary of what
        ran, what was cached, how long it took and what the store now holds. No logging
        setup is needed for any of that — a collection lends the `matchlab` logger a
        console handler where the application hasn't configured one, and leaves an
        application that has entirely alone. See `matchlab.core.logging.audible`.

        Args:
            store: Where to read and write artifacts. Defaults to the module-level
                store (a DuckDB store in the user cache directory).
            interactive: Whether someone is watching. `None`, the default, takes a
                terminal or a notebook as a yes. When they are, the plan is drawn as a
                live tree redrawn in place, and not logged. The tree on screen is the
                key those `[step N]` records need, and it stays there. When they are
                not, the plan is logged instead. See `matchlab.progress`.

        Returns:
            This step, now collected.
        """
        store = store or default_store()
        # One walk, used for both. It fixes each step's position, so a `step 7` the
        # reporter logs is the node it drew as `[7]`.
        steps = lineage.walk(self)
        # Outside the report, so it is still standing when the report logs its closing
        # summary on the way out. Does nothing where the application configured logging
        # itself; see `matchlab.core.logging.audible`.
        with mlog.audible(), report(self, steps, interactive, store) as reporter:
            for step in steps:
                reporter.begin(step)
                try:
                    status = step._ensure(store)
                except Exception:
                    reporter.end(step, StepStatus.FAILED)
                    raise
                reporter.end(step, status)
        return self

    def lineage(self) -> list["Step"]:
        """Return this step and all its inputs, upstream-first."""
        return lineage.walk(self)

    def draw(self) -> str:
        """Render this step's sub-plan as a tree."""
        return lineage.draw(self)

    def fingerprints(self) -> set[Fingerprint]:
        """Address every artifact this plan is made of, its own and its inputs'.

        Which artifacts belong to a plan is the plan's own business, so this is where
        a store gets told: `store.prune(keep=plan.fingerprints())` hands storage a set
        of addresses it already understands, rather than a graph it would have to learn
        to walk.

        Returns:
            One fingerprint per step in `lineage()`. A set, because two steps in one
            plan can address the same artifact. Identical specs over identical
            inputs is the same bytes, and it is stored once.

        Raises:
            RuntimeError: If any step has not been collected. An uncollected plan names
                no artifacts at all, so answering with a smaller set would quietly tell
                a caller that less is worth keeping than they think.
        """
        return {step._collected()[1] for step in self.lineage()}

    # -- helpers for subclasses -------------------------------------------------------

    def _collected(self) -> tuple[Store, Fingerprint]:
        """Return the store this step was collected into, and its fingerprint.

        Both together, because everything that reads a stored artifact needs both and
        neither exists before collection. Returning the pair is what lets a caller
        use the fingerprint without re-checking that it is there.
        """
        if self._store is None or self._fp is None:
            raise RuntimeError(
                f"This {self.kind} has not been collected. Call collect() first."
            )
        return self._store, self._fp

    def _require_store(self) -> Store:
        """Return the store this step was collected into."""
        return self._collected()[0]
