# Resolvers

The `Resolver` step. A resolver's methodology takes the output of a [model][matchlab.models.models] and resolves it into entities. It is a [`RecordStep`][matchlab.recordstep] that can be used in further entity resolution processes, or output as your final lookup.

::: matchlab.resolvers.resolvers
    options:
        show_root_heading: true
        show_root_full_path: true
        members_order: source
        show_if_no_docstring: true
        docstring_style: google
        show_signature_annotations: true
        separate_signature: true
        filters:
            - "!^[A-Z]$"
            - "!^_"

::: matchlab.resolvers.base
    options:
        show_root_heading: true
        show_root_full_path: true
        members_order: source
        show_if_no_docstring: true
        docstring_style: google
        show_signature_annotations: true
        separate_signature: true
        filters:
            - "!^[A-Z]$"
            - "!^_"
