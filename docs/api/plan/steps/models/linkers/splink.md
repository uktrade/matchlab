`SplinkLinker` uses Splink 5. See the [Splink documentation](https://moj-analytical-services.github.io/splink/) for configuration and usage.

::: matchlab.models.linkers.splinklinker
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
