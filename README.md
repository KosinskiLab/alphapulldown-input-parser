# alphapulldown-input-parser

Reusable parser for AlphaPulldown-style fold specifications. Install it with:

```bash
pip install alphapulldown-input-parser
```

or, for local development:

```bash
pip install -e /path/to/alphapulldown-input-parser
```

The package exposes two helpers:

* `parse_fold(...)` – mirrors the historical AlphaPulldown helper and performs
  feature existence checks.
* `expand_fold_specification(...)` – expands a single fold string without
  raising if features are missing.

The parser is dependency-free and works across AlphaPulldown, the Snakemake
pipeline, or any other tooling that consumes the same fold syntax.
