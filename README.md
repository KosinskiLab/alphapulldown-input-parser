# alphapulldown-input-parser

Reusable parser for AlphaPulldown-style fold specifications. Install it with:

```bash
pip install "alphapulldown-input-parser>=0.4.0"
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

As of `0.4.0`, AF3 JSON feature files support the same copy/range suffixes as
classic AlphaPulldown feature pickles. For example:

```python
parse_fold(
    ["P01258_af3_input.json:1-100:2"],
    features_directory=["/path/to/features"],
    protein_delimiter="+",
)
```

This expands to two folding entries for the same AF3 JSON feature file, each
restricted to residues `1-100`.
