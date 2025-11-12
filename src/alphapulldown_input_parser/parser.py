"""Standalone parser for AlphaPulldown fold specifications."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Dict, Iterable, List, NamedTuple, Optional, Sequence, Tuple, Union

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _deduplicate_preserve_order(items: Iterable[str]) -> Tuple[str, ...]:
    """Return a tuple containing the first occurrence of every item."""
    return tuple(dict.fromkeys(items))


def _strip_path_and_extension(value: str) -> str:
    return Path(value).stem


def _format_error(spec: str, msg: str | None = None) -> None:
    """Mirror the historical AlphaPulldown error message."""
    base = f"Your format: {spec} is wrong. The program will terminate."
    detail = f" {msg}" if msg else ""
    raise FormatError(f"{base}{detail}")


def _read_nonempty_lines(path: Path) -> List[str]:
    with path.open(mode="r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


# ---------------------------------------------------------------------------
# Core data structures
# ---------------------------------------------------------------------------


class FormatError(ValueError):
    """Raised when a fold specification cannot be parsed."""


@dataclass(frozen=True)
class Region:
    """1-based closed interval over the protein sequence."""

    start: int
    end: int

    def __post_init__(self) -> None:
        if self.start < 1 or self.end < 1:
            raise ValueError("Region boundaries must be positive integers (1-based).")
        if self.start > self.end:
            raise ValueError("Region start must not exceed region end.")


@dataclass(frozen=True)
class RegionSelection:
    """Container describing the region selection for a fold."""

    regions: Tuple[Region, ...] | None = None

    @property
    def is_all(self) -> bool:
        return self.regions is None

    @classmethod
    def all(cls) -> "RegionSelection":
        return cls(regions=None)


# Either {"json_input": "/path/to.json"} or {"CHAIN_A": RegionSelection(...)}
FoldEntry = Dict[str, Union[str, RegionSelection]]


class ExpandResult(NamedTuple):
    formatted_folds: List[FoldEntry]
    missing_features: List[str]


# ---------------------------------------------------------------------------
# Fold dataset
# ---------------------------------------------------------------------------


@dataclass
class FoldDataset:
    """Container encapsulating parsed fold specifications."""

    fold_specifications: Tuple[str, ...]
    sequences_by_origin: Dict[str, Tuple[str, ...]]
    sequences_by_fold: Dict[str, Tuple[str, ...]]

    @property
    def unique_sequences(self) -> Tuple[str, ...]:
        ordered: List[str] = []
        for spec in self.fold_specifications:
            ordered.extend(self.sequences_by_fold.get(spec, ()))
        return _deduplicate_preserve_order(ordered)

    def symlink_local_files(self, output_directory: Union[str, Path]) -> None:
        output = Path(output_directory).expanduser().resolve()
        output.mkdir(parents=True, exist_ok=True)
        for file in self.sequences_by_origin.get("local", ()):
            source = Path(file).expanduser()
            target = output / Path(file).name
            if target.exists() or target.is_symlink():
                target.unlink()
            target.symlink_to(source)

    @classmethod
    def from_fold_specifications(
        cls,
        fold_specifications: Sequence[str],
        *,
        protein_delimiter: str = "+",
    ) -> "FoldDataset":
        normalized_specs: List[str] = []
        sequences_by_fold: Dict[str, Tuple[str, ...]] = {}
        referenced_sequences: List[str] = []

        for specification in fold_specifications:
            tokens = [
                token.strip()
                for token in specification.split(protein_delimiter)
                if token.strip()
            ]
            if not tokens:
                continue

            normalized_tokens: List[str] = []
            per_fold_sequences: List[str] = []
            for token in tokens:
                parts = [part.strip() for part in token.split(":")]
                protein_reference = parts[0]
                referenced_sequences.append(protein_reference)

                base_name = _strip_path_and_extension(protein_reference)
                per_fold_sequences.append(base_name)
                suffix_components = [part for part in parts[1:] if part]
                normalized_tokens.append(
                    ":".join([base_name, *suffix_components]) if suffix_components else base_name
                )

            normalized_spec = protein_delimiter.join(normalized_tokens)
            normalized_specs.append(normalized_spec)
            sequences_by_fold[normalized_spec] = _deduplicate_preserve_order(per_fold_sequences)

        unique_inputs = _deduplicate_preserve_order(referenced_sequences)
        sequences_by_origin: Dict[str, List[str]] = {"uniprot": [], "local": []}
        for sequence in unique_inputs:
            path = Path(sequence).expanduser()
            has_separator = "/" in sequence or "\\" in sequence
            has_suffix = path.suffix != ""
            exists = path.exists() and (path.is_file() or path.is_symlink())
            is_local = exists and (has_separator or has_suffix)
            if is_local:
                sequences_by_origin["local"].append(str(path.resolve()))
            else:
                sequences_by_origin["uniprot"].append(sequence)

        return cls(
            fold_specifications=_deduplicate_preserve_order(normalized_specs),
            sequences_by_origin={
                key: tuple(values) for key, values in sequences_by_origin.items()
            },
            sequences_by_fold=sequences_by_fold,
        )

    @classmethod
    def from_file(
        cls,
        filepath: Union[str, Path],
        *,
        protein_delimiter: str = "+",
    ) -> "FoldDataset":
        path = Path(filepath).expanduser().resolve()
        specifications = _deduplicate_preserve_order(_read_nonempty_lines(path))
        return cls.from_fold_specifications(
            specifications,
            protein_delimiter=protein_delimiter,
        )

# ---------------------------------------------------------------------------
# Feature index
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FeatureIndex:
    """Pre-indexed feature availability for faster lookups."""

    pkl: Dict[str, Tuple[str, ...]]
    json: Dict[str, str]

    def has_pkl(self, name: str) -> bool:
        return name in self.pkl

    def json_path(self, name: str) -> Optional[str]:
        return self.json.get(name)


def _build_feature_index(directories: Sequence[Path]) -> FeatureIndex:
    pkl: Dict[str, List[str]] = {}
    json_files: Dict[str, str] = {}

    for directory in directories:
        if not directory.is_dir():
            continue
        for entry in directory.iterdir():
            if not entry.is_file():
                continue
            filename = entry.name
            if filename.endswith(".json"):
                keys = {entry.name, entry.stem}
                for key in keys:
                    json_files.setdefault(key, str(entry))
            elif filename.endswith(".pkl"):
                base = filename[:-4]
                keys = {base, entry.name, entry.stem}
                for key in keys:
                    pkl.setdefault(key, []).append(str(entry))
            elif filename.endswith(".pkl.xz"):
                base = filename[:-7]
                keys = {base, entry.name, Path(filename[:-3]).stem}
                for key in keys:
                    pkl.setdefault(key, []).append(str(entry))

    return FeatureIndex(
        # deduplicate paths while preserving order
        pkl={name: tuple(dict.fromkeys(paths)) for name, paths in pkl.items()},
        json=json_files,
    )


# ---------------------------------------------------------------------------
# Expansion helpers
# ---------------------------------------------------------------------------


def _extract_copy_and_regions(tokens: Sequence[str], spec: str) -> Tuple[int, Sequence[str]]:
    """Return copy count and the remaining region tokens."""
    if len(tokens) > 1:
        try:
            return int(tokens[1]), tokens[2:]
        except ValueError:
            pass
        try:
            return int(tokens[-1]), tokens[1:-1]
        except ValueError:
            pass
    return 1, tokens[1:]


def _parse_regions(region_tokens: Sequence[str], spec: str) -> RegionSelection:
    """Parse optional region tokens into a RegionSelection."""
    if not region_tokens:
        return RegionSelection.all()
    regions: List[Region] = []
    for tok in region_tokens:
        parts = tok.split("-")
        if len(parts) != 2:
            _format_error(spec, msg=f"Region token '{tok}' is not of form start-stop.")
        try:
            start, end = map(int, parts)
        except ValueError:
            _format_error(spec, msg=f"Region token '{tok}' contains non-integer bounds.")
        try:
            regions.append(Region(start=start, end=end))
        except ValueError as exc:
            _format_error(spec, msg=str(exc))
    return RegionSelection(regions=tuple(regions))


# ---------------------------------------------------------------------------
# Expansion logic
# ---------------------------------------------------------------------------


def expand_fold_specification(
    spec: str,
    features_directory: Iterable[str],
    protein_delimiter: str,
    *,
    feature_index: FeatureIndex | None = None,
) -> ExpandResult:
    """Expand a single fold specification into fold entries.

    Example:
        >>> expand_fold_specification("protA:1-10+protB", ["/features"], "+", feature_index=index)

    Returns:
        ExpandResult with fold entries and missing feature names. Pure function.
    """

    index = feature_index
    if index is None:
        directories = tuple(Path(d).expanduser().resolve() for d in features_directory)
        index = _build_feature_index(directories)

    formatted_folds: List[FoldEntry] = []
    missing_features: List[str] = []

    for raw_pf in spec.split(protein_delimiter):
        pf = raw_pf.strip()
        if not pf:
            continue

        if pf.endswith(".json"):
            path_pf = Path(pf)
            json_path: Optional[str] = None
            for json_key in (path_pf.name, path_pf.stem):
                json_path = index.json_path(json_key)
                if json_path:
                    formatted_folds.append({"json_input": json_path})
                    break
            if json_path:
                continue
            missing_features.append(path_pf.name)
            continue

        tokens = [token.strip() for token in pf.split(":")]
        if not tokens or not tokens[0]:
            _format_error(spec, msg="Protein token is empty.")

        name = tokens[0]
        name_path = Path(name)
        name_candidates = [
            name,
            name_path.name,
            name_path.stem,
        ]
        # copy count can be either second or last token
        number, region_tokens = _extract_copy_and_regions(tokens, spec)
        regions = _parse_regions(region_tokens, spec)

        # try different name representations against prebuilt index
        canonical_name = next((candidate for candidate in name_candidates if index.has_pkl(candidate)), None)
        if canonical_name is None:
            missing_features.append(name)
            continue

        for _ in range(number):
            formatted_folds.append({canonical_name: regions})

    return ExpandResult(formatted_folds=formatted_folds, missing_features=missing_features)


def parse_fold(
    input_list: List[str],
    features_directory: Iterable[str],
    protein_delimiter: str,
) -> List[List[FoldEntry]]:
    """Parse a list of fold specifications into folding jobs.

    Example:
        >>> parse_fold(["protA+protB"], ["/features"], "+")

    Returns:
        List of jobs (each job is a list of FoldEntry). Pure function.
    """

    directories = tuple(features_directory)
    directory_labels = [str(d) for d in directories]
    directory_paths = tuple(Path(d).expanduser().resolve() for d in directories)
    feature_index = _build_feature_index(directory_paths)

    all_folding_jobs: List[List[FoldEntry]] = []
    missing_features = set()

    for spec in input_list:
        result = expand_fold_specification(
            spec=spec,
            features_directory=directories,
            protein_delimiter=protein_delimiter,
            feature_index=feature_index,
        )
        missing_features.update(result.missing_features)
        if result.formatted_folds:
            all_folding_jobs.append(result.formatted_folds)

    if missing_features:
        raise FileNotFoundError(f"{sorted(missing_features)} not found in {directory_labels}")

    return all_folding_jobs


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_fold_specifications(
    input_files: Sequence[Union[str, Path]],
    *,
    delimiter: str = "+",
    exclude_permutations: bool = True,
    output_path: Optional[Union[str, Path]] = None,
) -> List[str]:
    """Compute the Cartesian product of specification files.

    Args:
        input_files: Paths to text files containing one specification per line.
        delimiter: Delimiter used to join the combination into a specification string.
        exclude_permutations: When True, filter out combinations that are permutations
            of entries that already appear.
        output_path: Optional destination to persist the resulting specifications.

    Returns:
        List of joined specification strings.
    """

    paths = [Path(p).expanduser().resolve() for p in input_files]
    lines_per_file: List[List[str]] = []
    for path in paths:
        lines = _read_nonempty_lines(path)
        if not lines:
            warnings.warn(
                f"Input file '{path}' contains no specifications; skipping combination generation.",
                RuntimeWarning,
            )
            return []
        lines_per_file.append(lines)

    if not lines_per_file:
        combinations: List[Tuple[str, ...]] = []
    else:
        combinations_iter = product(*lines_per_file)
        seen: set[Tuple[str, ...]] = set()
        combinations = []
        for combo in combinations_iter:
            typed_combo = tuple(map(str, combo))
            if exclude_permutations:
                normalized = tuple(sorted(typed_combo))
                if normalized in seen:
                    continue
                seen.add(normalized)
            combinations.append(typed_combo)

    specifications = [delimiter.join(combo) for combo in combinations]

    if output_path:
        output = Path(output_path).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("\n".join(specifications) + ("\n" if specifications else ""), encoding="utf-8")

    return specifications
