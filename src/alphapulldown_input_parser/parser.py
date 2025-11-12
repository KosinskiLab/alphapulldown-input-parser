"""Standalone parser for AlphaPulldown fold specifications."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Dict, Iterable, List, NamedTuple, Optional, Sequence, Tuple, Union


class FormatError(ValueError):
    """Raised when a fold specification cannot be parsed."""


def _format_error(spec: str, msg: str | None = None) -> None:
    """Mirror the historical AlphaPulldown error message."""
    base = f"Your format: {spec} is wrong. The program will terminate."
    detail = f" {msg}" if msg else ""
    raise FormatError(f"{base}{detail}")


@dataclass(frozen=True)
class Region:
    """Closed interval over the protein sequence."""

    start: int
    end: int

    def __post_init__(self) -> None:
        if self.start < 0 or self.end < 0:
            raise ValueError("Region boundaries must be non-negative integers.")
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


FoldEntry = Dict[str, Union[str, RegionSelection]]


class ExpandResult(NamedTuple):
    formatted_folds: List[FoldEntry]
    missing_features: List[str]


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
                json_files.setdefault(filename, str(entry))
            elif filename.endswith(".pkl"):
                base = filename[:-4]
                pkl.setdefault(base, []).append(str(entry))
            elif filename.endswith(".pkl.xz"):
                base = filename[:-7]
                pkl.setdefault(base, []).append(str(entry))

    return FeatureIndex(
        pkl={name: tuple(paths) for name, paths in pkl.items()},
        json=json_files,
    )


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


def expand_fold_specification(
    spec: str,
    features_directory: Iterable[str],
    protein_delimiter: str,
    *,
    feature_index: FeatureIndex | None = None,
) -> ExpandResult:
    """Expand a single fold specification.

    Returns a tuple of (formatted_folds, missing_features).
    """

    index = feature_index
    if index is None:
        directories = tuple(Path(d).expanduser().resolve() for d in features_directory)
        index = _build_feature_index(directories)

    formatted_folds: List[FoldEntry] = []
    missing_features: List[str] = []

    for pf in spec.split(protein_delimiter):
        if pf.endswith(".json"):
            json_name = pf
            json_path = index.json_path(json_name)
            if json_path:
                formatted_folds.append({"json_input": json_path})
            else:
                missing_features.append(json_name)
            continue

        tokens = pf.split(":")
        if not tokens or not tokens[0]:
            _format_error(spec, msg="Protein token is empty.")

        name = tokens[0]
        number, region_tokens = _extract_copy_and_regions(tokens, spec)
        regions = _parse_regions(region_tokens, spec)

        if not index.has_pkl(name):
            missing_features.append(name)
            continue

        for _ in range(number):
            formatted_folds.append({name: regions})

    return ExpandResult(formatted_folds=formatted_folds, missing_features=missing_features)


def parse_fold(
    input_list: List[str],
    features_directory: Iterable[str],
    protein_delimiter: str,
) -> List[List[FoldEntry]]:
    """Parse a list of fold specifications into folding jobs."""

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


def _read_nonempty_lines(path: Path) -> List[str]:
    with path.open(mode="r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


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

    combinations = list(product(*lines_per_file)) if lines_per_file else []

    if exclude_permutations:
        filtered: List[Tuple[str, ...]] = []
        seen: set[Tuple[str, ...]] = set()
        for combo in combinations:
            normalized = tuple(sorted(map(str, combo)))
            if normalized in seen:
                continue
            seen.add(normalized)
            filtered.append(tuple(map(str, combo)))
        combinations = filtered
    else:
        combinations = [tuple(map(str, combo)) for combo in combinations]

    specifications = [delimiter.join(combo) for combo in combinations]

    if output_path:
        output = Path(output_path).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("\n".join(specifications) + ("\n" if specifications else ""), encoding="utf-8")

    return specifications
