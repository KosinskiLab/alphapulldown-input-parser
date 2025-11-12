from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pytest  # type: ignore[import-not-found]

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from alphapulldown_input_parser import (
    FormatError,
    FeatureIndex,
    Region,
    RegionSelection,
    generate_fold_specifications,
    parse_fold,
)


def selection_all() -> RegionSelection:
    return RegionSelection.all()


def selection_ranges(*ranges: tuple[int, int]) -> RegionSelection:
    return RegionSelection(regions=tuple(Region(start, end) for start, end in ranges))


@pytest.fixture
def patch_feature_index(monkeypatch):
    def _patch(
        pkl_entries: Dict[str, Tuple[str, ...]] | None = None,
        json_entries: Dict[str, str] | None = None,
    ) -> None:
        index = FeatureIndex(
            pkl={name: tuple(paths) for name, paths in (pkl_entries or {}).items()},
            json=dict(json_entries or {}),
        )

        def fake_build_feature_index(_directories):
            return index

        monkeypatch.setattr(
            "alphapulldown_input_parser.parser._build_feature_index",
            fake_build_feature_index,
        )

    return _patch


@pytest.mark.parametrize(
    (
        "input_list",
        "features_directory",
        "protein_delimiter",
        "feature_index_data",
        "expected_result",
        "expected_exception",
        "expected_message",
    ),
    [
        pytest.param(
            ["protein1"],
            ["dir1"],
            "_",
            {"pkl": {"protein1": ("dir1/protein1.pkl",)}},
            [[{"protein1": selection_all()}]],
            None,
            None,
            id="single_protein_no_copy",
        ),
        pytest.param(
            ["protein1:2"],
            ["dir1"],
            "_",
            {"pkl": {"protein1": ("dir1/protein1.pkl",)}},
            [[{"protein1": selection_all()}, {"protein1": selection_all()}]],
            None,
            None,
            id="single_protein_with_copy_number",
        ),
        pytest.param(
            ["protein1:1-10"],
            ["dir1"],
            "_",
            {"pkl": {"protein1": ("dir1/protein1.pkl",)}},
            [[{"protein1": selection_ranges((1, 10))}]],
            None,
            None,
            id="single_protein_with_region",
        ),
        pytest.param(
            ["protein1:2:1-10:20-30"],
            ["dir1"],
            "_",
            {"pkl": {"protein1": ("dir1/protein1.pkl",)}},
            [
                [
                    {"protein1": selection_ranges((1, 10), (20, 30))},
                    {"protein1": selection_ranges((1, 10), (20, 30))},
                ]
            ],
            None,
            None,
            id="single_protein_with_copy_and_regions",
        ),
        pytest.param(
            ["protein1:1-10:20-30:2"],
            ["dir1"],
            "_",
            {"pkl": {"protein1": ("dir1/protein1.pkl",)}},
            [
                [
                    {"protein1": selection_ranges((1, 10), (20, 30))},
                    {"protein1": selection_ranges((1, 10), (20, 30))},
                ]
            ],
            None,
            None,
            id="single_protein_with_region_and_copy",
        ),
        pytest.param(
            ["protein1:2_protein2:1-50"],
            ["dir1"],
            "_",
            {"pkl": {"protein1": ("dir1/protein1.pkl",), "protein2": ("dir1/protein2.pkl",)}},
            [
                [
                    {"protein1": selection_all()},
                    {"protein1": selection_all()},
                    {"protein2": selection_ranges((1, 50))},
                ]
            ],
            None,
            None,
            id="multiple_proteins",
        ),
        pytest.param(
            ["protein1", "protein2"],
            ["dir1"],
            "_",
            {"pkl": {}},
            None,
            FileNotFoundError,
            "['protein1', 'protein2'] not found in ['dir1']",
            id="missing_features",
        ),
        pytest.param(
            ["protein1::1-10"],
            ["dir1"],
            "_",
            {"pkl": {}},
            None,
            FormatError,
            "Your format: protein1::1-10 is wrong. The program will terminate. Region token '' is not of form start-stop.",
            id="invalid_format",
        ),
        pytest.param(
            ["protein1"],
            ["dir1", "dir2"],
            "_",
            {"pkl": {"protein1": ("dir2/protein1.pkl",)}},
            [[{"protein1": selection_all()}]],
            None,
            None,
            id="feature_exists_in_multiple_dirs",
        ),
        pytest.param(
            ["rna.json"],
            ["dir1"],
            "+",
            {"pkl": {}, "json": {"rna.json": "dir1/rna.json"}},
            [[{"json_input": "dir1/rna.json"}]],
            None,
            None,
            id="single_json_file",
        ),
        pytest.param(
            ["protein1+rna.json"],
            ["dir1"],
            "+",
            {
                "pkl": {"protein1": ("dir1/protein1.pkl",)},
                "json": {"rna.json": "dir1/rna.json"},
            },
            [[{"protein1": selection_all()}, {"json_input": "dir1/rna.json"}]],
            None,
            None,
            id="json_with_protein",
        ),
        pytest.param(
            ["rna.json"],
            ["dir1"],
            "+",
            {"pkl": {}, "json": {}},
            None,
            FileNotFoundError,
            "['rna.json'] not found in ['dir1']",
            id="missing_json_file",
        ),
        pytest.param(
            ["rna.json"],
            ["dir1", "dir2"],
            "+",
            {"pkl": {}, "json": {"rna.json": "dir2/rna.json"}},
            [[{"json_input": "dir2/rna.json"}]],
            None,
            None,
            id="json_in_multiple_dirs",
        ),
    ],
)
def test_parse_fold(
    patch_feature_index,
    input_list: List[str],
    features_directory: Iterable[str],
    protein_delimiter: str,
    feature_index_data: Dict[str, Dict[str, object]],
    expected_result: Optional[List[List[Dict[str, object]]]],
    expected_exception: Optional[type[BaseException]],
    expected_message: Optional[str],
) -> None:
    pkl_entries = feature_index_data.get("pkl", {})
    json_entries = feature_index_data.get("json", {})
    patch_feature_index(pkl_entries=pkl_entries, json_entries=json_entries)

    if expected_exception:
        with pytest.raises(expected_exception) as excinfo:
            parse_fold(input_list, features_directory, protein_delimiter)

        if expected_message:
            assert str(excinfo.value) == expected_message
    else:
        result = parse_fold(input_list, features_directory, protein_delimiter)
        assert result == expected_result


def _write_lines(directory: Path, filename: str, lines: List[str]) -> Path:
    path = directory / filename
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_generate_fold_specifications_excludes_permutations(tmp_path: Path) -> None:
    file_a = _write_lines(tmp_path, "a.txt", ["p1", "p2"])
    file_b = _write_lines(tmp_path, "b.txt", ["p2", "p1"])

    result = generate_fold_specifications(
        [file_a, file_b],
        delimiter="+",
        exclude_permutations=True,
    )

    assert result == ["p1+p2", "p1+p1", "p2+p2"]


def test_generate_fold_specifications_includes_permutations(tmp_path: Path) -> None:
    file_a = _write_lines(tmp_path, "a.txt", ["p1"])
    file_b = _write_lines(tmp_path, "b.txt", ["p2", "p1"])

    result = generate_fold_specifications(
        [file_a, file_b],
        delimiter="+",
        exclude_permutations=False,
    )

    assert result == ["p1+p2", "p1+p1"]


def test_generate_fold_specifications_writes_to_disk(tmp_path: Path) -> None:
    file_a = _write_lines(tmp_path, "a.txt", ["p1"])
    file_b = _write_lines(tmp_path, "b.txt", ["p2"])
    output_path = tmp_path / "output.txt"

    result = generate_fold_specifications(
        [file_a, file_b],
        delimiter="+",
        output_path=output_path,
    )

    assert result == ["p1+p2"]
    assert output_path.read_text(encoding="utf-8") == "p1+p2\n"
