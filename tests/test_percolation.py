from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

MODULE_PATH = Path(__file__).resolve().parents[1] / "percolation.py"
SPEC = importlib.util.spec_from_file_location("percolation_under_test", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Failed to load percolation module from {MODULE_PATH}")
PERCOLATION = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PERCOLATION
SPEC.loader.exec_module(PERCOLATION)

LammpsData = PERCOLATION.LammpsData
analyze_percolation = PERCOLATION.analyze_percolation
compute_report = PERCOLATION.compute_report
print_report = PERCOLATION.print_report
read_lammps_data = PERCOLATION.read_lammps_data
_check_translation_independent = PERCOLATION._check_translation_independent

REAL_DATA_PATH = Path(
    "/Users/arunsrikanthsridhar/Downloads/hydrogel_simulation/workspace/06fd0ccbfeedbfdf793a16b5d43fd918/data.lammps"
)


def _write_lammps_data(
    tmp_path: Path,
    atoms_header: str,
    atom_lines: list[str],
    bond_lines: list[str],
) -> Path:
    data_path = tmp_path / "data.perc.lammps"
    data_path.write_text(
        "\n".join(
            [
                "LAMMPS data file via unit test",
                "",
                f"{len(atom_lines)} atoms",
                f"{len(bond_lines)} bonds",
                "0 angles",
                "0 dihedrals",
                "0 impropers",
                "",
                "1 atom types",
                "1 bond types",
                "0 angle types",
                "0 dihedral types",
                "0 improper types",
                "",
                "0.0 10.0 xlo xhi",
                "0.0 10.0 ylo yhi",
                "0.0 10.0 zlo zhi",
                "",
                "Masses",
                "",
                "1 12.010",
                "",
                atoms_header,
                "",
                *atom_lines,
                "",
                "Bonds",
                "",
                *bond_lines,
                "",
            ]
        )
        + "\n"
    )
    return data_path


def test_read_lammps_data_returns_stable_named_payload(tmp_path: Path) -> None:
    data_path = _write_lammps_data(
        tmp_path,
        atoms_header="Atoms # full",
        atom_lines=[
            "1 1 1 0.0 1.0 1.0 1.0",
            "2 1 1 0.0 2.0 1.0 1.0",
        ],
        bond_lines=["1 1 1 2"],
    )

    parsed_default = read_lammps_data(str(data_path))
    assert isinstance(parsed_default, LammpsData)
    assert set(parsed_default.atom_data) == {1, 2}
    np.testing.assert_allclose(parsed_default.box, np.array([10.0, 10.0, 10.0]))
    np.testing.assert_array_equal(parsed_default.bond_translations[(1, 2)], np.array([0, 0, 0]))
    np.testing.assert_array_equal(parsed_default.bond_translations[(2, 1)], np.array([0, 0, 0]))


def test_read_lammps_data_accepts_optional_image_flags_but_uses_local_bond_crossings(
    tmp_path: Path,
) -> None:
    data_path = _write_lammps_data(
        tmp_path,
        atoms_header="Atoms # full",
        atom_lines=[
            "1 1 1 0.0 9.5 1.0 1.0 0 0 0",
            "2 1 1 0.0 1.0 1.0 1.0 3 0 0",
        ],
        bond_lines=["1 1 1 2"],
    )

    parsed = read_lammps_data(str(data_path))

    # The parser reads image flags, but the percolation edge field is defined from
    # the local wrapped bond crossing, not the absolute image-label difference.
    np.testing.assert_array_equal(parsed.bond_translations[(1, 2)], np.array([1, 0, 0]))
    np.testing.assert_array_equal(parsed.bond_translations[(2, 1)], np.array([-1, 0, 0]))


def test_read_lammps_data_rejects_non_full_atoms_style(tmp_path: Path) -> None:
    data_path = _write_lammps_data(
        tmp_path,
        atoms_header="Atoms # atomic",
        atom_lines=[
            "1 1 1.0 1.0 1.0",
            "2 1 2.0 1.0 1.0",
        ],
        bond_lines=["1 1 1 2"],
    )

    with pytest.raises(ValueError, match="Unsupported Atoms section style"):
        read_lammps_data(str(data_path))


def test_analyze_percolation_detects_x_wrapping_only_for_periodic_cycle() -> None:
    atom_data = {
        1: np.array([1.0, 1.0, 1.0]),
        2: np.array([3.0, 1.0, 1.0]),
        3: np.array([5.0, 1.0, 1.0]),
        4: np.array([9.5, 1.0, 1.0]),
    }
    bond_list = [
        (1, 1, 1, 2),
        (2, 1, 2, 3),
        (3, 1, 3, 4),
        (4, 1, 4, 1),
    ]
    neighbors = {
        1: [2, 4],
        2: [1, 3],
        3: [2, 4],
        4: [3, 1],
    }
    box = np.array([10.0, 10.0, 10.0])
    bond_translations = {
        (1, 2): np.array([0, 0, 0]),
        (2, 1): np.array([0, 0, 0]),
        (2, 3): np.array([0, 0, 0]),
        (3, 2): np.array([0, 0, 0]),
        (3, 4): np.array([0, 0, 0]),
        (4, 3): np.array([0, 0, 0]),
        (4, 1): np.array([1, 0, 0]),
        (1, 4): np.array([-1, 0, 0]),
    }

    components = analyze_percolation(
        atom_data,
        bond_list,
        neighbors,
        box,
        bond_translations=bond_translations,
    )
    assert len(components) == 1
    comp = components[0]
    assert comp["percolation_dim"] == 1
    assert len(comp["basis_vectors"]) == 1
    np.testing.assert_array_equal(comp["percolates"], np.array([True, False, False]))

    report = compute_report(components)
    np.testing.assert_array_equal(report["system_percolates"], np.array([True, False, False]))
    assert report["largest_component_percolation_dim"] == 1


def test_translation_basis_rejects_dependent_vectors() -> None:
    existing_basis = [np.array([-1, 0, 0])]

    assert not _check_translation_independent(existing_basis, np.array([3, 0, 0]))
    assert _check_translation_independent(existing_basis, np.array([0, 1, 0]))


def test_print_report_writes_report_file(tmp_path: Path) -> None:
    atom_data = {
        1: np.array([1.0, 1.0, 1.0]),
        2: np.array([3.0, 1.0, 1.0]),
        3: np.array([5.0, 1.0, 1.0]),
        4: np.array([9.5, 1.0, 1.0]),
    }
    bond_list = [
        (1, 1, 1, 2),
        (2, 1, 2, 3),
        (3, 1, 3, 4),
        (4, 1, 4, 1),
    ]
    neighbors = {
        1: [2, 4],
        2: [1, 3],
        3: [2, 4],
        4: [3, 1],
    }
    box = np.array([10.0, 10.0, 10.0])
    bond_translations = {
        (1, 2): np.array([0, 0, 0]),
        (2, 1): np.array([0, 0, 0]),
        (2, 3): np.array([0, 0, 0]),
        (3, 2): np.array([0, 0, 0]),
        (3, 4): np.array([0, 0, 0]),
        (4, 3): np.array([0, 0, 0]),
        (4, 1): np.array([1, 0, 0]),
        (1, 4): np.array([-1, 0, 0]),
    }

    components = analyze_percolation(
        atom_data,
        bond_list,
        neighbors,
        box,
        bond_translations=bond_translations,
    )

    report_path = tmp_path / 'percolation_report.txt'
    report = print_report(components, report_out=str(report_path))

    assert report_path.exists()
    report_text = report_path.read_text()
    assert 'PERCOLATION ANALYSIS REPORT' in report_text
    assert 'Component   0:' in report_text
    assert report['largest_component_percolation_dim'] == 1


def test_real_data_file_smoke_regression() -> None:
    if not REAL_DATA_PATH.exists():
        pytest.skip(f"Real regression file not found: {REAL_DATA_PATH}")

    parsed = read_lammps_data(str(REAL_DATA_PATH))

    assert len(parsed.atom_data) == 27510
    assert len(parsed.bond_list) == 27300
    assert len(parsed.bond_translations) == 2 * len(parsed.bond_list)

    components = analyze_percolation(
        parsed.atom_data,
        parsed.bond_list,
        parsed.neighbors,
        parsed.box,
        bond_translations=parsed.bond_translations,
    )
    report = compute_report(components)

    assert len(components) == 630
    assert report["largest_component_percolation_dim"] == 0
    np.testing.assert_array_equal(report["system_percolates"], np.array([False, False, False]))
