#!/usr/bin/env python3
"""
Percolation analysis for LAMMPS data files.

This module:
1) Parses a LAMMPS data file into atoms, bonds, neighbor graph, and box vectors.
2) Finds connected bond components and detects periodic wrapping via BFS image offsets.
3) Computes per-component and largest-component percolation summaries.
4) Optionally renders a human-readable report and a component-colored data file.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import re
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)



@dataclass(frozen=True)
class LammpsData:
    """Parsed LAMMPS data-file payload used by percolation analysis."""

    atom_data: dict[int, np.ndarray]
    bond_list: list[tuple[int, int, int, int]]
    neighbors: dict[int, list[int]]
    box: np.ndarray
    bond_translations: dict[tuple[int, int], np.ndarray]


def _validate_atoms_header(header_line: str) -> None:
    """Reject unsupported atom styles before parsing coordinates."""
    lower = header_line.lower()
    if "#" in lower and "full" not in lower:
        raise ValueError(
            "Unsupported Atoms section style. Expected 'Atoms # full' (or plain "
            "'Atoms' with full-style columns), got: "
            f"{header_line.strip()!r}"
        )


def _format_dim_flags(flags: np.ndarray) -> str:
    return "".join(["X" if flags[0] else "-", "Y" if flags[1] else "-", "Z" if flags[2] else "-"])


def _check_translation_independent(
    existing_basis: list[np.ndarray],
    new_vector: np.ndarray,
) -> bool:
    """Return True if ``new_vector`` adds a new independent periodic direction."""
    new_vector = np.asarray(new_vector, dtype=int)
    if not np.any(new_vector):
        return False
    if len(existing_basis) >= 3:
        return False
    if not existing_basis:
        return True

    # Only rank-increasing loop translations add a new percolation direction.
    # For example, [3, 0, 0] is dependent if [-1, 0, 0] is already in the basis.
    old_matrix = np.vstack([np.asarray(vec, dtype=float) for vec in existing_basis])
    new_matrix = np.vstack([old_matrix, new_vector.astype(float)])
    return np.linalg.matrix_rank(new_matrix) > np.linalg.matrix_rank(old_matrix)


def _basis_axes(basis_vectors: list[np.ndarray]) -> np.ndarray:
    """Map independent basis vectors back to x/y/z axis flags for reporting."""
    if not basis_vectors:
        return np.array([False, False, False], dtype=bool)
    basis = np.vstack([np.asarray(vec, dtype=int) for vec in basis_vectors])
    return np.any(basis != 0, axis=0)


def _infer_bond_translation(
    atom1_position: np.ndarray,
    atom2_position: np.ndarray,
    box: np.ndarray,
) -> np.ndarray:
    """Infer the periodic translation from atom1 to atom2 in an orthorhombic box."""
    diff = atom2_position - atom1_position
    return -np.round(diff / box).astype(int)


def _build_bond_translations(
    atom_data: dict[int, np.ndarray],
    bond_list: list[tuple[int, int, int, int]],
    box: np.ndarray,
) -> dict[tuple[int, int], np.ndarray]:
    """Build directed bond translations from local wrapped bond geometry."""
    translations: dict[tuple[int, int], np.ndarray] = {}
    for _bond_id, _bond_type, a1, a2 in bond_list:
        crossing = _infer_bond_translation(atom_data[a1], atom_data[a2], box)
        # Percolation needs a local edge translation field. Using absolute atom-image
        # differences here can telescope around closed loops and suppress the nonzero
        # lattice translations that define periodic percolation.
        translations[(a1, a2)] = crossing.copy()
        translations[(a2, a1)] = (-crossing).copy()
    return translations


def read_lammps_data(filename: str) -> LammpsData:
    """Read a LAMMPS data file and return a stable, named payload."""

    logger.info("Reading: %s", filename)
    with open(filename) as f:
        lines = f.readlines()

    atoms_line = None
    atoms_header = None
    bonds_line = None
    box_lo = np.zeros(3)
    box_hi = np.zeros(3)

    for i, line in enumerate(lines):
        s = line.strip()
        if s.startswith("Atoms"):
            atoms_line = i
            atoms_header = s
        elif s == "Bonds" or s.startswith("Bonds "):
            bonds_line = i
        elif "xlo xhi" in s:
            parts = s.split()
            box_lo[0], box_hi[0] = float(parts[0]), float(parts[1])
        elif "ylo yhi" in s:
            parts = s.split()
            box_lo[1], box_hi[1] = float(parts[0]), float(parts[1])
        elif "zlo zhi" in s:
            parts = s.split()
            box_lo[2], box_hi[2] = float(parts[0]), float(parts[1])

    if atoms_line is None or atoms_header is None:
        raise ValueError("Failed to locate 'Atoms' section in LAMMPS data file.")
    if bonds_line is None:
        raise ValueError("Failed to locate 'Bonds' section in LAMMPS data file.")
    _validate_atoms_header(atoms_header)

    box = box_hi - box_lo
    logger.info(
        "Box dimensions: %.3f x %.3f x %.3f Angstrom",
        box[0],
        box[1],
        box[2],
    )

    atom_data: dict[int, np.ndarray] = {}
    atom_images: dict[int, np.ndarray] = {}
    i = atoms_line + 2
    while i < len(lines):
        raw = lines[i]
        line = raw.strip()
        if not line or not line[0].isdigit():
            break
        parts = line.split()
        if len(parts) < 7:
            raise ValueError(
                "Atoms row has fewer than 7 columns. This parser expects atom_style "
                f"full-like rows with x/y/z at columns 5/6/7. Bad row at line {i+1}: "
                f"{raw.rstrip()!r}"
            )
        try:
            aid = int(parts[0])
            x = float(parts[4]) - box_lo[0]
            y = float(parts[5]) - box_lo[1]
            z = float(parts[6]) - box_lo[2]
        except ValueError as exc:
            raise ValueError(
                f"Failed parsing Atoms row at line {i+1}: {raw.rstrip()!r}"
            ) from exc
        atom_data[aid] = np.array([x, y, z])
        if len(parts) >= 10:
            try:
                atom_images[aid] = np.array(
                    [int(parts[7]), int(parts[8]), int(parts[9])],
                    dtype=int,
                )
            except ValueError as exc:
                raise ValueError(
                    f"Failed parsing optional image flags in Atoms row at line {i+1}: {raw.rstrip()!r}"
                ) from exc
        i += 1

    bond_list: list[tuple[int, int, int, int]] = []
    neighbors: defaultdict[int, list[int]] = defaultdict(list)
    i = bonds_line + 2
    while i < len(lines):
        raw = lines[i]
        line = raw.strip()
        if not line or not line[0].isdigit():
            break
        parts = line.split()
        if len(parts) < 4:
            raise ValueError(
                f"Bonds row has fewer than 4 columns at line {i+1}: {raw.rstrip()!r}"
            )
        try:
            bond_id = int(parts[0])
            bond_type = int(parts[1])
            a1 = int(parts[2])
            a2 = int(parts[3])
        except ValueError as exc:
            raise ValueError(
                f"Failed parsing Bonds row at line {i+1}: {raw.rstrip()!r}"
            ) from exc
        bond_list.append((bond_id, bond_type, a1, a2))
        neighbors[a1].append(a2)
        neighbors[a2].append(a1)
        i += 1

    if atom_images:
        logger.info(
            "Parsed image flags for %d atoms; percolation edge translations are still inferred from local wrapped bond geometry.",
            len(atom_images),
        )

    bond_translations = _build_bond_translations(
        atom_data=atom_data,
        bond_list=bond_list,
        box=box,
    )

    logger.info("Atoms: %d, Bonds: %d", len(atom_data), len(bond_list))
    return LammpsData(
        atom_data=atom_data,
        bond_list=bond_list,
        neighbors=dict(neighbors),
        box=box,
        bond_translations=bond_translations,
    )


def analyze_percolation(
    atom_data: dict[int, np.ndarray],
    bond_list: list[tuple[int, int, int, int]],
    neighbors: dict[int, list[int]],
    box: np.ndarray,
    bond_translations: dict[tuple[int, int], np.ndarray] | None = None,
) -> dict[int, dict[str, Any]]:
    """BFS through the bond graph using explicit edge translations to detect wrapping."""
    if bond_translations is None:
        bond_translations = _build_bond_translations(atom_data, bond_list, box)

    all_atoms = sorted(atom_data.keys())
    visited: dict[int, int] = {}
    components: dict[int, dict[str, Any]] = {}
    component_id = 0

    for start in all_atoms:
        if start in visited:
            continue

        # image_offset stores which periodic copy of each atom this BFS path reaches.
        image_offset = {start: np.array([0, 0, 0], dtype=int)}
        queue = deque([start])
        comp_atoms = [start]
        wrapping = {0: [], 1: [], 2: []}
        # basis_vectors holds only the independent loop translations for this component.
        basis_vectors: list[np.ndarray] = []

        while queue:
            a1 = queue.popleft()
            p1 = atom_data[a1]
            off1 = image_offset[a1]

            for a2 in neighbors.get(a1, []):
                p2 = atom_data[a2]
                crossing = bond_translations.get((a1, a2))
                if crossing is None:
                    crossing = _infer_bond_translation(p1, p2, box)
                # Follow the edge translation to the periodic copy of a2 reached from a1.
                off2_expected = off1 + crossing

                if a2 not in image_offset:
                    image_offset[a2] = off2_expected
                    queue.append(a2)
                    comp_atoms.append(a2)
                else:
                    # Reaching the same atom with a different offset closes a periodic loop.
                    # delta is the translation between the two copies of that atom.
                    delta = off2_expected - image_offset[a2]
                    for dim in range(3):
                        if delta[dim] != 0:
                            wrapping[dim].append((a1, a2, int(delta[dim])))
                    if _check_translation_independent(basis_vectors, delta):
                        basis_vectors.append(delta.copy())

        for aid in comp_atoms:
            visited[aid] = component_id

        offsets = np.array(list(image_offset.values()))
        span = offsets.max(axis=0) - offsets.min(axis=0)

        components[component_id] = {
            "atoms": sorted(comp_atoms),
            "n_atoms": len(comp_atoms),
            "wrapping": wrapping,
            "offset_span": span,
            "basis_vectors": [vec.copy() for vec in basis_vectors],
            "percolation_dim": len(basis_vectors),
            "percolates": _basis_axes(basis_vectors),
        }
        component_id += 1

    return components


def _component_rank_map(components: dict[int, dict[str, Any]]) -> dict[int, int]:
    """Return mapping from component id -> 1-based rank by descending size."""
    sorted_components = sorted(
        components.items(),
        key=lambda x: (-x[1]["n_atoms"], x[0]),
    )
    return {cid: rank for rank, (cid, _comp) in enumerate(sorted_components, start=1)}


def _find_atoms_bounds(lines: list[str]) -> tuple[int, int]:
    """Return ``(atoms_start_line_idx, atoms_end_line_idx_exclusive)``."""
    atoms_start = None
    for i, line in enumerate(lines):
        s = line.strip()
        if s.startswith("Atoms"):
            atoms_start = i
            break
    if atoms_start is None:
        raise RuntimeError("Failed to find Atoms section.")

    i = atoms_start + 2
    while i < len(lines):
        s = lines[i].strip()
        if not s or not s[0].isdigit():
            break
        i += 1
    return atoms_start, i


def write_component_type_data_file(
    data_file: str,
    output_file: str,
    components: dict[int, dict[str, Any]],
) -> None:
    """Write data file where atom type equals component rank (largest=1, next=2, ...)."""
    if output_file == data_file:
        raise ValueError("Refusing to overwrite input data file; choose a new output path.")

    rank_map = _component_rank_map(components)
    atom_to_component: dict[int, int] = {}
    for cid, comp in components.items():
        for atom_id in comp["atoms"]:
            atom_to_component[int(atom_id)] = int(cid)

    with open(data_file) as f:
        lines = f.readlines()

    current_n_types = None
    atom_types_re = re.compile(r"^\s*(\d+)\s+atom\s+types\b", re.IGNORECASE)
    atom_types_line_idx = None
    for i, line in enumerate(lines):
        m = atom_types_re.match(line)
        if m:
            current_n_types = int(m.group(1))
            atom_types_line_idx = i
            break

    n_components = len(components)
    target_n_types = max(current_n_types or 0, n_components)
    if (
        current_n_types is not None
        and atom_types_line_idx is not None
        and target_n_types != current_n_types
    ):
        lines[atom_types_line_idx] = re.sub(
            r"^\s*\d+",
            str(target_n_types),
            lines[atom_types_line_idx],
            count=1,
        )

    atoms_start, atoms_end = _find_atoms_bounds(lines)
    for i in range(atoms_start + 2, atoms_end):
        parts = lines[i].split()
        if len(parts) < 7:
            continue
        atom_id = int(parts[0])
        cid = atom_to_component.get(atom_id)
        if cid is None:
            continue
        parts[2] = str(rank_map[cid])
        lines[i] = " ".join(parts) + "\n"

    with open(output_file, "w") as f:
        f.writelines(lines)

    logger.info("Wrote component-type data file: %s", output_file)
    logger.info(
        "  Type assignment: largest component -> 1 ... smallest -> %d",
        n_components,
    )


def compute_report(
    components: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    """Pure report computation with no I/O side effects."""
    n_components = len(components)
    sorted_components = sorted(components.items(), key=lambda x: -x[1]["n_atoms"])
    sizes = [comp["n_atoms"] for _cid, comp in sorted_components]
    total_atoms = sum(sizes)
    largest_size = sizes[0] if sizes else 0
    disconnected_component_count = max(0, n_components - 1)
    disconnected_atom_count = total_atoms - largest_size
    singleton_count = sum(1 for size in sizes if size == 1)
    multi_atom_component_count = sum(1 for size in sizes if size > 1)

    component_summaries = []
    n_percolating = 0
    system_percolates = np.array([False, False, False])
    total_wrapping = {0: 0, 1: 0, 2: 0}
    for cid, comp in sorted_components:
        percolates = comp["percolates"]
        percolation_dim = comp.get("percolation_dim", int(any(percolates)))
        wrap_counts = [len(comp["wrapping"][d]) for d in range(3)]
        span = comp["offset_span"]
        if percolation_dim > 0:
            n_percolating += 1
        system_percolates |= percolates
        for dim in range(3):
            total_wrapping[dim] += wrap_counts[dim]
        component_summaries.append(
            {
                "id": cid,
                "n_atoms": comp["n_atoms"],

                "percolation_dim": percolation_dim,
                "percolates": percolates,
                "wrapping_counts": wrap_counts,
                "offset_span": span,
            }
        )

    if sorted_components:
        largest_cid, largest_comp = sorted_components[0]
        largest_fraction = largest_comp["n_atoms"] / total_atoms
        largest_percolates = largest_comp["percolates"]
        largest_percolation_dim = largest_comp.get(
            "percolation_dim",
            int(any(largest_percolates)),
        )
        largest_component_spans_xyz = bool(all(largest_percolates))
    else:
        largest_cid = None
        largest_fraction = 0.0
        largest_percolates = np.array([False, False, False])
        largest_percolation_dim = 0
        largest_component_spans_xyz = False

    return {
        "n_components": n_components,
        "n_percolating_components": n_percolating,
        "disconnected_component_count": disconnected_component_count,
        "singleton_count": singleton_count,
        "multi_atom_component_count": multi_atom_component_count,
        "sizes": sizes,
        "total_atoms": total_atoms,
        "largest_size": largest_size,
        "disconnected_atom_count": disconnected_atom_count,
        "component_summaries": component_summaries,
        "system_percolates": system_percolates,
        "total_wrapping": total_wrapping,
        "largest_component_id": largest_cid,
        "largest_component_fraction": largest_fraction,
        "largest_component_percolation_dim": largest_percolation_dim,
        "largest_component_percolates": largest_percolates,
        "largest_component_spans_xyz": largest_component_spans_xyz,

    }

def format_report(report: dict[str, Any]) -> list[str]:
    """Format a precomputed report dictionary for human-readable output."""
    lines: list[str] = []
    lines.append("=" * 60)
    lines.append("PERCOLATION ANALYSIS REPORT")
    lines.append("=" * 60)
    lines.append("")
    lines.append("Bond graph definition:")
    lines.append("  Nodes: all atoms in the Atoms section")
    lines.append("  Edges: all bonds in the Bonds section")
    lines.append("")
    lines.append(f"Connected components: {report['n_components']}")
    lines.append(
        f"Disconnected components (excluding largest): "
        f"{report['disconnected_component_count']}"
    )
    lines.append(f"Percolating components: {report['n_percolating_components']}")
    lines.append(f"Single-atom components: {report['singleton_count']}")
    lines.append(f"Multi-atom components: {report['multi_atom_component_count']}")
    lines.append("")
    lines.append("Component size distribution:")
    lines.append(f"  Largest:  {report['largest_size']} atoms")
    sizes = report["sizes"]
    if len(sizes) > 1:
        lines.append(f"  2nd:      {sizes[1]} atoms")
    if len(sizes) > 2:
        lines.append(f"  3rd:      {sizes[2]} atoms")
    if len(sizes) > 3:
        lines.append(f"  Smallest: {sizes[-1]} atoms")
    total_atoms = report["total_atoms"]
    largest_size = report["largest_size"]
    if total_atoms > 0:
        lines.append(f"  Fraction in largest: {largest_size / total_atoms * 100:.1f}%")
    else:
        lines.append("  Fraction in largest: N/A")
    lines.append(f"  Atoms outside largest: {report['disconnected_atom_count']}")
    lines.append("")
    lines.append("Per-component percolation:")
    for comp in report["component_summaries"]:
        perc = comp["percolates"]
        dims = _format_dim_flags(perc)
        wrap_counts = comp["wrapping_counts"]
        span = comp["offset_span"]
        lines.append(
            f"  Component {comp['id']:3d}: {comp['n_atoms']:6d} atoms | "
            f"dim={comp['percolation_dim']} | "
            f"percolates [{dims}] | "
            f"wrapping bonds X:{wrap_counts[0]} Y:{wrap_counts[1]} Z:{wrap_counts[2]} | "
            f"span X:{span[0]} Y:{span[1]} Z:{span[2]}"
        )
    lines.append("")
    lines.append("=" * 60)
    lines.append("SYSTEM-LEVEL PERCOLATION SUMMARY")
    lines.append("=" * 60)
    system_percolates = report["system_percolates"]
    total_wrapping = report["total_wrapping"]
    for dim, label in enumerate(["X", "Y", "Z"]):
        status = "YES" if system_percolates[dim] else "NO"
        lines.append(
            f"  Percolates in {label}: {status}  ({total_wrapping[dim]} wrapping bonds)"
        )
    lines.append("")
    lines.append(
        f"  Fully percolated (X+Y+Z): {'YES' if all(system_percolates) else 'NO'}"
    )
    lines.append("=" * 60)
    lines.append("")
    lines.append("LARGEST COMPONENT SUMMARY")
    lines.append("=" * 60)
    largest_cid = report["largest_component_id"]
    if largest_cid is None:
        lines.append("  Largest component id: N/A")
    else:
        lines.append(
            f"  Largest component id: {largest_cid} "
            f"({report['largest_size']} atoms, {report['largest_component_fraction'] * 100:.1f}% of system)"
        )
    lines.append(
        f"  Largest component percolation: "
        f"dim={report['largest_component_percolation_dim']} "
        f"[{_format_dim_flags(report['largest_component_percolates'])}]"
    )
    lines.append(
        "  Largest component spans X+Y+Z: "
        f"{'YES' if report['largest_component_spans_xyz'] else 'NO'}"
    )
    lines.append(f"  Largest component fraction: {report['largest_component_fraction']:.4f}")
    lines.append("=" * 60)
    lines.append("")
    lines.append(
        "Summary: X=%s Y=%s Z=%s"
        % (
            "YES" if system_percolates[0] else "NO",
            "YES" if system_percolates[1] else "NO",
            "YES" if system_percolates[2] else "NO",
        )
    )
    return lines

def _write_report_output(report_out: str, lines: list[str]) -> None:
    """Write the human-readable report to disk."""
    output_path = Path(report_out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n")
    logger.info("Wrote report file: %s", output_path)

def print_report(
    components: dict[int, dict[str, Any]],
    report_out: str = '',
) -> dict[str, Any]:
    """Compute the percolation report, emit it, and optionally persist it."""
    report = compute_report(components=components)
    report_lines = format_report(report)
    for line in report_lines:
        logger.info(line)

    if report_out:
        _write_report_output(report_out, report_lines)
    return report

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze percolation from a LAMMPS data file bond network and print "
            "a connected-component/percolation summary."
        )
    )
    parser.add_argument("data_file", help="LAMMPS data file")
    parser.add_argument(
        "--component-type-data-out",
        default="",
        help=(
            "Optional LAMMPS data output path where atom type is replaced by component "
            "rank (largest component -> type 1, second largest -> type 2, ...)."
        ),
    )
    parser.add_argument(
        "--report-out",
        default="",
        help=(
            "Optional path to write the human-readable percolation report that is "
            "printed to the terminal."
        ),
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    data = read_lammps_data(args.data_file)
    components = analyze_percolation(
        data.atom_data,
        data.bond_list,
        data.neighbors,
        data.box,
        bond_translations=data.bond_translations,
    )
    print_report(
        components=components,
        report_out=args.report_out,
    )

    if args.component_type_data_out:
        write_component_type_data_file(
            data_file=args.data_file,
            output_file=args.component_type_data_out,
            components=components,
        )


if __name__ == "__main__":
    main()
