#!/usr/bin/env python3
"""
Percolation analysis for LAMMPS data files.

This module:
1) Parses a LAMMPS data file into atoms, bonds, neighbor graph, and box vectors.
2) Finds connected bond components and detects periodic wrapping via BFS image offsets.
3) Computes a gel-like percolation criterion on the largest covalent component.
4) Optionally renders a human-readable report and a component-colored data file.
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_GEL_FRACTION_THRESHOLD = 0.90


@dataclass(frozen=True)
class LammpsData:
    """Parsed LAMMPS data-file payload used by percolation analysis."""

    atom_data: dict[int, np.ndarray]
    bond_list: list[tuple[int, int, int, int]]
    neighbors: dict[int, list[int]]
    box: np.ndarray
    atom_to_molecule: dict[int, int]


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


def read_lammps_data(filename: str, include_molecules: bool | None = None) -> LammpsData:
    """Read a LAMMPS data file and return a stable, named payload.

    The return signature is always stable (a :class:`LammpsData` dataclass), regardless
    of caller preferences. ``include_molecules`` is retained only for backward
    compatibility and is ignored.
    """
    if include_molecules is not None:
        logger.debug(
            "read_lammps_data(include_molecules=...) is deprecated; "
            "return shape is always stable."
        )

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
    atom_to_molecule: dict[int, int] = {}
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
            mol_id = int(parts[1])
            x = float(parts[4]) - box_lo[0]
            y = float(parts[5]) - box_lo[1]
            z = float(parts[6]) - box_lo[2]
        except ValueError as exc:
            raise ValueError(
                f"Failed parsing Atoms row at line {i+1}: {raw.rstrip()!r}"
            ) from exc
        atom_data[aid] = np.array([x, y, z])
        atom_to_molecule[aid] = mol_id
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

    logger.info("Atoms: %d, Bonds: %d", len(atom_data), len(bond_list))
    return LammpsData(
        atom_data=atom_data,
        bond_list=bond_list,
        neighbors=dict(neighbors),
        box=box,
        atom_to_molecule=atom_to_molecule,
    )


def analyze_percolation(
    atom_data: dict[int, np.ndarray],
    bond_list: list[tuple[int, int, int, int]],
    neighbors: dict[int, list[int]],
    box: np.ndarray,
    atom_to_molecule: dict[int, int] | None = None,
) -> dict[int, dict[str, Any]]:
    """BFS through the bond graph and track image offsets to detect wrapping."""
    all_atoms = sorted(atom_data.keys())
    visited: dict[int, int] = {}
    components: dict[int, dict[str, Any]] = {}
    component_id = 0

    for start in all_atoms:
        if start in visited:
            continue

        image_offset = {start: np.array([0, 0, 0], dtype=int)}
        queue = deque([start])
        comp_atoms = [start]
        wrapping = {0: [], 1: [], 2: []}

        while queue:
            a1 = queue.popleft()
            p1 = atom_data[a1]
            off1 = image_offset[a1]

            for a2 in neighbors.get(a1, []):
                p2 = atom_data[a2]
                diff = p2 - p1
                crossing = -np.round(diff / box).astype(int)
                off2_expected = off1 + crossing

                if a2 not in image_offset:
                    image_offset[a2] = off2_expected
                    queue.append(a2)
                    comp_atoms.append(a2)
                else:
                    delta = off2_expected - image_offset[a2]
                    for dim in range(3):
                        if delta[dim] != 0:
                            wrapping[dim].append((a1, a2, int(delta[dim])))

        for aid in comp_atoms:
            visited[aid] = component_id

        offsets = np.array(list(image_offset.values()))
        span = offsets.max(axis=0) - offsets.min(axis=0)

        n_unique_molecules = None
        n_intermolecular_bonds = None
        if atom_to_molecule is not None:
            comp_atom_set = set(comp_atoms)
            n_unique_molecules = len(
                {atom_to_molecule[aid] for aid in comp_atoms if aid in atom_to_molecule}
            )
            n_intermolecular_bonds = sum(
                1
                for _bid, _btype, a1, a2 in bond_list
                if a1 in comp_atom_set
                and a2 in comp_atom_set
                and atom_to_molecule.get(a1) is not None
                and atom_to_molecule.get(a2) is not None
                and atom_to_molecule[a1] != atom_to_molecule[a2]
            )

        components[component_id] = {
            "atoms": sorted(comp_atoms),
            "n_atoms": len(comp_atoms),
            "n_unique_molecules": n_unique_molecules,
            "n_intermolecular_bonds": n_intermolecular_bonds,
            "wrapping": wrapping,
            "offset_span": span,
            "percolates": np.array([len(wrapping[d]) > 0 for d in range(3)]),
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
    bond_list: list[tuple[int, int, int, int]],
    gel_fraction_threshold: float = DEFAULT_GEL_FRACTION_THRESHOLD,
    min_unique_molecules_in_largest: int = 1,
    min_intermolecular_bonds_in_largest: int = 1,
    crosslink_bond_ids: tuple[int, ...] = (),
    crosslink_bond_types: tuple[int, ...] = (),
    min_crosslink_bonds_in_largest: int = 0,
) -> dict[str, Any]:
    """Pure report computation with no I/O side effects."""
    if not (0.0 <= gel_fraction_threshold <= 1.0):
        raise ValueError(
            f"gel_fraction_threshold must be within [0, 1], got {gel_fraction_threshold}"
        )
    if min_unique_molecules_in_largest < 1:
        raise ValueError(
            "min_unique_molecules_in_largest must be >= 1, "
            f"got {min_unique_molecules_in_largest}"
        )
    if min_intermolecular_bonds_in_largest < 0:
        raise ValueError(
            "min_intermolecular_bonds_in_largest must be >= 0, "
            f"got {min_intermolecular_bonds_in_largest}"
        )
    if min_crosslink_bonds_in_largest < 0:
        raise ValueError(
            "min_crosslink_bonds_in_largest must be >= 0, "
            f"got {min_crosslink_bonds_in_largest}"
        )
    if (
        min_crosslink_bonds_in_largest > 0
        and not crosslink_bond_types
        and not crosslink_bond_ids
    ):
        raise ValueError(
            "crosslink_bond_ids or crosslink_bond_types must be provided when "
            "min_crosslink_bonds_in_largest > 0"
        )

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
        wrap_counts = [len(comp["wrapping"][d]) for d in range(3)]
        span = comp["offset_span"]
        if any(percolates):
            n_percolating += 1
        system_percolates |= percolates
        for dim in range(3):
            total_wrapping[dim] += wrap_counts[dim]
        component_summaries.append(
            {
                "id": cid,
                "n_atoms": comp["n_atoms"],
                "percolates": percolates,
                "wrapping_counts": wrap_counts,
                "offset_span": span,
            }
        )

    if sorted_components:
        largest_cid, largest_comp = sorted_components[0]
        largest_fraction = largest_comp["n_atoms"] / total_atoms
        largest_percolates = largest_comp["percolates"]
        largest_component_spans_xyz = bool(all(largest_percolates))
        largest_n_unique_molecules = largest_comp.get("n_unique_molecules")
        largest_n_intermolecular_bonds = largest_comp.get("n_intermolecular_bonds")
        largest_atoms_set = set(largest_comp["atoms"])
    else:
        largest_cid = None
        largest_fraction = 0.0
        largest_percolates = np.array([False, False, False])
        largest_component_spans_xyz = False
        largest_n_unique_molecules = None
        largest_n_intermolecular_bonds = None
        largest_atoms_set = set()

    crosslink_bond_id_set = set(int(bid) for bid in crosslink_bond_ids)
    crosslink_bond_type_set = set(int(t) for t in crosslink_bond_types)
    largest_crosslink_bond_count = 0
    if crosslink_bond_id_set or crosslink_bond_type_set:
        largest_crosslink_bond_count = sum(
            1
            for bond_id, bond_type, a1, a2 in bond_list
            if (
                int(bond_id) in crosslink_bond_id_set
                or int(bond_type) in crosslink_bond_type_set
            )
            and a1 in largest_atoms_set
            and a2 in largest_atoms_set
        )

    has_required_unique_molecules = (
        min_unique_molecules_in_largest <= 1
        or (
            largest_n_unique_molecules is not None
            and largest_n_unique_molecules >= min_unique_molecules_in_largest
        )
    )
    has_required_intermolecular_bonds = (
        min_intermolecular_bonds_in_largest <= 0
        or (
            largest_n_intermolecular_bonds is not None
            and largest_n_intermolecular_bonds >= min_intermolecular_bonds_in_largest
        )
    )
    has_required_crosslink_bonds = (
        largest_crosslink_bond_count >= min_crosslink_bonds_in_largest
    )
    gel_like_percolation_pass = bool(
        largest_component_spans_xyz
        and largest_fraction >= gel_fraction_threshold
        and has_required_unique_molecules
        and has_required_intermolecular_bonds
        and has_required_crosslink_bonds
    )

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
        "gel_fraction": largest_fraction,
        "largest_component_fraction": largest_fraction,
        "largest_component_percolates": largest_percolates,
        "largest_component_spans_xyz": largest_component_spans_xyz,
        "largest_component_unique_molecules": largest_n_unique_molecules,
        "min_unique_molecules_in_largest": min_unique_molecules_in_largest,
        "largest_component_intermolecular_bonds": largest_n_intermolecular_bonds,
        "min_intermolecular_bonds_in_largest": min_intermolecular_bonds_in_largest,
        "crosslink_bond_ids": tuple(sorted(crosslink_bond_id_set)),
        "crosslink_bond_types": tuple(sorted(crosslink_bond_type_set)),
        "largest_component_crosslink_bond_count": largest_crosslink_bond_count,
        "min_crosslink_bonds_in_largest": min_crosslink_bonds_in_largest,
        "gel_fraction_threshold": gel_fraction_threshold,
        "gel_like_percolation_pass": gel_like_percolation_pass,
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
            f"({report['largest_size']} atoms, {report['gel_fraction'] * 100:.1f}% of system)"
        )
    lines.append(
        f"  Largest component percolation: "
        f"[{_format_dim_flags(report['largest_component_percolates'])}]"
    )
    lines.append(
        "  Largest component spans X+Y+Z: "
        f"{'YES' if report['largest_component_spans_xyz'] else 'NO'}"
    )
    lines.append(f"  Largest component fraction: {report['largest_component_fraction']:.4f}")
    lines.append(f"  Gel fraction: {report['gel_fraction']:.4f}")
    unique_mols = report["largest_component_unique_molecules"]
    lines.append(
        "  Largest component unique molecules: "
        f"{unique_mols if unique_mols is not None else 'N/A'}"
    )
    lines.append(
        "  Required unique molecules in largest: "
        f"{report['min_unique_molecules_in_largest']}"
    )
    intermol = report["largest_component_intermolecular_bonds"]
    lines.append(
        "  Largest component intermolecular bonds: "
        f"{intermol if intermol is not None else 'N/A'}"
    )
    lines.append(
        "  Required intermolecular bonds in largest: "
        f"{report['min_intermolecular_bonds_in_largest']}"
    )
    lines.append(f"  Crosslink bond IDs considered: {list(report['crosslink_bond_ids']) or 'N/A'}")
    lines.append(
        f"  Crosslink bond types considered: {list(report['crosslink_bond_types']) or 'N/A'}"
    )
    lines.append(
        f"  Largest component crosslink bonds: {report['largest_component_crosslink_bond_count']}"
    )
    lines.append(
        f"  Required crosslink bonds in largest: {report['min_crosslink_bonds_in_largest']}"
    )
    lines.append(f"  Required fraction threshold: {report['gel_fraction_threshold']:.4f}")
    lines.append("=" * 60)
    lines.append("")
    lines.append("GEL-LIKE PERCOLATION CRITERION")
    lines.append("=" * 60)
    lines.append(
        "  Definition: largest covalent component spans X+Y+Z and exceeds fraction "
        "threshold, optional unique-molecule threshold, optional intermolecular-bond "
        "threshold, and optional crosslink-bond threshold"
    )
    lines.append(
        f"  Gel-like percolation: "
        f"{'YES' if report['gel_like_percolation_pass'] else 'NO'}"
    )
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


def print_report(
    components: dict[int, dict[str, Any]],
    bond_list: list[tuple[int, int, int, int]],
    gel_fraction_threshold: float = DEFAULT_GEL_FRACTION_THRESHOLD,
    min_unique_molecules_in_largest: int = 1,
    min_intermolecular_bonds_in_largest: int = 1,
    crosslink_bond_ids: tuple[int, ...] = (),
    crosslink_bond_types: tuple[int, ...] = (),
    min_crosslink_bonds_in_largest: int = 0,
) -> dict[str, Any]:
    """Compatibility wrapper: compute report, then emit formatted lines."""
    report = compute_report(
        components=components,
        bond_list=bond_list,
        gel_fraction_threshold=gel_fraction_threshold,
        min_unique_molecules_in_largest=min_unique_molecules_in_largest,
        min_intermolecular_bonds_in_largest=min_intermolecular_bonds_in_largest,
        crosslink_bond_ids=crosslink_bond_ids,
        crosslink_bond_types=crosslink_bond_types,
        min_crosslink_bonds_in_largest=min_crosslink_bonds_in_largest,
    )
    for line in format_report(report):
        logger.info(line)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze percolation from a LAMMPS data file bond network. "
            "CLI pass/fail uses gel-like criterion based on the largest component."
        )
    )
    parser.add_argument("data_file", help="LAMMPS data file")
    parser.add_argument(
        "--gel-fraction-threshold",
        type=float,
        default=DEFAULT_GEL_FRACTION_THRESHOLD,
        help=(
            "Largest-component fraction threshold for gel-like percolation pass "
            f"(default: {DEFAULT_GEL_FRACTION_THRESHOLD})."
        ),
    )
    parser.add_argument(
        "--min-unique-molecules-in-largest",
        type=int,
        default=1,
        help=(
            "Minimum distinct molecule IDs required in the largest component "
            "for gel-like percolation pass (default: 1, i.e. disabled)."
        ),
    )
    parser.add_argument(
        "--min-intermolecular-bonds-in-largest",
        type=int,
        default=1,
        help=(
            "Minimum number of intermolecular covalent bonds required in the largest "
            "component for gel-like percolation pass (default: 1)."
        ),
    )
    parser.add_argument(
        "--crosslink-bond-ids",
        nargs="*",
        type=int,
        default=[],
        help=(
            "Optional explicit bond IDs considered as crosslinks when applying "
            "--min-crosslink-bonds-in-largest."
        ),
    )
    parser.add_argument(
        "--crosslink-bond-types",
        nargs="*",
        type=int,
        default=[],
        help=(
            "Optional bond type IDs considered as crosslinks when applying "
            "--min-crosslink-bonds-in-largest."
        ),
    )
    parser.add_argument(
        "--min-crosslink-bonds-in-largest",
        type=int,
        default=0,
        help=(
            "Optional minimum number of selected crosslink bonds required in the "
            "largest component for gel-like percolation pass (default: 0, disabled)."
        ),
    )
    parser.add_argument(
        "--component-type-data-out",
        default="",
        help=(
            "Optional LAMMPS data output path where atom type is replaced by component "
            "rank (largest component -> type 1, second largest -> type 2, ...)."
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
        atom_to_molecule=data.atom_to_molecule,
    )
    report = print_report(
        components=components,
        bond_list=data.bond_list,
        gel_fraction_threshold=args.gel_fraction_threshold,
        min_unique_molecules_in_largest=args.min_unique_molecules_in_largest,
        min_intermolecular_bonds_in_largest=args.min_intermolecular_bonds_in_largest,
        crosslink_bond_ids=tuple(args.crosslink_bond_ids),
        crosslink_bond_types=tuple(args.crosslink_bond_types),
        min_crosslink_bonds_in_largest=args.min_crosslink_bonds_in_largest,
    )

    if args.component_type_data_out:
        write_component_type_data_file(
            data_file=args.data_file,
            output_file=args.component_type_data_out,
            components=components,
        )

    if not report["gel_like_percolation_pass"]:
        logger.error("ERROR: Gel-like percolation criterion failed.")
        logger.error(
            "Largest component spans: [%s]",
            _format_dim_flags(report["largest_component_percolates"]),
        )
        logger.error(
            "Gel fraction: %.4f (threshold %.4f)",
            report["gel_fraction"],
            report["gel_fraction_threshold"],
        )
        logger.error(
            "Largest component intermolecular bonds: %s (required >= %d)",
            report["largest_component_intermolecular_bonds"],
            report["min_intermolecular_bonds_in_largest"],
        )
        logger.error(
            "Largest component crosslink bonds: %d (required >= %d, ids=%s, types=%s)",
            report["largest_component_crosslink_bond_count"],
            report["min_crosslink_bonds_in_largest"],
            list(report["crosslink_bond_ids"]),
            list(report["crosslink_bond_types"]),
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
