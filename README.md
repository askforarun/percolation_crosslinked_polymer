# Percolation Analysis for LAMMPS Data

This repository contains `percolation.py`, a command-line tool to analyze bond-network percolation in a LAMMPS `data` file and evaluate a gel-like pass/fail criterion.

## What the script does

1. Parses a LAMMPS `data` file (`Atoms`, `Bonds`, and box bounds).
2. Builds connected components from covalent bonds.
3. Detects periodic wrapping/percolation (X, Y, Z) with a BFS image-offset method.
4. Computes system and largest-component metrics.
5. Evaluates a gel-like percolation criterion.
6. Prints a report and optionally writes a component-colored data file.

## Input assumptions

- LAMMPS `atom_style full` layout is expected (`Atoms # full` or equivalent full-style columns).
- Coordinates are read from columns 5/6/7 (`x y z`).
- Molecule ID is read from column 2.

## Molecule ID usage (important)

- Raw wrapping/percolation detection is based on geometry + bond graph and does **not** require molecule IDs.
- Molecule IDs are used for gel-like qualification metrics:
  - unique molecule count in the largest component
  - intermolecular bond count in the largest component (`mol_id(atom1) != mol_id(atom2)`)
- With defaults, gel pass requires at least one intermolecular bond in the largest component.

Practical recommendation:
- Keep distinct molecule IDs per physical molecule (e.g., each PVA chain and each GLU molecule) so intermolecular counts are meaningful.
- If molecule IDs are collapsed/rewritten, the intermolecular-bond criterion can become misleading.
- Caveat: if your crosslinking workflow alters molecule IDs during topology updates, gel-related metrics (especially intermolecular-bond-based checks) may report incorrect pass/fail outcomes.

## Gel-like pass criterion

`gel_like_percolation_pass` is true only if all conditions are true:

- Largest covalent component spans X+Y+Z
- Largest-component fraction `>= gel_fraction_threshold` (default `0.90`)
- Minimum unique-molecule requirement is met (default `1`, effectively disabled)
- Minimum intermolecular-bond requirement is met (default `1`)
- Optional selected crosslink-bond count requirement is met (default disabled)

## CLI usage

Basic:

```bash
python percolation.py data.crosslinked_updated
```

With explicit thresholds:

```bash
python percolation.py data.crosslinked_updated \
  --gel-fraction-threshold 0.90 \
  --min-intermolecular-bonds-in-largest 1
```

With optional crosslink filters:

```bash
python percolation.py data.crosslinked_updated \
  --crosslink-bond-types 12 13 \
  --min-crosslink-bonds-in-largest 2
```

Optional component-colored output:

```bash
python percolation.py data.crosslinked_updated \
  --component-type-data-out data.components
```

## Output behavior

- Prints a detailed report (component stats + system summary + gel criterion).
- Exit code is `0` on gel pass and `1` on gel fail.
