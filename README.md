# Overview

Python utility for analyzing bond-network percolation in LAMMPS `data` files through a CLI workflow built around explicit periodic bond translations and basis-vector percolation detection.

The core percolation logic follows the same main idea as Mattia Livraghi, Kevin Höllring, Christian R. Wick, David M. Smith, and Ana-Sunčana Smith, *An Exact Algorithm to Detect the Percolation Transition in Molecular Dynamics Simulations of Cross-Linking Polymer Networks*, *Journal of Chemical Theory and Computation* (2021), DOI: [10.1021/acs.jctc.1c00423](https://doi.org/10.1021/acs.jctc.1c00423), and the reference implementation in [`puls-group/percolation-analyzer`](https://github.com/puls-group/percolation-analyzer).

## Key Features

- **Periodic Graph Analysis**: Detect percolation from the covalent bond graph under periodic boundary conditions
- **Explicit Bond Translations**: Build directed per-bond periodic translations from local wrapped bond crossings
- **Percolation Dimension**: Report `percolation_dim = 0/1/2/3` from independent loop translations, not just raw wrapped bonds
- **Component-Level Reporting**: Summarize connected components, component sizes, axis flags, and system-level spanning
- **CLI + Report Output**: Print a readable report and optionally write a component-colored LAMMPS data file
- **Regression Tests**: Includes parser tests, periodic-loop tests, and a smoke regression against a real `data.lammps` file when available

## What This Tool Does

This tool reads a LAMMPS `data` file, constructs the covalent bond graph, tracks periodic image offsets while traversing that graph, and determines whether any connected component percolates through the periodic cell.

**Typical workflow:**
1. Start with a LAMMPS `data` file containing `Atoms` and `Bonds` sections.
2. Parse atom coordinates, box dimensions, bonds, and optional image flags.
3. Build local bond translations for each directed bond from the wrapped bond geometry.
4. Run BFS on each connected component and collect independent periodic loop directions.
5. Compute `percolation_dim`, axis flags, and component statistics.
6. Review the report or write a component-ranked output file for inspection.

**Input -> Output Mapping**
- **LAMMPS `data` file**: atoms, bonds, box bounds, optional image flags -> internal periodic bond graph
- **Bond graph + translations**: connected components and periodic loop structure -> `percolation_dim`, axis flags, wrapping information
- **Outputs**: terminal report and optional component-colored `data` file

## Table of Contents
- [Installation](#installation)
- [Input Requirements](#input-requirements)
- [How It Works](#how-it-works)
  - [Periodic Percolation Detection](#periodic-percolation-detection)
  - [Bond Translations](#bond-translations)
- [CLI Usage](#cli-usage)
- [Output Behavior](#output-behavior)
- [Tests](#tests)
  - [Run the Tests](#run-the-tests)
  - [Real Regression Test](#real-regression-test)
- [Limitations](#limitations)
- [References](#references)
- [Algorithm Walkthrough](./ALGORITHM_WALKTHROUGH.md)

## Installation

This repository is a standalone Python script, not an installable package.

**Required dependency:**
- `numpy`

**Optional dependency:**
- `pytest` for running the test suite

### Quick setup with `venv`

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install numpy pytest
```

### Quick setup with `conda`

```bash
conda create -n percolation python=3.11 numpy pytest
conda activate percolation
```

After installation, run the script from the repository root:

```bash
python percolation.py data.crosslinked_updated
```

## Input Requirements

The script currently expects:
- a LAMMPS `data` file
- an `Atoms` section in `atom_style full` layout
- a `Bonds` section
- orthorhombic box bounds (`xlo xhi`, `ylo yhi`, `zlo zhi`)

The parser uses:
- coordinates from columns 5/6/7 (`x y z`)

Optional image flags are also supported.
If present, the parser reads `nx ny nz` from columns 8/9/10. They are accepted but not required for the percolation analysis.

## How It Works

The percolation logic implemented here follows the same core idea as the percolation-detection method described by Livraghi et al. and the reference implementation in `puls-group/percolation-analyzer`; see the References section below.

### Periodic Percolation Detection

The script does **not** define percolation as "I saw one wrapped bond".
That is too weak.

Instead, it:
1. walks through the bond graph
2. tracks which periodic image of each atom is reached
3. checks whether the same atom can be reached again through a different image
4. stores only the **independent** loop directions

This gives a `percolation_dim` value:
- `0` = no percolation
- `1` = one independent periodic direction
- `2` = two independent periodic directions
- `3` = three independent periodic directions

The script also reports `percolates` as axis flags (`X`, `Y`, `Z`) for easier reading.

### Bond Translations

The current code builds **explicit per-bond translation vectors** from the local wrapped bond geometry.

That means each directed bond stores a periodic crossing like:
- `[0, 0, 0]` = no cell crossing
- `[1, 0, 0]` = crosses one box in `+x`
- `[-1, 0, 0]` = crosses one box in `-x`

This point is critical:

**Bond translations are the per-edge periodic steps used during BFS.**

The code uses them to answer this question for every bonded neighbor:
- "If I am at atom `a1` in periodic image `off1`, which periodic copy of atom `a2` do I reach when I cross this bond?"

In code terms, BFS does:

```text
off2_expected = off1 + bond_translation(a1 -> a2)
```

So bond translations are what allow the code to:
1. propagate image offsets through the network
2. detect when the same atom is reached again in a different periodic image
3. compute the loop translation difference `delta`
4. decide whether that `delta` adds a new percolation direction

Without bond translations, the code would only have a bond graph. It would not know whether traversing a bond keeps the path in the same periodic cell or moves it into a neighboring image.

## CLI Usage

Basic usage:

```bash
python percolation.py data.crosslinked_updated
```

Write a component-colored output file:

```bash
python percolation.py data.crosslinked_updated \
  --component-type-data-out data.components
```

Write the human-readable report to a file as well:

```bash
python percolation.py data.crosslinked_updated \
  --report-out percolation_report.txt
```

## Output Behavior

The script prints:
- connected component counts
- component sizes
- percolation dimension per component
- percolation axis flags
- largest-component summary
- system-level summary

If `--report-out` is provided, the same human-readable report is also written to disk.

### Example Output

The following output was produced by running:

```bash
python percolation.py data.crosslinked_updated
```

on the file [`data.crosslinked_updated`](./data.crosslinked_updated).

```text
Reading: /Users/arunsrikanthsridhar/Downloads/percolation_repo/data.crosslinked_updated
Box dimensions: 119.048 x 119.298 x 119.185 Angstrom
Parsed image flags for 86310 atoms; percolation edge translations are still inferred from local wrapped bond geometry.
Atoms: 86310, Bonds: 86940
============================================================
PERCOLATION ANALYSIS REPORT
============================================================

Bond graph definition:
  Nodes: all atoms in the Atoms section
  Edges: all bonds in the Bonds section

Connected components: 1
Disconnected components (excluding largest): 0
Percolating components: 1
Single-atom components: 0
Multi-atom components: 1

Component size distribution:
  Largest:  86310 atoms
  Fraction in largest: 100.0%
  Atoms outside largest: 0

Per-component percolation:
  Component   0:  86310 atoms | dim=3 | percolates [XYZ] | span X:8 Y:10 Z:7

============================================================
SYSTEM-LEVEL PERCOLATION SUMMARY
============================================================
  Percolates in X: YES
  Percolates in Y: YES
  Percolates in Z: YES

  Fully percolated (X+Y+Z): YES
============================================================

LARGEST COMPONENT SUMMARY
============================================================
  Largest component id: 0 (86310 atoms, 100.0% of system)
  Largest component percolation: dim=3 [XYZ]
  Largest component spans X+Y+Z: YES
  Largest component fraction: 1.0000
============================================================

Summary: X=YES Y=YES Z=YES
```

## Tests

A local test suite is included in `tests/test_percolation.py`.

It covers:
- stable parsing of LAMMPS input
- optional image-flag parsing
- explicit bond translations
- periodic x-wrap regression
- basis-vector independence logic
- one smoke regression against a real `data.lammps` file in the local workspace

### Run the Tests

The default Python on this machine did not have `pytest`, so the tested command used an environment that does:

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate AmberTools23
PYTHONDONTWRITEBYTECODE=1 python -m pytest tests/test_percolation.py -q -p no:cacheprovider --basetemp=/tmp/percolation-repo-pytest
```

### Real Regression Test

One test uses this local file if it exists:

```text
/Users/arunsrikanthsridhar/Downloads/hydrogel_simulation/workspace/06fd0ccbfeedbfdf793a16b5d43fd918/data.lammps
```

If that file is not available, the test is skipped automatically.

## Limitations

Current limitations include:
- orthorhombic box handling only
- LAMMPS `atom_style full` assumption
- no bundled trajectory reader
- no bundled LAMMPS examples inside this repository itself

## References

- Mattia Livraghi, Kevin Höllring, Christian R. Wick, David M. Smith, and Ana-Sunčana Smith, *An Exact Algorithm to Detect the Percolation Transition in Molecular Dynamics Simulations of Cross-Linking Polymer Networks*, Journal of Chemical Theory and Computation (2021). DOI: [10.1021/acs.jctc.1c00423](https://doi.org/10.1021/acs.jctc.1c00423)
- `puls-group/percolation-analyzer` reference repository: [README](https://github.com/puls-group/percolation-analyzer/blob/master/Readme.md)
