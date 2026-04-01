# Algorithm Walkthrough

This note explains how `percolation.py` detects periodic percolation from a LAMMPS `data` file.

For the short operational explanation of what bond translations do during BFS, see the `How Bond Translations Are Handled` section in `README.md`.

The implementation follows the same core idea as:
- Livraghi et al., *An Exact Algorithm to Detect the Percolation Transition in Molecular Dynamics Simulations of Cross-Linking Polymer Networks* ([DOI](https://doi.org/10.1021/acs.jctc.1c00423))
- the `puls-group/percolation-analyzer` repository ([README](https://github.com/puls-group/percolation-analyzer/blob/master/Readme.md))

The difference is interface:
- the paper/repo describe a periodic graph with translation vectors on the edges
- this repository reads a LAMMPS `data` file and builds that graph automatically

## Step 1: Read the LAMMPS Data File

`read_lammps_data()` loads:
- atom coordinates
- molecule IDs
- bonds
- box lengths
- optional LAMMPS image flags

If image flags are present, they are used to build explicit bond translations.
Otherwise the code infers the translation once from wrapped coordinates.

## Step 2: Build Explicit Bond Translations

Each directed bond stores a periodic translation such as:
- `[0, 0, 0]` = no crossing
- `[1, 0, 0]` = cross one box in `+x`
- `[-1, 0, 0]` = cross one box in `-x`

This makes the graph look like the one described in the paper and the sample repo.

## Step 3: BFS Over a Connected Component

During BFS, the code stores `image_offset[atom]`, which means:

- which periodic copy of that atom was reached
- relative to the starting atom of the component

If atom 7 is reached with offset `[1, 0, 0]`, that means the current path reached the copy of atom 7 one periodic box away in `x`.

## Toy Example

Use a 4-atom ring with one periodic bond crossing in `x`:

```text
1 -- 2 -- 3 -- 4
|              |
+--------------+  (bond 4 -> 1 crosses +x)
```

Directed edge translations:
- `(1, 2) = [0, 0, 0]`
- `(2, 3) = [0, 0, 0]`
- `(3, 4) = [0, 0, 0]`
- `(4, 1) = [1, 0, 0]`
- reverse edges use the negative translation

Start BFS at atom 1:

```text
image_offset[1] = [0, 0, 0]
```

One path reaches atom 3 through `1 -> 2 -> 3`:

```text
image_offset[3] = [0, 0, 0]
```

Another path reaches atom 3 through `1 -> 4 -> 3`:

```text
expected offset for 3 = [-1, 0, 0]
```

Now the same atom is reached in two different periodic images, so the code computes:

```text
delta = [-1, 0, 0]
```

That means the component connects to itself shifted by one box in `x`.

The basis set becomes:

```text
basis_vectors = [[-1, 0, 0]]
```

So:
- `percolation_dim = 1`
- `percolates = [True, False, False]`

If the code later sees `[3, 0, 0]`, that is not a new dimension because it is linearly dependent on `[-1, 0, 0]`.

## Step 4: Build the Basis of Independent Loop Directions

Whenever the same atom is revisited with a different image offset, the code computes:

```text
delta = off2_expected - image_offset[a2]
```

This `delta` is a periodic loop translation.

The code adds `delta` to `basis_vectors` only if it increases the rank of the basis.
That is how `percolation_dim` is defined:

- `0` = no periodic loop translations
- `1` = one independent direction
- `2` = two independent directions
- `3` = three independent directions

## Real Data Example

The test suite includes a smoke regression using this local file when present:

```text
/Users/arunsrikanthsridhar/Downloads/hydrogel_simulation/workspace/06fd0ccbfeedbfdf793a16b5d43fd918/data.lammps
```

Observed values for that file:
- atoms: `27510`
- bonds: `27300`
- connected components: `630`
- largest component percolation dimension: `0`
- system percolation flags: `---`

So for this specific example, the code finds no spanning periodic network.

## Step 5: Convert Percolation Into Gel-Style Reporting

After component analysis, `compute_report()` identifies the largest covalent component and calculates:
- size fraction
- percolation dimension
- X/Y/Z spanning flags
- molecule-count and intermolecular-bond metrics
- optional crosslink-bond counts

The final `gel_like_percolation_pass` is stricter than pure percolation.
It requires the largest component to span `X+Y+Z` and pass user-selected thresholds.

## What to Look at in the Code

The most important functions are:
- `read_lammps_data()`
- `_build_bond_translations()`
- `analyze_percolation()`
- `_check_translation_independent()`
- `compute_report()`

These correspond to the same conceptual stages described in the paper and the `puls-group/percolation-analyzer` example implementation.
