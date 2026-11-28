# Worldline-cluster, annealed-spectral, and telescoping obstructions to a phonon-frequency pairing ceiling in the Holstein model
Research record: [Worldline-cluster, annealed-spectral, and telescoping obstructions to a phonon-frequency pairing ceiling in the Holstein model](https://supermatics.io/research/worldline-telescoping-ceiling-obstructions).

Davis Rens, SuperMatics Inc., [ORCID 0000-0001-5839-5197](https://orcid.org/0000-0001-5839-5197)

| | |
| --- | --- |
| Preprint | [10.5281/zenodo.24425677](https://doi.org/10.5281/zenodo.24425677) |
| Replay supplement | [10.5281/zenodo.24425679](https://doi.org/10.5281/zenodo.24425679) |
| Version | 1.0.0, 21 November 2026 |
| Manuscript | [`paper/main.pdf`](paper/main.pdf), 20 pages |

This repository is the tagged release tree of that preprint. It holds the reviewed manuscript
source, the released PDF and the complete replay supplement, pinned byte for byte through
[`MANIFEST.sha256`](MANIFEST.sha256) and [`replay/SHA256SUMS`](replay/SHA256SUMS).

## What the paper does

Whether the pair-correlation decay temperature `T_pair` of the spinful Holstein model obeys a
rigorous phonon-frequency ceiling `k_B T_pair <= C(t,U) Omega`, uniformly in the volume and in
the chemical potential, is open, and this paper does not close it. It consolidates five
independently checked technical components that fix what can be proved on both sides of the
remaining gap and where the leading closure strategies break down.

1. **Charged-skeleton influence bounds and route obstructions.** An exact Parseval identity and
   a disjoint-bump Gram inequality for the Lang-Firsov phonon action in the full unrestricted
   Fock space, an additive multi-transition barrier lower bound with constant `3/22` (improving
   to `3/19` for half-width transitions and robust to short opposite-charge blips), and two
   structural obstructions at `beta t^2/U >> 1`: signless majorization fails on 16 of the 160
   four-hop neutral plaquette loops, and unexpanded-bath boundary subtraction cannot cancel an
   `O(beta t^2/U)` extensive mismatch.
2. **Telescoping obstruction and determinant positivity.** An exact telescoping bound that caps
   bare Lang-Firsov phonon suppression at a single walk-length-independent endpoint factor,
   Hubbard-Stratonovich determinant positivity for all real chemical potential with exact dimer
   cancellation, and a refutation of the literal strong-coupling bound of the preceding
   framework by converged grand-canonical and canonical exact diagonalization.
3. **Annealed static-plus-Gaussian package.** The static spectral package and a
   `(beta Omega)^2/96` continuum first-variation bound, the coherent-state pressure sandwich and
   the quantum zero-mode variance identity, a dimer counterexample that rules out unrestricted
   low-temperature transfer of static level counting, and the rejection of direct unweighted
   second-chaos normalization into the static force square.
4. **Annealed endpoint closure.** An unweighted annealed endpoint inequality yields only `l^1`
   summability, whereas a spatially weighted exponential-moment closing estimate implies the
   ceiling; the universal static Combes-Thomas base bound `K_0 = (1 + pi/4)^2`; an exact `t=0`
   barrier-to-well marginal suppression that annihilates the pathwise staggered instanton; an
   unconditional `V=2` operator-norm proof of the endpoint bound; and the exact `s=0` derivative
   bound `C_1^phys = 1/24 + zeta(3)/(8 pi^3)`.
5. **Neutral-dipole cluster expansion and collar boundary.** A volume-, temperature-,
   frequency-, chemical-potential- and background-uniform Poghosyan-Ueltschi cluster expansion
   for the neutral-loop gas and for its perturbed exterior outside an `Omega^-1` collar,
   together with the two open items that the collar leaves: interior screening `e^{+c U/Omega}`
   and four-hop virtual pair-return channels of order `beta t^4/(U^2 Omega)`.

## What the paper does not claim

No pairing ceiling is proved or disproved, and no bound is proved or disproved on any critical
temperature. No material is claimed, no superconductivity is claimed, and no phase transition is
proved or excluded. The route obstructions are obstructions for the specified estimates at the
specified constants, not impossibility statements for the underlying questions. The replay
scripts verify finite arithmetic in exact rational and outward-interval arithmetic; they are not
certificates for any thermodynamic statement and they check none of the analytic steps.

## Contents

| Path | What it is |
| --- | --- |
| `paper/main.pdf` | The released manuscript, 20 pages. |
| `paper/main.tex` | The LaTeX source that produces that PDF. |
| `paper/refs.bib` | The bibliography. |
| `paper/CLAIMS.md` | Index of every labeled statement, with its anchor in the source and the replay check that backs its finite arithmetic. |
| `replay/README.md` | What each replay script checks, how to run it and its byte custody. |
| `replay/replay-charged-skeleton.py` | Section 2: Parseval and disjoint-bump Gram bounds, the rational constants `3/22` and `3/19`, the four-hop fermion sign witness, converged 2-site to 4-site exact diagonalization, 8 mutants. |
| `replay/replay-telescope.py` | Section 3: multi-time Weyl phonon trace and telescoping, the offset action bound, Hubbard-Stratonovich positivity and dimer cancellation, 2-site and 3-site exact diagonalization, 7 mutants. |
| `replay/replay-annealed-spectral.py` | Section 4: static Wick and Ward identities, Matsubara covariance, the `1/96` bound, complete second-chaos Taylor jets and Gauss-Hermite and MCMC oracle checks, 7 mutants. |
| `replay/replay-annealed-endpoint.py` | Section 5: rational constants, interval Combes-Thomas and first-order bounds, monodromy checks, the `t=0` and `V=2` statements, `V=2,4,6` exact diagonalization and determinant quantum Monte Carlo, the summary tables, 13 mutants. |
| `replay/replay-neutral-dipole.py` | Section 6: Gaussian trace and double-finite-difference bounds, Poghosyan-Ueltschi constants, the collar screening witness, 2-site and 3-site exact diagonalization, 4-site worldline Monte Carlo, 10 mutants. |
| `replay/run_all.py` | Standard-library driver for all five scripts in plain, `--mutants` and `--tables` mode. |
| `replay/SHA256SUMS` | Pins for the seven other files of the replay directory. |
| `MANIFEST.sha256` | Pins for every other tracked file of this repository. |
| `BUILD.md` | How the PDF is built, with the compiler version, the pinned inputs and the observed log counts. |
| `CITATION.cff` | Citation metadata for the preprint and the supplement. |
| `LICENSE` | Apache-2.0 for the code, CC-BY-4.0 for the manuscript, as stated in its first lines. |
| `ci/verify_text.py` | Checks that the released PDF carries the title, identifiers, constants and the disclosure sentence. |

## Reproduce

### The replay

Python 3.12 or later with NumPy, SciPy, mpmath and sympy:

```sh
python -m pip install numpy scipy mpmath sympy
```

From the `replay/` directory of an unpacked release:

```sh
cd replay
python -B replay-charged-skeleton.py
python -B replay-charged-skeleton.py --mutants
python -B replay-telescope.py
python -B replay-telescope.py --mutants
python -B replay-annealed-spectral.py
python -B replay-annealed-spectral.py --mutants
python -B replay-annealed-endpoint.py
python -B replay-annealed-endpoint.py --mutants
python -B replay-annealed-endpoint.py --tables
python -B replay-neutral-dipole.py
python -B replay-neutral-dipole.py --mutants
python -B run_all.py
```

Every command exits `0`. The closing lines are `ALL CHARGED-SKELETON CHECKS PASSED (EXIT 0)`,
`ALL 8 MUTANTS CAUGHT WITH EXIT 1; DRIVER EXIT 0`, `ALL_CHECKS_PASSED`,
`MUTANTS all 7/7 caught with exit!=0; driver exit=0`, `REPLAY SUCCESS`,
`ALL 7/7 MUTANTS CAUGHT` followed by `REPLAY SUCCESS`, `ALL REPLAY CHECKS PASSED.` followed by
`Elapsed: ...`, `ALL 13 MUTANTS KILLED (exit 0).`, the `V = 6` Hubbard-limit row of the second
summary table for `--tables`, `[PASS] Step 5C: 4-site ring worldline MC vs analytical cluster
expansion (...)`, `Mutation summary: 10/10 mutants caught`, and `SUMMARY: 5/5 scripts passed in
...s`. Timings vary by more than an order of magnitude with the machine and its load; the
closing text does not. [`replay/README.md`](replay/README.md) maps every script to the section
it checks and lists the observed final lines, and `run_all.py` runs the whole battery in about
six minutes on a single laptop core.

### The paper

```sh
cd paper
SOURCE_DATE_EPOCH=1795132800 tectonic -X compile --keep-logs main.tex
```

Tectonic 0.17.0 with that fixed epoch reproduces the released PDF byte for byte (SHA-256
`52a80ea0b6b837a0286eeb467a4ad1b5af9afcaaf0601192fb4a1f5cb4d14dc9`); the log has no overfull
box, no undefined reference and no warning, and holds the two underfull boxes documented in
[`BUILD.md`](BUILD.md).

### Continuous integration

[`.github/workflows/ci.yml`](.github/workflows/ci.yml) has two jobs. The first installs the
pinned Tectonic release after checking the digest of its archive, rebuilds the manuscript with
the epoch above, requires zero overfull boxes, zero undefined references and no warning,
requires the rebuilt PDF to equal the released PDF byte for byte, and checks its metadata and
text. The second installs NumPy, SciPy, mpmath and sympy, verifies both pin files, builds the
release archive from the commit with `git archive`, unpacks it, and runs all five scripts in
plain, mutant and table mode plus the combined driver on the unpacked copy, requiring each
documented closing line.

## Byte custody

Every released file is a copy of reviewed bytes, and `MANIFEST.sha256` plus `replay/SHA256SUMS`
pin all of them except the two manifests themselves:

```sh
sha256sum -c MANIFEST.sha256
(cd replay && sha256sum -c SHA256SUMS)
```

The deposited supplement archive is `git archive` of the `v1.0.0` tag, so its members are the
files listed by `git ls-files` at that commit, unchanged.

## Licences

The manuscript (`paper/main.tex`, `paper/refs.bib`, `paper/main.pdf`) is licensed under Creative
Commons Attribution 4.0 International (CC-BY-4.0). The replay scripts and all remaining files
are licensed under the Apache License 2.0. The first lines of [`LICENSE`](LICENSE) state that
split; the full Apache-2.0 text follows.

## Review and AI assistance

AI tools were used for drafting, calculation and checking under D. Rens' direction and review,
and separate AI systems carried out independent reviews of the argument and of the replay
scripts, whose comments led to several corrections. This preprint has not been peer reviewed.
