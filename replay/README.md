# Replay supplement

Five standalone replay scripts and a standard-library driver reproduce every
finite-arithmetic, exact-rational, interval-enclosure, exact-diagonalization,
determinant quantum Monte Carlo, and worldline Monte Carlo check referenced in
the accompanying paper. Each replay script is a byte-for-byte copy of the script
run during verification. `SHA256SUMS` in this directory lists the SHA-256
digest of every other file in this directory, and the digests of the scripts are
repeated at the end of this file. None of the five replay scripts imports
another, and none reads any file outside this directory.

| File | Paper topic | Primary checks |
|---|---|---|
| `replay-charged-skeleton.py` | Charged-skeleton influence bound | Parseval and disjoint-bump Gram bounds, rational constants `3/22` and `3/19`, 4-hop fermion sign witness, converged 2-site to 4-site ED, and 8 mutants |
| `replay-telescope.py` | Telescoping obstruction | multi-time Weyl phonon trace and telescoping, offset action bound, HS positivity for all real $\mu$ and dimer cancellation, 2-site and 3-site ED, and 7 mutants |
| `replay-annealed-spectral.py` | Annealed spectral slice | static Wick and Ward identities, Matsubara covariance, `1/96` bound, complete second-chaos Taylor jets and Gauss-Hermite / MCMC oracle checks, and 7 mutants |
| `replay-annealed-endpoint.py` | Annealed endpoint closure | rational constants, interval Combes-Thomas and first-order bounds, SPD monodromy checks, `t=0` and `V=2` proofs, `V=2,4,6` ED/DQMC, tables, and 13 mutants |
| `replay-neutral-dipole.py` | Neutral-dipole cluster expansion | Gaussian trace and double-finite-difference bounds, Poghosyan-Ueltschi constants, collar screening witness, 2-site/3-site ED, 4-site worldline MC, and 10 mutants |
| `run_all.py` | Combined driver | runs all five scripts in plain mode, `--mutants`, and `--tables` using only the Python standard library |

## Script descriptions

`replay-charged-skeleton.py` verifies the exact Parseval kernel-variance
identity and the disjoint-bump Gram matrix positive semidefiniteness bound in
arbitrary precision, checks the exact rational additive multi-transition lower
bound constants `3/22` and `3/19` together with the regime window comparison,
enumerates the four-hop neutral plaquette fermion signs (finding 16 negative
and 144 positive loops among 160 four-hop neutral loops, alongside 70 positive
disjoint braiding loops), and runs converged 2-site grand-canonical and 3-site
and 4-site canonical exact diagonalizations comparing pair correlations against
the bare single-bridge kernel and neutral-dressed pair-flip overlaps. With
`--mutants`, it executes 8 deliberately broken variants and checks that each
fails with exit status 1.

`replay-telescope.py` checks the multi-time Weyl phonon displacement trace
identity and verifies that bare Lang-Firsov displacements along a synchronized
pair-hopping walk telescope to an endpoint-only trace independent of walk
length while the closed physical correlator collapses as the endpoint
separation vanishes. It also checks the near-synchronized offset action bound,
verifies Hubbard-Stratonovich determinant positivity for arbitrary real chemical potential (at zero and nonzero $\mu$, including non-bipartite graphs)
and exact dimer determinant cancellation, evaluates pre-Lang-Firsov exact
diagonalization on 2-site and 3-site chains demonstrating `(t/U)^(2d)` scaling
against exponential per-step suppression claims, and with `--mutants` catches
all 7 broken variants.

`replay-annealed-spectral.py` verifies the static annealed pair identity, spin
factor, Matsubara covariance, `1/96` first-variation bound, and exact Ward
identity connecting the observable gradient to the force covariance. It
computes complete Gaussian Taylor jets including non-frequency-conserving
second-chaos pairs on the dimer, 4-site ring, and 2x2 torus, cross-checks
those expansions against 60-digit high-precision evaluations, Gauss-Hermite
quadrature, and MCMC normalization-oracle runs showing the connected covariance
and observable-derivative terms, and with `--mutants` catches all 7 broken
variants.

`replay-annealed-endpoint.py` checks the exact rational counterexample
scaling `C(t, U)`, outward interval arithmetic for the Combes-Thomas resolvent
constant `K_0 = (1 + pi/4)^2`, SPD monodromy bounds alongside the normal `SO(2)`
counterexample and zero-mode Gaussian integration by parts, the exact `t=0`
nine-component marginal and `V=2` half-filled inequality, outward interval
bounds for the first-order dynamic coefficient `C_1^phys`, and the `V=6` Schur
row-sum interval certificate. It also compares finite-volume Fock-space exact
diagonalization, exact Hubbard-limit diagonalization, and UDT-stabilized
determinant quantum Monte Carlo on lattices of sizes `V = 2, 4, 6`. With
`--mutants` it kills all 13 broken variants, and with `--tables` it regenerates
the two summary numerical tables.

`replay-neutral-dipole.py` verifies the Gaussian phonon trace, Parseval
representation, double-finite-difference identity across interleaved, nested,
and periodic-wrapping intervals, and the `L^1` shift bound. It computes the
explicit temperature-, volume-, frequency-, and chemical-potential-uniform
Poghosyan-Ueltschi cluster-expansion constants for vertex degrees 2, 4, and 6,
verifies the exterior potential bound `|V_z| <= 0.05 U` outside the `1/Omega`
collar and the interior exponential screening witness inside it, matches 2-site
and 3-site exact Lang-Firsov diagonalizations (including the 4-hop virtual
pair-return shift and the 3-site sign change), and compares 4-site ring
continuous-time worldline Monte Carlo against the analytical 1-loop and 2-loop
cluster expansion including negative 4-hop exchange signs. With `--mutants`, it
catches all 10 broken variants.

`run_all.py` is a small standard-library-only driver that executes all five
scripts in plain mode, executes `--mutants` on all five scripts, and executes
`--tables` on `replay-annealed-endpoint.py` via `sys.executable -B`. It prints
a `PASS` or `FAIL` line for each check and each script and exits nonzero if any
invocation fails.

## Requirements and commands

The scripts require Python 3.12 or newer (verified with Python 3.13.9) with
`numpy`, `scipy`, `mpmath`, and `sympy`. From this directory in a POSIX shell
with `uv`, run the individual checks as follows:

```sh
uv run --no-project --with numpy --with scipy --with mpmath --with sympy python replay-charged-skeleton.py
uv run --no-project --with numpy --with scipy --with mpmath --with sympy python replay-charged-skeleton.py --mutants
uv run --no-project --with numpy --with scipy --with mpmath --with sympy python replay-telescope.py
uv run --no-project --with numpy --with scipy --with mpmath --with sympy python replay-telescope.py --mutants
uv run --no-project --with numpy --with scipy --with mpmath --with sympy python replay-annealed-spectral.py
uv run --no-project --with numpy --with scipy --with mpmath --with sympy python replay-annealed-spectral.py --mutants
uv run --no-project --with numpy --with scipy --with mpmath --with sympy python replay-annealed-endpoint.py
uv run --no-project --with numpy --with scipy --with mpmath --with sympy python replay-annealed-endpoint.py --mutants
uv run --no-project --with numpy --with scipy --with mpmath --with sympy python replay-annealed-endpoint.py --tables
uv run --no-project --with numpy --with scipy --with mpmath --with sympy python replay-neutral-dipole.py
uv run --no-project --with numpy --with scipy --with mpmath --with sympy python replay-neutral-dipole.py --mutants
```

To run the entire suite in one command:

```sh
uv run --no-project --with numpy --with scipy --with mpmath --with sympy python run_all.py
```

Setting `OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1`, and `MKL_NUM_THREADS=1`
is recommended on multi-core machines to avoid BLAS thread contention, and
`PYTHONDONTWRITEBYTECODE=1` keeps this directory free of `__pycache__`
artifacts (`run_all.py` sets these defaults automatically for child processes).

## Expected output and approximate run times

All commands exit with status 0. Observed final output lines and approximate
single-threaded laptop wall times are listed below:

- `replay-charged-skeleton.py` (~56 s): `ALL CHARGED-SKELETON CHECKS PASSED (EXIT 0)`
- `replay-charged-skeleton.py --mutants` (~30 s): `ALL 8 MUTANTS CAUGHT WITH EXIT 1; DRIVER EXIT 0`
- `replay-telescope.py` (~62 s): `ALL_CHECKS_PASSED`
- `replay-telescope.py --mutants` (~5 s): `MUTANTS all 7/7 caught with exit!=0; driver exit=0`
- `replay-annealed-spectral.py` (~9 s): `REPLAY SUCCESS`
- `replay-annealed-spectral.py --mutants` (~2 s): `ALL 7/7 MUTANTS CAUGHT` followed by `REPLAY SUCCESS`
- `replay-annealed-endpoint.py` (~7 s): `ALL REPLAY CHECKS PASSED.` followed by `Elapsed: <seconds>`
- `replay-annealed-endpoint.py --mutants` (~25 s): `ALL 13 MUTANTS KILLED (exit 0).`
- `replay-annealed-endpoint.py --tables` (~27 s): ends with the `V = 6` Hubbard-limit row of Table 2 (`Exact Hubbard limit $\Omega=\infty$, $\mu=0.5$`)
- `replay-neutral-dipole.py` (~50 s): `[PASS] Step 5C: 4-site ring worldline MC vs analytical cluster expansion (<seconds>)`
- `replay-neutral-dipole.py --mutants` (~150 s): `Mutation summary: 10/10 mutants caught`
- `run_all.py` (~6 minutes total): `SUMMARY: 5/5 scripts passed in <seconds>`

## What these scripts do not establish

Numerical checks support but do not replace the proofs in the paper;
finite-size exact diagonalization, determinant quantum Monte Carlo, and
worldline Monte Carlo are numerical evidence rather than thermodynamic proofs.
Exact-rational, outward-interval, finite-dimensional, and mutation checks
verify finite arithmetic and finite-volume witnesses only, and passing them
establishes none of the open thermodynamic lemmas stated in the paper.

## SHA-256 of the scripts in this directory

```
4fe93b2bd0fdf3c11a75aef7cbe415ed9a4015a8a682a38c00adba5c65d8368d  replay-charged-skeleton.py
800d6e12d196de3f585ad7d0c295b3be20006292eaeddc56b951e919d95b094b  replay-telescope.py
d33e788fe95ed1139f34372b50e0c6c879e2f41eee734db598f494b00d74e8fd  replay-annealed-spectral.py
d013cda90dde118358b489e382a587bfc1d740c0586427f6057fd827d2582595  replay-annealed-endpoint.py
e4f07397205b0343686369f0665c4648ed289eeaffe16aa0683adbe933aeea6a  replay-neutral-dipole.py
45647e15f8b6a38cbb2593341fda6b9fc6350671172e2cb71a37ec1657acde7b  run_all.py
```

`SHA256SUMS` additionally includes the digest of this `README.md` file and can be
verified from this directory with `sha256sum -c SHA256SUMS`.
