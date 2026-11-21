# Claim map

Companion index mapping every labeled theorem, lemma, corollary, proposition, observation, route
obstruction, definition, remark and summary table of `paper/main.tex` to its anchor in the source
and to the replay script that rechecks the finite arithmetic behind it. Line numbers refer to the
released `paper/main.tex`.

## Section 1: Introduction and program overview (`sec:intro`, `main.tex:86`)

- **Definition (`def:tpair`, `main.tex:169`)**: Pair-correlation decay temperature
  `T_pair` (`eq:tpair-def`, `main.tex:176`), restated from the preceding framework paper.

## Section 2: Charged-skeleton influence bounds and route obstructions (`sec:charged-skeleton`, `main.tex:186`)

- **Lemma (`lem:cs-gram`, `main.tex:244`)**: Parseval identity (`eq:cs-parseval-h1`,
  `main.tex:265`) and disjoint-bump Gram inequality `G <= D`; `replay/replay-charged-skeleton.py`
  Checks 1-2 (`[0, 1]`, `1e-12` tolerance).
- **Lemma (`lem:cs-barrier`, `main.tex:305`)**: Additive multi-transition barrier lower bound
  (`3/22`, `3/19`, `1/8` blip tolerance; `eq:cs-barrier`, `main.tex:321`);
  `replay/replay-charged-skeleton.py` Check 3 (exact `Fraction` arithmetic).
- **Remark (`rem:cs-window`, `main.tex:362`)**: Parameter window against the released
  strong-coupling ceiling and the open window `tau ~ Omega << t^2/U`; notes the bare
  single-bridge kernel comparison (Observation 2.1, `10^33` ratio without refuting the
  unconstrained neutral skeleton).
- **Route Obstruction (`obs:cs-sign`, `main.tex:380`)**: Neutral four-hop fermion sign witness
  (16 of 160 negative loops on a `2 x 2` plaquette); `replay/replay-charged-skeleton.py` Check 4.
- **Route Obstruction (`obs:cs-boundary`, `main.tex:394`)**: Unexpanded neutral-loop boundary
  coupling at `beta t^2/U >> 1` (`tau ~ Omega << t^2/U`);
  `replay/replay-charged-skeleton.py` Check 5 (2-, 3- and 4-site exact diagonalization).

## Section 3: Telescoping obstruction, determinant positivity, and ED refutation (`sec:telescope`, `main.tex:406`)

- **Theorem (`thm:tel-telescope`, `main.tex:425`)**: Exact synchronized pair-transfer
  telescoping (`W_open,sync = exp[-2X coth(beta Omega/2)]` and `W_phys,sync -> 1` as
  `s_1 < ... < s_L -> 0^+`); `replay/replay-telescope.py` Check 1.
- **Theorem (`thm:tel-offset`, `main.tex:473`)**: Near-synchronized time-offset action bound
  (`eq:tel-offset`, `main.tex:485`); `replay/replay-telescope.py` Check 1.
- **Lemma (`lem:tel-hs`, `main.tex:505`)**: Generalized Hubbard-Stratonovich positivity
  (`det^2 >= 0` for any finite graph and real `mu`) and exact dimer (`N=2`) cancellation;
  `replay/replay-telescope.py` Check 2.
- **Observation (`obs:tel-ed`, `main.tex:551`)**: Exact-diagonalization refutation of the literal
  per-step physical bound (`eq:tel-fg-conj`, `main.tex:556`) with `O(1)` neutral prefactor
  (`1.60e-19`, `6.40e-20`, `1.57e-9` after correction); `replay/replay-telescope.py` Checks 3-4.

## Section 4: Annealed spectral estimates and the second-chaos obstruction (`sec:annealed-spectral`, `main.tex:593`)

- **Theorem (`thm:as-static`, `main.tex:621`)**: Static annealed spectral package: static pair
  identity, sum rule, Kubo bound, uniform Fermi-kernel decay, finite-shift relative-entropy level
  counting `N(E) <= min(1, 8E/U)` with `2 + 4 log 2`, annealed resolvent, log Cooper bubble and a
  single nonstatic Cooper rung.
- **Theorem (`thm:as-firstvar`, `main.tex:689`)**: Continuum first-variation bound
  `V^-1 E_{nu_0} E_G ||delta C||_HS^2 <= (beta Omega)^2/96` (`eq:as-1over96`, `main.tex:695`);
  `replay/replay-annealed-spectral.py` Check 1.
- **Theorem (`thm:as-sandwich`, `main.tex:712`)**: Coherent-state pressure sandwich
  (`eq:as-sandwich`, `main.tex:725`).
- **Theorem (`thm:as-dimer`, `main.tex:740`)**: Quantum zero-mode variance identity
  (`eq:as-nuqvar`, `main.tex:745`) and the low-temperature dimer level-counting obstruction
  (`U = 16t`, `E = 3t/2`); `replay/replay-annealed-spectral.py` Check 3.
- **Route Obstruction (`obs:as-chaos`, `main.tex:787`)**: Negative verdict on direct unweighted
  second-chaos normalization and its exact replacement (`eq:as-ward`, `main.tex:809`);
  `replay/replay-annealed-spectral.py` Checks 2 and 4.

## Section 5: Annealed endpoint reduction, static Combes-Thomas base, and instanton annihilation (`sec:annealed-endpoint`, `main.tex:830`)

- **Theorem (`thm:ae-cond`, `main.tex:869`)**: Conditional linear-in-`Omega` reduction from
  Closing Estimate (2) (`eq:ae-closing2`, `main.tex:864`) and insufficiency of unweighted `l^1`
  summability (`eq:ae-conj1`, `main.tex:855`).
- **Theorem (`thm:ae-k0`, `main.tex:910`)**: Universal static-limit Combes-Thomas base bound
  `K_0 = (1 + pi/4)^2` (`eq:ae-k0`, `main.tex:921`) and the `V=6` outward interval Schur
  certificate; `replay/replay-annealed-endpoint.py` Check 1.
- **Theorem (`thm:ae-v2`, `main.tex:963`)**: Instanton weight suppression (`eq:ae-r0`,
  `main.tex:979`, `C_bar(beta U) in (1, 16)` with exact supremum `B_*`) and the unconditional
  `V=2` operator-norm proof `E_{nu_1}[S_1] < 1`; `replay/replay-annealed-endpoint.py` Checks 2-4.
- **Theorem (`thm:ae-s0`, `main.tex:1027`)**: Static strictness and the exact first-order `s=0`
  derivative `C_1^phys = 1/24 + zeta(3)/(8 pi^3)` (`eq:ae-c1phys`, `main.tex:1041`);
  `replay/replay-annealed-endpoint.py` Check 5.
- **Proposition (`prop:ae-hubbard`, `main.tex:1063`)**: High-frequency (`Omega -> infinity`)
  attractive Hubbard anchor and the two global barriers that keep a non-asymptotic closure out of
  reach at the specified constants.

## Section 6: Volume-uniform neutral-dipole cluster expansion and collar boundary (`sec:neutral-dipole`, `main.tex:1089`)

- **Theorem (`thm:nd-conv`, `main.tex:1133`)**: Volume-, `beta`-, `Omega`-, `mu`- and
  `eta`-uniform Poghosyan-Ueltschi convergence for `t/U <= c_0(Delta)` through Lemmas 1.1-2.2.
  The measure includes the atomic factor and cyclic divisor, `b = Phi_beta` cancels the phonon
  weight in Assumption 2', and the soft interaction is integrated only over compatible shifts;
  qualitative integrability holds for each finite graph and is not a uniform bound on the total
  integral. `replay/replay-neutral-dipole.py` Checks 1-3.
- **Corollary (`cor:nd-press`, `main.tex:1178`)**: Pure-neutral pressure analyticity
  `|p_neut| <= C_press(Delta) t^2/U` and the site-pinning bound.
- **Lemma (`lem:nd-ratios`, `main.tex:1193`)**: Unperturbed neutral-sector ratio bounds
  (`C_diff(Delta) = 2 C_press(Delta)` and `C_col,2(Delta)`).
- **Theorem (`thm:nd-ext`, `main.tex:1239`)**: Perturbed exterior cluster expansion and the exact
  collar-ratio interface for `t/U <= c_0^ext(Delta)`, with the constants table
  (`tab:nd-constants`, `main.tex:1296`, including `Delta = 4` values
  `c_0^gen(4) = 0.0151` and `C_press^gen(4) = 26.7837`);
  `replay/replay-neutral-dipole.py` Check 4.
- **Remark (`rem:nd-scope`, `main.tex:1310`)**: The collar boundary stated openly: screening
  `e^{+c U/Omega}` inside `R_col` and four-hop virtual pair-return channels
  `O(beta t^4/(U^2 Omega))` remain open.

## Section 7: Scope of verification, open problems, and disclosures (`sec:scope`, `main.tex:1334`)

- **Table (`tab:scope`, `main.tex:1382`)**: Unified summary of the established results, the
  ruled-out directions and the open gaps across the five components.

## Replay coverage

`replay/run_all.py` executes every script of the supplement in plain mode, every `--mutants` mode
and the `--tables` mode of `replay/replay-annealed-endpoint.py`, and `replay/README.md` states
what each script checks. The scripts verify finite arithmetic only; no entry of this map is
established by a script, and the open statements listed above are not closed by any of them.
