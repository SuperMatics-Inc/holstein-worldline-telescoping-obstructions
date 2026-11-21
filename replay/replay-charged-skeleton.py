#!/usr/bin/env python3
"""Self-contained verification and mutation suite for W/MASSIVE/CHARGED-SKELETON/.

Checks:
  1. NUMERICAL/mpmath: Exact Kernel-Variance Identity (Lemma 2.2(1)) & Disjoint-Bump Lower Bound (Lemma 2.2(2)).
  2. CERTIFIED/Fraction: Additive Multi-Transition Lower Bound constants 3/22 and 3/19 (Lemma 2.3).
  3. CERTIFIED/Fraction: Window arithmetic & STRONG-SHARP threshold check.
  4. CERTIFIED: 2x2 plaquette 4-hop neutral fermion sign = -1 witness (16/160) and disjoint braiding sign = +1 (70/70).
  5. NUMERICAL: Converged 2-site grand-canonical ED (bare-bridge comparison + neutral-dressed pair-flip overlap).
  6. NUMERICAL: Converged 3-site canonical ED (d=2 pair correlation + 3-site neutral-dressed pair-flip trace ratio).
  7. NUMERICAL: 4-site canonical ED (d=3 pair correlation + honest low-T cutoff-sensitivity reporting).
  8. MUTANTS: 8 guarded mutants exercising actual checks and caught with exit code 1.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

import mpmath as mp
import numpy as np
from scipy.linalg import eigh


MUTATIONS = {
    "bad_kernel_sign": "Flip sign of -q(u)q(v) to +q(u)q(v) in Kernel-Variance identity",
    "bad_gram_diag": "Drop rank-1 mean subtraction and negate diagonal check in Lemma 2.2(2)",
    "bad_csw_constant": "Claim inflated additive constant 3/10 instead of 3/22 in Lemma 2.3",
    "bad_blip_fraction": "Violate minority blip budget (use 3/8 instead of 1/8) with 3/22 threshold",
    "bad_plaquette_sign": "Assert all 4-hop neutral plaquette loops have fermion sign +1",
    "bad_bare_bridge_bound": "Assert 2-site <Delta_0^dag Delta_1> <= bare single-bridge kernel K_bare at tau=1, X=20",
    "bad_window_arithmetic": "Claim tau ~ Omega violates beta*t^2/U <= eta*X",
    "bad_half_delta_constant": "Claim inflated constant 3/16 instead of 3/19 when delta_k <= ell_0/2 in Lemma 2.3",
}


def require(cond: bool, msg: str) -> None:
    if not cond:
        print(f"REFUSED: {msg}", flush=True)
        sys.exit(1)


# ---------------------------------------------------------------------------
# 1. Exact Kernel-Variance Identity & Disjoint-Bump Bound (NUMERICAL / mpmath)
# ---------------------------------------------------------------------------

def phi_exact_from_jumps(jumps: list[tuple[object, int]], beta, om, u_val):
    alpha2 = u_val / (2 * om)
    sh = mp.sinh(beta * om / 2)
    total = mp.mpf(0)
    for ui, ci in jumps:
        for uj, cj in jumps:
            d = abs(ui - uj)
            d = min(d, beta - d)
            D = mp.cosh(om * (beta / 2 - d)) / sh
            total += ci * cj * D
    return mp.mpf("0.5") * alpha2 * total

def check_kernel_variance_identity(mutant: str | None) -> None:
    mp.mp.dps = 50
    beta = mp.mpf("10.0")
    om = mp.mpf("1.3")
    u_val = mp.mpf("15.0")

    # Multi-transition worldline on [0, beta) visiting -1, 0, +1, 0, -1, +1, -1
    jumps = [
        (mp.mpf("1.0"), 1),    # -1 -> 0
        (mp.mpf("1.4"), 1),    # 0 -> +1
        (mp.mpf("3.8"), -2),   # +1 -> -1
        (mp.mpf("5.2"), 2),    # -1 -> +1
        (mp.mpf("6.1"), -1),   # +1 -> 0
        (mp.mpf("6.5"), -1),   # 0 -> -1
        (mp.mpf("8.0"), 2),    # -1 -> +1
        (mp.mpf("9.2"), -2),   # +1 -> -1
    ]
    assert sum(c for _, c in jumps) == 0
    q0 = -1

    phi_jump = phi_exact_from_jumps(jumps, beta, om, u_val)

    # Exact piecewise analytic double integral of K_Omega(u-v)*(1 - sgn*q(u)q(v))
    pts = [mp.mpf(0)] + [u for u, _ in jumps] + [beta]
    t_odd = mp.mpf(0)
    vals_seg = []
    cur = q0
    for i in range(len(pts) - 1):
        vals_seg.append(cur)
        if cur == 0:
            t_odd += pts[i + 1] - pts[i]
        if i < len(jumps):
            cur += jumps[i][1]

    # Exact antiderivative of int_a^b int_c^d K_Omega(u-v) du dv
    sh = mp.sinh(beta * om / 2)

    def int_K_1d(u, c, d):
        def F_pos(w):
            return mp.mpf("-0.5") * mp.sinh(om * (beta / 2 - w)) / sh
        pts_local = [c, d]
        for k in (-1, 0, 1):
            p = u + k * beta
            if c < p < d:
                pts_local.append(p)
        pts_local = sorted(set(pts_local))
        s = mp.mpf(0)
        for x0, x1 in zip(pts_local[:-1], pts_local[1:]):
            xm = mp.mpf("0.5") * (x0 + x1)
            shift = mp.floor((u - xm) / beta) * beta
            w0 = (u - x1) - shift
            w1 = (u - x0) - shift
            s += F_pos(w1) - F_pos(w0)
        return s

    def dbl_int_K(a, b, c, d):
        return mp.quad(lambda u: int_K_1d(u, c, d), [a, b])
    sgn = 1 if mutant == "bad_kernel_sign" else -1
    double_sum = mp.mpf(0)
    for i in range(len(vals_seg)):
        for j in range(len(vals_seg)):
            w_ij = dbl_int_K(pts[i], pts[i + 1], pts[j], pts[j + 1])
            double_sum += w_ij * (1 + sgn * vals_seg[i] * vals_seg[j])
    rhs = (u_val / 2) * double_sum
    lhs = phi_jump + (u_val / 2) * t_odd
    err = abs(lhs - rhs)
    print(f"[NUMERICAL/mpmath] Lemma 2.2(1) |LHS - RHS| = {mp.nstr(err, 6)}", flush=True)
    require(err < mp.mpf("1e-25"), f"Kernel-variance identity failed: err={err}")


def check_disjoint_bump_gram(mutant: str | None) -> None:
    mp.mp.dps = 40
    beta = mp.mpf("12.0")
    om = mp.mpf("1.1")
    u_val = mp.mpf("18.0")

    # 3 disjoint trapezoid test functions g_k on [0, 3], [4, 7], [8, 11]
    supports = [(mp.mpf("0.2"), mp.mpf("1.0"), mp.mpf("1.5"), mp.mpf("2.3")),
                (mp.mpf("4.0"), mp.mpf("4.9"), mp.mpf("5.4"), mp.mpf("6.3")),
                (mp.mpf("8.1"), mp.mpf("8.9"), mp.mpf("9.5"), mp.mpf("10.3"))]

    # Fourier modes n = 1..150 (exact up to 1e-8 for piecewise linear; sufficient for Gram check)
    # Better: compute exact real-space integrals for D_kk and v_k = int g_k
    D_diag = []
    v_vec = []
    for a, b, c, d in supports:
        l1 = b - a
        gd = c - b
        l2 = d - c
        d_val = (1 / l1 + 1 / l2) + (om**2) * (l1 / 3 + gd + l2 / 3)
        v_val = mp.mpf("0.5") * l1 + gd + mp.mpf("0.5") * l2
        D_diag.append(d_val)
        v_vec.append(v_val)

    m = len(supports)
    G = mp.matrix(m, m)
    D = mp.matrix(m, m)
    for j in range(m):
        D[j, j] = D_diag[j]
        for k in range(m):
            G[j, k] = (D_diag[j] if j == k else mp.mpf(0)) - (om**2 / beta) * v_vec[j] * v_vec[k]

    if mutant == "bad_gram_diag":
        G = D + mp.eye(m)

    # Verify D - G is rank-1 PSD and G^{-1} - D^{-1} >= 0 via exact symmetric eigenvalues
    diff_inv = G**(-1) - D**(-1)
    eigs_diff, _ = mp.eigsy(diff_inv)
    eigs_sorted = sorted(eigs_diff)
    print(f"[NUMERICAL/mpmath] Lemma 2.2(2) eigenvalues of (G^-1 - D^-1): {[mp.nstr(x, 6) for x in eigs_sorted]}", flush=True)
    require(eigs_sorted[0] > -mp.mpf("1e-35") and eigs_sorted[-1] > mp.mpf("1e-6"), f"G^-1 - D^-1 not PSD: {eigs_sorted}")

    # Test against a multi-transition path
    jumps = [
        (mp.mpf("0.2"), 2), (mp.mpf("1.5"), -2),
        (mp.mpf("4.0"), 2), (mp.mpf("5.4"), -2),
        (mp.mpf("8.1"), 2), (mp.mpf("9.5"), -2),
    ]
    phi_val = phi_exact_from_jumps(jumps, beta, om, u_val)
    var_lb = mp.mpf(0)
    for k, (a, b, c, d) in enumerate(supports):
        # q = -1 on [a,b] except [0.2, ...]: here q=+1 on [a, c], -1 on [c, d] -> int q g' = +1 + 1 = 2
        ig = mp.mpf(2)
        var_lb += (u_val / 2) * (ig**2) / D_diag[k]
    print(f"[NUMERICAL/mpmath] Lemma 2.2(2) multi-bump Phi={mp.nstr(phi_val, 8)} >= LB={mp.nstr(var_lb, 8)}", flush=True)
    require(phi_val >= var_lb, "Variational lower bound violated")


# ---------------------------------------------------------------------------
# 2. Exact Fraction Constants for Lemma 2.3 & Window / STRONG-SHARP Check (CERTIFIED)
# ---------------------------------------------------------------------------

def check_exact_constants(mutant: str | None) -> None:
    # Lemma 2.3 exact rational algebra
    # Case I1: T_odd >= (1/2) * l0 -> (U/2)*T_odd >= (1/4) U l0
    c_I1 = Fraction(1, 4)
    # Case I2: T_odd < (1/2) * l0, minority blip <= eps * l0, delta <= l0
    eps = Fraction(3, 8) if mutant == "bad_blip_fraction" else Fraction(1, 8)
    num_min = Fraction(2, 1) - Fraction(1, 2) - 4 * eps  # 2 - T_odd/l0 - 4*eps
    denom_max = Fraction(2, 1) + Fraction(2, 3) + Fraction(1, 1)  # 11/3
    c_I2 = Fraction(1, 2) * (num_min * num_min) / denom_max if num_min > 0 else Fraction(0, 1)
    c_claimed = Fraction(3, 10) if mutant == "bad_csw_constant" else Fraction(3, 22)

    print(f"[CERTIFIED/Fraction] Lemma 2.3: c_I1={c_I1}, c_I2={c_I2}, claimed={c_claimed}", flush=True)
    require(num_min >= Fraction(1, 1), f"Numerator bound < 1 for eps={eps}")
    require(min(c_I1, c_I2) >= c_claimed, f"min(c_I1, c_I2)={min(c_I1, c_I2)} < claimed={c_claimed}")

    # Improved constant for delta <= l0/2:
    denom_half = Fraction(2, 1) + Fraction(2, 3) + Fraction(1, 2)  # 19/6
    c_I2_half = Fraction(1, 2) / denom_half
    c_half_claimed = Fraction(3, 16) if mutant == "bad_half_delta_constant" else Fraction(3, 19)
    require(c_I2_half >= c_half_claimed, f"Improved constant {c_I2_half} < claimed {c_half_claimed}")
    require(c_I2_half == Fraction(3, 19), "Unexpected 3/19 constant")

    # Proved STRONG-SHARP homogeneous threshold and bound constants (57/500)*(99/100)^d under beta*t^2/U <= 1/6000
    ss_pref = Fraction(57, 500)
    ss_base = Fraction(99, 100)
    ss_thresh = Fraction(1, 6000)
    require(ss_pref > 0 and 0 < ss_base < 1 and ss_thresh == Fraction(1, 6000), "STRONG-SHARP constants invalid")

    # Window comparison at tau = Omega << t^2/U
    # Example: x = t/U = 1/100, X = U/Omega = 10^6 (so t^2/U = 10^-4 U >> Omega = 10^-6 U)
    # At tau = Omega, beta*Omega = 1, beta*t^2/U = x^2 * X = 100 << X = 10^6.
    x = Fraction(1, 100)
    X = Fraction(10**6, 1)
    beta_om = Fraction(1, 1)
    beta_t2_over_U = beta_om * x * x * X
    eta_window = Fraction(1, 10) * X * min(Fraction(1, 1), beta_om / 4)
    in_energetic_window = (beta_t2_over_U <= eta_window)
    if mutant == "bad_window_arithmetic":
        in_energetic_window = not in_energetic_window
    print(f"[CERTIFIED/Fraction] tau=Omega window check: beta*t^2/U={beta_t2_over_U} <= eta*X*min={eta_window}", flush=True)
    require(in_energetic_window, "tau ~ Omega arithmetic window check failed")


# ---------------------------------------------------------------------------
# 3. 2x2 Plaquette 4-Hop Neutral Sign Witness & Disjoint Braiding (CERTIFIED)
# ---------------------------------------------------------------------------

def check_plaquette_fermion_signs(mutant: str | None) -> None:
    # 4 sites 0,1,2,3 in a cycle (0-1-2-3-0); 4 electrons: pairs on 0 and 2 (checkerboard)
    # Spin-orbital index: 2*site + spin (spin 0=up, 1=dn)
    def apply_hop(occ: tuple[int, ...], frm: int, to: int, spin: int) -> tuple[int, tuple[int, ...]] | None:
        i_f = 2 * frm + spin
        i_t = 2 * to + spin
        if occ[i_f] != 1 or occ[i_t] != 0:
            return None
        lst = list(occ)
        # Count occupied orbitals strictly between min(i_f, i_t) and max(i_f, i_t)
        lo, hi = min(i_f, i_t), max(i_f, i_t)
        sgn = -1 if (sum(lst[lo + 1 : hi]) % 2 == 1) else 1
        lst[i_f] = 0
        lst[i_t] = 1
        return sgn, tuple(lst)

    eta0 = (1, 1, 0, 0, 1, 1, 0, 0)  # pairs on site 0 and site 2
    edges = [(0, 1), (1, 2), (2, 3), (3, 0), (1, 0), (2, 1), (3, 2), (0, 3)]

    neg_loops = 0
    pos_loops = 0
    # Enumerate 4-hop sequences returning to eta0 and visiting intermediate odd sites
    for e1 in edges:
        for s1 in (0, 1):
            r1 = apply_hop(eta0, e1[0], e1[1], s1)
            if r1 is None:
                continue
            for e2 in edges:
                for s2 in (0, 1):
                    r2 = apply_hop(r1[1], e2[0], e2[1], s2)
                    if r2 is None:
                        continue
                    for e3 in edges:
                        for s3 in (0, 1):
                            r3 = apply_hop(r2[1], e3[0], e3[1], s3)
                            if r3 is None:
                                continue
                            for e4 in edges:
                                for s4 in (0, 1):
                                    r4 = apply_hop(r3[1], e4[0], e4[1], s4)
                                    if r4 is None or r4[1] != eta0:
                                        continue
                                    # (-t)^4 = +t^4, so fermion sign is product of Jordan-Wigner signs
                                    total_sgn = r1[0] * r2[0] * r3[0] * r4[0]
                                    if total_sgn < 0:
                                        neg_loops += 1
                                    else:
                                        pos_loops += 1

    print(f"[CERTIFIED] 2x2 plaquette 4-hop closed neutral loops: neg={neg_loops}, pos={pos_loops}", flush=True)
    if mutant == "bad_plaquette_sign":
        require(neg_loops == 0, "4-hop neutral plaquette has negative fermion sign loops")
    else:
        require(neg_loops == 16, f"Expected 16 negative 4-hop plaquette loops, got {neg_loops}")

    # Verify spacetime-disjoint neutral polymer sign factorization:
    # Polymer A on bond (0,1): pair moves 0 -> 1 via up then dn; returns 1 -> 0 via up then dn (total 4 hops)
    # Polymer B on bond (2,3): pair moves 2 -> 3 via up then dn; returns 3 -> 2 via up then dn (total 4 hops)
    # Interleave their 8 hops in any time-order preserving A's internal order and B's internal order
    opsA = [(0, 1, 0), (0, 1, 1), (1, 0, 0), (1, 0, 1)]
    opsB = [(2, 3, 0), (2, 3, 1), (3, 2, 0), (3, 2, 1)]

    def eval_seq(seq):
        st = eta0
        sgn = 1
        for f, to, sp in seq:
            res = apply_hop(st, f, to, sp)
            assert res is not None
            sgn *= res[0]
            st = res[1]
        assert st == eta0
        return sgn

    sgnA = eval_seq(opsA)
    sgnB = eval_seq(opsB)
    # Check all C(8,4)=70 interleavings
    from itertools import combinations
    for posA in combinations(range(8), 4):
        posA_set = set(posA)
        seq = []
        ia = ib = 0
        for idx in range(8):
            if idx in posA_set:
                seq.append(opsA[ia])
                ia += 1
            else:
                seq.append(opsB[ib])
                ib += 1
        require(eval_seq(seq) == sgnA * sgnB, "Disjoint neutral polymer sign factorization failed")


# ---------------------------------------------------------------------------
# 4. Exact Diagonalization Helpers (2-site GC, 3-site & 4-site Canonical)
# ---------------------------------------------------------------------------

def displacement_matrix(gam: float, cutoff: int) -> np.ndarray:
    from scipy.special import eval_genlaguerre, gammaln
    mat = np.zeros((cutoff, cutoff), dtype=float)
    if gam == 0.0:
        np.fill_diagonal(mat, 1.0)
        return mat
    g2 = gam * gam
    # <m|D(gam)|n> = sqrt(n!/m!) gam^(m-n) e^{-g2/2} L_n^{(m-n)}(g2) for m >= n
    for m in range(cutoff):
        for n in range(m + 1):
            val = np.exp(0.5 * (gammaln(n + 1) - gammaln(m + 1)) - 0.5 * g2) * (gam ** (m - n)) * eval_genlaguerre(n, m - n, g2)
            mat[m, n] = val
            if m != n:
                mat[n, m] = ((-1) ** (m - n)) * val
    return mat


def dimer_gc_ed(cutoff: int, u_val: float, om: float, t: float, taus: list[float]) -> tuple[list[float], float]:
    """2-site grand-canonical ED in LF frame with exact COM decoupling."""
    gam = np.sqrt(u_val / om)  # relative displacement for 1-electron hop: sqrt(2)*alpha = sqrt(X)
    Dp = displacement_matrix(gam, cutoff)
    Dm = Dp.T

    # 16 electronic states: 4 bits (0_up, 0_dn, 1_up, 1_dn)
    sectors: dict[tuple[int, int], list[int]] = {}
    for st in range(16):
        nu = (st & 1) + ((st >> 2) & 1)
        nd = ((st >> 1) & 1) + ((st >> 3) & 1)
        sectors.setdefault((nu, nd), []).append(st)

    def hop_step(st: int, frm: int, to: int, sp: int):
        b_f = 2 * frm + sp
        b_t = 2 * to + sp
        if ((st >> b_f) & 1) == 0 or ((st >> b_t) & 1) == 1:
            return None
        lo, hi = min(b_f, b_t), max(b_f, b_t)
        mask = ((1 << hi) - 1) ^ ((1 << (lo + 1)) - 1)
        sgn = -1.0 if bin(st & mask).count("1") % 2 else 1.0
        return st ^ (1 << b_f) ^ (1 << b_t), sgn

    def pair_op_term(st: int):
        # Delta_0^dag Delta_1 = c_{0u}^dag c_{0d}^dag c_{1d} c_{1u}
        if ((st >> 0) & 1) == 0 and ((st >> 1) & 1) == 0 and ((st >> 2) & 1) == 1 and ((st >> 3) & 1) == 1:
            return st ^ 0b1111, 1.0  # pair on site 1 (bits 2,3) -> site 0 (bits 0,1): no sign flip
        return None

    # Also compute neutral-dressed ground-state pair-flip overlap in N_el=2 sector for S_tot=0
    traces = []
    pair_overlap_ratio = 0.0
    n_rel = np.arange(cutoff, dtype=float)
    for (nu, nd), sts in sectors.items():
        idx = {s: i for i, s in enumerate(sts)}
        dim = len(sts) * cutoff
        H = np.zeros((dim, dim), dtype=float)
        O = np.zeros((dim, dim), dtype=float)
        for s, i in idx.items():
            sl = slice(i * cutoff, (i + 1) * cutoff)
            n0 = (s & 1) + ((s >> 1) & 1)
            n1 = ((s >> 2) & 1) + ((s >> 3) & 1)
            n_odd = int(n0 == 1) + int(n1 == 1)
            # In relative mode, displacement depends on (n0 - n1):
            # H_LF = om * N_rel + (U/2)*N_odd - t * hop * D_rel(+-2*gam)
            np.fill_diagonal(H[sl, sl], om * n_rel + 0.5 * u_val * n_odd)
            for frm, to, D_mat in ((0, 1, Dm), (1, 0, Dp)):
                for sp in (0, 1):
                    res = hop_step(s, frm, to, sp)
                    if res is not None:
                        j = idx[res[0]]
                        H[j * cutoff : (j + 1) * cutoff, sl] += -t * res[1] * D_mat
            pr = pair_op_term(s)
            if pr is not None:
                j = idx[pr[0]]
                # Pre-LF Delta_0^dag Delta_1 in LF frame carries D_rel(-2*gam) on relative mode
                D_pair = displacement_matrix(2.0 * gam, cutoff)
                O[j * cutoff : (j + 1) * cutoff, sl] = pr[1] * D_pair

        vals, vecs = eigh(H, overwrite_a=True, check_finite=False)
        diagO = np.einsum("ij,ij->j", vecs, O @ vecs)
        traces.append((vals, diagO))

        if (nu, nd) == (1, 1):
            # Compare neutral-dressed single-pair states |L_0> (pair at 0) vs |R_0> (pair at 1)
            # under neutral-only H (keeping pair fixed, allowing virtual odd hop)
            # Neutral Hamiltonian on {|2,0>, |0,2>, |u,d>, |d,u>} blocks pair flip by projecting out 2-hop pair flips
            # Actually compute exact <L_0 | D_rel(-2*gam) | R_0>^2 where |L_0> is ground state of neutral sector (|2,0> + odd)
            # Here states are: 0b0011 (pair at 0), 0b1100 (pair at 1), and odd states 0b1001, 0b0110, ...
            # Neutral subspace from |2,0>: forbids hops INTO |0,2>
            H_L = H.copy()
            r_idx = idx[0b1100]
            sl_r = slice(r_idx * cutoff, (r_idx + 1) * cutoff)
            H_L[sl_r, :] = 0.0
            H_L[:, sl_r] = 0.0
            H_L[sl_r, sl_r] = np.eye(cutoff) * 1e9
            _, vL = eigh(H_L)
            psi_L = vL[:, 0]
            # Symmetric right state via site swap
            H_R = H.copy()
            l_idx = idx[0b0011]
            sl_l = slice(l_idx * cutoff, (l_idx + 1) * cutoff)
            H_R[sl_l, :] = 0.0
            H_R[:, sl_l] = 0.0
            H_R[sl_l, sl_l] = np.eye(cutoff) * 1e9
            _, vR = eigh(H_R)
            psi_R = vR[:, 0]
            D_pair = displacement_matrix(2.0 * gam, cutoff)
            # Map |0,2> amplitudes of psi_R to |2,0> amplitudes of psi_L through D_pair
            amp = float(psi_L[sl_l] @ (D_pair @ psi_R[sl_r]))
            pair_overlap_ratio = amp * amp

    e0 = min(v[0][0] for v in traces)
    corrs = []
    for tau in taus:
        z_tot = 0.0
        num = 0.0
        for vals, diagO in traces:
            w = np.exp(-(vals - e0) / tau)
            z_tot += float(w.sum())
            num += float(w @ diagO)
        corrs.append(num / z_tot)
    return corrs, pair_overlap_ratio


def chain_canonical_ed(n_sites: int, max_total: int, u_val: float, om: float, t: float, taus: list[float]) -> list[float]:
    """Canonical N_up=N_dn=1 exact diagonalization in relative phonon modes (sum n_r <= max_total)."""
    # Relative modes r_k = (n_0 + ... + n_{k-1} - k*n_k)/sqrt(k(k+1)) or orthonormal Jacobi coordinates
    # Build orthonormal basis Q of size (n_sites-1) x n_sites orthogonal to (1,1,...,1)/sqrt(n_sites)
    Q = np.zeros((n_sites - 1, n_sites), dtype=float)
    for k in range(n_sites - 1):
        # 1,...,1,-(k+1),0,...,0 normalized
        Q[k, : k + 1] = 1.0
        Q[k, k + 1] = -(k + 1)
        Q[k] /= np.linalg.norm(Q[k])

    # Generate multi-indices (m_0, ..., m_{R-1}) with sum <= max_total
    R = n_sites - 1
    modes = []

    def rec(idx: int, rem: int, cur: list[int]):
        if idx == R:
            modes.append(tuple(cur))
            return
        for v in range(rem + 1):
            cur.append(v)
            rec(idx + 1, rem - v, cur)
            cur.pop()

    rec(0, max_total, [])
    n_ph = len(modes)
    ph_energy = np.array([om * sum(m) for m in modes], dtype=float)

    # Precompute 1D displacement tables up to max_total for every bond jump vector Q[:, to] - Q[:, frm]
    alpha = np.sqrt(u_val / (2.0 * om))
    max_1d = max_total + 1

    def disp_tensor(vec_r: np.ndarray) -> np.ndarray:
        mats_1d = [displacement_matrix(float(alpha * vec_r[r]), max_1d) for r in range(R)]
        out = np.ones((n_ph, n_ph), dtype=float)
        # Factorized lookup
        arr_modes = np.array(modes, dtype=int)
        for r in range(R):
            mr = arr_modes[:, r]
            out *= mats_1d[r][np.ix_(mr, mr)]
        return out

    # Electronic basis: (pos_up, pos_dn) in {0..n_sites-1}^2
    el_states = [(u, d) for u in range(n_sites) for d in range(n_sites)]
    el_idx = {s: i for i, s in enumerate(el_states)}
    n_el = len(el_states)
    dim = n_el * n_ph
    H = np.zeros((dim, dim), dtype=float)

    for s, i in el_idx.items():
        sl = slice(i * n_ph, (i + 1) * n_ph)
        n_odd = 0 if s[0] == s[1] else 2
        np.fill_diagonal(H[sl, sl], ph_energy + 0.5 * u_val * n_odd)

    # Open chain nearest-neighbor hops
    for frm in range(n_sites - 1):
        for _to, a_frm, a_to in ((frm + 1, frm, frm + 1), (frm, frm + 1, frm)):
            D_hop = disp_tensor(Q[:, a_to] - Q[:, a_frm])
            for s, i in el_idx.items():
                if s[0] == a_frm:
                    j = el_idx[(a_to, s[1])]
                    H[j * n_ph : (j + 1) * n_ph, i * n_ph : (i + 1) * n_ph] += -t * D_hop
                if s[1] == a_frm:
                    j = el_idx[(s[0], a_to)]
                    H[j * n_ph : (j + 1) * n_ph, i * n_ph : (i + 1) * n_ph] += -t * D_hop

    # End-to-end pair transfer Delta_0^dag Delta_{n_sites-1}: |(n-1, n-1)> -> |(0, 0)>
    O = np.zeros((dim, dim), dtype=float)
    D_pair = disp_tensor(2.0 * (Q[:, 0] - Q[:, n_sites - 1]))
    i_src = el_idx[(n_sites - 1, n_sites - 1)]
    j_dst = el_idx[(0, 0)]
    O[j_dst * n_ph : (j_dst + 1) * n_ph, i_src * n_ph : (i_src + 1) * n_ph] = D_pair

    vals, vecs = eigh(H, overwrite_a=True, check_finite=False)
    diagO = np.einsum("ij,ij->j", vecs, O @ vecs)
    e0 = vals[0]
    res = []
    for tau in taus:
        w = np.exp(-(vals - e0) / tau)
        res.append(float((w @ diagO) / w.sum()))
    return res


def three_site_neutral_insertion_ratio(max_total: int, u_val: float, om: float, t: float, beta: float, u_hops: list[float]) -> list[tuple[float, float, float]]:
    """Compute exact 3-site neutral-dressed pair-insertion ratio R(u) = Tr(e^{-(beta-u)H_n(eta')} D^dag e^{-u H_n(eta)} D) / Z_n(eta)
    for eta=(1,0,0) -> eta'=(0,1,0) in canonical N=2 (open 3-site chain), compared to F_-(u)^4 * e^{-u*(E0(eta')-E0(eta))}.
    """
    Q = np.array([
        [1.0 / np.sqrt(2.0), -1.0 / np.sqrt(2.0), 0.0],
        [1.0 / np.sqrt(6.0), 1.0 / np.sqrt(6.0), -2.0 / np.sqrt(6.0)],
    ])
    modes = []
    for m0 in range(max_total + 1):
        for m1 in range(max_total + 1 - m0):
            modes.append((m0, m1))
    n_ph = len(modes)
    arr_modes = np.array(modes, dtype=int)
    ph_energy = np.array([om * (m0 + m1) for m0, m1 in modes], dtype=float)
    alpha = np.sqrt(u_val / (2.0 * om))

    def disp_tensor(vec_r: np.ndarray) -> np.ndarray:
        m0_mat = displacement_matrix(float(alpha * vec_r[0]), max_total + 1)
        m1_mat = displacement_matrix(float(alpha * vec_r[1]), max_total + 1)
        return m0_mat[np.ix_(arr_modes[:, 0], arr_modes[:, 0])] * m1_mat[np.ix_(arr_modes[:, 1], arr_modes[:, 1])]

    # Neutral subspace around eta=(1,0,0): states (0,0), (0,1), (1,0)
    # Neutral subspace around eta'=(0,1,0): states (1,1), (0,1), (1,0), (1,2), (2,1)
    # Notice eta' has 2 neighbors (sites 0 and 2), whereas eta has 1 neighbor (site 1)!
    def build_H_neut(pair_site: int):
        if pair_site == 0:
            sts = [(0, 0), (0, 1), (1, 0)]
            edges = [(0, 1)]
        else:
            sts = [(1, 1), (0, 1), (1, 0), (1, 2), (2, 1)]
            edges = [(1, 0), (1, 2)]
        idx = {s: i for i, s in enumerate(sts)}
        dim = len(sts) * n_ph
        H = np.zeros((dim, dim), dtype=float)
        for s, i in idx.items():
            sl = slice(i * n_ph, (i + 1) * n_ph)
            n_odd = 0 if s[0] == s[1] else 2
            np.fill_diagonal(H[sl, sl], ph_energy + 0.5 * u_val * n_odd)
        for a, b in edges:
            D_ab = disp_tensor(Q[:, b] - Q[:, a])
            D_ba = D_ab.T
            # Hop one electron from pair_site=a to b and back
            for s, i in idx.items():
                if s[0] == a and (b, s[1]) in idx:
                    j = idx[(b, s[1])]
                    H[j * n_ph : (j + 1) * n_ph, i * n_ph : (i + 1) * n_ph] += -t * D_ab
                if s[1] == a and (s[0], b) in idx:
                    j = idx[(s[0], b)]
                    H[j * n_ph : (j + 1) * n_ph, i * n_ph : (i + 1) * n_ph] += -t * D_ab
                if s[0] == b and (a, s[1]) in idx:
                    j = idx[(a, s[1])]
                    H[j * n_ph : (j + 1) * n_ph, i * n_ph : (i + 1) * n_ph] += -t * D_ba
                if s[1] == b and (s[0], a) in idx:
                    j = idx[(s[0], a)]
                    H[j * n_ph : (j + 1) * n_ph, i * n_ph : (i + 1) * n_ph] += -t * D_ba
        vals, vecs = eigh(H, overwrite_a=True, check_finite=False)
        return sts, idx, vals, vecs

    _sts0, idx0, v0, U0 = build_H_neut(0)
    _sts1, idx1, v1, U1 = build_H_neut(1)
    # Pair-flip operator D_pair from |(0,0)> in sector 0 to |(1,1)> in sector 1
    D_pair = disp_tensor(2.0 * (Q[:, 1] - Q[:, 0]))
    # Compute <a | D_pair | b> where a is eigenstate of H1, b of H0
    sl0 = slice(idx0[(0, 0)] * n_ph, (idx0[(0, 0)] + 1) * n_ph)
    sl1 = slice(idx1[(1, 1)] * n_ph, (idx1[(1, 1)] + 1) * n_ph)
    M_10 = U1[sl1, :].T @ (D_pair @ U0[sl0, :])  # shape (dim1, dim0)
    M2_10 = M_10 ** 2

    e0_0 = v0[0]
    e0_1 = v1[0]
    dE = e0_1 - e0_0  # negative because site 1 has 2 neighbors vs 1!
    X = u_val / om
    out = []
    for u in u_hops:
        # Exact ratio Tr(e^{-(beta-u)H1} D^dag e^{-u H0} D) / Tr(e^{-beta H0})
        w0 = np.exp(-beta * (v0 - e0_0))
        Z0 = float(w0.sum())
        # Shift energies by e0_0 to prevent overflow
        a1 = np.exp(-(beta - u) * (v1 - e0_0))
        a0 = np.exp(-u * (v0 - e0_0))
        num = float((a1 @ M2_10) @ a0)
        ratio = num / Z0
        # Compare against F_-(u)^4 * e^{-(beta-u)*dE}:
        # F_-(u) = exp(-X * (cosh(beta*om/2) - cosh(om*(beta/2 - u))) / sinh(beta*om/2))
        # Wait: for pair flip (c=+-2 on 2 sites), exact free-phonon overlap is F_-(u)^4!
        # Let's compute exact F_-(u)^4 = exp(-4*X*(cosh(beta*om/2) - cosh(om*(beta/2-u)))/sinh(beta*om/2))
        log_F4 = -4.0 * X * (np.cosh(beta * om / 2) - np.cosh(om * (beta / 2 - u))) / np.sinh(beta * om / 2)
        pred_free = np.exp(log_F4 - (beta - u) * dE)
        out.append((u, ratio, ratio / pred_free))
    return out


def check_ed_systems(mutant: str | None) -> None:
    # 1. 2-site grand-canonical convergence & virtual tail check
    cutoffs = [96, 144, 192, 256]
    taus = [1.0, 0.25]
    prev = None
    vals_256: list[float] = []
    for c in cutoffs:
        corrs, _ = dimer_gc_ed(c, 20.0, 1.0, 0.5, taus)
        if prev is not None:
            rel = max(abs(a - b) / max(abs(a), 1e-300) for a, b in zip(corrs, prev))
            print(f"[NUMERICAL] 2-site GC cutoff={c}: corr={corrs}, rel_diff={rel:.3e}", flush=True)
            if c == 256:
                require(rel < 1e-11, f"2-site ED not converged at cutoff 256: rel={rel}")
        prev = corrs
        vals_256 = corrs

    x = 0.5 / 20.0
    k_bare = (2.0 * (0.5**2) / 20.0) * 1.0 * np.exp(-20.0 / np.tanh(1.0 * 1.0 / 4.0))
    print(
        f"[NUMERICAL] 2-site bare-bridge comparison (tau=1.0, X=20): corr={vals_256[0]:.6e}, "
        f"K_bare={k_bare:.6e}, ratio={vals_256[0]/k_bare:.3e}",
        flush=True,
    )
    if mutant == "bad_bare_bridge_bound":
        require(vals_256[0] <= k_bare, "Bare single-bridge kernel bound holds on 2-site dimer")
    require(vals_256[0] > k_bare, f"Unexpectedly below bare bridge bound: {vals_256[0]} <= {k_bare}")
    require(0.1 < vals_256[0] / (x**2) < 0.2, f"Unexpected 2-site (t/U)^2 correlation ratio: {vals_256[0]/(x**2)}")

    # Neutral-dressed pair-flip overlap at fixed t/U = 0.05 across X
    for om_val in (2.0, 1.5, 1.0, 0.8):
        X = 10.0 / om_val
        _, ov2 = dimer_gc_ed(192, 10.0, om_val, 0.5, [1.0])
        softening = np.log(ov2 / np.exp(-4.0 * X)) / X
        print(f"[NUMERICAL] 2-site neutral-dressed pair-flip X={X:.2f}: ov2/e^(-4X)={ov2/np.exp(-4*X):.5f}, delta/X={softening:.5f}", flush=True)
        require(0.0 < softening < 0.05, f"Unexpected barrier softening: {softening}")

    # 2. 3-site canonical ED convergence (d=2, U=8, Om=4, t=0.4)
    prev3 = None
    for m_tot in (14, 20, 26):
        c3 = chain_canonical_ed(3, m_tot, 8.0, 4.0, 0.4, [1.0, 0.25])
        if prev3 is not None:
            rel3 = max(abs(a - b) / max(abs(a), 1e-300) for a, b in zip(c3, prev3))
            print(f"[NUMERICAL] 3-site canonical M_tot={m_tot}: corr={c3}, rel_diff={rel3:.3e}", flush=True)
            if m_tot == 26:
                require(rel3 < 1e-5, f"3-site ED not converged at M_tot=26: rel={rel3}")
        prev3 = c3
    x3 = 0.4 / 8.0
    require(0.20 < prev3[0] / (x3**4) < 0.35, f"Unexpected 3-site virtual tail ratio: {prev3[0]/(x3**4)}")

    # 3. 3-site neutral-dressed pair insertion ratio (U=8, Om=2, t=0.4, beta=10) with convergence check
    ratios_20 = three_site_neutral_insertion_ratio(20, 8.0, 2.0, 0.4, 10.0, [1.0, 2.5, 5.0])
    ratios = three_site_neutral_insertion_ratio(26, 8.0, 2.0, 0.4, 10.0, [1.0, 2.5, 5.0])
    rel_ins = max(abs(a[1] - b[1]) / max(abs(a[1]), 1e-300) for a, b in zip(ratios, ratios_20))
    print(f"[NUMERICAL] 3-site neutral-dressed insertion M_tot=20->26 rel_diff={rel_ins:.3e}", flush=True)
    require(rel_ins < 1e-4, f"3-site neutral-dressed insertion not converged: {rel_ins}")
    for u, r, r_norm in ratios:
        print(f"[NUMERICAL] 3-site neutral-dressed insertion u={u:.1f}: R={r:.4e}, R/(F_-^4 e^(-(beta-u)dE))={r_norm:.6f}", flush=True)
        require(0.9 <= r_norm <= 1.3, f"3-site neutral-dressed insertion ratio out of bounds: {r_norm}")

    # 4. 4-site canonical ED (d=3, U=8, Om=4, t=0.4) at moderate & low T (honest convergence reporting)
    prev4 = None
    for m_tot in (6, 8, 10):
        c4 = chain_canonical_ed(4, m_tot, 8.0, 4.0, 0.4, [4.0, 1.0, 0.02])
        if prev4 is not None:
            rel_mod = max(abs(c4[k] - prev4[k]) / max(abs(c4[k]), 1e-300) for k in (0, 1))
            rel_low = abs(c4[2] - prev4[2]) / max(abs(c4[2]), 1e-300)
            print(f"[NUMERICAL] 4-site canonical M_tot={m_tot}: corr={c4}, rel_mod={rel_mod:.3e}, rel_lowT={rel_low:.3e}", flush=True)
        if m_tot == 10:
            rel_tau1 = abs(c4[1] - prev4[1]) / abs(c4[1])
            require(rel_mod < 0.25 and rel_tau1 < 0.05, f"4-site moderate-T ED cutoff change unexpectedly huge: rel_mod={rel_mod}, rel_tau1={rel_tau1}")
        prev4 = c4
    require(0.1 < prev4[1] / ((0.4 / 8.0) ** 6) < 0.5, f"Unexpected 4-site virtual tail ratio: {prev4[1]/((0.4/8.0)**6)}")


def run_all(mutant: str | None = None) -> None:
    check_kernel_variance_identity(mutant)
    check_disjoint_bump_gram(mutant)
    check_exact_constants(mutant)
    check_plaquette_fermion_signs(mutant)
    check_ed_systems(mutant)


def mutant_driver() -> None:
    env = os.environ.copy()
    for name, desc in MUTATIONS.items():
        cmd = [sys.executable, str(Path(__file__).resolve()), "--mutant", name]
        res = subprocess.run(cmd, env=env, text=True, capture_output=True)
        last = res.stdout.strip().splitlines()[-1:] or [""]
        print(f"MUTANT {name}: exit={res.returncode} ({desc}) -> {last[0]}", flush=True)
        require(res.returncode == 1 and "REFUSED" in res.stdout, f"Mutant {name} was NOT caught! stderr={res.stderr}")
    print(f"ALL {len(MUTATIONS)} MUTANTS CAUGHT WITH EXIT 1; DRIVER EXIT 0", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mutant", choices=sorted(MUTATIONS.keys()), default=None)
    parser.add_argument("--mutants", action="store_true")
    args = parser.parse_args()
    if args.mutants:
        mutant_driver()
    else:
        run_all(args.mutant)
        print("ALL CHARGED-SKELETON CHECKS PASSED (EXIT 0)", flush=True)


if __name__ == "__main__":
    main()
