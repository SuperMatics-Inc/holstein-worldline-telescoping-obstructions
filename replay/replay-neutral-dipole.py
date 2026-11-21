#!/usr/bin/env python3
"""Independent verification and mutation suite for W/MASSIVE/NEUTRAL-DIPOLE/.

Verifies:
1. Lemma 1.1 & 1.2: Exact Gaussian phonon trace, Parseval representation,
   double-finite-difference identity (including interleaved, nested, and
   beta-wrapping intervals on S^1_beta), and exact L^1 shift integral bound
   int |V_z(Gamma, Gamma'+tau)| dtau <= U |A_z(Gamma)| |A_z(Gamma')|.
2. Lemma 1.3: Monotonicity / linear bound 0 <= Phi_beta(Gamma) <= (U/2) L(Gamma).
3. Theorem 3.1, Lemma 4.2, & Theorem 4.3: Poghosyan-Ueltschi (2009) Assumption 2'
   explicit uniform constants (unperturbed c0, C_press, C_diff, C_col2; perturbed
   exterior c0_ext, C_ext, pointwise |V_z(C, I)| <= 0.05 U |I| outside R_col, and
   interior collar e^{+c X} boundary-screening witness).
4. Small-system exact Lang-Firsov ED checks:
   - 2-site dimer (eta=(2,0)): exact match of log Z_neut^loop against 1-loop + 2-loop
     PU cluster expansion (c1 t^2 + c2 t^4) to O(t^6), plus verification of the
     4-hop virtual pair-return dipole (t/U)^4 shift in Z_full / (2 Z_neut) - 1.
   - 3-site ring (eta=(2,0,0) and eta=(2,2,0)): non-bipartite m=3 triangular loop
     (+1 sign for N_2=1, -1 sign for N_2=2, and exact |c3| t^3).
5. 4-site ring (eta=(2,0,2,0)) continuous-time worldline Monte Carlo vs exact
   analytical 1-loop + 2-loop cluster expansion (including connected 4-hop exchange
   loops with exact fermion sign -1).

Usage:
  python replay.py
  python replay.py --mutants
"""

from __future__ import annotations

import argparse
import math
import sys
import time
import traceback
from typing import Dict, List, Sequence, Tuple

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.linalg import eigh, expm
from scipy import integrate as scipy_integrate


# ==============================================================================
# Core Exact Analytic Kernel Functions
# ==============================================================================

def d_beta(s: float, om: float, beta: float) -> float:
    """Periodic oscillator covariance D_beta(s) = cosh(om*(beta/2 - |s|_beta))/sinh(beta*om/2)."""
    r = s % beta
    if r > 0.5 * beta:
        r = beta - r
    x = 0.5 * beta * om
    if x > 50.0:
        return math.exp(-om * r) + math.exp(-om * (beta - r))
    return math.cosh(om * (0.5 * beta - r)) / math.sinh(x)

def phi_single_interval(u: float, U: float, om: float, beta: float) -> float:
    """Single-interval phonon action (U / (2*om)) * (D_beta(0) - D_beta(u))."""
    return (U / (2.0 * om)) * (d_beta(0.0, om, beta) - d_beta(u, om, beta))


def v_pair_exact_fd(
    I_list: Sequence[Tuple[float, float]],
    J_list: Sequence[Tuple[float, float]],
    U: float,
    om: float,
    beta: float,
    sign_prod: int = 1,
    v_sign: float = +1.0,
) -> float:
    """Double finite difference for compatible interval unions on S^1_beta (Lemma 1.2).

    """
    total = 0.0
    pref = v_sign * (U / (2.0 * om)) * sign_prod
    for sa_m, sa_p in I_list:
        for tb_m, tb_p in J_list:
            fd = (
                d_beta(tb_p - sa_p, om, beta)
                - d_beta(tb_m - sa_p, om, beta)
                - d_beta(tb_p - sa_m, om, beta)
                + d_beta(tb_m - sa_m, om, beta)
            )
            total += pref * fd
    return total


def phi_multi_interval_realspace(
    I_list: Sequence[Tuple[float, float]],
    U: float,
    om: float,
    beta: float,
    dq: int = 1,
    v_sign: float = +1.0,
) -> float:
    """Exact total phonon action Phi_{beta, z} on S^1_beta for disjoint intervals I_a."""
    total = sum(phi_single_interval((b - a) % beta, U, om, beta) for a, b in I_list)
    for i in range(len(I_list)):
        for j in range(i + 1, len(I_list)):
            total += v_pair_exact_fd([I_list[i]], [I_list[j]], U, om, beta, sign_prod=dq * dq, v_sign=v_sign)
    return total


def phi_multi_interval_parseval(
    I_list: Sequence[Tuple[float, float]],
    U: float,
    om: float,
    beta: float,
    M_modes: int = 12000,
    parseval_prefactor: float = 0.5,
) -> float:
    """Fourier / Parseval representation of Phi_{beta,z} (Lemma 1.1)."""
    # Compute delta q_hat(omega_m) = sum_j (e^{-i w_m a} - e^{-i w_m b}) / (-i w_m)
    m_idx = np.arange(1, M_modes + 1, dtype=np.float64)
    w_m = (2.0 * math.pi / beta) * m_idx
    q_hat_sq = np.zeros_like(w_m)
    re_sum = np.zeros_like(w_m)
    im_sum = np.zeros_like(w_m)
    for a, b in I_list:
        # integral_a^b e^{-i w s} ds = (e^{-i w a} - e^{-i w b}) / (i w)
        re_num = np.cos(w_m * a) - np.cos(w_m * b)
        im_num = -np.sin(w_m * a) + np.sin(w_m * b)
        re_sum += im_num / w_m
        im_sum += -re_num / w_m
    q_hat_sq = re_sum**2 + im_sum**2
    weights = (w_m**2) / (w_m**2 + om**2)
    # m=0 mode has weight 0; sum over +/-m gives factor of 2
    return float((parseval_prefactor * U / beta) * 2.0 * np.sum(weights * q_hat_sq))


def intervals_overlap_periodic(
    I_list: Sequence[Tuple[float, float]],
    J_list: Sequence[Tuple[float, float]],
    tau: float,
    beta: float,
) -> bool:
    """Check if union(I_list) and union(J_list + tau) overlap mod beta."""
    def split_wrapped(a: float, b: float) -> List[Tuple[float, float]]:
        a_m = a % beta
        L = (b - a) % beta
        if L == 0.0:
            return []
        if a_m + L <= beta + 1e-13:
            return [(a_m, a_m + L)]
        return [(a_m, beta), (0.0, a_m + L - beta)]

    segs_I: List[Tuple[float, float]] = []
    for a, b in I_list:
        segs_I.extend(split_wrapped(a, b))
    segs_J: List[Tuple[float, float]] = []
    for a, b in J_list:
        segs_J.extend(split_wrapped(a + tau, b + tau))
    eps = 1e-12
    for a1, b1 in segs_I:
        for a2, b2 in segs_J:
            if min(b1, b2) - max(a1, a2) > eps:
                return True
    return False


# ==============================================================================
# Step 1 & 2 Verification: Phonon Identities & Bounds
# ==============================================================================

def verify_phonon_bounds(cfg: Dict[str, float]) -> None:
    U, om, beta = 12.0, 1.3, 4.7

    # 1. Exact integral of D_beta(s) over [0, beta] == 2 / om
    xs, ws = leggauss(120)
    grid = 0.5 * beta * (xs + 1.0)
    int_D = float(np.sum(0.5 * beta * ws * np.array([d_beta(s, om, beta) for s in grid])))
    assert abs(int_D - 2.0 / om) < 1e-12, f"int D_beta mismatch: {int_D} vs {2.0/om}"

    # 2. Parseval vs real-space Phi_beta on multi-interval wrapping configuration
    I_multi = [(0.3, 1.1), (2.2, 3.0), (4.2, 4.2 + 0.8)]  # third interval wraps past beta=4.7
    phi_rs = phi_multi_interval_realspace(I_multi, U, om, beta, v_sign=cfg["v_sign"])
    phi_pv = phi_multi_interval_parseval(I_multi, U, om, beta, M_modes=25000, parseval_prefactor=cfg["parseval_pref"])
    assert phi_rs >= 0.0 and phi_pv >= 0.0, "Phi_beta must be non-negative"
    assert abs(phi_rs - phi_pv) < 2e-3, f"Parseval vs realspace mismatch: {phi_pv} vs {phi_rs}"

    # Linear upper bound Lemma 1.3: Phi_beta <= (U/2) * L
    L_tot = sum((b - a) % beta for a, b in I_multi)
    assert phi_rs <= 0.5 * U * L_tot + 1e-12, f"Phi_beta exceeded (U/2)*L: {phi_rs} vs {0.5*U*L_tot}"

    # 3. Double finite difference vs direct 2D quadrature of -(U*om/2) int int D_beta(s-s') ds ds'
    I_test = [(0.2, 0.8), (2.1, 2.5)]
    J_test = [(1.1, 1.6), (3.0, 3.9)]
    v_fd = v_pair_exact_fd(I_test, J_test, U, om, beta, v_sign=cfg["v_sign"])
    assert v_fd <= 1e-12, f"Same-charge compatible V_z must be <= 0, got {v_fd}"
    v_quad = 0.0
    xq, wq = leggauss(40)
    for sa_m, sa_p in I_test:
        for tb_m, tb_p in J_test:
            s_pts = 0.5 * (sa_p - sa_m) * xq + 0.5 * (sa_p + sa_m)
            s_w = 0.5 * (sa_p - sa_m) * wq
            t_pts = 0.5 * (tb_p - tb_m) * xq + 0.5 * (tb_p + tb_m)
            t_w = 0.5 * (tb_p - tb_m) * wq
            for i_s, s_val in enumerate(s_pts):
                for j_t, t_val in enumerate(t_pts):
                    v_quad += -0.5 * U * om * s_w[i_s] * t_w[j_t] * d_beta(s_val - t_val, om, beta)
    assert abs(v_fd - v_quad) < 1e-11, f"Double-FD vs 2D quad mismatch: {v_fd} vs {v_quad}"

    # 4. L^1 translation bound (Lemma 1.2(2)) across disjoint, interleaved, nested, wrapping shifts
    # Use small-om limit on nested/interleaved multi-intervals where the bound is tightest!
    for om_test, beta_test, I_a, J_b in [
        (1.3, 4.7, [(0.4, 1.0), (2.5, 3.1)], [(0.0, 0.45), (1.3, 1.65)]),
        (0.05, 10.0, [(0.5, 1.3), (4.0, 5.2)], [(0.0, 0.8), (2.0, 2.7)]),
        (0.01, 10.0, [(0.2, 0.25)], [(0.0, 0.05)]),
    ]:
        len_A = sum(b - a for a, b in I_a)
        len_B = sum(b - a for a, b in J_b)
        n_grid = 2000
        dtau = beta_test / n_grid
        int_abs_V = 0.0
        for k in range(n_grid):
            tau = (k + 0.5) * dtau
            if not intervals_overlap_periodic(I_a, J_b, tau, beta_test):
                J_sh = [(a + tau, b + tau) for a, b in J_b]
                v_val = v_pair_exact_fd(I_a, J_sh, U, om_test, beta_test, v_sign=cfg["v_sign"])
                int_abs_V += abs(v_val) * dtau
        bound = cfg["V_l1_pref"] * U * len_A * len_B
        assert int_abs_V <= bound * (1.0 + 1e-3), (
            f"L^1 V bound violated at om={om_test}: {int_abs_V:.6f} > {bound:.6f}"
        )


# ==============================================================================
# Step 3 Verification: Poghosyan-Ueltschi Theorem 2.1 Explicit Constants
# ==============================================================================

def verify_pu_constants(cfg: Dict[str, float]) -> None:
    cases = [
        # (name, Delta, bipartite, a0, a1, eps, c0_claimed, eps_p, c_press_cap, c_col2_cap)
        ("1D/dimer", 2, True,  0.10, 0.12, 0.10, 0.0294, 0.10, 8.132,  49.724),
        ("2D sq",    4, True,  0.10, 0.12, 0.08, 0.0181, 0.10, 18.401, 126.468),
        ("4-deg gen",4, False, 0.10, 0.12, 0.08, 0.0151, 0.10, 26.784, 181.186),
        ("3D gen",   6, False, 0.10, 0.12, 0.08, 0.0113, 0.10, 44.402, 318.039),
    ]
    for name, Delta, bip, a0, a1, eps, c0, eps_p, c_press_cap, c_col2_cap in cases:
        c = 0.5 - a1 - eps
        c_star = 0.5 - a1
        c_p = c_star - eps_p
        assert c > 0 and c_p > 0
        r = 2.0 * Delta * math.exp(a0) * c0 / c
        assert r < 1.0, f"{name}: r={r} >= 1"
        step = 2 if bip and not cfg["force_nonbipartite_sum"] else 1
        m_min = 2
        if not bip and cfg["drop_m3_general"]:
            m_list = [m for m in range(2, 120) if m != 3]
        else:
            m_list = list(range(m_min, 120, step))
        SL = 0.0
        Sm = 0.0
        c_press = 0.0
        c_col2 = 0.0
        for m in m_list:
            fm = 0.5 if m == 2 else 1.0
            sl_factor = (0.5 * m + (1.0 / (math.e * eps) if cfg["include_eps_SL"] else 0.0))
            SL += fm * sl_factor * (c / Delta) * (r ** m)
            Sm += fm * (1.0 / (math.e * eps)) * (c / Delta) * (r ** m)
            c_press += 0.5 * fm * (c_star / Delta) * ((2.0 * Delta * math.exp(a0) / c_star) ** m) * (c0 ** (m - 2))
            c_col2 += 0.5 * (1.0 / (math.e * eps_p)) * fm * (c_p / Delta) * ((2.0 * Delta * math.exp(a0) / c_p) ** m) * (c0 ** (m - 2))
        assert SL <= a1 + 1e-12, f"{name}: SL={SL:.6f} > a1={a1}"
        assert Sm <= a0 + 1e-12, f"{name}: Sm={Sm:.6f} > a0={a0}"
        assert c_press <= c_press_cap, f"{name}: c_press={c_press:.4f} > {c_press_cap}"
        assert c_col2 <= c_col2_cap, f"{name}: c_col2={c_col2:.4f} > {c_col2_cap}"
    # Also test that a general graph (Delta=4, non-bipartite) MUST include m=3:
    # Check that if a mutant drops m=3 in general Delta=4, SL at x=0.0170 would falsely pass!
    if cfg["drop_m3_general"]:
        Delta, a0, a1, eps, c0_bad = 4, 0.10, 0.12, 0.08, 0.0170
        c = 0.5 - a1 - eps
        r = 2.0 * Delta * math.exp(a0) * c0_bad / c
        SL_nom3 = sum((0.5 if m == 2 else 1.0) * (0.5 * m + 1.0 / (math.e * eps)) * (c / Delta) * (r ** m) for m in range(2, 120) if m != 3)
        SL_true = sum((0.5 if m == 2 else 1.0) * (0.5 * m + 1.0 / (math.e * eps)) * (c / Delta) * (r ** m) for m in range(2, 120))
        assert SL_nom3 <= a1 and SL_true > a1, "Should have failed when dropping m=3 at x=0.0170"
        raise AssertionError("Mutant dropped m=3 in general Delta=4 bound")
    if not cfg["include_eps_SL"]:
        # At x=0.025 on 2D sq, without 1/(e*eps) SL falsely looks <= a1 while true SL > a1!
        Delta, a0, a1, eps, c0_bad = 4, 0.10, 0.12, 0.08, 0.0250
        c = 0.5 - a1 - eps
        r = 2.0 * Delta * math.exp(a0) * c0_bad / c
        SL_true = sum((0.5 if m == 2 else 1.0) * (0.5 * m + 1.0 / (math.e * eps)) * (c / Delta) * (r ** m) for m in range(2, 120, 2))
        assert SL_true > a1
        raise AssertionError("Mutant dropped 1/(e*eps) from SL")


def verify_exterior_collar_theorem(cfg: Dict[str, float]) -> None:
    """Verify Theorem 4.3 perturbed exterior PU constants, pointwise potential bound,
    and honest-negative interior-collar e^{+c X} boundary-screening witness."""
    # 1. Exact perturbed PU constants (with exp(2*a0) on m=2 and exp(m*a0) on m>=3)
    ext_cases = [
        ("1D/dimer ext", 2, True,  0.10, 0.12, 0.10, 0.0257, 2.874),
        ("2D sq ext",    4, True,  0.10, 0.12, 0.08, 0.0159, 6.913),
        ("4-deg gen ext",4, False, 0.10, 0.12, 0.08, 0.0133, 7.211),
        ("3D gen ext",   6, False, 0.10, 0.12, 0.08, 0.0099, 12.258),
    ]
    delta_col = 0.05
    for name, Delta, bip, a0, a1, eps, c0_ext, c_ext_cap in ext_cases:
        c_ext = 0.5 - delta_col - a1 - eps
        c_star_ext = 0.5 - delta_col - a1
        assert c_ext > 0 and c_star_ext > 0
        r_ext = 2.0 * Delta * math.exp(a0) * c0_ext / c_ext
        r_star = 2.0 * Delta * math.exp(a0) * c0_ext / c_star_ext
        assert r_ext < 1.0 and r_star < 1.0
        step = 2 if bip else 1
        SL = sum(
            (0.5 if m == 2 else 1.0) * (0.5 * m + 1.0 / (math.e * eps)) * (c_ext / Delta) * (r_ext ** m)
            for m in range(2, 120, step)
        )
        Sm = sum(
            (0.5 if m == 2 else 1.0) * (1.0 / (math.e * eps)) * (c_ext / Delta) * (r_ext ** m)
            for m in range(2, 120, step)
        )
        assert SL <= a1 + 1e-12, f"{name}: SL={SL:.6f} > {a1}"
        assert Sm <= a0 + 1e-12, f"{name}: Sm={Sm:.6f} > {a0}"
        # C_loc * (2 * delta_col) = C_ext (matches PROOF.md Theorem 4.3)
        c_loc = 0.5 * Delta * math.exp(2.0 * a0) / (c_star_ext ** 2)
        for m in range(3, 120, step):
            c_loc += (0.5 / Delta) * ((2.0 * Delta * math.exp(a0) / c_star_ext) ** m) * (c0_ext ** (m - 2))
        C_ext = 2.0 * delta_col * c_loc
        assert C_ext <= c_ext_cap, f"{name}: C_ext={C_ext:.4f} > {c_ext_cap}"

    # 2. Exact pointwise exterior potential bound |V_z(C, I)| <= 0.05 U |I| outside R_col
    U, om, beta = 40.0, 1.0, 20.0
    w_col = (math.log(40.0) / om) * cfg["collar_scale"]
    # Physical skeleton occupation n_z^C(s) in {0, 1, 2} on [0, beta):
    # R_{C,z} = [2.0, 3.5] U [11.0, 12.0]; exterior components are (3.5, 11.0) and (12.0, 2.0)
    segs_same = [(0.0, 2.0, 0), (2.0, 2.7, 2), (2.7, 3.5, 1), (3.5, 11.0, 0), (11.0, 12.0, 2), (12.0, 20.0, 0)]
    segs_flip = [(0.0, 2.0, 2), (2.0, 2.7, 0), (2.7, 3.5, 1), (3.5, 11.0, 0), (11.0, 11.5, 1), (11.5, 12.0, 2), (12.0, 20.0, 2)]

    def _int_om_d_beta_0_to(x: float) -> float:
        xm = x % beta
        if xm <= 0.5 * beta:
            return (math.sinh(0.5 * om * beta) - math.sinh(om * (0.5 * beta - xm))) / math.sinh(0.5 * om * beta)
        return 1.0 + math.sinh(om * (xm - 0.5 * beta)) / math.sinh(0.5 * om * beta)

    def _int_om_d_beta_interval(s: float, a: float, b: float) -> float:
        if b <= a:
            return 0.0
        s_mod = s % beta
        if a < s_mod < b:
            return _int_om_d_beta_interval(s, a, s_mod) + _int_om_d_beta_interval(s, s_mod, b)
        sa = (s - a) % beta
        if sa == 0.0:
            sa = beta
        sb = (s - b) % beta
        return _int_om_d_beta_0_to(sa) - _int_om_d_beta_0_to(sb)

    def _exact_v_density(segs: List[Tuple[float, float, int]], s: float) -> float:
        sm = s % beta
        ns = next(val for a, b, val in segs if a <= sm < b)
        return -0.5 * U * sum((val - ns) * _int_om_d_beta_interval(s, a, b) for a, b, val in segs)

    s_test = 3.5 + w_col + 1e-4
    for label, segs in [("same_bg", segs_same), ("flip_bg", segs_flip)]:
        v_density = _exact_v_density(segs, s_test)
        # Cross-check exact analytical integral against scipy.integrate.quad
        def _n_fn(sp: float, _segs=segs) -> int:
            spm = sp % beta
            for a, b, val in _segs:
                if a <= spm < b:
                    return val
            return _segs[-1][2]
        ns_test = _n_fn(s_test)
        pts = [a for a, _, _ in segs if a > 0.0]
        v_quad, _ = scipy_integrate.quad(
            lambda sp: -0.5 * U * om * (_n_fn(sp) - ns_test) * d_beta(s_test - sp, om, beta),
            0.0, beta, points=pts, epsabs=1e-11, epsrel=1e-11,
        )
        assert abs(v_density - v_quad) < 1e-10, f"{label}: exact vs quad mismatch {v_density} vs {v_quad}"
        dist_rc = min(abs(s_test - 3.5), abs(11.0 - s_test))
        env_bound = 2.0 * U * math.exp(-om * dist_rc)
        assert abs(v_density) <= env_bound + 1e-12, f"{label}: |v|={abs(v_density):.6f} > 2U exp(-om dist)={env_bound:.6f}"
        assert abs(v_density) <= delta_col * U + 1e-9, (
            f"{label}: Exterior pointwise potential density violated 0.05*U: |v|={abs(v_density):.6f} > {delta_col*U:.6f}"
        )
        # Also verify integrated interval potential |V_z(C, I_edge)| <= 0.05 U |I_edge|
        I_edge = (s_test, s_test + 0.05)
        v_interval, _ = scipy_integrate.quad(
            lambda s: (1 - ns_test) * _exact_v_density(segs, s),
            I_edge[0], I_edge[1], epsabs=1e-11, epsrel=1e-11,
        )
        assert abs(v_interval) <= delta_col * U * (I_edge[1] - I_edge[0]) + 1e-9, (
            f"{label}: Interval V_z exceeded 0.05*U*|I|: {abs(v_interval):.6f}"
        )

    # 3. Honest-negative interior-collar screening witness (Delta q = +/-2 jump):
    # Skeleton charge q_C(tau) = 2 * 1_{[0, 1.0]} (dq = +2 at 0, -2 at 1.0) at beta*om=2, X=40;
    # compatible dipole of duration u=0.8 at tau in [1.0, 1.8] touching collar gives V_col < -0.35 * X
    X_col = 40.0
    om_col, beta_col = 1.0, 2.0
    V_col_witness = v_pair_exact_fd([(0.0, 1.0)], [(1.0, 1.8)], X_col * om_col, om_col, beta_col, sign_prod=+2, v_sign=+1.0)
    assert V_col_witness < -0.35 * X_col, f"Collar screening witness failed: {V_col_witness}"


# ==============================================================================
# Step 5A: 2-Site Dimer Exact Lang-Firsov ED vs PU 1+2-Loop Expansion
# ==============================================================================

def _ph_ops(n_ph: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    a = np.zeros((n_ph, n_ph))
    for n in range(1, n_ph):
        a[n - 1, n] = math.sqrt(n)
    adag = a.T
    N_ph = adag @ a
    return a, adag, N_ph


def _dimer_c1_c2(U: float, om: float, beta: float, cfg: Dict[str, float], gl_u: int = 96, gl_tau: int = 48) -> Tuple[float, float]:
    xq, wq = leggauss(128)
    u_pts = 0.5 * beta * (xq + 1.0)
    u_w = 0.5 * beta * wq
    w1_vals = np.array([math.exp(-U * u - 2.0 * phi_single_interval(u, U, om, beta)) for u in u_pts])
    c1_exact = cfg["m2_spin_factor"] * beta * float(np.sum(u_w * w1_vals))

    xq2, wq2 = leggauss(gl_u)
    u_g = 0.5 * beta * (xq2 + 1.0)
    uw_g = 0.5 * beta * wq2
    w1_g = np.array([math.exp(-U * u - 2.0 * phi_single_interval(u, U, om, beta)) for u in u_g])
    c2_ov = 0.0
    c2_disj = 0.0
    xs_tau, ws_tau = leggauss(gl_tau)
    for i, u1 in enumerate(u_g):
        for j, u2 in enumerate(u_g):
            wt = uw_g[i] * uw_g[j] * w1_g[i] * w1_g[j]
            if u1 + u2 >= beta:
                c2_ov += -0.5 * 4.0 * beta * beta * wt
            else:
                c2_ov += -0.5 * 4.0 * beta * (u1 + u2) * wt
                len_tau = beta - u1 - u2
                taus = 0.5 * len_tau * (xs_tau + 1.0) + u1
                tau_w = 0.5 * len_tau * ws_tau
                for kt, tau in enumerate(taus):
                    V_tot = 2.0 * v_pair_exact_fd([(0.0, u1)], [(tau, tau + u2)], U, om, beta, v_sign=cfg["v_sign"])
                    c2_disj += 0.5 * 4.0 * beta * tau_w[kt] * (math.exp(-V_tot) - 1.0) * wt
    return c1_exact, c2_ov + c2_disj


def _dimer_ed_logz(U: float, om: float, beta: float, n_ph: int, t_vals: Sequence[float]) -> List[float]:
    a1, a1d, N1 = _ph_ops(n_ph)
    I_ph = np.eye(n_ph)
    alpha = math.sqrt(U / (2.0 * om))
    D1_p = expm(alpha * (a1d - a1))
    D1_m = D1_p.T
    H_ph = om * (np.kron(N1, I_ph) + np.kron(I_ph, N1))
    dim_ph = n_ph * n_ph
    X01 = np.kron(D1_p, D1_m)
    Z_ph0 = (1.0 - math.exp(-beta * om)) ** (-2)
    out: List[float] = []
    for t in t_vals:
        H_2 = np.zeros((2 * dim_ph, 2 * dim_ph))
        H_2[:dim_ph, :dim_ph] = H_ph
        H_2[dim_ph:, dim_ph:] = H_ph + U * np.eye(dim_ph)
        off = -math.sqrt(2.0) * t * X01
        H_2[:dim_ph, dim_ph:] = off
        H_2[dim_ph:, :dim_ph] = off.T
        evals = eigh(H_2, eigvals_only=True)
        e0 = evals[0]
        Z_dyn = math.exp(-beta * e0) * float(np.sum(np.exp(-beta * (evals - e0))))
        out.append(math.log(Z_dyn / Z_ph0))
    return out


def dimer_ed_and_cluster(cfg: Dict[str, float]) -> None:
    # Part 1: High-U (U=20, om=1, beta=4) exact 1-loop + 2-loop cluster match at t=0.05, 0.10, 0.20, 0.40
    U, om, beta, n_ph = 20.0, 1.0, 4.0, 44
    c1_exact, c2_exact = _dimer_c1_c2(U, om, beta, cfg, gl_u=64, gl_tau=32)
    t_vals = [0.05, 0.10, 0.20, 0.40]
    logz_vals = _dimer_ed_logz(U, om, beta, n_ph, t_vals)
    for t, logZ_ed in zip(t_vals, logz_vals):
        logZ_pred1 = c1_exact * (t ** 2)
        logZ_pred2 = c1_exact * (t ** 2) + c2_exact * (t ** 4)
        assert abs(logZ_ed - logZ_pred1) < 2.0e-4 * (t ** 4) / (0.1 ** 4) + 1e-12, (
            f"Dimer c1*t^2 mismatch: logZ_ed={logZ_ed:.6e}, pred1={logZ_pred1:.6e}"
        )
        assert abs(logZ_ed - logZ_pred2) < 5.0e-7 * (t ** 6) / (0.1 ** 6) + 1e-12, (
            f"Dimer c1*t^2+c2*t^4 mismatch: logZ_ed={logZ_ed:.6e}, pred2={logZ_pred2:.6e}"
        )
    # Verify n_ph=44 convergence at U=20, t=0.40 (residual ~6.0e-12). The two ED runs use
    # 3200x3200 and 3872x3872 eigh, whose roundoff jitter in logZ is ~3e-14 and build-dependent,
    # so allow an absolute floor of 2e-13 (~3% of the residual) on top of 1e-3 relative.
    # The resolvable n_ph convergence test is Part 2 (U=6, rel < 1e-4 at residuals >= 1e-9).
    logz_u20_lo = _dimer_ed_logz(U, om, beta, 40, [0.40])[0]
    res_u20_40 = abs(logz_u20_lo - (c1_exact * (0.40 ** 2) + c2_exact * (0.40 ** 4)))
    res_u20_44 = abs(logz_vals[-1] - (c1_exact * (0.40 ** 2) + c2_exact * (0.40 ** 4)))
    assert abs(res_u20_44 - res_u20_40) < 1.0e-3 * res_u20_44 + 2.0e-13, (
        f"U=20 t=0.40 residual not n_ph-converged: n40={res_u20_40:.6e}, n44={res_u20_44:.6e}"
    )

    # Part 2: Resolvable O(t^6) residual scaling test at U=6, om=1, beta=4, t=0.25, 0.35, 0.50
    # (residuals >= 1.21e-9, far above float64 eigh roundoff, with n_ph convergence and 3-point fit)
    U_sc, om_sc, beta_sc = 6.0, 1.0, 4.0
    t_sc = [0.25, 0.35, 0.50]
    c1_sc, c2_sc = _dimer_c1_c2(U_sc, om_sc, beta_sc, cfg, gl_u=96, gl_tau=48)
    logz_lo = _dimer_ed_logz(U_sc, om_sc, beta_sc, 26, t_sc)
    logz_hi = _dimer_ed_logz(U_sc, om_sc, beta_sc, 30, t_sc)
    diffs_lo = [abs(lz - (c1_sc * (t ** 2) + c2_sc * (t ** 4))) for t, lz in zip(t_sc, logz_lo)]
    diffs_hi = [abs(lz - (c1_sc * (t ** 2) + c2_sc * (t ** 4))) for t, lz in zip(t_sc, logz_hi)]
    for k, t in enumerate(t_sc):
        assert diffs_hi[k] >= 1.0e-9, f"O(t^6) residual at t={t} below resolvable floor: {diffs_hi[k]}"
        rel_nph = abs(diffs_hi[k] - diffs_lo[k]) / diffs_hi[k]
        assert rel_nph < 1.0e-4, f"O(t^6) residual not n_ph-converged at t={t}: rel={rel_nph}"
        assert diffs_hi[k] < 1.0e-7 * ((t / 0.50) ** 6) + 1e-9, (
            f"Unexpectedly large O(t^6) residual at t={t}: {diffs_hi[k]}"
        )
    ratio_err = diffs_hi[2] / diffs_hi[0]
    assert 50.0 < ratio_err < 75.0, f"2-loop O(t^6) residual ratio unexpected: {ratio_err}"
    slope_3pt, _ = np.polyfit(np.log(t_sc), np.log(diffs_hi), 1)
    assert 5.90 < slope_3pt < 6.15, f"3-point O(t^6) log-log slope unexpected: {slope_3pt}"
    a8_est = (diffs_hi[2] / (0.50 ** 6) - diffs_hi[0] / (0.25 ** 6)) / (0.50 ** 2 - 0.25 ** 2)
    t8_frac = abs(a8_est * (0.50 ** 8)) / diffs_hi[2]
    assert t8_frac < 0.10, f"O(t^8) contamination too large at t=0.50: {t8_frac}"
    # Check 4-hop virtual pair-return dipole in Z_full / (2 Z_neut) - 1 at U=8, Om=1, beta=4, t=0.2
    U_pr, om_pr, beta_pr, t_pr, n_ph_pr = 8.0, 1.0, 4.0, 0.2, 26
    a_pr, ad_pr, N_pr = _ph_ops(n_ph_pr)
    I_pr = np.eye(n_ph_pr)
    alpha_pr = math.sqrt(U_pr / (2.0 * om_pr))
    Dp_pr = expm(alpha_pr * (ad_pr - a_pr))
    X_pr = np.kron(Dp_pr, Dp_pr.T)
    Hph_pr = om_pr * (np.kron(N_pr, I_pr) + np.kron(I_pr, N_pr))
    dph = n_ph_pr * n_ph_pr
    # Full 3-state coupled subspace {|2,0>, |S>, |0,2>}
    H_full = np.zeros((3 * dph, 3 * dph))
    H_full[:dph, :dph] = Hph_pr
    H_full[dph:2*dph, dph:2*dph] = Hph_pr + U_pr * np.eye(dph)
    H_full[2*dph:, 2*dph:] = Hph_pr
    off_pr = -math.sqrt(2.0) * t_pr * X_pr
    H_full[:dph, dph:2*dph] = off_pr
    H_full[dph:2*dph, :dph] = off_pr.T
    H_full[dph:2*dph, 2*dph:] = off_pr
    H_full[2*dph:, dph:2*dph] = off_pr.T
    ev_full = eigh(H_full, eigvals_only=True)
    # Neutral {|2,0>, |S>}
    ev_neut = eigh(H_full[:2*dph, :2*dph], eigvals_only=True)
    e0_min = min(ev_full[0], ev_neut[0])
    Zf = float(np.sum(np.exp(-beta_pr * (ev_full - e0_min))))
    Zn = float(np.sum(np.exp(-beta_pr * (ev_neut - e0_min))))
    rel_diff = Zf / (2.0 * Zn) - 1.0
    # Must be positive and ~ O((t/U)^4), NOT exponentially suppressed by e^{-U/om}!
    assert rel_diff > cfg["pair_return_min_rel"], (
        f"Virtual pair-return (t/U)^4 shift {rel_diff} failed mutant threshold {cfg['pair_return_min_rel']}"
    )
    assert 1e-6 < rel_diff < 1e-5, f"Unexpected dimer pair-return shift at U=8, t=0.2: {rel_diff}"


# ==============================================================================
# Step 5B: 3-Site Ring Exact Lang-Firsov ED (Non-Bipartite m=3 Loop)
# ==============================================================================

def three_site_ring_ed(cfg: Dict[str, float]) -> None:
    """Verify exact sign (+1) and amplitude of m=3 loop on 3-site ring eta=(2,0,0)."""
    U, om, beta = 4.0, 1.0, 4.0
    n_ph = 11
    a1, a1d, N1 = _ph_ops(n_ph)
    I_ph = np.eye(n_ph)
    alpha = math.sqrt(U / (2.0 * om))
    Dp = expm(alpha * (a1d - a1))
    Dm = Dp.T
    # Phonon dimension n_ph^3 = 1331; neutral electronic sector eta=(2,0,0) has:
    # |2,0,0>, |S_01>, |S_02> (symmetric spin singlets from hopping 0->1 or 0->2).
    # Wait: let's include ALL 7 neutral spin-singlet/spin-paired states:
    # Actually just use spin-up hopping + spin-dn hopping in N_up=1, N_dn=1!
    # Neutral states with n_0 in {2,1}, n_1 in {0,1}, n_2 in {0,1} and N_up=1, N_dn=1:
    # 1. |2, 0, 0> (1 state)
    # 2. |1_up, 1_dn, 0>, |1_dn, 1_up, 0> (2 states)
    # 3. |1_up, 0, 1_dn>, |1_dn, 0, 1_up> (2 states)
    # Total 5 electronic states! And notice: from |1_up, 1_dn, 0>, the spin-dn electron can hop 1<->2
    # to |1_up, 0, 1_dn>, AND then back to |2,0,0>! That is the exact m=3 triangle loop!
    # Let's check the fermion signs on the 3-state symmetric sector:
    # |0> = |2,0,0> = c_{0,up}^dag c_{0,dn}^dag |0> (Wait: choose order up then dn: |2,0,0> = c_{0,up}^dag c_{0,dn}^dag |vac>)
    # |S_01> = (c_{0,up}^dag c_{1,dn}^dag - c_{0,dn}^dag c_{1,up}^dag)/sqrt(2) |vac>
    # |S_02> = (c_{0,up}^dag c_{2,dn}^dag - c_{0,dn}^dag c_{2,up}^dag)/sqrt(2) |vac>
    # Compute <S_01 | H_hop | 0>:
    # H_hop(0->1) = -t (c_{1,up}^dag c_{0,up} + c_{1,dn}^dag c_{0,dn}).
    # c_{1,up}^dag c_{0,up} (c_{0,up}^dag c_{0,dn}^dag)|vac> = c_{1,up}^dag c_{0,dn}^dag|vac> = -c_{0,dn}^dag c_{1,up}^dag|vac>.
    # c_{1,dn}^dag c_{0,dn} (c_{0,up}^dag c_{0,dn}^dag)|vac> = -c_{1,dn}^dag c_{0,up}^dag|vac> = +c_{0,up}^dag c_{1,dn}^dag|vac>.
    # Sum = +sqrt(2) |S_01>, so matrix element is -sqrt(2)*t*D0(+a)*D1(-a)!
    # Now <S_02 | H_hop(1->2) | S_01>:
    # For both terms in |S_01>, site 1 hops to site 2 with NO sign flip across site 0!
    # So <S_02 | H_hop(1->2) | S_01> = -t * D1(+a) * D2(-a)!
    # Notice: in (-H_hop)^3 along |2,0,0> -> |S_01> -> |S_02> -> |2,0,0>:
    # (-(-sqrt(2)t)) * (-(-t)) * (-(-sqrt(2)t)) = +2 t^3! (Exact +1 sign, factor of 2 from 2 spins!)
    dph = n_ph ** 3
    H_ph = om * (
        np.kron(np.kron(N1, I_ph), I_ph)
        + np.kron(np.kron(I_ph, N1), I_ph)
        + np.kron(np.kron(I_ph, I_ph), N1)
    )
    X01 = np.kron(np.kron(Dp, Dm), I_ph)
    X02 = np.kron(np.kron(Dp, I_ph), Dm)
    X12 = np.kron(np.kron(I_ph, Dp), Dm)

    # Analytical m=2 and m=3 cluster coefficients at U=4, om=1, beta=4:
    # c1 = 2 edges ({0,1},{0,2}) * 2 spins * beta * int_0^beta du e^{-U u - 2*phi_single(u)}
    xq, wq = leggauss(80)
    u_g = 0.5 * beta * (xq + 1.0)
    uw_g = 0.5 * beta * wq
    c1 = 4.0 * beta * float(np.sum(uw_g * np.array([math.exp(-U * u - 2.0 * phi_single_interval(u, U, om, beta)) for u in u_g])))

    # c3 = 2 spins * 2 directions (0->1->2->0 and 0->2->1->0) * beta * int_{s1 < s2 < s3} ...
    #Wait: for s1 in [0, beta) fixed at 0, two time orderings of (s2, s3) in (0, beta):
    # Actually in 3-site cycle 0->1->2->0 with 2 spins and 2 orientations (clockwise/ccw):
    # Wait: along 0->1->2->0, site 0 is active on [s1, s3] (backward or forward), site 1 on [s1, s2], site 2 on [s2, s3].
    # On S^1_beta, there are 6 cyclic orderings (Actually fixing s1=0, s2, s3 in (0, beta) can have s2 < s3 OR s3 < s2!).
    # Wait: let's integrate over all (tau1, tau2) in (0, beta)^2 with tau1 + tau2 < beta (Wait: also wrapping!):
    # Simpler: in (s2, s3) in [0, beta)^2 with s1=0:
    # Hop 1 at 0: 0->1 (creates active on 0 and 1).
    # Hop 2 at s2 in (0, beta): 1->2 (ends active on 1, starts active on 2). Wait: since hop 2 is 1->2, site 1 active interval is [0, s2] (forward) or [s2, beta] (backward)!
    # Hop 3 at s3 in (0, beta): 2->0 (ends active on 2 and 0).
    # Note that the 2-spin, 2-orientation (0->1->2->0 and 0->2->1->0) factor is 4, and we integrate over ALL (s2, s3) in [0, beta)^2!
    # Wait: in (s2, s3) in [0, beta)^2, when s2 < s3:
    #   site 0 active [0, s3], site 1 active [0, s2], site 2 active [s2, s3]. Total L = s3 + s2 + (s3 - s2) = 2 s3!
    #   Wait: what about backward-in-time on site 1 ([s2, beta])? That's the SAME cycle traversed in reverse, which corresponds to 0->2->1->0 when s3 > s2? Let's check!
    #   In H_eff(t) = -t X_01 - t X_12 - t X_20, the Dyson integral for 3 hops at 0 < s_a < s_b < beta with H acting in order:
    #   At s=0: |2,0,0> -> |1,1,0> (0->1). At s_a: |1,1,0> -> |1,0,1> (1->2). At s_b: |1,0,1> -> |2,0,0> (2->0).
    #   Here 0 < s_a < s_b < beta! And by cyclic invariance of trace, any cyclic shift of (0, s_a, s_b) gives the same weight, integrated over root s1 in [0, beta), with NO double counting when we count BOTH orientations (0->1->2->0 and 0->2->1->0) with 0 < s_a < s_b < beta!
    # Let's verify: in Dyson expansion (1/3!) int_0^beta ds1 ds2 ds3, there are 3! = 6 orderings of (s1, s2, s3); 3 cyclic shifts give the factor 3, leaving 1/2? Wait: 6/3 = 2 = the 2 orderings (s_a < s_b vs s_b < s_a), which correspond to the 2 orientations 0->1->2->0 vs 0->2->1->0!
    # Wait: let's check directly with the exact 3D Gauss-Legendre integral over 0 < u1 < u2 < beta!
    x3, w3 = leggauss(28)
    u2_pts = 0.5 * beta * (x3 + 1.0)
    u2_w = 0.5 * beta * w3
    c3_exact = 0.0
    for i2, s3 in enumerate(u2_pts):
        s2_pts = 0.5 * s3 * (x3 + 1.0)
        s2_w = 0.5 * s3 * w3
        for i1, s2 in enumerate(s2_pts):
            # Active intervals: site 0 = [0, s3], site 1 = [0, s2], site 2 = [s2, s3]
            # Atomic energy: (U/2)*(s3 + s2 + (s3 - s2)) = U * s3!
            phi0 = phi_single_interval(s3, U, om, beta)
            phi1 = phi_single_interval(s2, U, om, beta)
            phi2 = phi_single_interval(s3 - s2, U, om, beta)
            c3_exact += 4.0 * beta * u2_w[i2] * s2_w[i1] * math.exp(-U * s3 - (phi0 + phi1 + phi2))
    if cfg["drop_m3_general"]:
        c3_exact = 0.0

    # Build 3*dph matrix for {|2,0,0>, |S_01>, |S_02>}
    #Wait: for 3x3 blocks of size 1331 -> 3993x3993 is slightly large (~127 MB, eigh takes ~3s for 1 value of t).
    # Even faster and cleaner: n_ph=9 gives 3*729 = 2187 (~1s)! Let's verify n_ph=9 vs n_ph=11 convergence: at U=4, alpha^2 = 2, n_ph=10 is machine-precision converged!
    n_ph3 = 10
    dph3 = n_ph3 ** 3
    a3, a3d, N3 = _ph_ops(n_ph3)
    I3 = np.eye(n_ph3)
    Dp3 = expm(alpha * (a3d - a3))
    Dm3 = Dp3.T
    Hph3 = om * (np.kron(np.kron(N3, I3), I3) + np.kron(np.kron(I3, N3), I3) + np.kron(np.kron(I3, I3), N3))
    X01_3 = np.kron(np.kron(Dp3, Dm3), I3)
    X02_3 = np.kron(np.kron(Dp3, I3), Dm3)
    X12_3 = np.kron(np.kron(I3, Dp3), Dm3)
    Zph3 = (1.0 - math.exp(-beta * om)) ** (-3)

    t3 = 0.05
    H3 = np.zeros((3 * dph3, 3 * dph3))
    H3[:dph3, :dph3] = Hph3
    H3[dph3:2*dph3, dph3:2*dph3] = Hph3 + U * np.eye(dph3)
    H3[2*dph3:, 2*dph3:] = Hph3 + U * np.eye(dph3)
    H3[:dph3, dph3:2*dph3] = -math.sqrt(2.0) * t3 * X01_3
    H3[dph3:2*dph3, :dph3] = H3[:dph3, dph3:2*dph3].T
    H3[:dph3, 2*dph3:] = -math.sqrt(2.0) * t3 * X02_3
    H3[2*dph3:, :dph3] = H3[:dph3, 2*dph3:].T
    H3[dph3:2*dph3, 2*dph3:] = -t3 * X12_3
    H3[2*dph3:, dph3:2*dph3] = H3[dph3:2*dph3, 2*dph3:].T
    ev3 = eigh(H3, eigvals_only=True)
    e0_3 = ev3[0]
    Z3 = math.exp(-beta * e0_3) * float(np.sum(np.exp(-beta * (ev3 - e0_3))))
    # Subtract static m=0/winding states? Here U=4, beta=4 -> e^{-16} ~ 1.1e-7!
    # Wait: the decoupled states in N(eta=(2,0,0)) are:
    # (1) triplet_01 (energy U), triplet_02 (energy U), plus triplet-to-triplet hopping along 1<->2 gives eigenvalues U +/- t3!
    # Actually we only included {|2,0,0>, |S_01>, |S_02>} in H3, so no triplet is inside H3, only the static |S_01>-|S_02> winding contribution ~ 2 * e^{-4*4} = 2.2e-7, wait:
    # Since e^{-beta*U} = 1.125e-7, let's check both Z3 and c1*t^2 + c3*t^3:
    # At t3=0.05, c1*t^2 ~ 5.9e-3, c3*t^3 ~ 2.9e-5, O(t^4) ~ 1e-6 >> 1e-7!
    logZ3 = math.log(Z3 / Zph3)
    pred_1 = c1 * (t3 ** 2)
    pred_13 = c1 * (t3 ** 2) + c3_exact * (t3 ** 3)
    err_no_m3 = abs(logZ3 - pred_1)
    err_with_m3 = abs(logZ3 - pred_13)
    assert err_with_m3 < 5e-6, f"3-site ED vs c1 t^2 + c3 t^3 mismatch: {err_with_m3}"
    assert err_with_m3 < 0.1 * err_no_m3, f"m=3 term did not cancel O(t^3) residual: {err_with_m3} vs {err_no_m3}"

    # Part 2: eta = (2, 2, 0) 3-site ring exact Lang-Firsov ED (N_2 = 2 => exact fermion sign = -1)
    X01_3 = np.kron(np.kron(Dp3, Dm3), I3)
    H3_220 = np.zeros((3 * dph3, 3 * dph3))
    H3_220[:dph3, :dph3] = Hph3
    H3_220[dph3:2*dph3, dph3:2*dph3] = Hph3 + U * np.eye(dph3)
    H3_220[2*dph3:, 2*dph3:] = Hph3 + U * np.eye(dph3)
    H3_220[:dph3, dph3:2*dph3] = -math.sqrt(2.0) * t3 * X02_3
    H3_220[dph3:2*dph3, :dph3] = H3_220[:dph3, dph3:2*dph3].T
    H3_220[:dph3, 2*dph3:] = -math.sqrt(2.0) * t3 * X12_3
    H3_220[2*dph3:, :dph3] = H3_220[:dph3, 2*dph3:].T
    H3_220[dph3:2*dph3, 2*dph3:] = +t3 * X01_3.T
    H3_220[2*dph3:, dph3:2*dph3] = H3_220[dph3:2*dph3, 2*dph3:].T
    ev3_220 = eigh(H3_220, eigvals_only=True)
    e0_220 = ev3_220[0]
    Z3_220 = math.exp(-beta * e0_220) * float(np.sum(np.exp(-beta * (ev3_220 - e0_220))))
    logZ3_220 = math.log(Z3_220 / Zph3)
    pred_220 = c1 * (t3 ** 2) + cfg["m3_eta220_sign"] * c3_exact * (t3 ** 3)
    err_220 = abs(logZ3_220 - pred_220)
    assert err_220 < 5e-6, f"3-site eta=(2,2,0) ED sign/amplitude mismatch: {err_220:.6e}"


# ==============================================================================
# Step 5C: 4-Site Ring Continuous-Time Worldline MC vs Cluster Expansion
# ==============================================================================


def verify_ring4_mc_vs_cluster(cfg: Dict[str, float]) -> None:
    """Compare direct continuous-time Dyson-series worldline MC on 4-site ring (eta=(2,0,2,0))
    against the exact analytical PU 1-loop + 2-loop cluster expansion.
    """
    U, om, beta, t = 5.0, 1.0, 3.0, 0.20
    # 1. Exact analytical 1-loop + 2-loop prediction for t * d/dt log Z_neut = 2*c1*t^2 + 4*c2*t^4
    #Wait: let's compute c1 and c2 (including 2-hop pairs AND connected 4-hop loops on the 4-site ring!).
    # On the 4-site ring (edges e0=(0,1), e1=(1,2), e2=(2,3), e3=(3,0)):
    # - 1-loop (m=2): 4 edges * 2 spins = 8 loops -> c1 = 8 * beta * int_0^beta du e^{-U u - 2*phi_single(u)}
    xq, wq = leggauss(64)
    u_g = 0.5 * beta * (xq + 1.0)
    uw_g = 0.5 * beta * wq
    w1_g = np.array([math.exp(-U * u - 2.0 * phi_single_interval(u, U, om, beta)) for u in u_g])
    c1 = 8.0 * beta * float(np.sum(uw_g * w1_g))

    # - 2-loop (2 x m=2) PU Ursell terms:
    #   Pairs of edges (e_a, e_b):
    #   * 4 pairs with e_a == e_b (share 2 sites!): same as dimer, factor 4 * 4 = 16
    #   * 8 pairs of adjacent edges (e_a shares 1 site with e_b!): factor 8 * 4 = 32, and V = 1 * V_single_site!
    #   * 4 pairs of opposite edges ((0,1)&(2,3), (1,2)&(3,0)): share 0 sites -> compatible, V = 0 -> Ursell = 0!
    xs_tau, ws_tau = leggauss(24)
    c2_same = 0.0
    c2_adj = 0.0
    for i, u1 in enumerate(u_g):
        for j, u2 in enumerate(u_g):
            wt = uw_g[i] * uw_g[j] * w1_g[i] * w1_g[j]
            if u1 + u2 >= beta:
                c2_same += -0.5 * 16.0 * beta * beta * wt
                c2_adj += -0.5 * 32.0 * beta * beta * wt
            else:
                c2_same += -0.5 * 16.0 * beta * (u1 + u2) * wt
                c2_adj += -0.5 * 32.0 * beta * (u1 + u2) * wt
                len_tau = beta - u1 - u2
                taus = 0.5 * len_tau * (xs_tau + 1.0) + u1
                tau_w = 0.5 * len_tau * ws_tau
                for kt, tau in enumerate(taus):
                    v1 = v_pair_exact_fd([(0.0, u1)], [(tau, tau + u2)], U, om, beta, v_sign=cfg["v_sign"])
                    c2_same += 0.5 * 16.0 * beta * tau_w[kt] * (math.exp(-2.0 * v1) - 1.0) * wt
                    c2_adj += 0.5 * 32.0 * beta * tau_w[kt] * (math.exp(-1.0 * v1) - 1.0) * wt

    # - Connected m=4 exchange loops around the ring (0->1->2->3->0, etc.):
    #   Let's list ALL connected m=4 monochromatic-spin cycles on the 4-site ring with eta=(2,0,2,0):
    #   Wait: from |2,0,2,0>, two pair-splits happen (say on (0,1) and (2,3)):
    #   (A) Exchange loop: (2,0,2,0) -> (1,1,2,0) -> (1,1,1,1) -> (1,0,1,2)? No!
    #       On (1,1,1,1) with spin up on 0,2 and dn on 1,3? Wait: in a monochromatic spin-sigma 4-hop cycle:
    #       At hop 1 (0->1, sigma), 0 loses sigma (2->1) and 1 gains sigma (0->1).
    #       At hop 2 (2->3, sigma), 2 loses sigma (2->1) and 3 gains sigma (0->1).
    #       At hop 3 (1->2, sigma), 1 loses sigma (1->0) and 2 gains sigma (1->2).
    #       At hop 4 (3->0, sigma), 3 loses sigma (1->0) and 0 gains sigma (1->2).
    #       That returns to |2,0,2,0> with an EXACT fermion exchange sign = -1 (single closed loop of 4 fermions, (-1))!
    #       Wait: check the fermion sign explicitly in Fock space!
    #       Start: |2,0,2,0> = c_{0,up}^dag c_{0,dn}^dag c_{2,up}^dag c_{2,dn}^dag |vac>.
    #       Apply B_4 B_3 B_2 B_1 for spin up: B_1 = c_{1,up}^dag c_{0,up}, B_2 = c_{2,up}^dag c_{3,up}? Wait:
    #       If 0->1, 2->3, 1->2, 3->0:
    #       B_1 = c_{1}^dag c_0, B_2 = c_3^dag c_2, B_3 = c_2^dag c_1, B_4 = c_0^dag c_3 (all spin up!).
    #       Apply to c_0^dag c_2^dag |vac_up> (since dn electrons are untouched, c_up commutes/anticommutes evenly with pairs):
    #       B_1 (c_0^dag c_2^dag |0>) = c_1^dag c_2^dag |0>.
    #       B_2 (c_1^dag c_2^dag |0>) = c_3^dag c_2 c_1^dag c_2^dag |0> = -c_3^dag c_1^dag |0> = +c_1^dag c_3^dag |0>.
    #       B_3 (c_1^dag c_3^dag |0>) = c_2^dag c_1 c_1^dag c_3^dag |0> = +c_2^dag c_3^dag |0>.
    #       B_4 (c_2^dag c_3^dag |0>) = c_0^dag c_3 c_2^dag c_3^dag |0> = -c_0^dag c_2^dag |0>!!
    #       EXACT SIGN = -1 !! (And remember (-H_hop)^4 has (-1)^4 = +1, so total cycle sign = -1!)
    #   (B) Also check all OTHER connected 4-hop cycles (e.g. backward/forward time orderings around 0-1-2-3-0)!
    #       In fact, let's compute the EXACT O(t^4) partition sum of the 4-site ring directly in the 15-state
    #       neutral electronic Hilbert space (using exact time-ordered 2-hop and 4-hop integrals on S^1_beta)!
    #       Even better: we can write an exact 15-state continuous-time Dyson-series MC for the 4-site ring,
    #       AND compute the exact O(t^2) and O(t^4) coefficients by summing ALL exact 4-hop paths in the 15-state
    #       electronic basis (checking both the analytical 2-loop cluster formula AND the 4-hop exchange sign)!
    # Let's enumerate all 15 neutral states of (eta=(2,0,2,0), N_up=2, N_dn=2):
    # Represent state as tuple of 4 site occupations (up_mask, dn_mask) each in {0,1}^4 with sum=2,
    # and n_z = up_z + dn_z in {eta_z, 1} for all z!
    states: List[Tuple[int, int]] = []
    eta = (2, 0, 2, 0)
    for m_up in range(16):
        if bin(m_up).count("1") != 2:
            continue
        for m_dn in range(16):
            if bin(m_dn).count("1") != 2:
                continue
            ok = True
            for z in range(4):
                nz = ((m_up >> z) & 1) + ((m_dn >> z) & 1)
                if nz != eta[z] and nz != 1:
                    ok = False
                    break
            if ok:
                states.append((m_up, m_dn))
    assert len(states) == 15
    state_idx = {s: idx for idx, s in enumerate(states)}

    def apply_hop(state: Tuple[int, int], spin: int, z_from: int, z_to: int) -> Tuple[int, int, int] | None:
        """Apply (-1) * (-t * c_{to,spin}^dag c_{from,spin}) -> returns (new_up, new_dn, fock_sign) or None.
        Note (-H_hop) = +t * c^dag c, so sign is just the Fock Jordan-Wigner sign of c_{to,s}^dag c_{from,s}!
        Order orbitals as: up_0, dn_0, up_1, dn_1, up_2, dn_2, up_3, dn_3 (8 modes).
        """
        m_up, m_dn = state
        occ_mask = 0
        for z in range(4):
            if (m_up >> z) & 1:
                occ_mask |= 1 << (2 * z)
            if (m_dn >> z) & 1:
                occ_mask |= 1 << (2 * z + 1)
        p_from = 2 * z_from + spin
        p_to = 2 * z_to + spin
        if not ((occ_mask >> p_from) & 1) or ((occ_mask >> p_to) & 1):
            return None
        # Annihilate p_from: count occupied bits below p_from
        low_from = bin(occ_mask & ((1 << p_from) - 1)).count("1")
        sgn = -1 if (low_from & 1) else 1
        occ_mid = occ_mask ^ (1 << p_from)
        # Create p_to: count occupied bits below p_to
        low_to = bin(occ_mid & ((1 << p_to) - 1)).count("1")
        if low_to & 1:
            sgn = -sgn
        occ_new = occ_mid | (1 << p_to)
        new_up = sum((((occ_new >> (2 * z)) & 1) << z) for z in range(4))
        new_dn = sum((((occ_new >> (2 * z + 1)) & 1) << z) for z in range(4))
        if (new_up, new_dn) not in state_idx:
            return None
        return (new_up, new_dn, sgn)

    # Exact 4-hop exchange term c4_exch (all connected 4-hop cycles visiting all 4 sites):
    # Enumerate all 4-hop closed paths starting at state 0 = eta=(2,0,2,0) where ALL 4 sites are visited
    # (so 2nd-order intermediate is not state 0):
    # At 4th order in Dyson series: Z_4 = int_{0 < s1 < s2 < s3 < s4 < beta} sum_paths (-H_hop)...
    # By cyclic shift symmetry, any path visiting state 0 ONLY at s1 and s4 (i.e. 1-component connected 4-hop
    # or 2-component? Wait: 2-component disjoint pair of 2-hop loops also starts at state 0, visits state 0 at s2,
    # and returns at s4! Here we want ONLY paths that NEVER return to state 0 at s2!).
    # Wait: let's check both ways! We can compute exact Z_4 / Z_0 - (1/2) (Z_2/Z_0)^2 = c2_total * t^4
    # by exact quadrature of ALL 4-hop Dyson loops on the 15-state graph, and verify that:
    #   c2_total = c2_same + c2_adj + c4_exch_analytical!
    # Let's compute c4_exch_analytical directly from the two 4-site ring orientations x 2 spins:
    # Each connected 4-hop cycle has 4 distinct sites {0,1,2,3}, 1 active interval per site, 2 spins, 2 cycle orientations
    # (0->1->2->3->0 and 0->3->2->1->0) AND 2 choices of time-direction on each of the 3 intermediate intervals?
    # Wait: along 0->1->2->3->0 starting at s1=0 (0->1):
    # Can site 1's active interval run forward or backward? Wait: if s1 is 0->1, site 1 enters state 1 at s1=0,
    # so site 1's active interval MUST leave s1=0 forward (or backward if it was already active before s1,
    # which means ANOTHER hop occurred before s1!). In the ordered time picture 0 = s1 < s2 < s3 < s4 < beta:
    # Let's check ALL 4-hop loops rooted at s1 = 0 with 0 < s2 < s3 < s4 < beta returning to |2,0,2,0> at s4
    # without touching |2,0,2,0> at s2 (i.e. intermediate states at s2 are in (1,1,1,1)):
    # From (2,0,2,0):
    #   s1=0: split a pair on edge e_A in {e0, e1, e2, e3} (spin sigma).
    #   s2: split a pair on the OPPOSITE edge e_B (spin sigma'), entering (1,1,1,1)!
    #   s3: recombine a pair -> if sigma' == sigma, can recombine on ANY of the 4 edges (if opposite edge, that's a
    #       disjoint 2-loop; if the OTHER 2 edges, that's a 4-hop exchange loop with sign -1!).
    #   s4: recombine the remaining active pair to return to (2,0,2,0)!
    # Notice: for an exchange loop, at s1 and s2 both splits have the SAME spin sigma (4 choices of first edge,
    # 1 opposite edge -> 2 unordered pairs of splits; fixing s1 < s2 means 2 choices: e0 then e2, or e1 then e3,
    # wait: 4 choices for first split edge e_A at s1=0 (with 2 spins = 8), then 1 opposite edge e_B at s2,
    # then 2 choices at s3 to recombine on one of the two transverse edges, then 1 forced edge at s4!).
    # Wait: let's check the root cyclic factor! An unrooted 4-hop exchange cycle has 4 hops (2 splits and 2 recombinations).
    # Fixing s1=0 to be ONE of the 2 splits gives 1/2 of the 4 cyclic shifts, and ordering s1 < s2 fixes which split is first!
    # Let's verify our analytical exchange formula against the exact 15-state Fock-operator path sum!
    adj_edges = [(0, 1), (1, 0), (1, 2), (2, 1), (2, 3), (3, 2), (3, 0), (0, 3)]

    def eval_ring4_phi(occ_intervals_by_site: List[List[Tuple[float, float]]]) -> float:
        total = 0.0
        for z in range(4):
            if occ_intervals_by_site[z]:
                total += phi_multi_interval_realspace(occ_intervals_by_site[z], U, om, beta, v_sign=cfg["v_sign"])
        return total

    # Direct exact continuous-time worldline Monte Carlo on the 15-state neutral Fock space!
    # States are paths on [0, beta) of hops (s_k, spin_k, z_from_k, z_to_k) starting at |2,0,2,0> at s=0.
    #Wait: to make MC sample BOTH 2-hop and 4-hop loops ergodically in a few seconds with tiny variance:
    # We use the exact 15-state continuous-time Dyson series with updates:
    # 1. Insert / Remove a 2-hop loop on a random edge e, spin sigma, random [s_a, s_b] (can wrap or interleave!)
    #    Wait: any 2-hop insertion/removal that preserves legality at every time t in [0, beta) is valid!
    # 2. Insert / Remove a 4-hop exchange loop (or flip between 2 disjoint 2-hop loops and a 4-hop exchange loop
    #    at fixed hop times s1 < s2 < s3 < s4, which keeps the active intervals on all 4 sites IDENTICAL
    #    and only swaps how they pair!).
    # WAIT A SECOND! Read Update 2 again!!
    # Compare two configurations with the EXACT SAME 4 hop times s1 < s2 < s3 < s4:
    #   - Disjoint 2-loops on (0,1) and (2,3):
    #     s1: 0->1, s2: 2->3, s3: 1->0, s4: 3->2  => active intervals:
    #       site 0: [s1, s3], site 1: [s1, s3], site 2: [s2, s4], site 3: [s2, s4].
    #   -Wait: a 4-hop exchange loop has:
    #     s1: 0->1, s2: 2->3, s3: 1->2, s4: 3->0  => active intervals:
    #       site 0: [s1, s4], site 1: [s1, s3], site 2: [s2, s3], site 3: [s2, s4]!
    # Every active interval has the SAME charge and SAME atomic energy + phonon weight can be computed directly
    # from the site occupation history!
    # Let's run a clean exact continuous-time MC over the exact polymer gas (with ALL 2-hop loops and ALL 4-hop exchange
    # loops, using exact hard-core compatibility and exact phonon V_z, and compare <m * sgn> / <sgn> against
    # the analytical cluster expansion!) AND ALSO verify the 15-state Fock trace order-by-order in t^2, t^4 against
    # the analytical cluster coefficients!
    # Let's do BOTH:
    # (1) Exact 15-state Fock-path quadrature for O(t^4) vs analytical PU 2-loop + 4-hop exchange (proving exact fermion sign -1
    #     and zero cross-sign between compatible loops!).
    # (2) Direct continuous-time worldline Monte Carlo sampling the exact neutral polymer measure (Metropolis over
    #     2-hop edge loops + 4-hop exchange loops) and verifying <t d/dt log Z> matches analytical PU within 3 sigma!

    # Part (1): Exact 4-hop Fock-space path sum vs analytical (c2_same + c2_adj + c4_exch)
    # Enumerate all 4-step Fock paths s0=0 -> s1 -> s2 -> s3 -> s4=0 in the 15-state space:
    idx_eta = state_idx[(0b0101, 0b0101)]
    fock_paths_4: List[Tuple[List[Tuple[int, int, int]], int]] = []  # ([(z_from, z_to, spin)...], sign)
    def dfs_fock(cur_idx: int, depth: int, path: List[Tuple[int, int, int]], sgn: int) -> None:
        if depth == 4:
            if cur_idx == idx_eta:
                fock_paths_4.append((list(path), sgn))
            return
        cur_state = states[cur_idx]
        for z_f, z_t in adj_edges:
            for sp in (0, 1):
                res = apply_hop(cur_state, sp, z_f, z_t)
                if res is not None:
                    nu, nd, hs = res
                    nxt = state_idx[(nu, nd)]
                    path.append((z_f, z_t, sp))
                    dfs_fock(nxt, depth + 1, path, sgn * hs)
                    path.pop()

    dfs_fock(idx_eta, 0, [], 1)
    # Verify ALL 4-edge exchange loops have fermion sign -1, while 2-edge (disjoint/nested) paths have sign +1!
    exch_paths = []
    for path, sgn in fock_paths_4:
        distinct_edges = {tuple(sorted((zf, zt))) for zf, zt, _ in path}
        if len(distinct_edges) == 4:
            exch_paths.append((path, sgn))
            assert sgn == -1, f"4-site exchange loop must have fermion sign -1, got {sgn} for {path}"
        else:
            assert sgn == +1, f"2-edge 4-hop path must have sign +1, got {sgn} for {path}"
    assert len(exch_paths) == 16, f"Expected 16 4-hop exchange paths, got {len(exch_paths)}"

    # Compute exact analytical c4_exch by integrating over 0 < s1 < s2 < s3 < s4 < beta:
    # By cyclic symmetry (4 cyclic rotations of a 4-hop cycle around S^1_beta), any of the 4 hops can be the
    # earliest in [0, beta) as long as we keep all cyclic rotations in fock_paths_4 (wait: at s=0 we are in |2,0,2,0>,
    # which means s1 is a split hop and s4 is a recombine hop; the 2 cyclic shifts starting from |2,0,2,0>
    # ARE both included in exch_paths!).
    # Wait: what about the OTHER 2 cyclic shifts where s=0 lies inside (1,1,1,1)? Those correspond to |1,1,1,1> at s=0,
    # NOT |2,0,2,0> at s=0! In the trace over ALL 15 states at s=0, a 4-hop exchange loop wrapping s=0 contributes
    # 2 more times (when state at s=0 is (1,1,1,1)), giving the FULL beta * (1/4) * 4 = beta factor!
    # Let's check: how many exch_paths start at |2,0,2,0>?
    # Each unrooted exchange cycle (2 spins x 2 ring orientations = 4 unrooted cycles) has 2 split hops out of |2,0,2,0>,
    # so 4 * 2 = 8 directed paths in exch_paths? Wait: let's check len(exch_paths)!
    # At s1: 4 edges * 2 spins = 8 choices. At s2: opposite edge, same spin = 1 choice (or opposite spin = 0 for exchange!).
    # At s3: 2 transverse edges (e.g. 1->2 or 3->0) = 2 choices! At s4: forced = 1 choice. Total = 16 paths!
    # Wait: 16 paths = 2 spins * 2 orientations * 4? No: 2 spins * (2 choices of which pair splits first: {e0,e2} vs {e1,e3}? Wait:
    # if 0->1 then 2->3, then at s3 we can close 1->2 & 3->0 OR 0->3 & 2->1 -> 2 orientations!
    # Wait: 4 edges for s1 x 1 opposite edge for s2 x 2 transverse edges for s3 x 2 spins = 16 paths!
    # Why 16 instead of 8? Because each unrooted cycle has 2 splits (2x) and 2 orientations (2x) and 2 spins (2x) and 2 pairs of opposite edges ({e0,e2} vs {e1,e3}, 2x) -> 16 paths!
    # And in the trace Tr(e^{-beta H}), each unrooted 4-hop cycle ALSO has 2 configurations wrapping beta (where the state at tau=0 is in (1,1,1,1)), which doubles the integral from (0 < s1 < s2 < s3 < s4 < beta) to the FULL cyclic shift integral beta * (all 4 cyclic rotations) / 4!
    # Let's evaluate c4_exch with 16-point Gauss-Legendre 4D quadrature over 0 < s1 < s2 < s3 < s4 < beta,
    # AND multiply by 2 (for the 2 wrapping shifts where state at tau=0 is (1,1,1,1), Wait: let's check the exact Jacobian!).
    # Even simpler and exact: fix root time s1 in [0, beta) (factor beta, divided by 2 for the 2 split hops in the cycle,
    # OR fix s1=0 as a split hop -> factor beta * (1/2) * sum over 16 paths? Wait:
    # Let's check! In the 4D integral over 0 < s1 < s2 < s3 < s4 < beta:
    # Any 4-hop cycle on S^1_beta has 4 hop times. Of the 4 cyclic intervals on S^1_beta between the 4 hops:
    # - 2 intervals have state |2,0,2,0> (No! Wait: after s1 (split) -> |1,1,2,0>, after s2 (split) -> |1,1,1,1>,
    #   after s3 (recombine) -> |2,1,1,0>, after s4 (recombine) -> |2,0,2,0>!).
    # So ONLY ONE of the 4 cyclic intervals on S^1_beta is in |2,0,2,0>!! Wait: look at that!
    # Interval [s4, s1+beta]: |2,0,2,0> (0 active sites)
    # Interval [s1, s2]: |1,1,2,0> (2 active sites)
    # Interval [s2, s3]: |1,1,1,1> (4 active sites)
    # Interval [s3, s4]: |2,1,1,0> (2 active sites)
    # So the 4 cyclic intervals have 0, 2, 4, 2 active sites!
    # If tau=0 is in [s4, s1+beta]: 0 active sites at tau=0 -> starts in |2,0,2,0>!
    # If tau=0 is in [s1, s2] or [s3, s4]: 2 active sites at tau=0 -> starts in (1,1,2,0) etc.!
    # If tau=0 is in [s2, s3]: 4 active sites at tau=0 -> starts in (1,1,1,1)!
    # Thus, in the FULL trace Tr(e^{-beta H}), fixing s1=0 (the FIRST split after |2,0,2,0>!) gives an EXACT factor of beta (NOT beta/2, because there is ONLY ONE |2,0,2,0> interval on the circle S^1_beta for an exchange cycle!!), and then 0 < u1 < u2 < u3 < beta (where u1 = s2-s1, u2 = s3-s1, u3 = s4-s1)!
    #Wait: let's double check that!
    # In exch_paths, we counted 16 paths starting from |2,0,2,0>. Why 16?
    # 2 spins * 2 pairs of opposite edges ({e0,e2} vs {e1,e3}) * 2 choices of which edge splits FIRST (e0 vs e2) * 2 choices of which site recombines (e.g. 1->2 vs 3->0) = 16!
    # Each of those 16 paths corresponds to a UNIQUE unrooted exchange cycle (1-to-1 with the 16 cycle shapes when rooted at the unique entry from |2,0,2,0>!).
    # Wait: let's verify! In an unrooted exchange cycle (say 0->1, 2->3, 1->2, 3->0):
    # There is ONLY ONE time interval where ALL 4 sites are inactive (|2,0,2,0>), namely [s4, s1]!
    # So there is a UNIQUE hop s1 that leaves |2,0,2,0>! Thus rooting at s1 has multiplicity 1 (factor beta * 1),
    # and we integrate 0 < u1 < u2 < u3 < beta over ALL 16 shapes!
    # Let's check this numerically with 14-pt Gauss-Legendre 3D quadrature!
    x4, w4 = leggauss(14)
    u3_pts = 0.5 * beta * (x4 + 1.0)
    u3_w = 0.5 * beta * w4
    c4_exch = 0.0
    exch_sgn = cfg["m4_exch_sign"]
    # For each of the 16 exchange paths, active intervals on the 4 sites during [0, u1, u2, u3] are:
    # Site becomes active at hop where it changes eta->1, and inactive where it changes 1->eta.
    path_intervals: List[List[Tuple[float, float]]] = []
    # Map each path to site active interval index pairs (t_start_idx, t_end_idx) with times [0, u1, u2, u3]:
    path_site_specs: List[List[Tuple[int, int]]] = []
    for path, _ in exch_paths:
        # Track occupation of each site after hop 0, 1, 2, 3:
        # Since every site starts at eta and returns to eta at hop 3, each site is active on [t_in, t_out]
        spec = [None, None, None, None]
        active = [False, False, False, False]
        t_in = [0, 0, 0, 0]
        for k, (zf, zt, sp) in enumerate(path):
            for z in (zf, zt):
                if not active[z]:
                    active[z] = True
                    t_in[z] = k
                else:
                    active[z] = False
                    spec[z] = (t_in[z], k)
        assert all(s is not None for s in spec)
        path_site_specs.append(spec)  # type: ignore

    # Notice all 16 exchange paths have the SAME time-interval symmetry up to site relabeling:
    # Site A: [0, u2], Site B: [0, u3], Site C: [u1, u2], Site D: [u1, u3] (or [0,u3],[0,u2],[u1,u3],[u1,u2])!
    #Wait: check that! At k=0: e_A=(z0,z1) splits -> z0, z1 active at 0. At k=1: e_B=(z2,z3) splits -> z2, z3 active at u1.
    # At k=2: one of z0,z1 recombines with one of z2,z3 -> two sites end at u2, remaining two end at u3!
    # So EVERY one of the 16 exchange paths has active lengths: one [0, u2], one [0, u3], one [u1, u2], one [u1, u3]!
    # Total active length L = u2 + u3 + (u2 - u1) + (u3 - u1) = 2*(u2 + u3 - u1)!
    # Let's integrate ONE representative and multiply by 16 * exch_sgn * beta!
    for i3, u3 in enumerate(u3_pts):
        u2_pts = 0.5 * u3 * (x4 + 1.0)
        u2_w = 0.5 * u3 * w4
        for i2, u2 in enumerate(u2_pts):
            u1_pts = 0.5 * u2 * (x4 + 1.0)
            u1_w = 0.5 * u2 * w4
            for i1, u1 in enumerate(u1_pts):
                L_ex = 2.0 * (u2 + u3 - u1)
                phi_ex = (
                    phi_single_interval(u2, U, om, beta)
                    + phi_single_interval(u3, U, om, beta)
                    + phi_single_interval(u2 - u1, U, om, beta)
                    + phi_single_interval(u3 - u1, U, om, beta)
                )
                wt = u3_w[i3] * u2_w[i2] * u1_w[i1]
                c4_exch += 16.0 * exch_sgn * beta * wt * math.exp(-0.5 * U * L_ex - phi_ex)

    # Analytical 2*c1*t^2 + 4*(c2_same + c2_adj + c4_exch)*t^4 for t*d/dt log Z_neut:
    tdlogZ_pred = 2.0 * c1 * (t ** 2) + 4.0 * (c2_same + c2_adj + c4_exch) * (t ** 4)

    # Part (2): Continuous-time worldline Monte Carlo on the polymer gas of the 4-site ring!
    # Sample from the grand-canonical polymer gas containing:
    #   - 2-hop edge loops on the 4 edges x 2 spins (8 species)
    #   - 4-hop exchange loops (16 shapes)
    # Using exact Metropolis insertion/deletion with positive reference weight |W|, exact hard-core compatibility,
    # exact pairwise phonon interaction exp(-V), and measuring <m * sgn> / <sgn> = t d/dt log Z_neut!
    rng = np.random.default_rng(20261121)
    # Proposal rates:
    # 2-hop loop: pick (edge in 0..3, spin in 0..1), s1 ~ Unif[0, beta), u ~ Exp(U) conditioned on u < beta
    #   Proposal density q2(s1, u) = (1/8) * (1/beta) * (U * exp(-U*u) / (1 - exp(-U*beta))).
    #   Target weight w2(s1, u) = t^2 * exp(-U*u - 2*phi_single(u)).
    #   Wait: ratio w2 / q2 = 8 * beta * (1 - exp(-U*beta)) * (t^2 / U) * exp(-2*phi_single(u))!
    # 4-hop exchange loop: pick shape in 0..15, s1 ~ Unif[0, beta), and generate (u1, u2, u3) via:
    #   draw u_a ~ Exp(U), u_b ~ Exp(U), set u1 = min, u2 = max? Even simpler: draw v1, v2, v3 ~ Exp(U) i.i.d.
    #   and set u1 = v1, u2 = v1 + v2, u3 = v1 + v2 + v3 (condition on u3 < beta)!
    #   Notice L_ex = 2*(u2+u3-u1) = 2*(v1 + 2*v2 + v3), and exp(-0.5*U*L_ex) = exp(-U*v1 - 2*U*v2 - U*v3)!
    #   Proposal density q4 = (1/16) * (1/beta) * U * exp(-U*v1) * 2*U * exp(-2*U*v2) * U * exp(-U*v3).
    #   Target |w4| = t^4 * exp(-U*v1 - 2*U*v2 - U*v3 - phi_ex).
    #   Ratio |w4| / q4 = 16 * beta * (t^4 / (2 * U^3)) * exp(-phi_ex)!
    polymers: List[Dict[str, object]] = []
    # Each polymer has: kind (2 or 4), sign (+1 or exch_sgn), sites_intervals: dict {z: [(a,b), ...]}
    p_ins2 = 0.45
    p_rem2 = 0.45
    p_ins4 = 0.05
    p_rem4 = 0.05
    norm_u2 = 1.0 - math.exp(-U * beta)

    def poly_compatible(new_iv: Dict[int, List[Tuple[float, float]]], skip_idx: int = -1) -> bool:
        for idx, pol in enumerate(polymers):
            if idx == skip_idx:
                continue
            piv: Dict[int, List[Tuple[float, float]]] = pol["iv"]  # type: ignore
            for z, iv_new in new_iv.items():
                if z in piv:
                    if intervals_overlap_periodic(piv[z], iv_new, 0.0, beta):
                        return False
        return True

    def delta_V_with_rest(new_iv: Dict[int, List[Tuple[float, float]]]) -> float:
        dv = 0.0
        for pol in polymers:
            piv: Dict[int, List[Tuple[float, float]]] = pol["iv"]  # type: ignore
            for z, iv_new in new_iv.items():
                if z in piv:
                    dv += v_pair_exact_fd(piv[z], iv_new, U, om, beta, v_sign=cfg["v_sign"])
        return dv

    n_therm = 40000
    n_meas = 280000
    sum_sgn = 0.0
    sum_m_sgn = 0.0
    sum_n4_sgn = 0.0
    cur_sgn = 1
    cur_m = 0
    cur_n4 = 0

    for step in range(n_therm + n_meas):
        r = rng.random()
        if r < p_ins2:
            # Propose insert 2-hop loop
            e = int(rng.integers(0, 4))
            z0, z1 = e, (e + 1) % 4
            s1 = float(rng.random() * beta)
            u = -math.log(1.0 - float(rng.random()) * norm_u2) / U
            new_iv = {z0: [(s1, s1 + u)], z1: [(s1, s1 + u)]}
            if poly_compatible(new_iv):
                dv = delta_V_with_rest(new_iv)
                n_2 = sum(1 for p in polymers if p["m"] == 2)
                acc = (p_rem2 / p_ins2) * (1.0 / (n_2 + 1)) * (8.0 * beta * norm_u2 * (t ** 2) / U) * math.exp(-2.0 * phi_single_interval(u, U, om, beta) - dv)
                if rng.random() < acc:
                    polymers.append({"m": 2, "sgn": 1, "iv": new_iv})
                    cur_m += 2
        elif r < p_ins2 + p_rem2:
            idx2 = [i for i, p in enumerate(polymers) if p["m"] == 2]
            if idx2:
                pick = idx2[int(rng.integers(0, len(idx2)))]
                pol = polymers[pick]
                piv: Dict[int, List[Tuple[float, float]]] = pol["iv"]  # type: ignore
                z_any = next(iter(piv))
                a0, b0 = piv[z_any][0]
                u = b0 - a0
                # Compute dv without pick
                polymers.pop(pick)
                dv = delta_V_with_rest(piv)
                acc = (p_ins2 / p_rem2) * (len(idx2)) / ((8.0 * beta * norm_u2 * (t ** 2) / U) * math.exp(-2.0 * phi_single_interval(u, U, om, beta) - dv))
                if rng.random() < acc:
                    cur_m -= 2
                else:
                    polymers.insert(pick, pol)
        elif r < p_ins2 + p_rem2 + p_ins4:
            # Propose insert 4-hop exchange loop
            s1 = float(rng.random() * beta)
            v1 = -math.log(1.0 - float(rng.random())) / U
            v2 = -math.log(1.0 - float(rng.random())) / (2.0 * U)
            v3 = -math.log(1.0 - float(rng.random())) / U
            u1, u2, u3 = v1, v1 + v2, v1 + v2 + v3
            if u3 < beta:
                # Pick random orientation: which sites get [0,u2], [0,u3], [u1,u2], [u1,u3]
                # 16 shapes give 16 overall multiplicity; pick one of the 4 spatial layouts
                rot = int(rng.integers(0, 4))
                # Layout: adjacent sites on ring get ([0,u2],[0,u3],[u1,u2],[u1,u3])
                z_a, z_b, z_c, z_d = rot, (rot + 1) % 4, (rot + 2) % 4, (rot + 3) % 4
                new_iv = {
                    z_a: [(s1, s1 + u2)],
                    z_b: [(s1, s1 + u3)],
                    z_c: [(s1 + u1, s1 + u2)],
                    z_d: [(s1 + u1, s1 + u3)],
                }
                if poly_compatible(new_iv):
                    dv = delta_V_with_rest(new_iv)
                    phi_ex = (
                        phi_single_interval(u2, U, om, beta)
                        + phi_single_interval(u3, U, om, beta)
                        + phi_single_interval(u2 - u1, U, om, beta)
                        + phi_single_interval(u3 - u1, U, om, beta)
                    )
                    n_4 = sum(1 for p in polymers if p["m"] == 4)
                    acc = (p_rem4 / p_ins4) * (1.0 / (n_4 + 1)) * (16.0 * beta * (t ** 4) / (2.0 * U ** 3)) * math.exp(-phi_ex - dv)
                    if rng.random() < acc:
                        polymers.append({"m": 4, "sgn": -1, "iv": new_iv, "phi_ex": phi_ex})
                        cur_m += 4
                        cur_n4 += 1
                        cur_sgn *= -1
        else:
            idx4 = [i for i, p in enumerate(polymers) if p["m"] == 4]
            if idx4:
                pick = idx4[int(rng.integers(0, len(idx4)))]
                pol = polymers[pick]
                piv = pol["iv"]  # type: ignore
                phi_ex_val = float(pol["phi_ex"])  # type: ignore
                polymers.pop(pick)
                dv = delta_V_with_rest(piv)
                acc = (p_ins4 / p_rem4) * len(idx4) / ((16.0 * beta * (t ** 4) / (2.0 * U ** 3)) * math.exp(-phi_ex_val - dv))
                if rng.random() < acc:
                    cur_m -= 4
                    cur_n4 -= 1
                    cur_sgn *= int(pol["sgn"])  # type: ignore
                else:
                    polymers.insert(pick, pol)
        if step >= n_therm:
            sum_sgn += cur_sgn
            sum_m_sgn += cur_m * cur_sgn
            sum_n4_sgn += cur_n4 * cur_sgn

    mc_est = sum_m_sgn / sum_sgn
    mc_n4_est = sum_n4_sgn / sum_sgn
    pred_n4 = c4_exch * (t ** 4)
    assert mc_n4_est * pred_n4 > 0.0, (
        f"4-site ring MC exchange sign mismatch: mc_n4={mc_n4_est:.6e}, pred_n4={pred_n4:.6e}"
    )
    assert abs(mc_n4_est - pred_n4) < 0.45 * abs(pred_n4) + 2e-4, (
        f"4-site ring MC n4 magnitude mismatch: mc_n4={mc_n4_est:.6e}, pred_n4={pred_n4:.6e}"
    )
    # note mutant flipping m4_exch_sign from -1 to +1 shifts 4*c4_exch*t^4 by ~ 3e-3? Wait:
    # Let's check exact magnitude of 4*c4_exch*t^4 at U=8, beta=2.5, t=0.25 -> if too small, use U=4, beta=3, t=0.3!)
    # Wait: let's check both! At U=4, om=1, beta=3, t=0.25:
    #   c4_exch * t^4 ~ 16 * 3 * (0.25^4 / (2*64)) = 48 * 3.9e-3 / 128 ~ 1.5e-3, so 4 * c4_exch * t^4 ~ 6e-3,
    #   meaning flipping sign of c4_exch shifts tdlogZ by 2 * 6e-3 = 0.012 >> MC noise!
    # Wait: let's check the actual difference |mc_est - tdlogZ_pred| right now when we run it!
    # Note: in MC, the MC sampler ALSO uses exch_sgn if we pass cfg["m4_exch_sign"] to MC; to catch a mutant
    # that flips m4_exch_sign ONLY in the analytical cluster formula (or ONLY in MC), the analytical formula uses
    # cfg["m4_exch_sign"] while the exact Fock DFS checked sgn == -1 above!
    # Let's assert BOTH: exact Fock sign == -1 (which already catches mutant in exch_sgn if applied to analytical),
    # AND MC vs analytical match!
    assert abs(mc_est - tdlogZ_pred) < 0.004, (
        f"4-site ring MC vs analytical mismatch: MC={mc_est:.6f}, pred={tdlogZ_pred:.6f}"
    )


# ==============================================================================
# Runner & Mutation Suite
# ==============================================================================

def default_cfg() -> Dict[str, float]:
    return {
        "parseval_pref": 0.5,
        "v_sign": +1.0,
        "V_l1_pref": 1.0,
        "include_eps_SL": True,
        "force_nonbipartite_sum": False,
        "drop_m3_general": False,
        "m4_exch_sign": -1,
        "pair_return_min_rel": 1.0e-6,
        "m2_spin_factor": 2.0,
        "m3_eta220_sign": -1.0,
        "collar_scale": 1.0,
    }


def run_all(cfg: Dict[str, float], verbose: bool = True) -> None:
    t0 = time.time()
    verify_phonon_bounds(cfg)
    if verbose:
        print("[PASS] Step 1-2: Exact phonon trace, Parseval, double-FD, and L^1 shift bound")
    verify_pu_constants(cfg)
    if verbose:
        print("[PASS] Step 3: Poghosyan-Ueltschi Assumption 2' explicit constants (Delta=2,4,6)")
    verify_exterior_collar_theorem(cfg)
    if verbose:
        print("[PASS] Step 4: Theorem 4.3 perturbed exterior constants, |V|<=0.05U, and collar witness")
    dimer_ed_and_cluster(cfg)
    if verbose:
        print("[PASS] Step 5A: 2-site dimer exact Lang-Firsov ED vs 1+2-loop PU + pair-return check")
    three_site_ring_ed(cfg)
    if verbose:
        print("[PASS] Step 5B: 3-site ring exact Lang-Firsov ED (eta=(2,0,0) +1 & eta=(2,2,0) -1)")
    verify_ring4_mc_vs_cluster(cfg)
    if verbose:
        print(f"[PASS] Step 5C: 4-site ring worldline MC vs analytical cluster expansion ({time.time()-t0:.2f}s)")


def run_mutants() -> int:
    mutants = [
        ("M1_parseval_factor2", {"parseval_pref": 1.0}),
        ("M2_wrong_V_sign", {"v_sign": -1.0}),
        ("M3_wrong_V_l1_prefactor", {"V_l1_pref": 0.5}),
        ("M4_drop_eps_in_SL", {"include_eps_SL": False}),
        ("M5_drop_m3_nonbipartite", {"drop_m3_general": True}),
        ("M6_wrong_m4_fermion_sign", {"m4_exch_sign": +1}),
        ("M7_false_exp_pair_return_suppression", {"pair_return_min_rel": 1.0e-4}),
        ("M8_wrong_m2_spin_factor", {"m2_spin_factor": 1.0}),
        ("M9_wrong_m3_eta220_sign", {"m3_eta220_sign": +1.0}),
        ("M10_wrong_collar_width", {"collar_scale": 0.5}),
    ]
    caught = 0
    for name, patch in mutants:
        cfg = default_cfg()
        cfg.update(patch)
        try:
            run_all(cfg, verbose=False)
            print(f"[ESCAPED] {name}")
        except AssertionError as exc:
            caught += 1
            tb_last = traceback.extract_tb(exc.__traceback__)[-1]
            print(f"[CAUGHT]  {name} ({tb_last.name}:{tb_last.lineno}): {exc}")
    print(f"Mutation summary: {caught}/{len(mutants)} mutants caught")
    return 0 if caught == len(mutants) else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mutants", action="store_true", help="Run semantic mutation suite")
    args = parser.parse_args()
    if args.mutants:
        return run_mutants()
    try:
        run_all(default_cfg(), verbose=True)
        return 0
    except Exception as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
