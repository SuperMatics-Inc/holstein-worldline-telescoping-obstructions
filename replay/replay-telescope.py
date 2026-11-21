"""Verification replay for W/MASSIVE/TELESCOPE/.

Checks:
1. Exact multi-time Weyl phonon trace identity and telescoping (open bare-LF & physical closed).
2. Near-synchronized offset action bound Phi_beta <= U(|Delta u_up|_beta + |Delta u_dn|_beta).
3. Exact HS determinant positivity at mu=0 and dimer determinant cancellation.
4. Pre-LF exact diagonalization table on N=2 (d=1) and N=3 (d=2) chains showing (t/U)^(2d) scaling.
"""

import argparse
import itertools
import math
import os
import subprocess
import sys

import numpy as np
from scipy.linalg import eigh, expm
from scipy.special import eval_genlaguerre, gammaln


MUTATIONS = (
    "wrong_trace_sign",
    "wrong_telescope_L",
    "wrong_fminus_power",
    "wrong_offset_factor",
    "wrong_hs_dimer_cancel",
    "wrong_hs_nonzero_mu",
    "wrong_ed_exp_ceiling",
)


def active_mutation():
    return os.environ.get("TELESCOPE_MUTATION", "")


def displacement(gamma, cutoff):
    m_idx = np.arange(cutoff)[:, None]
    n_idx = np.arange(cutoff)[None, :]
    lo = np.minimum(m_idx, n_idx)
    hi = np.maximum(m_idx, n_idx)
    power = hi - lo
    signed = np.where(m_idx >= n_idx, gamma, -gamma)
    g2 = gamma * gamma
    return (
        np.exp(-0.5 * g2 + 0.5 * (gammaln(lo + 1) - gammaln(hi + 1)))
        * (signed ** power)
        * eval_genlaguerre(lo, power, g2)
    )


def d_beta(s, beta, om):
    s_abs = abs(s) % beta
    s_per = min(s_abs, beta - s_abs)
    return math.cosh(om * (beta / 2.0 - s_per)) / math.sinh(beta * om / 2.0)


def check_trace_and_telescoping(mut=""):
    om = 1.7
    beta = 1.3
    cut = 90
    energies = om * np.arange(cut)
    z_ph = float(np.sum(np.exp(-beta * energies)))

    # 1a. General multi-time Weyl trace identity
    us = [1.1, 0.85, 0.4, 0.15]
    for label, rs in [("sum=0", [1.2, -0.7, 0.4, -0.9]), ("sum!=0", [1.2, -0.7, 0.4, -0.3])]:
        mat = np.diag(np.exp(-(beta - us[0]) * energies))
        for k in range(len(us)):
            mat = mat @ displacement(rs[k], cut)
            dt = (us[k] - us[k + 1]) if k + 1 < len(us) else us[k]
            mat = mat @ np.diag(np.exp(-dt * energies))
        num = float(np.trace(mat)) / z_ph
        sign = 0.5 if mut == "wrong_trace_sign" else -0.5
        expo = 0.0
        for j in range(len(us)):
            for k in range(len(us)):
                expo += sign * rs[j] * rs[k] * d_beta(us[j] - us[k], beta, om)
        ana = math.exp(expo)
        rel = abs(num - ana) / max(abs(num), 1e-300)
        if rel > 1e-12:
            raise AssertionError(f"trace mismatch ({label}): rel={rel}")
        print(f"TRACE {label}: num={num:.12e} ana={ana:.12e} rel_err={rel:.3e}", flush=True)

    # 1b. Telescoping check: open bare-LF product along walk of length L is independent of L
    u_val = 6.0
    alpha = math.sqrt(u_val / (2.0 * om))
    d_p = displacement(2.0 * alpha, cut)
    d_m = displacement(-2.0 * alpha, cut)
    rho = np.diag(np.exp(-beta * energies)) / z_ph
    tr_p = float(np.trace(rho @ d_p))
    tr_m = float(np.trace(rho @ d_m))
    # Synchronized product along walk 0->1->...->L factors by site: tr(D_0(+2a)) * 1 * ... * tr(D_L(-2a))
    # Verify middle site cancellation D(-2a) D(+2a) = I on low states
    prod_mid = d_m @ d_p
    mid_err = float(np.max(np.abs(prod_mid[:20, :20] - np.eye(20))))
    if mid_err > 1e-12:
        raise AssertionError(f"D(-2a)D(+2a) != I: err={mid_err}")
    target = math.exp(-2.0 * (u_val / om) / math.tanh(beta * om / 2.0))
    for L in (1, 2, 3, 4):
        scale = L if mut == "wrong_telescope_L" else 1.0
        val = (tr_p * tr_m) ** scale
        rel = abs(val - target) / target
        if rel > 1e-12:
            raise AssertionError(f"telescoping failed at L={L}: val={val} target={target}")
    print(f"TELESCOPE open bare-LF L=1..4: trace={tr_p * tr_m:.12e} analytic={target:.12e}", flush=True)

    # 1c. Physical closed trace with endpoint displacements at u=0 approaches 1 as s_L -> 0+
    # In FULL-GIBBS-KERNEL/PROOF.md §2 convention, F_-(u) = exp(-(U/Om)*(D(0)-D(u))),
    # so each site gives F_-(du)^2 = exp(-(2U/Om)*(D(0)-D(du))).
    eps = 1e-5
    s_list = [eps * (k + 1) / 4.0 for k in range(3)]
    d0 = d_beta(0.0, beta, om)
    f_closed = 1.0
    f_analytic = 1.0
    f_pow = 1.0 if mut == "wrong_fminus_power" else 2.0
    for r0, t0, r1, t1 in [(-2*alpha, 0.0, 2*alpha, s_list[0]),
                           (-2*alpha, s_list[0], 2*alpha, s_list[1]),
                           (-2*alpha, s_list[1], 2*alpha, s_list[2]),
                           (-2*alpha, s_list[2], 2*alpha, 0.0)]:
        m_site = (
            np.diag(np.exp(-(beta - max(t0, t1)) * energies))
            @ displacement(r1 if t1 > t0 else r0, cut)
            @ np.diag(np.exp(-abs(t1 - t0) * energies))
            @ displacement(r0 if t1 > t0 else r1, cut)
            @ np.diag(np.exp(-min(t0, t1) * energies))
        )
        f_closed *= float(np.trace(m_site)) / z_ph
        du = abs(t1 - t0)
        f_analytic *= math.exp(-f_pow * (u_val / om) * (d0 - d_beta(du, beta, om)))
    rel_phys = abs(f_closed - f_analytic) / f_analytic
    if rel_phys > 1e-12 or abs(f_closed - 1.0) > 1e-3:
        raise AssertionError(f"closed physical trace check failed: W={f_closed} ana={f_analytic} rel={rel_phys}")
    print(f"TELESCOPE closed physical s_L={s_list[-1]:.1e}: W_phys={f_closed:.12f} rel_F2={rel_phys:.3e}", flush=True)


def check_offset_bound(mut=""):
    rng = np.random.default_rng(20261121)
    u_val = 12.0
    max_ratio = 0.0
    factor = 0.5 if mut == "wrong_offset_factor" else 1.0
    cs = (-1.0, -1.0, 1.0, 1.0)
    for _ in range(5000):
        om = float(rng.uniform(0.2, 5.0))
        beta = float(rng.uniform(0.2, 6.0))
        ts = rng.uniform(0.0, beta, size=4)
        phi = 0.0
        for a in range(4):
            for b in range(4):
                phi += cs[a] * cs[b] * d_beta(ts[a] - ts[b], beta, om)
        phi *= u_val / (4.0 * om)
        du_up = min(abs(ts[0] - ts[2]), beta - abs(ts[0] - ts[2]))
        du_dn = min(abs(ts[1] - ts[3]), beta - abs(ts[1] - ts[3]))
        rhs = factor * u_val * (du_up + du_dn)
        if rhs > 1e-12:
            ratio = phi / rhs
            if ratio > max_ratio:
                max_ratio = ratio
            if phi > rhs + 1e-11:
                raise AssertionError(f"offset bound violated: phi={phi} > rhs={rhs}")
    print(f"OFFSET_BOUND 5000 samples: max(Phi/RHS)={max_ratio:.8f} <= 1.0", flush=True)


def check_hs_and_dimer(mut=""):
    rng = np.random.default_rng(7)
    beta = 1.3
    t = 0.7
    steps = 20
    dt = beta / steps

    # Check N=2 dimer determinant cancellation and N=3 chain & K3 non-bipartite positivity at mu=0 and mu!=0
    configs = [
        ("chain_N2_mu0", 2, False, 0.0),
        ("chain_N2_mu-0.4", 2, False, -0.4),
        ("chain_N3_mu0.3", 3, False, 0.3),
        ("nonbip_K3_mu-0.4", 3, True, -0.4),
    ]
    for label, sites, complete, mu in configs:
        adj = np.zeros((sites, sites))
        for z in range(sites - 1):
            adj[z, z + 1] = adj[z + 1, z] = 1.0
        if complete:
            adj[0, sites - 1] = adj[sites - 1, 0] = 1.0
        u_mat0 = np.eye(sites)
        phi_bar = np.zeros(sites)
        for _ in range(steps):
            phi = rng.normal(size=sites)
            phi_bar += dt * phi
            h0 = -t * adj - np.diag(phi)
            u_mat0 = expm(-dt * h0) @ u_mat0
        u_mat = math.exp(beta * mu) * u_mat0
        det_val = float(np.linalg.det(np.eye(sites) + u_mat))
        w_val = math.exp(-float(np.sum(phi_bar))) * (det_val ** 2)
        g_mat = np.linalg.solve(np.eye(sites) + u_mat, u_mat)
        corr_num = w_val * (g_mat[-1, 0] ** 2)
        if w_val < 0 or corr_num < 0:
            raise AssertionError(f"HS positivity violated for {label}")
        if sites == 2:
            mu_eff = -mu if mut == "wrong_hs_nonzero_mu" else mu
            rhs = math.exp(2.0 * beta * mu_eff - float(np.sum(phi_bar))) * (u_mat0[1, 0] ** 2)
            if mut == "wrong_hs_dimer_cancel":
                rhs *= det_val
            rel = abs(corr_num - rhs) / max(rhs, 1e-300)
            if rel > 1e-12:
                raise AssertionError(f"dimer cancellation failed ({label}): corr_num={corr_num} rhs={rhs}")
            print(f"HS_DIMER_CANCEL ({label}): W*G(1,0)^2={corr_num:.12e} rhs={rhs:.12e} rel={rel:.3e}", flush=True)
        else:
            print(f"HS_POS ({label}): W={w_val:.6e} >= 0, W*G^2={corr_num:.6e} >= 0", flush=True)

def pre_lf_canonical(sites, max_total, u_val, om, t, temperatures):
    modes = sites - 1
    ph_states = [ns for ns in itertools.product(range(max_total + 1), repeat=modes) if sum(ns) <= max_total]
    ph_index = {ns: i for i, ns in enumerate(ph_states)}
    phdim = len(ph_states)
    rel_mat = np.zeros((sites, modes))
    for j in range(1, sites):
        rel_mat[:j, j - 1] = 1.0 / math.sqrt(j * (j + 1))
        rel_mat[j, j - 1] = -j / math.sqrt(j * (j + 1))
    g = math.sqrt(0.5 * u_val * om)
    ph_diag = om * np.sum(np.array(ph_states, dtype=int), axis=1)

    x_ph = [np.zeros((phdim, phdim), dtype=float) for _ in range(modes)]
    for i, ns in enumerate(ph_states):
        for m in range(modes):
            if ns[m] + 1 <= max_total and sum(ns) + 1 <= max_total:
                nxt = list(ns)
                nxt[m] += 1
                j = ph_index[tuple(nxt)]
                val = math.sqrt(ns[m] + 1)
                x_ph[m][i, j] = val
                x_ph[m][j, i] = val

    el_states = [(up, dn) for up in range(sites) for dn in range(sites)]
    el_index = {st: i for i, st in enumerate(el_states)}
    dim = len(el_states) * phdim
    ham = np.zeros((dim, dim), dtype=float)
    eye_ph = np.eye(phdim)

    for (up, dn), i in el_index.items():
        sl = slice(i * phdim, (i + 1) * phdim)
        q = np.full(sites, -1.0)
        q[up] += 1.0
        q[dn] += 1.0
        q_rel = rel_mat.T @ q
        block = np.diag(ph_diag)
        for m in range(modes):
            if abs(q_rel[m]) > 1e-15:
                block -= g * q_rel[m] * x_ph[m]
        ham[sl, sl] = block
        for spin, pos in enumerate((up, dn)):
            for dst in (pos - 1, pos + 1):
                if 0 <= dst < sites:
                    nxt = (dst, dn) if spin == 0 else (up, dst)
                    j = el_index[nxt]
                    ham[j * phdim:(j + 1) * phdim, sl] -= t * eye_ph

    vals, vecs = eigh(ham, overwrite_a=True, check_finite=False)
    out = {}
    for d in range(1, sites):
        i0 = el_index[(0, 0)]
        id_ = el_index[(d, d)]
        v0 = vecs[i0 * phdim:(i0 + 1) * phdim, :]
        vd = vecs[id_ * phdim:(id_ + 1) * phdim, :]
        diag_phys = np.einsum("ai,ai->i", v0, vd)
        res_tau = []
        for tau in temperatures:
            w = np.exp(-(vals - vals[0]) / tau)
            res_tau.append((tau, float((w @ diag_phys) / w.sum())))
        out[d] = res_tau
    return out


def dimer_gc_mu0(mt, u_val, om, t, temperatures):
    g = math.sqrt(0.5 * u_val * om)
    phdim = mt + 1
    x_ph = np.zeros((phdim, phdim), dtype=float)
    for n in range(mt):
        val = math.sqrt(n + 1)
        x_ph[n, n + 1] = val
        x_ph[n + 1, n] = val
    ph_diag = om * np.arange(phdim, dtype=float)
    eye_ph = np.eye(phdim)

    # (1,1) sector: 4 electronic states, COM mode uncoupled
    el_states = [(up, dn) for up in range(2) for dn in range(2)]
    el_idx = {st: i for i, st in enumerate(el_states)}
    h11 = np.zeros((4 * phdim, 4 * phdim), dtype=float)
    for (up, dn), i in el_idx.items():
        sl = slice(i * phdim, (i + 1) * phdim)
        q_rel = ((1.0 if up == 0 else 0.0) + (1.0 if dn == 0 else 0.0)
                 - (1.0 if up == 1 else 0.0) - (1.0 if dn == 1 else 0.0)) / math.sqrt(2.0)
        h11[sl, sl] = np.diag(ph_diag) - g * q_rel * x_ph
        for sp, pos in enumerate((up, dn)):
            dst = 1 - pos
            nxt = (dst, dn) if sp == 0 else (up, dst)
            j = el_idx[nxt]
            h11[j * phdim:(j + 1) * phdim, sl] -= t * eye_ph
    vals11, vecs11 = eigh(h11, overwrite_a=True, check_finite=False)
    v0 = vecs11[el_idx[(0, 0)] * phdim:(el_idx[(0, 0)] + 1) * phdim, :]
    v1 = vecs11[el_idx[(1, 1)] * phdim:(el_idx[(1, 1)] + 1) * phdim, :]
    diag_phys = np.einsum("ai,ai->i", v0, v1)

    # 1-electron (and 3-electron) sectors: 2 electronic states, COM shift -U/4
    h10 = np.zeros((2 * phdim, 2 * phdim), dtype=float)
    for pos in range(2):
        sl = slice(pos * phdim, (pos + 1) * phdim)
        q_rel = (1.0 if pos == 0 else -1.0) / math.sqrt(2.0)
        h10[sl, sl] = np.diag(ph_diag) - g * q_rel * x_ph
        h10[(1 - pos) * phdim:(2 - pos) * phdim, sl] -= t * eye_ph
    vals10 = eigh(h10, eigvals_only=True, overwrite_a=True, check_finite=False)

    e0 = vals11[0]
    out = []
    for tau in temperatures:
        z_ph_rel = float(np.sum(np.exp(-ph_diag / tau)))
        w11 = np.exp(-(vals11 - e0) / tau)
        num11 = float(w11 @ diag_phys)
        z11 = float(w11.sum())
        z00 = math.exp((u_val + e0) / tau) * z_ph_rel
        z10 = math.exp((0.25 * u_val + e0) / tau) * float(np.sum(np.exp(-vals10 / tau)))
        z20 = math.exp(e0 / tau) * z_ph_rel
        z_gc = z11 + 2.0 * z00 + 4.0 * z10 + 2.0 * z20
        out.append((tau, num11 / z_gc, z11 / z_gc))
    return out


def check_ed_n2_n3(mut=""):
    cases = [
        (4.0, 1.0, 0.25, [(2, 30, 40, [0.25, 1.0]), (3, 22, 28, [0.25, 1.0])]),
        (10.0, 2.0, 0.4, [(2, 30, 40, [0.1, 1.0]), (3, 22, 28, [0.1, 1.0])]),
        (8.0, 1.0, 0.25, [(2, 30, 40, [0.25, 1.0]), (3, 25, 29, [0.25])]),
        (12.0, 1.0, 0.25, [(2, 30, 40, [0.25, 1.0])]),
        (10.0, 0.5, 1.0, [(2, 40, 50, [0.05, 0.25])]),
    ]
    for u_val, om, t, configs in cases:
        x_ratio = u_val / om
        t_over_u = t / u_val
        for sites, mt_lo, mt_hi, taus in configs:
            d = sites - 1
            r_lo = pre_lf_canonical(sites, mt_lo, u_val, om, t, taus)[d]
            r_hi = pre_lf_canonical(sites, mt_hi, u_val, om, t, taus)[d]
            for (tau, v_lo), (_, v_hi) in zip(r_lo, r_hi):
                diff = abs(v_hi - v_lo) / max(abs(v_hi), 1e-300)
                scaled = v_hi / (t_over_u ** (2 * d))
                print(
                    f"ED_CAN N={sites} d={d} U={u_val} Om={om} t={t} X={x_ratio:.1f} "
                    f"tau={tau} mt={mt_lo}->{mt_hi}: val={v_hi:.6e} "
                    f"/(t/U)^{2*d}={scaled:.4f} cut_rel={diff:.2e}",
                    flush=True,
                )
                if diff > 1e-2:
                    raise AssertionError(f"ED cutoff did not converge within 1%: rel={diff}")
                if mut == "wrong_ed_exp_ceiling":
                    beta = 1.0 / tau
                    ceil = ((t * t / (u_val * tau)) * math.exp(-x_ratio / math.tanh(beta * om / 4.0))) ** ((d + 1) // 2)
                    if v_hi > ceil:
                        raise AssertionError(f"Refuted literal exp ceiling violated: {v_hi} > {ceil}")

    gc_cases = [
        (12.0, 1.0, 0.25, 40, 50, [0.25, 1.0]),
        (10.0, 0.5, 1.0, 40, 50, [0.05, 0.25]),
    ]
    for u_val, om, t, mt_lo, mt_hi, taus in gc_cases:
        x_ratio = u_val / om
        t_over_u = t / u_val
        r_lo = dimer_gc_mu0(mt_lo, u_val, om, t, taus)
        r_hi = dimer_gc_mu0(mt_hi, u_val, om, t, taus)
        for (tau, v_lo, _), (_, v_hi, w11) in zip(r_lo, r_hi):
            diff = abs(v_hi - v_lo) / max(abs(v_hi), 1e-300)
            scaled = v_hi / (t_over_u ** 2)
            print(
                f"ED_GC(mu=0) N=2 d=1 U={u_val} Om={om} t={t} X={x_ratio:.1f} "
                f"tau={tau} mt={mt_lo}->{mt_hi}: val={v_hi:.6e} /(t/U)^2={scaled:.4f} "
                f"Z11/Zgc={w11:.4f} cut_rel={diff:.2e}",
                flush=True,
            )
            if diff > 1e-2:
                raise AssertionError(f"GC ED cutoff did not converge within 1%: rel={diff}")


def run_all(mut=""):
    check_trace_and_telescoping(mut)
    check_offset_bound(mut)
    check_hs_and_dimer(mut)
    check_ed_n2_n3(mut)
    print("ALL_CHECKS_PASSED", flush=True)


def run_mutants():
    for name in MUTATIONS:
        env = os.environ.copy()
        env["TELESCOPE_MUTATION"] = name
        proc = subprocess.run(
            [sys.executable, __file__],
            env=env,
            capture_output=True,
            text=True,
        )
        if proc.returncode == 0:
            print(f"MUTANT_SURVIVED: {name}", flush=True)
            return 1
        print(f"MUTANT_CAUGHT: {name} rc={proc.returncode}", flush=True)
    print(f"MUTANTS all {len(MUTATIONS)}/{len(MUTATIONS)} caught with exit!=0; driver exit=0", flush=True)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mutants", action="store_true", help="Run mutation suite")
    args = parser.parse_args()
    if args.mutants:
        return run_mutants()
    try:
        run_all(active_mutation())
    except Exception as exc:
        print(f"CHECK_FAILED: {exc}", file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
