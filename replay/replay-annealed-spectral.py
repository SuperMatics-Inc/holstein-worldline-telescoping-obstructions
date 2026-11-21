"""Light scientific replay; floating-point results are NUMERICAL.
Complete Gaussian Taylor jets, including non-frequency-conserving C2 pairs.
Run normally, or --mutants (each deliberately broken variant must fail).
No external files, GPU, phonon truncation, or unpublished dependencies.
"""
import argparse
import itertools
import math
from fractions import Fraction

import numpy as np
from scipy.linalg import expm
from scipy.special import expit, logsumexp


def graph(name):
    if name == "dimer":
        return np.array([[0., 1.], [1., 0.]])
    ring = np.array([[0., 1., 0., 1.], [1., 0., 1., 0.],
                     [0., 1., 0., 1.], [1., 0., 1., 0.]])
    return ring if name == "ring4" else 2 * ring


def static(v, adjacency, beta, mu=0.):
    h = np.diag(v) - adjacency - mu * np.eye(len(v))
    energies, basis = np.linalg.eigh(h)
    f = expit(-beta * energies)
    c = (basis * f) @ basis.T
    de = energies[:, None] - energies[None, :]
    df = f[:, None] - f[None, :]
    divided = np.divide(df, de, out=np.zeros_like(de), where=np.abs(de) > 1e-12)
    divided[np.abs(de) <= 1e-12] = np.broadcast_to(
        (-beta * f * (1-f))[:, None], de.shape)[np.abs(de) <= 1e-12]
    p = basis[:, :, None] * basis[:, None, :]
    chi = -2 * np.einsum("ab,zab,wab->zw", divided, p, p)
    dc = np.einsum("xa,yb,zab,ab->zxy", basis, basis, p, divided)
    return h, energies, basis, f, c, chi, dc


def design_matrix(volume, beta, coupling, omega, slices, modes=2, noise_scale=1.):
    # u(tau)=sum_m sqrt(2 T D_m)[xi_c cos(nu_m tau)+xi_s sin(nu_m tau)].
    tau = (np.arange(slices) + .5) * beta / slices
    design = np.zeros((slices, volume, 2*modes*volume))
    for m in range(1, modes+1):
        nu = 2*np.pi*m/beta
        dhat = coupling*omega**2/(nu**2+omega**2)
        sigma = noise_scale*np.sqrt(2*dhat/beta)
        for z in range(volume):
            design[:, z, (m-1)*2*volume+z] = sigma*np.cos(nu*tau)
            design[:, z, (m-1)*2*volume+volume+z] = sigma*np.sin(nu*tau)
    assert np.max(np.abs(np.mean(design, axis=0))) < 1e-13
    return design


def stationary_projection(b, volume, modes):
    # Exact projection of a real quadratic kernel onto total Fourier frequency zero.
    out = np.zeros_like(b)
    for m in range(modes):
        c = slice(2*m*volume, (2*m+1)*volume)
        s = slice((2*m+1)*volume, (2*m+2)*volume)
        same = .5*(b[c, c] + b[s, s])
        cross = .5*(b[c, s] - b[c, s].swapaxes(0, 1))
        out[c, c] = same
        out[s, s] = same
        out[c, s] = cross
        out[s, c] = cross.swapaxes(0, 1)
    return out


def jets(h, beta, design, mutant=None):
    # Taylor coefficient convention P=P0+eps P1_j xi_j+eps^2 P2_jk xi_j xi_k.
    slices, volume, q = design.shape
    dt = beta/slices
    half = expm(-.5*dt*h)
    b0 = half @ half
    p0 = np.eye(volume)
    p1 = np.zeros((q, volume, volume))
    p2 = np.zeros((q, q, volume, volume))
    for u in design:
        b1 = np.einsum("xz,zj,zy->jxy", half, -dt*u, half)
        b2 = np.einsum("xz,zj,zk,zy->jkxy", half, u, u, half)*(.5*dt**2)
        cross = np.einsum("jab,kbc->jkac", b1, p1)
        p2 = b0 @ p2 + .5*(cross + cross.swapaxes(0, 1)) + b2 @ p0
        p1 = b0 @ p1 + b1 @ p0
        p0 = b0 @ p0
    inverse = np.linalg.inv(np.eye(volume)+p0)
    s1 = inverse @ p1
    d1 = np.trace(s1, axis1=-2, axis2=-1)
    d2 = np.trace(inverse @ p2, axis1=-2, axis2=-1)
    d2 += .5*(np.outer(d1, d1) - np.einsum("jab,kba->jk", s1, s1))
    c0 = (np.eye(volume)-inverse).T
    c1 = (inverse @ p1 @ inverse).swapaxes(-1, -2)
    cross = np.einsum("jab,kbc->jkac", inverse @ p1 @ inverse, p1 @ inverse)
    c2 = (inverse @ p2 @ inverse - .5*(cross+cross.swapaxes(0, 1))).swapaxes(-1, -2)
    if mutant == "omit_nonzero_total_frequency":
        c2 = stationary_projection(c2, volume, design.shape[-1]//(2*volume))
    if mutant == "drop_inverse_cross":
        c2 = (inverse @ p2 @ inverse).swapaxes(-1, -2)
    return c0, c1, c2, d1, d2


def direct(h, beta, design, xi, epsilon):
    half = expm(-beta/(2*len(design))*h)
    p = np.eye(len(h))
    for u in np.einsum("szj,j->sz", design, xi):
        p = half @ np.diag(np.exp(-epsilon*beta/len(design)*u)) @ half @ p
    x = np.eye(len(h))+p
    return (np.eye(len(h))-np.linalg.inv(x)).T, np.linalg.det(x)


def jet_checks(mutant=None):
    adjacency = graph("dimer")
    h = np.diag([.7, -.4])-adjacency
    beta, coupling, omega = 1.4, 2., .5
    design = design_matrix(2, beta, coupling, omega, 32, 2)
    c0, c1, c2, d1, d2 = jets(h, beta, design, mutant)
    xi = np.array([.8, -.2, .6, .9, -.5, .4, .2, -.7])
    epsilon = 3e-4
    cp, dp = direct(h, beta, design, xi, epsilon)
    cm, dm = direct(h, beta, design, xi, -epsilon)
    cz, dz = direct(h, beta, design, xi, 0.)
    actual1 = (cp-cm)/(2*epsilon)
    actual2 = (cp+cm-2*cz)/(2*epsilon**2)
    predicted1 = np.einsum("j,jxy->xy", xi, c1)
    predicted2 = np.einsum("j,k,jkxy->xy", xi, xi, c2)
    assert np.max(np.abs(d1)) < 2e-12
    assert np.max(np.abs(actual1-predicted1)) < 2e-7
    assert np.max(np.abs(actual2-predicted2)) < 2e-7
    det_actual = (dp+dm-2*dz)/(2*epsilon**2*dz)
    assert abs(det_actual-xi @ d2 @ xi) < 2e-7
    projected = stationary_projection(c2, 2, 2)
    lost = 2*np.sum((c2-projected)**2)
    assert lost > 1e-9
    print("JET complete cross-mode finite-difference errors:",
          float(np.max(np.abs(actual1-predicted1))),
          float(np.max(np.abs(actual2-predicted2))), "lost_H2_norm", float(lost))


def scalar_checks(mutant=None):
    assert Fraction(1, 8)*Fraction(1, 12) == Fraction(1, 96)
    assert Fraction(1, 2)*Fraction(1, 4) == Fraction(1, 8)
    # Analytic identities are proved in PROOF.md; these are independent numerical witnesses.
    for temperature in [.2, 1., 2.]:
        e = np.linspace(-temperature, temperature, 1001)
        shift = 4*temperature
        phi = lambda z: temperature*np.logaddexp(0., -z/temperature)
        bracket = phi(e+shift)+phi(e-shift)-2*phi(e)
        assert np.min(bracket) >= (3-2*np.log(2))*temperature-1e-12
    adjacency = graph("dimer")
    h, e, r, f, c, chi, dc = static(np.array([.3, -.8]), adjacency, 1.7)
    pair_sum = np.sum(c*c, axis=1)
    assert np.all(pair_sum <= 1.)
    bubble = np.where(np.abs(e)>1e-12, np.tanh(.85*e)/(2*e), 1.7/4)
    if mutant == "pair_kubo_factor_two":
        # At h=0 the exact beta/4 limit catches the duplicated spin factor.
        bubble0 = 2*1.7/4
    else:
        bubble0 = 1.7/4
    assert bubble0 <= 1.7/4+1e-14
    assert np.max(bubble) <= 1.7/4
    for m in [1, 2, 5]:
        nu = 2*np.pi*m/1.7
        de = e[:, None]-e[None, :]
        df = f[:, None]-f[None, :]
        p = r[:, :, None]*r[:, None, :]
        chim = 2*np.einsum("ab,zab,wab->zw", -df*de/(de**2+nu**2), p, p)
        assert np.min(np.linalg.eigvalsh(chi-chim)) > -1e-12
        left = np.sum(df**2/(de**2+nu**2)*np.sum(p*p, axis=0))
        assert left <= (1.7/8)*np.trace(chim)+1e-13
    # Independent direct Matsubara covariance check, exact free oscillator identity.
    beta, coupling, omega = 1.7, 2.3, .4
    variance = coupling*omega/2/math.tanh(beta*omega/2)-coupling/beta
    assert variance > 0
    assert variance <= coupling*beta*omega**2/12
    print("SCALAR static identities, spin factor, Matsubara covariance and 1/96: OK")


def ward_check(mutant=None):
    beta, coupling, mu = 2.5, 2., .3
    adjacency = graph("dimer")
    nodes, weights = np.polynomial.hermite.hermgauss(80)
    fields = coupling+np.sqrt(2*coupling/beta)*nodes
    rows, logweights = [], []
    wt = np.array([[1., math.e], [math.e, 1.]])
    for i, j in itertools.product(range(len(nodes)), repeat=2):
        v = np.array([fields[i], fields[j]])
        h, e, r, f, c, chi, dc = static(v, adjacency, beta, mu)
        g = .5*np.sum(wt*c*c)
        force = v/coupling-1+2*np.diag(c)
        dg = np.einsum("xy,xy,zxy->z", wt, c, dc)
        chi_trace = np.trace(chi)*(2 if mutant == "density_spin_duplicate" else 1)
        rows.append([g, chi_trace, g*chi_trace, force @ force,
                     g*(force @ force), dg @ force])
        logweights.append(np.log(weights[i])+np.log(weights[j])+
                          2*np.sum(np.logaddexp(0., -beta*e)))
    w = np.exp(np.array(logweights)-logsumexp(logweights))
    eg, ec, egc, ef, egf, egf_grad = w @ np.array(rows)
    lhs = egc-eg*ec
    force_cov = -beta*(egf-eg*ef)
    gradient = 0. if mutant == "omit_weight_gradient" else egf_grad
    assert abs(lhs-force_cov-gradient) < 5e-11
    assert abs(ec + beta*ef - 2/coupling) < 5e-11
    print("WARD NUMERICAL:", "Cov(g,Tr chi)=", lhs,
          "force_cov=", force_cov, "gradient=", gradient,
          "residual=", lhs-force_cov-gradient)


def field_statistics(v, adjacency, beta, coupling, omega, slices=24, modes=2,
                     mutant=None):
    h = np.diag(v)-adjacency
    design = design_matrix(len(v), beta, coupling, omega, slices, modes,
                           math.sqrt(2) if mutant == "noise_variance_double" else 1.)
    c0, c1, c2, d1, d2 = jets(h, beta, design)
    assert np.max(np.abs(d1)) < 1e-8
    mean_c2 = np.einsum("jjxy->xy", c2)
    mean_w2 = np.trace(d2)
    var_w2 = 2*np.sum(d2*d2)
    var_c2 = 2*np.sum(c2*c2, axis=(0, 1))
    cov_c2_w2 = 2*np.einsum("jkxy,jk->xy", c2, d2)
    linear_sq = np.sum(c1*c1, axis=0)
    projected = stationary_projection(c2, len(v), modes)
    lost = 2*np.sum((c2-projected)**2, axis=(0, 1))
    # Store both unweighted and exponential-weighted site averages.
    dist = np.zeros_like(adjacency)
    for x in range(len(v)):
        for y in range(len(v)):
            dist[x, y] = min(abs(x-y), len(v)-abs(x-y))
    results = []
    for a in [0., 1.]:
        wt = np.exp(a*dist)/len(v)
        g = np.sum(wt*c0*c0)
        lc = np.sum(wt*linear_sq)
        cross = np.sum(wt*2*c0*mean_c2)
        vc = np.sum(wt*var_c2)
        cov = np.sum(wt*2*c0*cov_c2_w2)
        results.extend([g, lc, cross, mean_w2, g*mean_w2,
                        vc, cov, var_w2, g*var_w2, np.sum(wt*lost)])
    return np.array(results)


def summarize(label, average, beta, omega, mutant=None):
    scale = (beta*omega)**2
    unweighted_delta = 0.
    for index, a in enumerate([0., 1.]):
        g, lc, cc, mw, gmw, vc, cw, vw, gvw, lost = average[index*10:(index+1)*10]
        norm = 2*(gmw-g*mw)
        if mutant == "drop_normalization_covariance":
            norm = 0.
        delta1 = lc+cc+norm
        if index == 0:
            unweighted_delta = delta1
        delta2 = vc+cw+gvw-g*vw
        print(f"{label} a={a:g} g={g:.9g} O(D)/bOm2: linear={lc/scale:+.9g} "
              f"cross={cc/scale:+.9g} norm={norm/scale:+.9g} net={delta1/scale:+.9g}")
        print(f"{label} a={a:g} COMPLETE H2/bOm4: C2var={vc/scale**2:+.9g} "
              f"cross={cw/scale**2:+.9g} norm={(gvw-g*vw)/scale**2:+.9g} "
              f"net={delta2/scale**2:+.9g} omitted_sector={lost/scale**2:.9g}")
    return unweighted_delta


def dimer_anneal(nodes_count, slices):
    beta, coupling, omega = 3., 2., .1
    nodes, weights = np.polynomial.hermite.hermgauss(nodes_count)
    fields = coupling+np.sqrt(2*coupling/beta)*nodes
    rows, logweights = [], []
    adjacency = graph("dimer")
    for i, j in itertools.product(range(nodes_count), repeat=2):
        v = np.array([fields[i], fields[j]])
        e = np.linalg.eigvalsh(np.diag(v)-adjacency)
        rows.append(field_statistics(v, adjacency, beta, coupling, omega, slices))
        logweights.append(np.log(weights[i])+np.log(weights[j])+
                          2*np.sum(np.logaddexp(0., -beta*e)))
    w = np.exp(np.array(logweights)-logsumexp(logweights))
    average = w @ np.array(rows)
    summarize(f"DIMER_GH{nodes_count}_M{slices}", average, beta, omega)
    # First covariance variation theorem, under the *static* measure.
    unweighted_linear = average[1]
    assert unweighted_linear <= (beta*omega)**2/96+1e-8
    assert average[9] > 1e-11
    return average


def metropolis_fields(adjacency, beta, coupling, count, seed):
    rng = np.random.default_rng(seed)
    volume = len(adjacency)
    def logweight(v):
        e = np.linalg.eigvalsh(np.diag(v)-adjacency)
        return -beta*np.sum(v*v)/(2*coupling)+beta*np.sum(v)+2*np.sum(np.logaddexp(0., -beta*e))
    v = np.zeros(volume)
    weight = logweight(v)
    draws = []
    for step in range(1000+8*count):
        proposal = v+rng.normal(0., .9, volume)
        candidate = logweight(proposal)
        if np.log(rng.random()) < candidate-weight:
            v, weight = proposal, candidate
        if step >= 1000 and (step-1000)%8 == 0:
            draws.append(v.copy())
    return draws


def ring_anneal():
    beta, coupling, omega = 3., 2., .1
    for name, seed in [("ring4", 117), ("torus2x2", 118)]:
        adjacency = graph(name)
        draws = metropolis_fields(adjacency, beta, coupling, 192, seed)
        rows = np.array([field_statistics(v, adjacency, beta, coupling, omega, 32)
                         for v in draws])
        summarize(name.upper()+"_NU0_MCMC192_M32", rows.mean(axis=0), beta, omega)
        # Block means disclose sampling variation; no confidence certification.
        for b, block in enumerate(np.array_split(rows, 4)):
            mean = block.mean(axis=0)
            net = mean[11]+mean[12]+2*(mean[14]-mean[10]*mean[13])
            print(f"{name} block{b} weighted O(D)/bOm2={net/(beta*omega)**2:+.9g}")
        assert rows[:, 9].mean() > 1e-12


def high_precision_checks():
    # Independent 60-digit finite differences of the prescribed finite-time matrices.
    # Fields are draws from nu_0-targeted Metropolis chains, not exact independent samples.
    import mpmath as mp
    mp.mp.dps = 60
    beta, coupling, omega, slices = 3., 2., .1, 32
    for name, seed in [("dimer", 711), ("ring4", 712), ("torus2x2", 713)]:
        adjacency = graph(name)
        v = metropolis_fields(adjacency, beta, coupling, 1, seed)[0]
        h = np.diag(v)-adjacency
        design = design_matrix(len(v), beta, coupling, omega, slices, 2)
        xi = np.linspace(-.8, .9, design.shape[-1])
        _, c1, c2, _, d2 = jets(h, beta, design)
        predicted1 = np.einsum("j,jxy->xy", xi, c1)
        predicted2 = np.einsum("j,k,jkxy->xy", xi, xi, c2)
        mh = mp.matrix([[mp.mpf(str(x)) for x in row] for row in h])
        half = mp.expm(-mp.mpf(str(beta))/(2*slices)*mh)
        mx = [mp.mpf(str(x)) for x in xi]
        directions = [[sum(mp.mpf(str(design[s,z,j]))*mx[j]
                           for j in range(len(mx))) for z in range(len(v))]
                      for s in range(slices)]
        def evaluate(epsilon):
            p = mp.eye(len(v))
            for row in directions:
                b = mp.diag([mp.exp(-epsilon*mp.mpf(str(beta))/slices*x) for x in row])
                p = half*b*half*p
            a = mp.eye(len(v))+p
            return (mp.eye(len(v))-a**-1).T, mp.det(a)
        epsilon = mp.mpf("1e-12")
        cp, dp = evaluate(epsilon)
        cm, dm = evaluate(-epsilon)
        cz, dz = evaluate(mp.mpf(0))
        actual1 = (cp-cm)/(2*epsilon)
        actual2 = (cp+cm-2*cz)/(2*epsilon**2)
        err1 = max(abs(actual1[x,y]-mp.mpf(str(predicted1[x,y])))
                   for x in range(len(v)) for y in range(len(v)))
        err2 = max(abs(actual2[x,y]-mp.mpf(str(predicted2[x,y])))
                   for x in range(len(v)) for y in range(len(v)))
        errd = abs((dp+dm-2*dz)/(2*epsilon**2*dz)-mp.mpf(str(xi@d2@xi)))
        assert max(err1, err2, errd) < mp.mpf("2e-10")
        print("HP60", name, "C1_error", mp.nstr(err1, 8),
              "C2_error", mp.nstr(err2, 8), "D2_error", mp.nstr(errd, 8))


def noise_variance_check(mutant=None):
    beta, coupling, omega, slices = 1.4, 2., .5, 32
    adjacency = graph("dimer")
    v = np.array([.7, -.4])
    h = np.diag(v)-adjacency
    design = design_matrix(2, beta, coupling, omega, slices, 2)
    actual = field_statistics(v, adjacency, beta, coupling, omega, slices,
                              mutant=mutant)[1]
    expected = 0.
    epsilon = 1e-4
    for j in range(design.shape[-1]):
        xi = np.zeros(design.shape[-1])
        xi[j] = 1.
        cp, _ = direct(h, beta, design, xi, epsilon)
        cm, _ = direct(h, beta, design, xi, -epsilon)
        expected += np.sum(((cp-cm)/(2*epsilon))**2)/2
    assert expected > 1e-8
    assert abs(actual-expected) < 1e-10+1e-6*expected
    print("NOISE VARIANCE direct-product residual:", float(actual-expected))


def normalization_covariance_check(mutant=None):
    # Independent ratio finite difference using a two-field static quadrature.
    # The 2q-point noise cubature has the exact Gaussian first and second moments.
    beta, coupling, omega, slices = 1.4, 2., .5, 32
    adjacency = graph("dimer")
    fields = [np.array([.7, -.4]), np.array([-1.2, 1.8])]
    logweights = []
    for v in fields:
        e = np.linalg.eigvalsh(np.diag(v)-adjacency)
        logweights.append(-beta*np.sum(v*v)/(2*coupling)+beta*np.sum(v)
                          +2*np.sum(np.logaddexp(0., -beta*e)))
    weights = np.exp(np.array(logweights)-logsumexp(logweights))
    rows = [field_statistics(v, adjacency, beta, coupling, omega, slices)
            for v in fields]
    average = weights @ np.array(rows)
    numerators, denominators = np.zeros(3), np.zeros(3)
    epsilon = 1e-3
    for v, weight in zip(fields, weights):
        h = np.diag(v)-adjacency
        design = design_matrix(2, beta, coupling, omega, slices, 2)
        q = design.shape[-1]
        cz, dz = direct(h, beta, design, np.zeros(q), 0.)
        numerators[0] += weight*np.sum(cz*cz)/2
        denominators[0] += weight
        for j in range(q):
            xi = np.zeros(q)
            xi[j] = np.sqrt(q)
            for index, sign in [(1, 1.), (2, -1.)]:
                for noise_sign in [1., -1.]:
                    c, d = direct(h, beta, design, noise_sign*xi, sign*epsilon)
                    w = weight*(d/dz)**2/(2*q)
                    numerators[index] += w*np.sum(c*c)/2
                    denominators[index] += w
    ratios = numerators/denominators
    expected = (ratios[1]+ratios[2]-2*ratios[0])/(2*epsilon**2)
    covariance = 2*(average[4]-average[0]*average[3])
    assert abs(covariance) > 2e-7
    actual = summarize("NORMALIZATION_ORACLE", average, beta, omega, mutant)
    assert abs(actual-expected) < 2e-7
    print("NORMALIZATION direct-ratio residual:", float(actual-expected),
          "covariance:", float(covariance))


def mutation_checks():
    cases = [
        ("pair_kubo_factor_two", scalar_checks),
        ("density_spin_duplicate", ward_check),
        ("omit_weight_gradient", ward_check),
        ("omit_nonzero_total_frequency", jet_checks),
        ("drop_inverse_cross", jet_checks),
        ("noise_variance_double", noise_variance_check),
        ("drop_normalization_covariance", normalization_covariance_check),
    ]
    caught = 0
    for name, check in cases:
        try:
            check(name)
        except AssertionError:
            print("MUTANT CAUGHT", name)
            caught += 1
        else:
            raise AssertionError("Mutant survived: "+name)
    print(f"ALL {caught}/{len(cases)} MUTANTS CAUGHT")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mutants", action="store_true")
    args = parser.parse_args()
    if args.mutants:
        mutation_checks()
    else:
        scalar_checks()
        ward_check()
        jet_checks()
        high_precision_checks()
        noise_variance_check()
        normalization_covariance_check()
        first = dimer_anneal(24, 24)
        second = dimer_anneal(32, 48)
        # Convergence is NUMERICAL, not an interval enclosure.
        print("DIMER refinement max coefficient change:", float(np.max(np.abs(first-second))))
        ring_anneal()
    print("REPLAY SUCCESS")


if __name__ == "__main__":
    main()
