"""
fast.py -- Produit matrice-vecteur rapide et solveur iteratif.

Methode : FFT precorrigee (pFFT), voir J. R. Phillips et J. K. White,
"A precorrected-FFT method for electrostatic analysis of complicated 3-D
structures", IEEE Trans. CAD 16, 1059 (1997). Meme famille que l'acceleration
multipolaire de FastHenry, adaptee ici a des nappes planes.

Pour un vecteur de sources q_T (une composante de A_T * K_T) on veut

    (G q)_T = Somme_T'  G_TT' q_T' ,   G_TT' = IntInt dA dA' / |r - r'|

1. Champ lointain. Chaque triangle est projete sur une grille reguliere par
   ses trois points de quadrature et des poids bilineaires (operateur creux
   S). Le produit par le noyau 1/r sur la grille est une convolution, donc une
   FFT. Le retour aux triangles utilise S^T, ce qui rend l'operateur
   symetrique :  G_far = S K S^T.
2. Champ proche. Pour les paires de triangles a moins de `near_cells` mailles
   de grille, on retranche ce que la grille a calcule et on ajoute la valeur
   exacte (quadrature 3 points, disque equivalent sur la diagonale).
   G = S K S^T + (G_exact - S K S^T)|_proche.

L'energie E = k.W.k/2 est alors minimisee par gradient conjugue, avec pour
preconditionneur la factorisation LU creuse de la partie locale de
l'operateur (terme cinetique + champ proche exact), qui joue le role de la
"sparsified matrix" de FastHenry.

Cout par iteration : O(N log N). Memoire : O(N).
"""

from __future__ import annotations

import time
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.spatial import cKDTree

try:
    from .core import (Assembly, MU0, PHI0, resistive_field, stream_operator,
                       build_dof_map)
    from .solver import (Solution, _node_loads, _fundamental_cycles,
                         contact_current_operator, contact_currents,
                         _pillar_block)
    from .accel import get_backend
    from .near_pairs import near_pairs
except ImportError:
    from core import (Assembly, MU0, PHI0, resistive_field, stream_operator,
                      build_dof_map)
    from solver import (Solution, _node_loads, _fundamental_cycles,
                        contact_current_operator, contact_currents,
                        _pillar_block)
    from accel import get_backend
    from near_pairs import near_pairs


# ======================================================================
class FastKernel:
    """Operateur q -> G q par FFT precorrigee."""

    def __init__(self, asm: Assembly, grid=None, near_cells=3, prec_cells=1.5,
                 near_hmax=None, backend=None, verbose=True):
        """near_hmax : plafond (um) sur la taille de triangle utilisee pour
        le rayon de precorrection. Sur un maillage gradue, un grand triangle
        adjacent a une zone fine peut avoir des dizaines de milliers de
        voisins dans near_cells * h ; ce plafond borne le nombre de paires.
        None = pas de plafond."""
        t0 = time.time()
        self.asm = asm
        self.bk = backend if backend is not None else get_backend("auto", verbose)
        # coordonnees relatives pour les backends simple precision
        self.cen3 = np.column_stack([asm.centroid, asm.tri_z])
        self.qrel = asm.quad - asm.centroid[:, None, :]
        h_med = float(np.median(asm.h_tri))
        self.hg = float(grid) if grid else 1.0 * h_med
        self.near_cells = int(near_cells)

        # ---------------- plans en z
        zs = np.unique(np.round(asm.tri_z, 9))
        self.planes = zs
        self.tri_plane = np.searchsorted(zs, np.round(asm.tri_z, 9))
        npl = len(zs)

        # ---------------- grille
        pts = asm.points
        lo = pts.min(axis=0) - 2 * self.hg
        hi = pts.max(axis=0) + 2 * self.hg
        self.x0, self.y0 = lo
        self.nx = int(np.ceil((hi[0] - lo[0]) / self.hg)) + 2
        self.ny = int(np.ceil((hi[1] - lo[1]) / self.hg)) + 2
        ng = self.nx * self.ny

        # ---------------- projection bilineaire des points de quadrature
        # footprint : 12 noeuds (3 points x 4 coins), poids * A_T/3
        Q = asm.quad.reshape(-1, 2)                        # (3M,2)
        u = (Q[:, 0] - self.x0) / self.hg
        v = (Q[:, 1] - self.y0) / self.hg
        i0 = np.floor(u).astype(int)
        j0 = np.floor(v).astype(int)
        fu, fv = u - i0, v - j0
        wts = np.stack([(1 - fu) * (1 - fv), fu * (1 - fv),
                        (1 - fu) * fv, fu * fv], axis=1)              # (3M,4)
        ii = np.stack([i0, i0 + 1, i0, i0 + 1], axis=1)
        jj = np.stack([j0, j0, j0 + 1, j0 + 1], axis=1)
        gidx = ii * self.ny + jj                                       # (3M,4)
        tri_idx = np.repeat(np.arange(asm.n_tri), 3)
        wts = wts * (asm.area[tri_idx] / 3.0)[:, None]         # poids * A_T/3
        rows = np.repeat(tri_idx, 4)
        cols = gidx.ravel()
        vals = wts.ravel()
        S = sp.coo_matrix((vals, (rows, cols)), shape=(asm.n_tri, ng)).tocsr()
        S.sum_duplicates()
        self.S = S
        # empreinte par triangle (indices entiers) pour la precorrection
        self.fp_i = ii.reshape(asm.n_tri, 12)
        self.fp_j = jj.reshape(asm.n_tri, 12)
        self.fp_w = wts.reshape(asm.n_tri, 12)

        # ---------------- noyaux FFT par paire de plans
        NX, NY = 2 * self.nx, 2 * self.ny
        if verbose:
            nker = npl * (npl + 1) // 2
            mem = NX * NY * 8 * 4 + nker * NX * (NY // 2 + 1) * 16
            print(f"  pFFT: grid {self.nx}x{self.ny}, FFT {NX}x{NY}, "
                  f"{nker} kernel(s), ~{mem/1e9:.1f} GB for the kernels",
                  flush=True)
        ix = np.fft.fftfreq(NX, d=1.0 / NX)          # -> 0..nx-1, -nx..-1
        iy = np.fft.fftfreq(NY, d=1.0 / NY)
        X = ix[:, None] * self.hg
        Y = iy[None, :] * self.hg
        R2 = X * X + Y * Y
        self.Khat = {}
        self.k0 = {}
        for a in range(npl):
            for b in range(a, npl):
                dz = zs[b] - zs[a]
                K = 1.0 / np.sqrt(R2 + dz * dz + 1e-300)
                if dz == 0:
                    # valeur au noeud origine, distance inverse moyenne
                    # dans une maille carree (~ 2.97/h), arbitraire car
                    # corrigee en champ proche
                    K[0, 0] = 2.973 / self.hg
                self.Khat[(a, b)] = self.bk.prepare_kernel(K)
                self.Khat[(b, a)] = self.Khat[(a, b)]
        self.NX, self.NY = NX, NY

        # ---------------- paires proches
        # rayon adapte a la taille locale : la quadrature a 3 points et la
        # projection bilineaire ne sont fiables qu'a une distance de quelques
        # tailles de triangle ET de quelques mailles de grille
        h = asm.h_tri
        h_eff = h if near_hmax is None else np.minimum(h, float(near_hmax))
        rad = self.near_cells * np.maximum(self.hg, h_eff)
        C3 = np.column_stack([asm.centroid, asm.tri_z])
        I, J = near_pairs(C3, rad, verbose=verbose)
        self.n_near = len(I)
        if verbose:
            print(f"  near pairs: {self.n_near} after symmetrisation, "
                  f"~{self.n_near * 56 / 1e9:.1f} GB for the correction",
                  flush=True)

        # table du noyau grille-grille assez large pour les plus grands
        R = int(np.ceil((rad.max() + 2.0 * h.max()) / self.hg)) + 2
        self.Rtab = R
        d = np.arange(-R, R + 1) * self.hg
        D2 = d[:, None] ** 2 + d[None, :] ** 2
        self.Ktab = {}
        for a in range(npl):
            for b in range(npl):
                dz = zs[b] - zs[a]
                T = 1.0 / np.sqrt(D2 + dz * dz + 1e-300)
                if dz == 0:
                    T[R, R] = 2.973 / self.hg
                self.Ktab[(a, b)] = T

        Gex = self.bk.exact_pairs(self.cen3, self.qrel, asm.tri_z, asm.area,
                                  I, J)
        tab = np.empty((npl, npl, 2 * R + 1, 2 * R + 1))
        for (a, b), T in self.Ktab.items():
            tab[a, b] = T
        Ggr = self.bk.grid_pairs(self.fp_i, self.fp_j, self.fp_w,
                                 self.tri_plane, tab, R, I, J)
        self.G_corr = sp.coo_matrix((Gex - Ggr, (I, J)),
                                    shape=(asm.n_tri, asm.n_tri)).tocsr()
        # sous-ensemble plus compact pour le preconditionneur
        dist = np.linalg.norm(C3[I] - C3[J], axis=1)
        mp = dist <= prec_cells * np.maximum(self.hg, np.maximum(h[I], h[J]))
        self.G_near_exact = sp.coo_matrix((Gex[mp], (I[mp], J[mp])),
                                          shape=(asm.n_tri, asm.n_tri)).tocsr()
        del Gex, Ggr, dist, mp
        if verbose:
            print(f"  pFFT: grid {self.nx}x{self.ny} (h={self.hg:.3g} um, "
                  f"FFT {2*self.nx}x{2*self.ny}), {npl} plane(s), "
                  f"{self.n_near} near pairs ({self.n_near/asm.n_tri:.0f}"
                  f"/triangle), backend {self.bk.name}, {time.time()-t0:.1f} s")

    # ------------------------------------------------------------------
    def apply(self, q):
        """G q pour q (n_tri,) ou (n_tri, m)."""
        q = np.asarray(q, dtype=float)
        one_d = q.ndim == 1
        if one_d:
            q = q[:, None]
        m = q.shape[1]
        out = np.zeros_like(q)
        npl = len(self.planes)
        # sources sur la grille, par plan
        Qg = []
        for a in range(npl):
            sel = self.tri_plane == a
            qa = np.zeros_like(q)
            qa[sel] = q[sel]
            g = (self.S.T @ qa).reshape(self.nx, self.ny, m)
            Qg.append(g)
        # convolutions
        for b in range(npl):
            pot = np.zeros((self.nx, self.ny, m))
            for a in range(npl):
                for c in range(m):
                    if not np.any(Qg[a][:, :, c]):
                        continue
                    pot[:, :, c] += self.bk.conv(Qg[a][:, :, c],
                                                 self.Khat[(a, b)],
                                                 self.NX, self.NY)
            sel = self.tri_plane == b
            full = self.S @ pot.reshape(-1, m)
            out[sel] += full[sel]
        out += self.G_corr @ q
        return out[:, 0] if one_d else out


# ======================================================================
def solve_currents_fast(asm: Assembly, terminals, contacts=(), fluxoid=None,
                        terminal_profile="uniform", grid=None, near_cells=4,
                        prec_cells=1.5, near_hmax=None, tol=None, maxiter=400,
                        kernel=None, backend=None, pillar_L=None,
                        drive=None, verbose=True):
    """Meme probleme que solver.solve_currents, resolu iterativement.

    drive : (n_tri, 2) linear term a_T of the energy, G = E + Sum_T a_T.K_T,
             see core.external_field_drive (uniform applied field). The
             returned energy is the self energy E = K.W.K/2, the work term
             Sum a.K is stored in info['drive_energy'].

    kernel : FastKernel deja construit (le noyau ne depend que de la
             geometrie, on peut le reutiliser quand seul Lambda change).
    pillar_L : matrice des inductances des piliers, voir solver.solve_currents.
    """
    t0 = time.time()

    # ------------------------------------------------ amorcage et boucles
    loads = np.zeros(asm.n_nodes)
    for t in terminals:
        loads += _node_loads(asm, t.nodes, t.current, terminal_profile)
    if abs(loads.sum()) > 1e-12 * max(1.0, np.abs(loads).sum()):
        raise ValueError("terminal currents do not sum to zero")
    merge = [np.concatenate([c.nodes_a, c.nodes_b]) for c in contacts]
    Sstiff = asm.stiffness()
    if np.any(loads):
        kseed = resistive_field(asm, loads, merge_groups=merge, K_stiff=Sstiff)
    else:
        kseed = np.zeros((asm.n_tri, 2))     # no injection (field response)

    ncomp, lab = asm.components()
    edges = [(int(lab[c.nodes_a[0]]), int(lab[c.nodes_b[0]])) for c in contacts]
    cycles = _fundamental_cycles(ncomp, edges) if contacts else []
    nloop = len(cycles)
    B = np.zeros((asm.n_tri, 2, nloop))
    for ci, cyc in enumerate(cycles):
        f = np.zeros(asm.n_nodes)
        for k, s in cyc:
            c = contacts[k]
            f -= _node_loads(asm, c.nodes_a, s, "uniform")
            f += _node_loads(asm, c.nodes_b, s, "uniform")
        B[:, :, ci] = resistive_field(asm, f, merge_groups=None, K_stiff=Sstiff)

    node_dof, ndof, hole_dofs, loops = build_dof_map(asm)
    Cx, Cy = stream_operator(asm, node_dof, ndof)
    ntot = ndof + nloop
    if verbose:
        print(f"  {asm.n_tri} triangles, {ndof} dofs, {len(hole_dofs)} holes, "
              f"{nloop} bridge loop(s)")

    # ------------------------------------------------ operateur rapide
    FK = kernel if kernel is not None else FastKernel(
        asm, grid=grid, near_cells=near_cells, prec_cells=prec_cells,
        near_hmax=near_hmax, backend=backend, verbose=verbose)
    if tol is None:
        tol = 1e-8 if FK.bk.dtype == np.float64 else 3e-6
    wkin = MU0 * asm.tri_Lambda * asm.area
    cmag = MU0 / (4.0 * np.pi)

    # piliers : terme de rang fini D^T L D, D = operateur courant de contact
    Lp = _pillar_block(pillar_L, contacts)
    if Lp is not None:
        Dx, Dy = contact_current_operator(asm, contacts)

    def W_apply(k):
        """k (n_tri,2) -> W k."""
        out = wkin[:, None] * k + cmag * FK.apply(k)
        if Lp is not None:
            v = Lp @ (Dx @ k[:, 0] + Dy @ k[:, 1])
            out[:, 0] += Dx.T @ v
            out[:, 1] += Dy.T @ v
        return out

    def expand(x):
        g, al = x[:ndof], x[ndof:]
        k = np.column_stack([Cx @ g, Cy @ g])
        if nloop:
            k += np.einsum('mdc,c->md', B, al)
        return k

    def contract(y):
        out = np.empty(ntot)
        out[:ndof] = Cx.T @ y[:, 0] + Cy.T @ y[:, 1]
        if nloop:
            out[ndof:] = np.einsum('mdc,md->c', B, y)
        return out

    def matvec(x):
        return contract(W_apply(expand(x)))

    A = spla.LinearOperator((ntot, ntot), matvec=matvec, dtype=float)

    rhs = -contract(W_apply(kseed))
    if drive is not None:
        drive = np.asarray(drive, dtype=float)
        rhs -= contract(drive)
    if fluxoid:
        for dofi, loopi in hole_dofs:
            n = fluxoid.get(loopi, 0)
            if n:
                rhs[dofi] += n * PHI0

    # ------------------------------------------------ preconditionneur
    Wloc = sp.diags(wkin) + cmag * FK.G_near_exact
    Hloc = (Cx.T @ Wloc @ Cx + Cy.T @ Wloc @ Cy).tocsc()
    if nloop:
        # bloc dense des boucles, traite par complement de Schur approche
        Hbb = np.zeros((nloop, nloop))
        Hbd = np.zeros((nloop, ndof))
        for c in range(nloop):
            wb = Wloc @ B[:, 0, c], Wloc @ B[:, 1, c]
            Hbd[c] = Cx.T @ wb[0] + Cy.T @ wb[1]
            Hbb[c] = B[:, 0, :].T @ wb[0] + B[:, 1, :].T @ wb[1]
        if Lp is not None:
            RB = Dx @ B[:, 0, :] + Dy @ B[:, 1, :]      # (n_contacts, nloop)
            Hbb += RB.T @ Lp @ RB
        Hloc = sp.bmat([[Hloc, sp.csc_matrix(Hbd.T)],
                        [sp.csc_matrix(Hbd), sp.csc_matrix(Hbb)]]).tocsc()
    lu, nnz, kind = _factorize(Hloc)
    M = spla.LinearOperator((ntot, ntot), matvec=lu, dtype=float)
    if verbose:
        print(f"  preconditioner {kind}, nnz={nnz}, {time.time()-t0:.1f} s")

    # ------------------------------------------------ gradient conjugue
    it = [0]

    def cb(xk):
        it[0] += 1

    x, info = _cg(A, rhs, M, tol, maxiter, cb)
    res = np.linalg.norm(A @ x - rhs) / max(np.linalg.norm(rhs), 1e-300)
    if info != 0 or res > 100 * tol:
        if verbose:
            print(f"  CG did not converge (res={res:.2e}), switching to MINRES")
        x, info = _minres(A, rhs, x, M, tol, 4 * maxiter)
        res = np.linalg.norm(A @ x - rhs) / max(np.linalg.norm(rhs), 1e-300)
    if verbose:
        print(f"  iterative solver: {it[0]} CG iterations, relative residual "
              f"{res:.2e}, total {time.time()-t0:.1f} s")

    K = kseed + expand(x)
    info_d = dict(node_dof=node_dof, loops=loops, hole_dofs=hole_dofs,
                  g=x[:ndof], alpha=x[ndof:], kseed=kseed, cycles=cycles,
                  residual=res, iterations=it[0], kernel=FK, W_apply=W_apply)
    # fluxoide reel = H x + b = matvec(x) - rhs_sans_flux
    info_d['fluxoids'] = matvec(x) + contract(W_apply(kseed))
    if drive is not None:
        info_d['fluxoids'] += contract(drive)
        info_d['drive_energy'] = float(np.sum(drive * K))
    if contacts:
        info_d['contact_currents'] = contact_currents(asm, K, contacts)
    sol = Solution(asm, K, None, info_d)
    sol.energy = total_energy_fast(asm, K, W_apply)
    return sol


def _cg(A, b, M, tol, maxiter, cb):
    """scipy >= 1.12 utilise rtol, les versions anterieures tol."""
    try:
        return spla.cg(A, b, M=M, rtol=tol, maxiter=maxiter, callback=cb)
    except TypeError:
        return spla.cg(A, b, M=M, tol=tol, atol=0.0, maxiter=maxiter,
                       callback=cb)


def _minres(A, b, x0, M, tol, maxiter):
    try:
        return spla.minres(A, b, x0=x0, M=M, rtol=tol, maxiter=maxiter)
    except TypeError:
        return spla.minres(A, b, x0=x0, M=M, tol=tol, maxiter=maxiter)


def _factorize(H):
    """Factorisation creuse de la matrice locale (SPD).

    CHOLMOD (scikit-sparse, multithread via SuiteSparse et le BLAS) si
    disponible, sinon SuperLU de scipy. Retourne (solve, nnz, description).
    """
    try:
        from sksparse.cholmod import cholesky
        F = cholesky(H.tocsc())
        L = F.L()
        return F.solve_A, int(L.nnz), "CHOLMOD Cholesky"
    except Exception:
        pass
    try:
        lu = spla.splu(H.tocsc())
        return lu.solve, int(lu.L.nnz + lu.U.nnz), "SuperLU LU"
    except RuntimeError:
        lu = spla.spilu(H.tocsc(), drop_tol=1e-5, fill_factor=20)
        return lu.solve, int(lu.L.nnz + lu.U.nnz), "incomplete LU"


def total_energy_fast(asm, K, W_apply):
    """E = k.W.k / 2 avec l'operateur rapide."""
    return 0.5 * float(np.sum(K * W_apply(K)))
