"""
core.py -- Noyau physique du solveur de densite de courant DC
dans des films supraconducteurs minces (modele de London 2D).

Unite de longueur : le micron. Courants en A, flux en Wb, inductances en H.

Le noyau ne depend que de numpy / scipy. Il travaille sur un maillage
triangulaire deja construit, ce qui permet de le tester independamment
de la chaine GDS -> polygones -> maillage.

Formulation
-----------
Chaque film est traite comme une feuille de courant K (A/um) situee a
une altitude z_s. L'energie totale s'ecrit

    E[K] = (mu0*Lambda/2) * Int |K|^2 dA
         + (mu0/(8*pi)) * IntInt K(r).K(r') / |r-r'| dA dA'

Le premier terme est l'energie cinetique des paires de Cooper
(Lambda = lambda*coth(d/lambda), avec la limite mince Lambda = lambda^2/d),
le second est l'energie magnetique du champ rayonne.

En regime DC etabli, E = 0 dans le supraconducteur, donc le reseau est
purement inductif. Si le circuit est refroidi en champ nul puis le courant
monte de 0 a I, la distribution finale est l'unique minimum de E sous les
contraintes de conservation du courant, d'injection aux terminaux, et de
quantification du fluxoide (n = 0 par defaut dans chaque trou).

La stationnarite de E redonne l'equation de London integree sur l'epaisseur

    mu0 * Lambda * K(r) + A(r) = - (hbar/(2e)) * grad(theta)

et la condition de fluxoide Int_C (mu0*Lambda*K + A).dl = n*Phi0 autour de
chaque trou.

Discretisation
--------------
K est constant par triangle. On ecrit K = K_seed + somme(alpha_b * F_b)
+ z x grad(g), ou
  - K_seed est une distribution admissible quelconque (obtenue par un
    probleme resistif auxiliaire) qui porte l'injection aux terminaux,
  - F_b sont les champs de boucle associes aux ponts (airbridges), qui
    portent le transfert vertical de courant entre feuilles,
  - g est une fonction de courant P1 nodale, constante sur chaque contour
    de chaque feuille (une constante libre par trou, fixee a 0 sur le
    contour exterieur de chaque composante connexe).

Le minimum de E est alors solution d'un systeme lineaire dense de taille
(nombre de degres de liberte de g) + (nombre de boucles de ponts).
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import scipy.sparse.csgraph  # noqa: F401

# ----------------------------------------------------------------------
# Constantes (longueurs en micron)
# ----------------------------------------------------------------------
MU0 = 4.0e-13 * np.pi          # H/um   (4*pi*1e-7 H/m)
PHI0 = 2.067833848e-15         # Wb


def sheet_inductance_to_Lambda(L_square: float) -> float:
    """Lambda (um) a partir de l'inductance cinetique par carre L_square (H).

    L_square = mu0 * Lambda, donc Lambda = L_square / mu0.
    Pour un film mince L_square = mu0 * lambda^2 / d = hbar * R_square / (pi * Delta).
    """
    return float(L_square) / MU0


# constantes Josephson
HBAR = 1.054571817e-34
E_CHARGE = 1.602176634e-19
H_PLANCK = 6.62607015e-34


def josephson_inductance(EJ_GHz=None, EJ_J=None, Ic_A=None, LJ_H=None,
                         Rn_Ohm=None, Delta_ueV=None):
    """Inductance Josephson lineaire L_J = Phi0 / (2 pi I_c) en henry.

    Une seule des entrees suffit. Conversions :
        E_J = Phi0 I_c / (2 pi)           E_J [GHz] = E_J / h * 1e-9
        L_J = (Phi0 / 2 pi)^2 / E_J
        Ambegaokar-Baratoff a T = 0 : I_c R_n = pi Delta / (2 e)
    Retourne (L_J, I_c, E_J_joule).
    """
    phi0_2pi = PHI0 / (2 * np.pi)
    if LJ_H is not None:
        LJ = float(LJ_H)
        Ic = phi0_2pi / LJ
    elif Ic_A is not None:
        Ic = float(Ic_A)
        LJ = phi0_2pi / Ic
    elif EJ_J is not None or EJ_GHz is not None:
        EJ = float(EJ_J) if EJ_J is not None else float(EJ_GHz) * 1e9 * H_PLANCK
        LJ = phi0_2pi ** 2 / EJ
        Ic = phi0_2pi / LJ
    elif Rn_Ohm is not None:
        if Delta_ueV is None:
            raise ValueError("Delta_ueV is required together with Rn_Ohm")
        Ic = np.pi * float(Delta_ueV) * 1e-6 * E_CHARGE / (2 * E_CHARGE) / float(Rn_Ohm)
        LJ = phi0_2pi / Ic
    else:
        raise ValueError("give EJ_GHz, EJ_J, Ic_A, LJ_H or (Rn_Ohm, Delta_ueV)")
    return LJ, Ic, phi0_2pi * Ic


def effective_penetration_depth(lambda_L: float, thickness: float) -> float:
    """Lambda = lambda * coth(d/lambda).

    Limite d << lambda : Lambda -> lambda^2/d.
    Limite d >> lambda : Lambda -> lambda.
    La longueur de Pearl correspondante vaut 2*Lambda.
    """
    x = thickness / lambda_L
    if x < 1e-6:
        return lambda_L / x
    return lambda_L / np.tanh(x)


# ----------------------------------------------------------------------
# Structures de maillage
# ----------------------------------------------------------------------
class Sheet:
    """Une feuille supraconductrice plane maillee en triangles.

    points    : (N,2) coordonnees en um
    triangles : (M,3) indices
    z         : altitude en um
    Lambda    : longueur de penetration effective en um
    name      : identifiant
    """

    def __init__(self, points, triangles, z=0.0, Lambda=0.1, name="sheet"):
        self.points = np.asarray(points, dtype=float)
        self.triangles = np.asarray(triangles, dtype=int)
        self.z = float(z)
        Lambda = np.asarray(Lambda, dtype=float)
        # scalaire (feuille homogene) ou tableau par triangle (jonctions,
        # materiaux differents)
        if Lambda.ndim == 0:
            self.Lambda = np.full(len(self.triangles), float(Lambda))
        else:
            if len(Lambda) != len(self.triangles):
                raise ValueError("Lambda must be a scalar or an array of size n_tri")
            self.Lambda = Lambda
        self.name = name


class Assembly:
    """Concatenation de plusieurs feuilles + operateurs geometriques."""

    def __init__(self, sheets):
        self.sheets = list(sheets)
        self._build()

    # ---------------- construction ----------------
    def _build(self):
        pts, tris, zs, lam = [], [], [], []
        self.node_offset, self.tri_offset, self.node_sheet, self.tri_sheet = [], [], [], []
        n0 = t0 = 0
        for si, s in enumerate(self.sheets):
            self.node_offset.append(n0)
            self.tri_offset.append(t0)
            pts.append(s.points)
            tris.append(s.triangles + n0)
            zs.append(np.full(len(s.triangles), s.z))
            lam.append(s.Lambda)
            self.node_sheet.append(np.full(len(s.points), si))
            self.tri_sheet.append(np.full(len(s.triangles), si))
            n0 += len(s.points)
            t0 += len(s.triangles)
        self.points = np.vstack(pts)
        self.triangles = np.vstack(tris)
        self.tri_z = np.concatenate(zs)
        self.tri_Lambda = np.concatenate(lam)
        self.node_sheet = np.concatenate(self.node_sheet)
        self.tri_sheet = np.concatenate(self.tri_sheet)
        self.n_nodes = len(self.points)
        self.n_tri = len(self.triangles)

        p = self.points[self.triangles]          # (M,3,2)
        x0, y0 = p[:, 0, 0], p[:, 0, 1]
        x1, y1 = p[:, 1, 0], p[:, 1, 1]
        x2, y2 = p[:, 2, 0], p[:, 2, 1]
        area2 = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)   # 2*aire signee
        # orientation directe imposee
        flip = area2 < 0
        if np.any(flip):
            self.triangles[flip] = self.triangles[flip][:, [0, 2, 1]]
            p = self.points[self.triangles]
            x0, y0 = p[:, 0, 0], p[:, 0, 1]
            x1, y1 = p[:, 1, 0], p[:, 1, 1]
            x2, y2 = p[:, 2, 0], p[:, 2, 1]
            area2 = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)
        self.area = 0.5 * area2
        self.centroid = p.mean(axis=1)

        # gradients des fonctions de forme P1 : (M,3,2)
        g = np.empty((self.n_tri, 3, 2))
        g[:, 0, 0] = (y1 - y2); g[:, 0, 1] = (x2 - x1)
        g[:, 1, 0] = (y2 - y0); g[:, 1, 1] = (x0 - x2)
        g[:, 2, 0] = (y0 - y1); g[:, 2, 1] = (x1 - x0)
        self.grad = g / area2[:, None, None]

        # aire nodale forfaitaire (pour repartir les injections)
        self.node_area = np.zeros(self.n_nodes)
        np.add.at(self.node_area, self.triangles.ravel(),
                  np.repeat(self.area / 3.0, 3))

        # points de quadrature a 3 points (barycentriques 2/3,1/6,1/6)
        bary = np.array([[2 / 3, 1 / 6, 1 / 6],
                         [1 / 6, 2 / 3, 1 / 6],
                         [1 / 6, 1 / 6, 2 / 3]])
        self.quad = np.einsum('qk,mkd->mqd', bary, p)          # (M,3,2)
        self.h_tri = np.sqrt(self.area)

    # ---------------- topologie ----------------
    def boundary_loops(self):
        """Retourne (loops, node_loop) : contours du maillage.

        loops     : liste de listes de noeuds ordonnes
        node_loop : tableau (n_nodes,) valant -1 hors contour
        """
        e = np.vstack([self.triangles[:, [0, 1]],
                       self.triangles[:, [1, 2]],
                       self.triangles[:, [2, 0]]])
        key = np.sort(e, axis=1)
        uniq, inv, cnt = np.unique(key, axis=0, return_inverse=True,
                                   return_counts=True)
        bmask = cnt[inv] == 1
        bedges = e[bmask]                     # orientees (bord a gauche)
        nxt = {int(a): int(b) for a, b in bedges}
        node_loop = np.full(self.n_nodes, -1, dtype=int)
        loops = []
        visited = set()
        for a in nxt:
            if a in visited:
                continue
            loop, cur = [], a
            while cur not in visited:
                visited.add(cur)
                loop.append(cur)
                cur = nxt.get(cur, None)
                if cur is None:
                    break
            if len(loop) >= 3:
                for n in loop:
                    node_loop[n] = len(loops)
                loops.append(loop)
        return loops, node_loop

    def components(self):
        """Composantes connexes du maillage (par noeud et par triangle)."""
        i = np.concatenate([self.triangles[:, 0], self.triangles[:, 1],
                            self.triangles[:, 2]])
        j = np.concatenate([self.triangles[:, 1], self.triangles[:, 2],
                            self.triangles[:, 0]])
        A = sp.coo_matrix((np.ones(len(i)), (i, j)),
                          shape=(self.n_nodes, self.n_nodes))
        ncomp, labels = sp.csgraph.connected_components(A, directed=False)
        return ncomp, labels

    def stiffness(self):
        """Matrice de rigidite Int grad(phi_i).grad(phi_j) dA."""
        rows, cols, vals = [], [], []
        for a in range(3):
            for b in range(3):
                rows.append(self.triangles[:, a])
                cols.append(self.triangles[:, b])
                vals.append(np.einsum('md,md->m', self.grad[:, a],
                                      self.grad[:, b]) * self.area)
        return sp.coo_matrix((np.concatenate(vals),
                              (np.concatenate(rows), np.concatenate(cols))),
                             shape=(self.n_nodes, self.n_nodes)).tocsr()


# ----------------------------------------------------------------------
# Probleme resistif auxiliaire (construction des champs admissibles)
# ----------------------------------------------------------------------
def resistive_field(asm: Assembly, loads, merge_groups=None, K_stiff=None):
    """Champ de courant conservatif porte par des charges nodales.

    loads : (n_nodes,) courant injecte a chaque noeud (somme nulle par
            composante connexe du reseau *apres* fusion).
    merge_groups : liste de listes de noeuds forcement equipotentiels
            (contacts parfaits entre feuilles).

    Retourne K (n_tri,2) tel que, au sens faible,
        Int grad(phi_i).K dA = -loads_i.
    """
    n = asm.n_nodes
    S = asm.stiffness() if K_stiff is None else K_stiff

    # application noeud -> ddl apres fusion
    dof = np.arange(n)
    if merge_groups:
        for grp in merge_groups:
            if len(grp) > 1:
                dof[np.asarray(grp, dtype=int)] = int(min(grp))
    uniq, dof = np.unique(dof, return_inverse=True)
    ndof = len(uniq)
    R = sp.coo_matrix((np.ones(n), (np.arange(n), dof)),
                      shape=(n, ndof)).tocsr()

    Sm = (R.T @ S @ R).tocsr()
    fm = R.T @ np.asarray(loads, dtype=float)

    # une contrainte de jauge par composante connexe du graphe fusionne
    ncomp, lab = sp.csgraph.connected_components(Sm, directed=False)
    resid = np.zeros(ncomp)
    np.add.at(resid, lab, fm)
    if np.max(np.abs(resid)) > 1e-9 * max(1.0, np.max(np.abs(fm))):
        bad = np.where(np.abs(resid) > 1e-9 * max(1.0, np.max(np.abs(fm))))[0]
        raise ValueError("unbalanced loads on a piece of metal not connected "
                         f"in DC to the rest (components {bad.tolist()}, "
                         f"residuals {resid[bad]}). A terminal lies on a piece "
                         "without return path, use solver.build_terminal_sets "
                         "to restrict the returns to the connected pads")

    Sm = Sm.tolil()
    pins = []
    for c in range(ncomp):
        k = int(np.argmax(lab == c))
        pins.append(k)
    for k in pins:
        Sm.rows[k] = [k]
        Sm.data[k] = [1.0]
        fm[k] = 0.0
    Sm = Sm.tocsc()
    Vm = spla.spsolve(Sm, fm)
    V = R @ Vm

    K = -np.einsum('mkd,mk->md', asm.grad, V[asm.triangles])
    return K


# ----------------------------------------------------------------------
# Noyau magnetique
# ----------------------------------------------------------------------
def _pair_kernel(asm: Assembly, rows):
    """Bloc G[rows, :] du noyau IntInt dA dA' / |r-r'| (3D).

    Quadrature a 3 points par triangle hors diagonale, valeur analytique
    du disque equivalent sur la diagonale :
        IntInt dA dA'/|r-r'| = (16*pi/3) a^3,  a = sqrt(A/pi)
    (la distance moyenne inverse entre deux points d'un disque vaut
     16/(3*pi*a)).
    """
    qi = asm.quad[rows]                     # (n,3,2)
    zi = asm.tri_z[rows]
    ai = asm.area[rows]
    qj = asm.quad                           # (M,3,2)
    zj = asm.tri_z
    aj = asm.area

    acc = np.zeros((len(rows), asm.n_tri))
    dz = zi[:, None] - zj[None, :]
    dz2 = dz * dz
    for p in range(3):
        for q in range(3):
            dx = qi[:, p, 0][:, None] - qj[None, :, q, 0]
            dy = qi[:, p, 1][:, None] - qj[None, :, q, 1]
            d2 = dx * dx + dy * dy + dz2
            np.maximum(d2, 1e-24, out=d2)
            acc += 1.0 / np.sqrt(d2)
    G = (ai[:, None] * aj[None, :]) * (acc / 9.0)

    # diagonale
    diag = (16.0 * np.pi / 3.0) * (asm.area[rows] / np.pi) ** 1.5
    G[np.arange(len(rows)), rows] = diag
    return G


def assemble_reduced_system(asm: Assembly, C, B, kseed, chunk=192,
                            progress=False):
    """Assemble la forme quadratique reduite.

    C : liste [Cx, Cy] matrices creuses (n_tri, ndof)  -- partie z x grad(g)
    B : (n_tri, 2, nloop) champs de boucle
    kseed : (n_tri, 2)

    Retourne H (ndof+nloop carre) et b (ndof+nloop) tels que
        E(x) = cste + b.x + x.H.x/2
    """
    ndof = C[0].shape[1]
    nloop = B.shape[2] if B is not None and B.size else 0
    ntot = ndof + nloop
    H = np.zeros((ntot, ntot))
    b = np.zeros(ntot)

    Cx, Cy = C[0].tocsr(), C[1].tocsr()
    Bx = B[:, 0, :] if nloop else np.zeros((asm.n_tri, 0))
    By = B[:, 1, :] if nloop else np.zeros((asm.n_tri, 0))

    # --- terme cinetique (diagonal en triangles) ---
    w = MU0 * asm.tri_Lambda * asm.area
    Wx = sp.diags(w) @ Cx
    Wy = sp.diags(w) @ Cy
    H[:ndof, :ndof] += (Cx.T @ Wx + Cy.T @ Wy).toarray()
    if nloop:
        Hdl = Bx.T @ Wx + By.T @ Wy           # (nloop, ndof)
        H[ndof:, :ndof] += Hdl
        H[:ndof, ndof:] += Hdl.T
        H[ndof:, ndof:] += (Bx * w[:, None]).T @ Bx + (By * w[:, None]).T @ By
    b[:ndof] += Cx.T @ (w * kseed[:, 0]) + Cy.T @ (w * kseed[:, 1])
    if nloop:
        b[ndof:] += Bx.T @ (w * kseed[:, 0]) + By.T @ (w * kseed[:, 1])

    # --- terme magnetique (dense) ---
    c = MU0 / (4.0 * np.pi)
    nT = asm.n_tri
    for s in range(0, nT, chunk):
        rows = np.arange(s, min(s + chunk, nT))
        G = _pair_kernel(asm, rows) * c
        GCx = G @ Cx          # (n, ndof) dense
        GCy = G @ Cy
        H[:ndof, :ndof] += Cx[rows].T @ GCx + Cy[rows].T @ GCy
        b[:ndof] += Cx[rows].T @ (G @ kseed[:, 0]) + \
                    Cy[rows].T @ (G @ kseed[:, 1])
        if nloop:
            GBx, GBy = G @ Bx, G @ By
            Hdl = Bx[rows].T @ GCx + By[rows].T @ GCy
            H[ndof:, :ndof] += Hdl
            H[:ndof, ndof:] += (Cx[rows].T @ GBx + Cy[rows].T @ GBy)
            H[ndof:, ndof:] += Bx[rows].T @ GBx + By[rows].T @ GBy
            b[ndof:] += Bx[rows].T @ (G @ kseed[:, 0]) + \
                        By[rows].T @ (G @ kseed[:, 1])
        if progress and (s // chunk) % 10 == 0:
            print(f"    kernel {s}/{nT}", flush=True)

    H = 0.5 * (H + H.T)
    return H, b


# ----------------------------------------------------------------------
# Operateur z x grad(g) et reduction des degres de liberte
# ----------------------------------------------------------------------
def stream_operator(asm: Assembly, node_dof, ndof):
    """Matrices creuses Cx, Cy telles que K = [Cx,Cy] @ g_dof.

    K = z x grad(g) = (-dg/dy, dg/dx).
    """
    rows = np.repeat(np.arange(asm.n_tri), 3)
    cols = node_dof[asm.triangles].ravel()
    gx = asm.grad[:, :, 0].ravel()
    gy = asm.grad[:, :, 1].ravel()
    keep = cols >= 0
    Cx = sp.coo_matrix((-gy[keep], (rows[keep], cols[keep])),
                       shape=(asm.n_tri, ndof)).tocsr()
    Cy = sp.coo_matrix((gx[keep], (rows[keep], cols[keep])),
                       shape=(asm.n_tri, ndof)).tocsr()
    return Cx, Cy


def build_dof_map(asm: Assembly):
    """Construit la numerotation des ddl de la fonction de courant.

    - noeuds interieurs : un ddl chacun
    - chaque contour : un seul ddl partage (g constant sur un bord)
    - contour exterieur de chaque composante connexe : ddl elimine (g = 0)

    Retourne (node_dof, ndof, hole_dofs, loops).
    hole_dofs : liste (ddl, indice_de_contour) pour les trous, utilisee
    pour imposer le nombre de fluxoides.
    """
    loops, node_loop = asm.boundary_loops()
    ncomp, lab = asm.components()

    # aire algebrique de chaque contour -> identification du bord exterieur
    loop_area, loop_comp = [], []
    for lp in loops:
        p = asm.points[lp]
        a = 0.5 * np.sum(p[:, 0] * np.roll(p[:, 1], -1)
                         - np.roll(p[:, 0], -1) * p[:, 1])
        loop_area.append(a)
        loop_comp.append(lab[lp[0]])
    loop_area = np.array(loop_area)
    loop_comp = np.array(loop_comp)

    outer = set()
    for c in range(ncomp):
        idx = np.where(loop_comp == c)[0]
        if len(idx):
            outer.add(int(idx[np.argmax(np.abs(loop_area[idx]))]))

    node_dof = np.full(asm.n_nodes, -1, dtype=int)
    counter = 0
    loop_dof = {}
    for li in range(len(loops)):
        if li in outer:
            continue
        loop_dof[li] = counter
        counter += 1
    for n in range(asm.n_nodes):
        li = node_loop[n]
        if li < 0:
            node_dof[n] = counter
            counter += 1
        elif li in loop_dof:
            node_dof[n] = loop_dof[li]
        else:
            node_dof[n] = -1                 # jauge g = 0
    hole_dofs = [(loop_dof[li], li) for li in loop_dof]
    return node_dof, counter, hole_dofs, loops


# ----------------------------------------------------------------------
# Potentiel vecteur et flux
# ----------------------------------------------------------------------
def vector_potential(asm: Assembly, K, pts, z=0.0, near=3.0):
    """A(r) = (mu0/4pi) * Somme_T  A_T * K_T / |r - r_T|  (Wb/um).

    pts : (P,2). Les triangles proches (distance < near*h_T) sont traites
    par la quadrature a 3 points.
    """
    pts = np.atleast_2d(np.asarray(pts, dtype=float))
    z = np.atleast_1d(np.asarray(z, dtype=float))
    if z.size == 1:
        z = np.full(len(pts), z[0])
    out = np.zeros((len(pts), 2))
    c = MU0 / (4.0 * np.pi)
    KA = K * asm.area[:, None]

    block = max(1, int(4e6 / max(asm.n_tri, 1)))
    for s in range(0, len(pts), block):
        P = pts[s:s + block]
        Z = z[s:s + block]
        dx = P[:, None, 0] - asm.centroid[None, :, 0]
        dy = P[:, None, 1] - asm.centroid[None, :, 1]
        dz = Z[:, None] - asm.tri_z[None, :]
        d = np.sqrt(dx * dx + dy * dy + dz * dz)
        inv = 1.0 / np.maximum(d, 1e-9)
        # raffinement au voisinage
        close = d < near * asm.h_tri[None, :]
        if np.any(close):
            ii, jj = np.nonzero(close)
            qx = asm.quad[jj, :, 0]
            qy = asm.quad[jj, :, 1]
            ddx = P[ii, 0][:, None] - qx
            ddy = P[ii, 1][:, None] - qy
            ddz = (Z[ii] - asm.tri_z[jj])[:, None]
            dd = np.sqrt(ddx ** 2 + ddy ** 2 + ddz ** 2)
            inv[ii, jj] = np.mean(1.0 / np.maximum(dd, 1e-9), axis=1)
        out[s:s + block, 0] = c * (inv @ KA[:, 0])
        out[s:s + block, 1] = c * (inv @ KA[:, 1])
    return out


def flux_through_polygon(asm: Assembly, K, ring, z=0.0, seg=None):
    """Phi = Contour_int A.dl  (Wb) sur un polygone ferme (N,2)."""
    ring = np.asarray(ring, dtype=float)
    if np.allclose(ring[0], ring[-1]):
        ring = ring[:-1]
    P = np.vstack([ring, ring[:1]])
    if seg is None:
        seg = float(np.median(asm.h_tri))
    mids, tans = [], []
    for a, b in zip(P[:-1], P[1:]):
        L = np.linalg.norm(b - a)
        n = max(1, int(np.ceil(L / seg)))
        t = (np.arange(n) + 0.5) / n
        mids.append(a + t[:, None] * (b - a))
        tans.append(np.tile((b - a) / n, (n, 1)))
    mids = np.vstack(mids)
    tans = np.vstack(tans)
    A = vector_potential(asm, K, mids, z=z)
    return float(np.sum(A * tans))
