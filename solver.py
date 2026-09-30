"""
solver.py -- Assemblage du probleme physique complet et resolution.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

try:
    from .core import (Assembly, MU0, PHI0, resistive_field, stream_operator,
                       build_dof_map, assemble_reduced_system,
                       vector_potential, flux_through_polygon)
except ImportError:                      # fichiers a plat, sans package
    from core import (Assembly, MU0, PHI0, resistive_field, stream_operator,
                      build_dof_map, assemble_reduced_system,
                      vector_potential, flux_through_polygon)


class Contact:
    """Contact vertical parfait entre deux feuilles (pied d'airbridge, via).

    nodes_a : indices globaux des noeuds de la feuille inferieure
    nodes_b : indices globaux des noeuds de la feuille superieure
    """

    def __init__(self, nodes_a, nodes_b, name=""):
        self.nodes_a = np.asarray(nodes_a, dtype=int)
        self.nodes_b = np.asarray(nodes_b, dtype=int)
        self.name = name


class Terminal:
    """Terminal d'injection ou d'extraction de courant.

    nodes   : indices globaux couverts par le plot de contact
    current : courant algebrique injecte dans le film (A)
    """

    def __init__(self, nodes, current, name=""):
        self.nodes = np.asarray(nodes, dtype=int)
        self.current = float(current)
        self.name = name


# ----------------------------------------------------------------------
def dc_clusters(asm: Assembly, contacts=()):
    """Label of the galvanic (DC) cluster of every node, shape (n_nodes,).

    Two nodes belong to the same cluster when a path joins them through the
    triangles of any sheet or through a vertical contact (bridge foot). The
    nodes of a contact are merged exactly as in resistive_field, so that
    the clusters coincide with the connected components on which the loads
    must balance.
    """
    from scipy.sparse.csgraph import connected_components
    ncomp, lab = asm.components()
    u, v = [], []
    for c in contacts:
        nd = np.concatenate([c.nodes_a, c.nodes_b])
        if len(nd) > 1:
            u.append(np.full(len(nd) - 1, lab[nd[0]]))
            v.append(lab[nd[1:]])
    if not u:
        return lab
    u, v = np.concatenate(u), np.concatenate(v)
    A = sp.coo_matrix((np.ones(len(u)), (u, v)), shape=(ncomp, ncomp))
    _, clab = connected_components(A, directed=False)
    return clab[lab]


def build_terminal_sets(asm: Assembly, source_nodes, ground_nodes, current,
                        conductances=None, contacts=(), source_names=None,
                        verbose=True):
    """One terminal set per injection polygon, the ground returns being
    restricted to the pads connected in DC to that polygon.

    In the DC regime the film is equipotential on each galvanic cluster. The
    pads k of the cluster C that contains source j are all at the potential
    V_C, hence I_k = G_k V_C and

        I_k = I G_k / sum_{l in C} G_l    for k in C,
        I_k = 0                           otherwise.

    A pad lying on a cluster without injection carries no net DC current.
    If a pad overlaps several clusters, only its part on the cluster of the
    source is used.

    source_nodes, ground_nodes : lists of arrays of global node indices.
    conductances : 1/R_k of the pads, default all equal.
    Returns (terminal_sets, shares), shares of shape (n_sources, n_ground).
    """
    ng = len(ground_nodes)
    G = np.ones(ng) if conductances is None else \
        np.asarray(conductances, dtype=float)
    if len(G) != ng:
        raise ValueError("one conductance per ground pad is required")
    names = list(source_names) if source_names is not None else \
        [f"S{j}" for j in range(len(source_nodes))]
    clus = dc_clusters(asm, contacts)
    gcl = [np.unique(clus[np.asarray(n, dtype=int)]) for n in ground_nodes]
    if verbose:
        for k, c in enumerate(gcl):
            if len(c) > 1:
                print(f"  warning: ground pad gnd{k} overlaps {len(c)} pieces "
                      "of metal that are not connected in DC, each source "
                      "uses only the part on its own piece")
    terminal_sets = []
    shares = np.zeros((len(source_nodes), ng))
    for j, (nd, nm) in enumerate(zip(source_nodes, names)):
        nd = np.asarray(nd, dtype=int)
        cs = np.unique(clus[nd])
        if len(cs) != 1:
            raise ValueError(f"injection polygon '{nm}' overlaps {len(cs)} "
                             "pieces of metal that are not connected in DC, "
                             "the split of the injected current between them "
                             "is undefined")
        c = cs[0]
        sel = [k for k in range(ng) if np.any(gcl[k] == c)]
        if not sel:
            raise ValueError(f"no ground pad is connected in DC to injection "
                             f"polygon '{nm}', the current has no return path")
        sel = np.array(sel, dtype=int)
        shares[j, sel] = G[sel] / G[sel].sum()
        terms = [Terminal(nd, current, nm)]
        for k in sel:
            gk = np.asarray(ground_nodes[k], dtype=int)
            terms.append(Terminal(gk[clus[gk] == c], -current * shares[j, k],
                                  f"gnd{k}"))
        terminal_sets.append(terms)
        if verbose and len(sel) < ng:
            off = [f"gnd{k}" for k in range(ng) if k not in sel]
            print(f"  source {nm}: {len(sel)} ground pad(s) connected in DC, "
                  f"no return current through {', '.join(off)}")
    return terminal_sets, shares


# ----------------------------------------------------------------------
def contact_current_operator(asm: Assembly, contacts):
    """Matrices creuses (Dx, Dy), de taille (n_contacts, n_tri), telles que
    le courant montant dans le pilier k vaille

        I_k = Dx[k] . K_x + Dy[k] . K_y = Somme_{i in nodes_a} Int grad(phi_i).K dA

    c'est-a-dire la divergence faible de K sur la feuille inferieure, sommee
    sur les noeuds du pied. Le champ z x grad(g) etant a divergence nulle et
    tangent aux bords, cette fonctionnelle ne depend que du seed et des
    champs de boucle, et vaut exactement +-1 sur le champ de boucle d'un
    cycle passant par le contact.
    """
    rows, cols, vx, vy = [], [], [], []
    for k, c in enumerate(contacts):
        mask = np.zeros(asm.n_nodes, dtype=bool)
        mask[c.nodes_a] = True
        for a in range(3):
            t = np.where(mask[asm.triangles[:, a]])[0]
            rows.append(np.full(len(t), k))
            cols.append(t)
            vx.append(asm.grad[t, a, 0] * asm.area[t])
            vy.append(asm.grad[t, a, 1] * asm.area[t])
    n = len(contacts)
    if n == 0:
        z = sp.csr_matrix((0, asm.n_tri))
        return z, z
    r, c_ = np.concatenate(rows), np.concatenate(cols)
    Dx = sp.coo_matrix((np.concatenate(vx), (r, c_)), shape=(n, asm.n_tri)).tocsr()
    Dy = sp.coo_matrix((np.concatenate(vy), (r, c_)), shape=(n, asm.n_tri)).tocsr()
    return Dx, Dy


def contact_currents(asm: Assembly, K, contacts):
    """Courant montant (A) dans chaque pilier pour une distribution K."""
    Dx, Dy = contact_current_operator(asm, contacts)
    return Dx @ K[:, 0] + Dy @ K[:, 1]


def _pillar_block(pillar_L, contacts):
    """Matrice L des piliers validee, ou None si terme absent."""
    if pillar_L is None or len(contacts) == 0:
        return None
    L = np.asarray(pillar_L, dtype=float)
    if L.shape != (len(contacts), len(contacts)):
        raise ValueError("pillar_L must be (n_contacts, n_contacts)")
    if not np.any(L):
        return None
    return L


# ----------------------------------------------------------------------
def _node_loads(asm: Assembly, nodes, total, profile="uniform"):
    """Repartit un courant total sur un ensemble de noeuds."""
    f = np.zeros(asm.n_nodes)
    if len(nodes) == 0:
        raise ValueError("empty terminal, no mesh node inside the contact region")
    if profile == "uniform":
        w = asm.node_area[nodes]
    else:
        w = np.ones(len(nodes))
    f[nodes] = total * w / w.sum()
    return f


def _fundamental_cycles(n_patch, edges):
    """Cycles fondamentaux d'un multigraphe.

    edges : liste de (u, v). Retourne une liste de cycles, chaque cycle
    etant une liste de (indice_arete, signe).
    """
    parent = list(range(n_patch))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    tree, extra = [], []
    for k, (u, v) in enumerate(edges):
        ru, rv = find(u), find(v)
        if ru != rv:
            parent[ru] = rv
            tree.append(k)
        else:
            extra.append(k)

    # adjacence de l'arbre couvrant
    adj = {i: [] for i in range(n_patch)}
    for k in tree:
        u, v = edges[k]
        adj[u].append((v, k, +1))
        adj[v].append((u, k, -1))

    def path(u, v):
        """Chemin u -> v dans l'arbre, liste de (arete, signe)."""
        prev = {u: None}
        stack = [u]
        while stack:
            a = stack.pop()
            if a == v:
                break
            for b, k, s in adj[a]:
                if b not in prev:
                    prev[b] = (a, k, s)
                    stack.append(b)
        if v not in prev:
            return None
        out, cur = [], v
        while prev[cur] is not None:
            a, k, s = prev[cur]
            out.append((k, s))
            cur = a
        return out[::-1]

    cycles = []
    for k in extra:
        u, v = edges[k]
        p = path(v, u)
        if p is None:
            continue
        cycles.append([(k, +1)] + p)
    return cycles


# ----------------------------------------------------------------------
class Solution:
    def __init__(self, asm, K, energy, info):
        self.asm = asm
        self.K = K
        self.energy = energy
        self.info = info

    def sheet_current_magnitude(self):
        return np.linalg.norm(self.K, axis=1)

    def current_density(self, thickness):
        """J = K / d  en A/um^2 (valeur moyennee sur l'epaisseur)."""
        return self.K / thickness


def solve_currents(asm: Assembly, terminals, contacts=(), fluxoid=None,
                   terminal_profile="uniform", chunk=192, pillar_L=None,
                   verbose=True):
    """Minimise l'energie sous contraintes et retourne la solution.

    fluxoid : dict {indice_de_trou: n} imposant Int (mu0*Lambda*K + A).dl
              = n*Phi0. Par defaut n = 0 partout (refroidissement en
              champ nul, pas de flux piege).
    pillar_L : matrice (n_contacts, n_contacts) des inductances propres et
              mutuelles des piliers verticaux. Ajoute (1/2) I^T L I a
              l'energie, I_k etant le courant dans le contact k.
    """
    # ------------------------------------------------- champ d'amorcage
    loads = np.zeros(asm.n_nodes)
    for t in terminals:
        loads += _node_loads(asm, t.nodes, t.current, terminal_profile)
    if abs(loads.sum()) > 1e-12 * max(1.0, np.abs(loads).sum()):
        raise ValueError("terminal currents do not sum to zero")

    merge = [np.concatenate([c.nodes_a, c.nodes_b]) for c in contacts]
    S = asm.stiffness()
    kseed = resistive_field(asm, loads, merge_groups=merge, K_stiff=S)

    # ------------------------------------------------- boucles de ponts
    ncomp, lab = asm.components()
    edges = [(int(lab[c.nodes_a[0]]), int(lab[c.nodes_b[0]])) for c in contacts]
    cycles = _fundamental_cycles(ncomp, edges) if contacts else []
    B = np.zeros((asm.n_tri, 2, len(cycles)))
    for ci, cyc in enumerate(cycles):
        f = np.zeros(asm.n_nodes)
        for k, s in cyc:
            c = contacts[k]
            f -= _node_loads(asm, c.nodes_a, s, "uniform")
            f += _node_loads(asm, c.nodes_b, s, "uniform")
        B[:, :, ci] = resistive_field(asm, f, merge_groups=None, K_stiff=S)

    # ------------------------------------------------- degres de liberte
    node_dof, ndof, hole_dofs, loops = build_dof_map(asm)
    Cx, Cy = stream_operator(asm, node_dof, ndof)
    if verbose:
        print(f"  {asm.n_tri} triangles, {ndof} stream-function dofs, "
              f"{len(hole_dofs)} holes, {len(cycles)} bridge loop(s)")

    H, b = assemble_reduced_system(asm, (Cx, Cy), B, kseed, chunk=chunk,
                                   progress=verbose)

    # ------------------------------------------------- piliers
    # I = I_seed + R x,  E_pil = (1/2) I^T L I  ->  H += R^T L R, b += R^T L I_seed
    Lp = _pillar_block(pillar_L, contacts)
    if Lp is not None:
        Dx, Dy = contact_current_operator(asm, contacts)
        R = np.hstack([(Dx @ Cx + Dy @ Cy).toarray(),
                       Dx @ B[:, 0, :] + Dy @ B[:, 1, :]])   # (n_contacts, ntot)
        Iseed = Dx @ kseed[:, 0] + Dy @ kseed[:, 1]
        H += R.T @ Lp @ R
        b += R.T @ (Lp @ Iseed)

    rhs = -b
    if fluxoid:
        for dofi, loopi in hole_dofs:
            n = fluxoid.get(loopi, 0)
            if n:
                rhs[dofi] += n * PHI0

    x = np.linalg.solve(H, rhs)
    g = x[:ndof]
    alpha = x[ndof:]

    K = kseed + np.column_stack([Cx @ g, Cy @ g])
    if len(cycles):
        K += np.einsum('mdc,c->md', B, alpha)

    energy = 0.5 * x @ H @ x + b @ x
    # energie absolue (avec le terme constant du seed)
    info = dict(node_dof=node_dof, loops=loops, hole_dofs=hole_dofs,
                g=g, alpha=alpha, kseed=kseed, cycles=cycles,
                fluxoids=(H @ x + b))
    if contacts:
        info['contact_currents'] = contact_currents(asm, K, contacts)
    return Solution(asm, K, energy, info)


# ----------------------------------------------------------------------
def total_energy(asm: Assembly, K, chunk=192, contacts=(), pillar_L=None):
    """Energie totale exacte E = E_cin + E_mag + E_piliers (J)."""
    try:
        from .core import _pair_kernel
    except ImportError:
        from core import _pair_kernel
    E = 0.5 * np.sum(MU0 * asm.tri_Lambda * asm.area
                     * np.sum(K ** 2, axis=1))
    c = MU0 / (8.0 * np.pi)
    acc = 0.0
    for s in range(0, asm.n_tri, chunk):
        rows = np.arange(s, min(s + chunk, asm.n_tri))
        G = _pair_kernel(asm, rows)
        acc += np.sum((G @ K[:, 0]) * K[rows, 0] + (G @ K[:, 1]) * K[rows, 1])
    E += c * acc
    Lp = _pillar_block(pillar_L, contacts)
    if Lp is not None:
        I = contact_currents(asm, K, contacts)
        E += 0.5 * I @ Lp @ I
    return E


def mutual_inductances(asm: Assembly, K, rings, current, z=0.0, seg=None):
    """M = Phi / I pour une liste de contours (N,2), en H."""
    out = []
    for r in rings:
        phi = flux_through_polygon(asm, K, r, z=z, seg=seg)
        out.append(phi / current)
    return np.array(out)


# ----------------------------------------------------------------------
def junction_currents(asm, K, junctions):
    """Courant traversant chaque jonction, I_J = Int_J K.grad(V) dA,
    V etant le potentiel de Laplace unitaire entre les deux contacts."""
    out = []
    for jn in junctions:
        t = jn['tri']
        out.append(np.sum(np.einsum('md,md->m', K[t], jn['gradV']) * asm.area[t]))
    return np.array(out)
