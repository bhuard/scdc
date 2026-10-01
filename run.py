"""
run.py -- Interface de haut niveau, boucle sur les sources et rapport.

Le layer 3 peut contenir plusieurs polygones d'injection. Le calcul est
mene pour chacun comme s'il etait seul (courant I injecte dans le polygone
j, retour par les polygones du layer 4). Le resultat principal est la
matrice des inductances mutuelles

    M[i, j] = Phi_i / I_j      (henry)

reliant le flux a travers la surface i du layer 5 au courant injecte dans
le polygone j du layer 3. Le rapport Markdown donne M en pH et, si M est
carree, les valeurs propres de M^-1 en mA/Phi0 avec Phi0 = h/2e.

Tous les fichiers produits et messages sont en anglais.
"""

from __future__ import annotations

import argparse
import datetime
import json
import re
import numpy as np

try:
    from .core import MU0, PHI0, flux_through_polygon
    from .solver import solve_currents, total_energy
except ImportError:
    from core import MU0, PHI0, flux_through_polygon
    from solver import solve_currents, total_energy


# ----------------------------------------------------------------------
def _orient(ring, ccw=True):
    r = np.asarray(ring, dtype=float)
    a = 0.5 * np.sum(r[:, 0] * np.roll(r[:, 1], -1)
                     - np.roll(r[:, 0], -1) * r[:, 1])
    if (a < 0) == ccw:
        r = r[::-1]
    return r


def surface_fluxes(asm, K, rings, z=0.0, seg=None):
    """Flux through each layer-5 surface (normal +z, exterior contour
    counter-clockwise, interior contours clockwise)."""
    out = []
    for r in rings:
        phi = flux_through_polygon(asm, K, _orient(r['exterior'], True),
                                   z=z, seg=seg)
        for h in r['interiors']:
            phi += flux_through_polygon(asm, K, _orient(h, False), z=z, seg=seg)
        out.append(phi)
    return np.array(out)


def _safe_name(name):
    """File-system friendly version of a GDS label."""
    s = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name)).strip("_")
    return s or "unnamed"


# ----------------------------------------------------------------------
def mesh_report(model, png=None, near_cells=4, cell=50.0, top=12,
                grid=None, near_hmax=None):
    """Size statistics of the layer-1 mesh and location of clusters of
    small triangles, which drive the number of near pairs.

    grid and near_hmax must be those of the run for the estimate of the
    near pairs to be the count that FastKernel will find. The radius is the
    one of FastKernel, R_i = near_cells * max(h_grid, min(h_i, near_hmax)),
    h_grid being --grid, or the median triangle size over all sheets when
    --grid is absent."""
    asm = model['assembly']
    s0 = asm.sheets[0]
    n0 = len(s0.triangles)
    h = asm.h_tri[:n0]
    c = asm.centroid[:n0]
    hg = float(grid) if grid else float(np.median(asm.h_tri))
    print()
    print(f"  layer-1 mesh: {n0} triangles, h = sqrt(area), pFFT grid "
          f"h_grid = {hg:.3g} um ({'--grid' if grid else 'median of h'})")
    q = np.percentile(h, [0, 1, 5, 10, 25, 50, 75, 90, 99, 100])
    print("  percentiles of h (um):  " + "  ".join(
        f"p{p:g}={v:.3g}" for p, v in zip([0, 1, 5, 10, 25, 50, 75, 90, 99, 100], q)))
    for f in (0.5, 0.2, 0.1, 0.02):
        n = int(np.sum(h < f * hg))
        print(f"  h < {f:g} h_grid ({f*hg:.3g} um): {n} triangles "
              f"({100*n/n0:.1f} %)")
    try:
        from .near_pairs import count_near_pairs
    except ImportError:
        from near_pairs import count_near_pairs
    from scipy.spatial import cKDTree
    C3 = np.column_stack([asm.centroid, asm.tri_z])
    h_eff = asm.h_tri if near_hmax is None else \
        np.minimum(asm.h_tri, float(near_hmax))
    rad = near_cells * np.maximum(hg, h_eff)
    lens = count_near_pairs(cKDTree(C3), C3, rad)
    nd = int(lens.sum())
    print(f"  estimated near pairs before symmetrisation: {nd} "
          f"({nd / asm.n_tri:.0f}/triangle, max {int(lens.max())}), radius "
          f"{rad.min():.3g} to {rad.max():.3g} um")
    print(f"  memory: ~{nd*40/1e9:.1f} GB for the search, at most "
          f"~{2*nd*56/1e9:.1f} GB for the correction (symmetrisation at most "
          "doubles the count)")
    # cells ranked by the number of near pairs they generate, which is what
    # sets the memory, rather than by a size threshold relative to the grid
    ij = np.floor(c / cell).astype(np.int64)
    key = ij[:, 0] * 10**7 + ij[:, 1]
    uk, inv = np.unique(key, return_inverse=True)
    inv = inv.ravel()
    pairs = np.bincount(inv, weights=lens[:n0].astype(float))
    ntri = np.bincount(inv)
    hmin = np.full(len(uk), np.inf)
    np.minimum.at(hmin, inv, h)
    nmax = np.zeros(len(uk), dtype=np.int64)
    np.maximum.at(nmax, inv, lens[:n0])
    order = np.argsort(-pairs)[:top]
    share = pairs[order] / max(float(lens[:n0].sum()), 1.0)
    print(f"  cells of {cell:g} um generating the most near pairs "
          f"({len(uk)} cells hold layer-1 triangles, the {len(order)} below "
          f"hold {100 * share.sum():.1f} % of the layer-1 pairs):")
    print("      x_min     y_min  triangles   h_min[um]   pairs (%)   "
          "max_neighbours")
    for k, sh in zip(order, share):
        ix, iy = uk[k] // 10**7, uk[k] % 10**7
        print(f"  {ix*cell:9.0f} {iy*cell:9.0f} {ntri[k]:10d} "
              f"{hmin[k]:11.3g} {100 * sh:11.2f} {nmax[k]:16d}")
    if png:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.tri as mtri
        T = mtri.Triangulation(s0.points[:, 0], s0.points[:, 1], s0.triangles)
        fig, ax = plt.subplots(figsize=(11, 9))
        pc = ax.tripcolor(T, facecolors=np.log10(h), cmap="viridis",
                          shading="flat", rasterized=True)
        fig.colorbar(pc, ax=ax, label="log10 h (um)")
        ax.set_aspect("equal")
        fig.tight_layout()
        fig.savefig(png, dpi=180)
        print("  size map:", png)


def nodal_current(asm, K, sheet_index=0):
    """Area-weighted average of the triangle currents at the nodes."""
    s = asm.sheets[sheet_index]
    n0 = asm.node_offset[sheet_index]
    t0 = asm.tri_offset[sheet_index]
    nt = len(s.triangles)
    tri = asm.triangles[t0:t0 + nt] - n0
    w = asm.area[t0:t0 + nt]
    acc = np.zeros((len(s.points), 2))
    wsum = np.zeros(len(s.points))
    for c in range(3):
        np.add.at(acc, tri[:, c], K[t0:t0 + nt] * w[:, None])
        np.add.at(wsum, tri[:, c], w)
    return acc / np.maximum(wsum, 1e-30)[:, None]


# ----------------------------------------------------------------------
def make_view(model, Ks, names, thickness=None, title=None, log_color=False,
              show=False):
    """CurrentView of the layer-1 sheet holding one current map per source.
    Ks : (n_sources, n_tri, 2)."""
    try:
        from .viewer import CurrentView
    except ImportError:
        from viewer import CurrentView
    if not show:
        import matplotlib
        matplotlib.use("Agg")
    asm = model['assembly']
    s0 = asm.sheets[0]
    nt0 = len(s0.triangles)
    Ks = np.asarray(Ks)
    return CurrentView(s0.points, asm.triangles[:nt0], Ks[:, :nt0],
                       asm.area[:nt0], overlays=model.get('overlays'),
                       thickness=thickness if thickness else None,
                       title=title, log_color=log_color, names=list(names))


def plot_solutions(model, Ks, names, out_prefix, thickness=None, title=None,
                   log_color=False, show=False, verbose=True):
    """One PNG per source, then optionally the interactive window with all
    maps (key i cycles through the sources)."""
    v = make_view(model, Ks, names, thickness, title, log_color, show)
    files = []
    for k in range(len(names)):
        v.set_map(k, redraw_lines=False)
        fn = f"{out_prefix}_current_{k}_{_safe_name(names[k])}.png"
        v.save(fn)
        v.clear_lines()
        files.append(fn)
        if verbose:
            print(f"  current map: {fn}")
    if show:
        v.set_map(0, redraw_lines=False)
        v.show()
    else:
        v.plt.close(v.fig)
    return files


# ----------------------------------------------------------------------
def inverse_eigenvalues(M):
    """Eigenvalues of M^-1 in mA/Phi0, M being in henry.

    M^-1 maps fluxes to currents. Its eigenvalue lambda_k (A/Wb) multiplied
    by Phi0 is the current, along the k-th eigen-direction, producing one
    flux quantum. Returns (values in mA/Phi0 or None, condition number or
    None, message)."""
    M = np.asarray(M, dtype=float)
    if M.ndim != 2 or M.size == 0:
        return None, None, "M is empty"
    nl, ns = M.shape
    if nl != ns:
        return None, None, (f"M is {nl} x {ns}, not square, so M^-1 is not "
                            "defined")
    cond = np.linalg.cond(M)
    if not np.isfinite(cond) or cond > 1e15:
        return None, cond, f"M is singular to working precision (cond = {cond:.3g})"
    w = np.linalg.eigvals(np.linalg.inv(M)) * PHI0 * 1e3
    w = w[np.argsort(np.abs(w))]
    return w, cond, "ok"


def _fmt_c(z, fmt="{:.6g}"):
    """Formats a possibly complex number."""
    z = complex(z)
    if abs(z.imag) <= 1e-9 * max(abs(z.real), 1e-300):
        return fmt.format(z.real)
    sgn = "+" if z.imag >= 0 else "-"
    return fmt.format(z.real) + f" {sgn} {fmt.format(abs(z.imag))} i"


def write_report(path, model, res, sols, args, files):
    """Markdown report. The mutual-inductance matrix and the eigenvalues of
    its inverse come first."""
    asm = model['assembly']
    snames = list(model['source_names'])
    lnames = [r['name'] for r in model['rings']]
    M = res['M']
    I = model['current']
    L = []
    L.append(f"# scdc report: {args.get('gds', '')}\n")
    L.append(f"Generated {datetime.datetime.now():%Y-%m-%d %H:%M}. "
             f"Injected current I = {I*1e3:g} mA in each layer-3 polygon in "
             f"turn, returned through the layer-4 polygons. All lengths in um.\n")

    # ---------------------------------------------- matrix M
    L.append("## Mutual-inductance matrix M = Phi / I (pH)\n")
    L.append("Rows are the flux surfaces of layer 5, columns the injection "
             "polygons of layer 3. M[i, j] is the flux through surface i when "
             "the current I is injected in polygon j alone, divided by I.\n")
    if M.size:
        L.append("| surface \\ source | " + " | ".join(snames) + " |")
        L.append("|---|" + "---|" * len(snames))
        for i, nm in enumerate(lnames):
            L.append(f"| {nm} | " + " | ".join(f"{m*1e12:.6g}" for m in M[i]) + " |")
        L.append("")
    else:
        L.append("No flux surface in layer 5, M is empty.\n")

    # ---------------------------------------------- eigenvalues of M^-1
    L.append("## Eigenvalues of M^-1 (mA / Phi0)\n")
    L.append("Phi0 = h / 2e = 2.067833848e-15 Wb. Each eigenvalue lambda_k of "
             "M^-1 (A/Wb) is expressed as lambda_k Phi0 in mA, the current "
             "along the k-th eigen-direction producing one flux quantum "
             "through the corresponding combination of surfaces. Sorted by "
             "increasing modulus.\n")
    w, cond, msg = res['inv_eigenvalues'], res['cond'], res['inv_message']
    if w is not None:
        L.append(f"Condition number of M: {cond:.4g}.\n")
        L.append("| k | eigenvalue (mA/Phi0) | modulus (mA/Phi0) |")
        L.append("|---|---|---|")
        for k, z in enumerate(w):
            L.append(f"| {k} | {_fmt_c(z)} | {abs(z):.6g} |")
        L.append("")
        if np.any(np.abs(np.imag(w)) > 1e-9 * np.abs(w)):
            L.append("M is not symmetric in general (flux surfaces and "
                     "injection polygons are distinct objects), hence complex "
                     "eigenvalues can occur.\n")
    else:
        L.append(f"Not computed. {msg}.\n")
        if M.size and M.ndim == 2:
            sv = np.linalg.svd(M, compute_uv=False)
            good = sv > sv.max() * 1e-15
            L.append("Singular values of the pseudo-inverse M^+ (mA/Phi0), "
                     "which coincide with the moduli of the eigenvalues of "
                     "M^-1 when M is square and normal:\n")
            L.append("| k | 1/sigma_k (mA/Phi0) |")
            L.append("|---|---|")
            for k, s in enumerate(sv[good]):
                L.append(f"| {k} | {PHI0 * 1e3 / s:.6g} |")
            L.append("")

    # ---------------------------------------------- parameters
    L.append("## Model parameters\n")
    L.append(f"- GDS file: `{args.get('gds', '')}`")
    if args.get('thickness') is not None and args.get('lambda_L') is not None:
        L.append(f"- Layer-1 film: thickness d = {args['thickness']:g} um, "
                 f"London depth lambda = {args['lambda_L']:g} um")
    if args.get('L_square') is not None:
        L.append(f"- Layer-1 sheet inductance: {args['L_square']*1e12:g} pH/square")
    L.append(f"- Effective penetration depth Lambda = {model['Lambda']:.4g} um, "
             f"L_square = mu0 Lambda = {MU0*model['Lambda']*1e12:.4g} pH/square")
    L.append(f"- Injected current I = {I*1e3:g} mA")
    gs = model.get('ground_shares')
    if gs is None:
        L.append(f"- Ground current shares (layer 4): "
                 f"{np.round(model['ground_share'], 6).tolist()}")
    else:
        L.append("- Ground current shares (layer 4), per source. A pad not "
                 "connected in DC to the source carries no return current:")
        for nm, row in zip(snames, np.atleast_2d(gs)):
            L.append(f"  - {nm}: {np.round(row, 6).tolist()}")
    L.append(f"- Flux surfaces evaluated at z = {args.get('loop_z', 0.0):g} um")
    L.append(f"- Edge segment length: {args.get('seg_len'):g} um near the region "
             f"of interest")
    L.append(f"- Solver: {'dense reference' if args.get('dense') else 'pFFT + conjugate gradient'}"
             + ("" if args.get('dense') else f", near_cells = {args.get('near_cells')}"))
    if model.get('contacts'):
        L.append(f"- Bridges: {len(model['bridges'])}, contacts (feet): "
                 f"{len(model['contacts'])}, pillar inductances "
                 f"{'included' if np.any(model.get('pillar_L')) else 'disabled'}")
    L.append("")

    # ---------------------------------------------- mesh
    L.append("## Mesh\n")
    L.append(f"- Sheets: {len(asm.sheets)} ({', '.join(s.name for s in asm.sheets)})")
    L.append(f"- Triangles: {asm.n_tri}, nodes: {asm.n_nodes}")
    h = asm.h_tri
    L.append(f"- Triangle size sqrt(area): min {h.min():.3g}, median "
             f"{np.median(h):.3g}, max {h.max():.3g} um")
    L.append("")

    # ---------------------------------------------- sources
    L.append("## Injection polygons (layer 3)\n")
    L.append("| source | x_c | y_c | area (um^2) | mesh nodes | energy E (J) | "
             "L = 2E/I^2 (pH) | CG iterations | residual |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for j, (src, sol) in enumerate(zip(model['sources'], sols)):
        it = sol.info.get('iterations', '-')
        rr = sol.info.get('residual', None)
        rr = f"{rr:.2e}" if rr is not None else "-"
        L.append(f"| {src['name']} | {src['centroid'][0]:.3f} | "
                 f"{src['centroid'][1]:.3f} | {src['area']:.4g} | "
                 f"{len(src['nodes'])} | {res['energy'][j]:.6e} | "
                 f"{res['L_eff'][j]*1e12:.4f} | {it} | {rr} |")
    L.append("")

    # ---------------------------------------------- flux surfaces
    L.append("## Flux surfaces (layer 5)\n")
    if lnames:
        L.append("| surface | x_c | y_c | area (um^2) | " +
                 " | ".join(f"Phi/Phi0 ({s})" for s in snames) + " |")
        L.append("|---|---|---|---|" + "---|" * len(snames))
        for i, r in enumerate(model['rings']):
            L.append(f"| {r['name']} | {r['centroid'][0]:.3f} | "
                     f"{r['centroid'][1]:.3f} | {r['area']:.4g} | " +
                     " | ".join(f"{p/PHI0:.6g}" for p in res['flux'][i]) + " |")
        L.append("")
    else:
        L.append("None.\n")

    # ---------------------------------------------- pillars
    if model.get('contacts') and all('contact_currents' in s.info for s in sols):
        L.append("## Pillar currents (uA, upward)\n")
        pL = model.get('pillar_L')
        L.append("| pillar | bridge | x_c | y_c | L_self (pH) | " +
                 " | ".join(snames) + " |")
        L.append("|---|---|---|---|---|" + "---|" * len(snames))
        for k, c in enumerate(model['contacts']):
            Lk = pL[k, k] * 1e12 if pL is not None and pL.size else 0.0
            vals = [s.info['contact_currents'][k] * 1e6 for s in sols]
            L.append(f"| {k} | {getattr(c, 'bridge', -1)} | {c.center[0]:.2f} | "
                     f"{c.center[1]:.2f} | {Lk:.3f} | " +
                     " | ".join(f"{v:.4f}" for v in vals) + " |")
        L.append("")

    # ---------------------------------------------- junctions
    junctions = model.get('junctions', [])
    if junctions:
        IJ = res['junction_current']            # (n_sources, n_junctions)
        LJ = res['junction_LJ']                 # (n_sources, n_junctions)
        L.append("## Josephson junctions (layer 6)\n")
        L.append(f"Nonlinear iteration L_J(I) = L_J0 / sqrt(1 - (I/I_c)^2): "
                 f"{'on' if args.get('nonlinear') else 'off'}.\n")
        for jn in junctions:
            L.append(f"### Junction {jn['index']}\n")
            L.append(f"L_J0 = {jn['LJ0']*1e9:.4g} nH, I_c = {jn['Ic']*1e6:.4g} uA, "
                     f"E_J/h = {jn['EJ']/6.62607015e-34/1e9:.4g} GHz, "
                     f"F = L/L_square = {jn['F']:.4g}\n")
            L.append("| source | I_J (uA) | I_J / I_c | phase (rad) | L_J (nH) |")
            L.append("|---|---|---|---|---|")
            for j, s in enumerate(snames):
                x = float(np.clip(IJ[j, jn['index']] / jn['Ic'], -1, 1))
                L.append(f"| {s} | {IJ[j, jn['index']]*1e6:.4f} | {x:.4f} | "
                         f"{np.arcsin(x):.5f} | {LJ[j, jn['index']]*1e9:.4g} |")
            L.append("")

    # ---------------------------------------------- files
    L.append("## Output files\n")
    for f in files:
        L.append(f"- `{f}`")
    L.append("")
    with open(path, "w") as f:
        f.write("\n".join(L))
    return path


# ----------------------------------------------------------------------
def solve_model(model, seg_len, out_prefix="scdc", loop_z=0.0, plot=True,
                show=False, dense=False, near_cells=4, near_hmax=None,
                grid=None, log_color=False, nonlinear=False, nl_tol=1e-4,
                nl_maxiter=20, backend="auto", thickness=None, verbose=True,
                report=True, report_args=None):
    """Solves the problem of `model` once per injection polygon, writes the
    outputs and returns (sols, res). `model` is the dictionary produced by
    geometry.build_model, or any dictionary with the same keys (this allows
    tests without the GDS tool chain).

    res contains
        M        (n_surfaces, n_sources) mutual inductances in henry
        flux     (n_surfaces, n_sources) fluxes in weber
        energy, L_eff (n_sources,)
        junction_current, junction_LJ (n_sources, n_junctions)
        inv_eigenvalues (n,) complex, in mA/Phi0, or None
    """
    asm = model['assembly']
    current = model['current']
    junctions = model.get('junctions', [])
    terminal_sets = model['terminal_sets']
    snames = list(model['source_names'])
    nsrc = len(terminal_sets)
    report_args = dict(report_args or {})
    report_args.update(seg_len=seg_len, loop_z=loop_z, dense=dense,
                       near_cells=near_cells, nonlinear=nonlinear,
                       thickness=thickness)

    if verbose:
        print(f"  Lambda = {model['Lambda']:.4g} um, "
              f"L_square = {MU0*model['Lambda']*1e12:.4g} pH/square")
        gs = model.get('ground_shares')
        if gs is None:
            print(f"  ground current shares: {np.round(model['ground_share'], 6)}")
        else:
            for nm, row in zip(snames, np.atleast_2d(gs)):
                print(f"  ground current shares for {nm}: {np.round(row, 6)}")
        print(f"  {nsrc} injection polygon(s): {', '.join(snames)}")

    try:
        from .fast import solve_currents_fast, FastKernel
        from .solver import junction_currents
        from .accel import get_backend
    except ImportError:
        from fast import solve_currents_fast, FastKernel
        from solver import junction_currents
        from accel import get_backend

    pL = model.get('pillar_L')
    contacts = model.get('contacts', [])

    def _solve(terminals, kernel=None):
        if dense:
            s = solve_currents(asm, terminals, contacts, pillar_L=pL,
                               verbose=verbose)
            s.energy = total_energy(asm, s.K, contacts=contacts, pillar_L=pL)
            return s
        return solve_currents_fast(asm, terminals, contacts,
                                   near_cells=near_cells, near_hmax=near_hmax,
                                   grid=grid, kernel=kernel, pillar_L=pL,
                                   verbose=verbose)

    # the pFFT kernel depends on the geometry only, it is built once and
    # shared by all sources and all nonlinear iterations
    kernel = None if dense else FastKernel(asm, grid=grid, near_cells=near_cells,
                                           near_hmax=near_hmax,
                                           backend=get_backend(backend, verbose),
                                           verbose=verbose)

    sols, Ks, energies, IJs, LJs = [], [], [], [], []
    for j, terminals in enumerate(terminal_sets):
        if verbose:
            print()
            print(f"  ---- source {j}: {snames[j]} ----")
        # each source is computed as if alone, the junctions restart from
        # their small-current inductance
        for jn in junctions:
            jn['LJ'] = jn['LJ0']
            jn['Lambda'] = jn['LJ'] / (MU0 * jn['F'])
            asm.tri_Lambda[jn['tri']] = jn['Lambda']
        sol = _solve(terminals, kernel)
        IJ = junction_currents(asm, sol.K, junctions) if junctions else np.zeros(0)
        if nonlinear and junctions:
            for it in range(nl_maxiter):
                change = 0.0
                for jn, I in zip(junctions, IJ):
                    x = min(abs(I) / jn['Ic'], 0.999)
                    LJ_new = jn['LJ0'] / np.sqrt(1.0 - x * x)
                    change = max(change, abs(LJ_new - jn['LJ']) / jn['LJ'])
                    jn['LJ'] = LJ_new
                    jn['Lambda'] = LJ_new / (MU0 * jn['F'])
                    asm.tri_Lambda[jn['tri']] = jn['Lambda']
                if verbose:
                    print(f"  nonlinear iteration {it+1}, max relative change "
                          f"of L_J {change:.2e}")
                if change < nl_tol:
                    break
                sol = _solve(terminals, kernel)
                IJ = junction_currents(asm, sol.K, junctions)
        sols.append(sol)
        Ks.append(sol.K)
        energies.append(sol.energy)
        IJs.append(IJ)
        LJs.append(np.array([jn['LJ'] for jn in junctions]))
        if verbose:
            print(f"  energy E = {sol.energy:.6e} J, "
                  f"L = 2E/I^2 = {2*sol.energy/current**2*1e12:.4f} pH")

    Ks = np.array(Ks)                                   # (nsrc, n_tri, 2)
    energies = np.array(energies)
    L_eff = 2 * energies / current ** 2
    IJs = np.array(IJs).reshape(nsrc, len(junctions))
    LJs = np.array(LJs).reshape(nsrc, len(junctions))

    # fluxes and mutual inductances, M[i, j] = Phi_i(source j) / I
    nl = len(model['rings'])
    flux = np.zeros((nl, nsrc))
    for j in range(nsrc):
        flux[:, j] = surface_fluxes(asm, Ks[j], model['rings'], z=loop_z,
                                    seg=0.5 * seg_len)
    M = flux / current
    w, cond, msg = inverse_eigenvalues(M)
    lnames = [r['name'] for r in model['rings']]

    if verbose:
        print()
        print("  mutual-inductance matrix M = Phi/I (pH), rows = surfaces, "
              "columns = sources")
        wid = max([len(s) for s in snames + lnames] + [12])
        print("  " + " " * wid + "  " + "  ".join(f"{s:>12s}" for s in snames))
        for i, nm in enumerate(lnames):
            print("  " + f"{nm:{wid}s}" + "  "
                  + "  ".join(f"{m*1e12:12.5f}" for m in M[i]))
        if w is not None:
            print("  eigenvalues of M^-1 (mA/Phi0): "
                  + ", ".join(_fmt_c(z, "{:.5g}") for z in w))
        else:
            print(f"  eigenvalues of M^-1 not computed, {msg}")
        if contacts and all('contact_currents' in s.info for s in sols):
            print()
            print("  pillar   bridge     x_c       y_c    L_self[pH]  " +
                  "  ".join(f"I[uA] {s}" for s in snames))
            for k, c in enumerate(contacts):
                Lk = pL[k, k] * 1e12 if pL is not None and pL.size else 0.0
                print(f"  {k:6d} {getattr(c, 'bridge', -1):8d} "
                      f"{c.center[0]:9.2f} {c.center[1]:9.2f} {Lk:11.3f}  "
                      + "  ".join(f"{s.info['contact_currents'][k]*1e6:12.4f}"
                                  for s in sols))
        if junctions:
            print()
            print("  source          junction   I_J[uA]    I_J/I_c   phase[rad]   L_J[nH]")
            for j, s in enumerate(snames):
                for jn in junctions:
                    x = np.clip(IJs[j, jn['index']] / jn['Ic'], -1, 1)
                    print(f"  {s:14s} {jn['index']:8d} "
                          f"{IJs[j, jn['index']]*1e6:10.4f} {x:10.4f} "
                          f"{np.arcsin(x):12.5f} {LJs[j, jn['index']]*1e9:10.4g}")

    # ------------------------------------------------ files
    files = []
    fn = f"{out_prefix}_solution.npz"
    np.savez_compressed(
        fn,
        points=asm.points, triangles=asm.triangles, tri_sheet=asm.tri_sheet,
        node_sheet=asm.node_sheet, K=Ks, area=asm.area,
        centroid=asm.centroid, tri_z=asm.tri_z, Lambda=model['Lambda'],
        current=current, energy=energies, L_eff=L_eff, flux=flux, M=M,
        source_names=np.array(snames, dtype=str),
        loop_names=np.array(lnames, dtype=str),
        thickness=thickness if thickness else 0.0,
        junction_current=IJs,
        overlays=np.array(json.dumps(model['overlays'])))
    files.append(fn)

    fn = f"{out_prefix}_mutual.csv"
    with open(fn, "w") as f:
        f.write("surface_index,surface_name,area_um2,xc_um,yc_um,"
                + ",".join(f"M_pH_{_safe_name(s)}" for s in snames) + "\n")
        for i, r in enumerate(model['rings']):
            f.write(f"{i},{r['name']},{r['area']:.6g},{r['centroid'][0]:.6g},"
                    f"{r['centroid'][1]:.6g},"
                    + ",".join(f"{m*1e12:.8e}" for m in M[i]) + "\n")
    files.append(fn)

    res = dict(M=M, flux=flux, energy=energies, L_eff=L_eff,
               junction_current=IJs, junction_LJ=LJs, inv_eigenvalues=w,
               cond=cond, inv_message=msg, source_names=snames,
               loop_names=lnames)

    if plot or show:
        title = (f"I = {current*1e3:g} mA, "
                 f"$\\Lambda$ = {model['Lambda']:.3g} $\\mu$m")
        files += plot_solutions(model, Ks, snames, out_prefix,
                                thickness=thickness, title=title,
                                log_color=log_color, show=show,
                                verbose=verbose)

    if report:
        fn = f"{out_prefix}_report.md"
        write_report(fn, model, res, sols, report_args, files + [fn])
        files.append(fn)
        if verbose:
            print(f"  report: {fn}")
    res['files'] = files
    return sols, res


# ----------------------------------------------------------------------
def run(gds, seg_len, thickness=None, lambda_L=None, L_square=None,
        current=1e-3, out_prefix="scdc", loop_z=0.0, plot=True, show=False,
        dense=False, near_cells=4, near_hmax=None, grid=None,
        log_color=False, nonlinear=False, mesh_only=False,
        nl_tol=1e-4, nl_maxiter=20, backend="auto", verbose=True, **kw):
    """Builds the model from the GDS and solves it once per injection
    polygon. Returns (model, sols, res) with sols a list of Solution, one per
    layer-3 polygon, and res the dictionary described in solve_model."""
    try:
        from .geometry import build_model
    except ImportError:
        from geometry import build_model

    model = build_model(gds, thickness=thickness, lambda_L=lambda_L,
                        L_square=L_square, seg_len=seg_len, current=current,
                        verbose=verbose, **kw)
    if mesh_only:
        mesh_report(model, f"{out_prefix}_mesh.png", near_cells=near_cells,
                    grid=grid, near_hmax=near_hmax)
        return model, None, None
    sols, res = solve_model(
        model, seg_len, out_prefix=out_prefix, loop_z=loop_z, plot=plot,
        show=show, dense=dense, near_cells=near_cells, near_hmax=near_hmax,
        grid=grid, log_color=log_color, nonlinear=nonlinear, nl_tol=nl_tol,
        nl_maxiter=nl_maxiter, backend=backend, thickness=thickness,
        verbose=verbose,
        report_args=dict(gds=gds, thickness=thickness, lambda_L=lambda_L,
                         L_square=L_square))
    return model, sols, res


def main(argv=None):
    p = argparse.ArgumentParser(
        description="DC current density and mutual inductances in a "
                    "superconducting circuit described by a GDS. Each "
                    "polygon of layer 3 is treated in turn as the single "
                    "injection point. All lengths in micrometres.")
    p.add_argument("gds")
    p.add_argument("--thickness", type=float, default=None,
                   help="thickness of the layer-1 superconductor (um)")
    p.add_argument("--lambda-london", type=float, default=None,
                   help="London penetration depth of layer 1 (um)")
    p.add_argument("--Lsq", type=float, default=None,
                   help="kinetic sheet inductance of layer 1 (pH/square), "
                        "replaces thickness and lambda-london")
    p.add_argument("--layer-junction", type=int, default=6)
    p.add_argument("--EJ", type=float, default=None,
                   help="default E_J/h of the junctions (GHz)")
    p.add_argument("--Ic", type=float, default=None,
                   help="default I_c of the junctions (uA)")
    p.add_argument("--LJ", type=float, default=None,
                   help="default L_J of the junctions (nH)")
    p.add_argument("--junction-file", type=str, default=None,
                   help="JSON list of {x, y, EJ_GHz | Ic_uA | LJ_nH | "
                        "Rn_Ohm+Delta_ueV}")
    p.add_argument("--nonlinear", action="store_true",
                   help="iterate L_J(I) = L_J0 / sqrt(1 - (I/I_c)^2)")
    p.add_argument("--bridge-Lsq", type=float, default=None,
                   help="sheet inductance of the bridges (pH/square)")
    p.add_argument("--seg", type=float, required=True,
                   help="edge segment length near the region of interest (um)")
    p.add_argument("--seg-far", type=float, default=None,
                   help="edge segments far from the region of interest, "
                        "default 5*seg")
    p.add_argument("--fine-radius", type=float, default=None,
                   help="radius of the region of interest around layers 5 "
                        "and 6 (um), default 200*seg/1.5")
    p.add_argument("--seg-max", type=float, default=None,
                   help="maximum size of interior triangles, default 30*seg")
    p.add_argument("--current", type=float, default=1e-3, help="current (A)")
    p.add_argument("--bridge-height", type=float, default=3.0,
                   help="height of the airbridges above the plane (um)")
    p.add_argument("--bridge-thickness", type=float, default=None)
    p.add_argument("--bridge-lambda", type=float, default=None)
    p.add_argument("--bridge-feet", type=str, default="ends",
                   choices=["ends", "all"],
                   help="feet of a bridge: 'ends' keeps the two most distant "
                        "overlap regions, 'all' keeps them all (default ends)")
    p.add_argument("--pillar-side", type=float, default=30.0,
                   help="side of the square pillar base (um)")
    p.add_argument("--pillar-height", type=float, default=None,
                   help="pillar height (um), default = bridge-height")
    p.add_argument("--pillar-lambda", type=float, default=None,
                   help="lambda_L of the pillars (um), default = bridges")
    p.add_argument("--no-pillars", action="store_true",
                   help="perfect vertical contacts, no pillar inductance")
    p.add_argument("--loop-z", type=float, default=0.0,
                   help="altitude of the layer-5 surfaces (um)")
    p.add_argument("--ground-resistances", type=str, default=None,
                   help="comma-separated list, default 0.01 Ohm each")
    p.add_argument("--layer-metal", type=int, default=1)
    p.add_argument("--layer-bridge", type=int, default=2)
    p.add_argument("--layer-source", type=int, default=3)
    p.add_argument("--layer-ground", type=int, default=4)
    p.add_argument("--layer-loop", type=int, default=5)
    p.add_argument("--cell", type=str, default=None)
    p.add_argument("--scale", type=float, default=1.0,
                   help="scale factor to micrometres")
    p.add_argument("--out", type=str, default="scdc")
    p.add_argument("--no-plot", action="store_true")
    p.add_argument("--show", action="store_true",
                   help="open the interactive window after the computation "
                        "(key i cycles through the sources)")
    p.add_argument("--log", action="store_true",
                   help="logarithmic colour scale")
    p.add_argument("--dense", action="store_true",
                   help="dense reference assembly (small meshes)")
    p.add_argument("--backend", type=str, default="auto",
                   help="numpy | mlx | torch | auto")
    p.add_argument("--near-cells", type=int, default=4,
                   help="pFFT precorrection radius in grid cells")
    p.add_argument("--near-hmax", type=float, default=None,
                   help="cap (um) on the triangle size used in the "
                        "precorrection radius, bounds the number of near "
                        "pairs on a graded mesh (e.g. 3*seg)")
    p.add_argument("--grid", type=float, default=None,
                   help="pFFT grid spacing (um), default = median triangle size")
    p.add_argument("--mesh-only", action="store_true",
                   help="stop after meshing, print size statistics, clusters "
                        "of small triangles and a map of log10(h)")
    p.add_argument("--min-seg", type=float, default=None,
                   help="simplify the contours to this tolerance (um) before "
                        "meshing, removes tiny segments")
    p.add_argument("--pad-tol", type=float, default=None,
                   help="simplify the layer-3 and layer-4 polygons to this "
                        "tolerance (um) before their intersection with the "
                        "metal, removes the small triangles of finely drawn "
                        "pads")
    a = p.parse_args(argv)

    gr = None
    if a.ground_resistances:
        gr = [float(x) for x in a.ground_resistances.split(",")]

    jd = None
    if a.EJ is not None:
        jd = dict(EJ_GHz=a.EJ)
    elif a.Ic is not None:
        jd = dict(Ic_uA=a.Ic)
    elif a.LJ is not None:
        jd = dict(LJ_nH=a.LJ)

    run(a.gds, a.seg, thickness=a.thickness, lambda_L=a.lambda_london,
        L_square=a.Lsq * 1e-12 if a.Lsq is not None else None,
        current=a.current, out_prefix=a.out, loop_z=a.loop_z,
        plot=not a.no_plot, nonlinear=a.nonlinear,
        layer_junction=a.layer_junction, junction_default=jd,
        seg_far=a.seg_far, fine_radius=a.fine_radius, seg_max=a.seg_max,
        junction_file=a.junction_file,
        bridge_L_square=a.bridge_Lsq * 1e-12 if a.bridge_Lsq is not None else None,
        show=a.show, dense=a.dense, near_cells=a.near_cells,
        near_hmax=a.near_hmax, grid=a.grid, min_seg=a.min_seg,
        pad_tol=a.pad_tol, log_color=a.log,
        mesh_only=a.mesh_only,
        backend=a.backend,
        bridge_height=a.bridge_height, bridge_thickness=a.bridge_thickness,
        bridge_lambda=a.bridge_lambda, ground_resistances=gr,
        bridge_feet=a.bridge_feet, pillar_side=a.pillar_side,
        pillar_height=a.pillar_height, pillar_lambda=a.pillar_lambda,
        pillars=not a.no_pillars,
        layer_metal=a.layer_metal, layer_bridge=a.layer_bridge,
        layer_source=a.layer_source, layer_ground=a.layer_ground,
        layer_loop=a.layer_loop, cell=a.cell, scale=a.scale)


if __name__ == "__main__":
    main()
