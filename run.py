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

With --Bext a uniform field B_ext along +z (perpendicular to the plane of
the GDS) is applied, without injected current, and the flux concentration
Phi_i / (A_i B_ext) of each layer-5 surface of net area A_i is reported.
The report starts with the program version and the command line.

Tous les fichiers produits et messages sont en anglais.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import re
import shlex
import sys
import numpy as np

try:
    from .core import (MU0, PHI0, flux_through_polygon, external_field_drive,
                       magnetic_moment_z)
    from .solver import solve_currents, total_energy
    from .version import __version__, version_info
except ImportError:
    from core import (MU0, PHI0, flux_through_polygon, external_field_drive,
                      magnetic_moment_z)
    from solver import solve_currents, total_energy
    from version import __version__, version_info


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


def ring_area(r):
    """Net area (um^2) of a layer-5 surface, exterior minus holes, from the
    same contours as those used for the flux."""
    def a(p):
        p = np.asarray(p, dtype=float)
        return abs(0.5 * np.sum(p[:, 0] * np.roll(p[:, 1], -1)
                                - np.roll(p[:, 0], -1) * p[:, 1]))
    return a(r['exterior']) - sum(a(h) for h in r['interiors'])


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
              show=False, captions=None, groups=None):
    """CurrentView of the layer-1 sheet holding one current map per source,
    and optionally the map of the response to the applied field.
    Ks : (n_maps, n_tri, 2)."""
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
                       title=title, log_color=log_color, names=list(names),
                       captions=captions, groups=groups)


def plot_solutions(model, Ks, names, out_prefix, thickness=None, title=None,
                   log_color=False, show=False, verbose=True, captions=None,
                   groups=None, file_tags=None):
    """One PNG per map, then optionally the interactive window with all
    maps (key i cycles through them). file_tags gives the part of the file
    name after "_current_", default "<k>_<name>"."""
    v = make_view(model, Ks, names, thickness, title, log_color, show,
                  captions=captions, groups=groups)
    files = []
    for k in range(len(names)):
        v.set_map(k, redraw_lines=False)
        tag = file_tags[k] if file_tags is not None else \
            f"{k}_{_safe_name(names[k])}"
        fn = f"{out_prefix}_current_{tag}.png"
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


def _bridge_table(fits, snap_deg=None):
    """Markdown lines describing the rectangles that replace the layer-2
    polygons."""
    L = ["## Bridges (layer 2) replaced by rectangles\n",
         "Each layer-2 polygon P is replaced by the rectangle R minimising "
         "|P xor R|. IoU = |P n R| / |P u R|, deviation = largest distance "
         "of a vertex of P to the boundary of R. The angle is that of the "
         "long side, in (-90, 90] deg."
         + (f" Rectangles tilted by at most {snap_deg:g} deg with respect "
            "to an axis are made exactly axis-aligned." if snap_deg else "")
         + "\n",
         "| bridge | vertices | x_c | y_c | length (um) | width (um) | "
         "angle (deg) | snapped from (deg) | IoU | deviation (um) |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for f in fits:
        sn = f"{f['tilt_before_snap']:+.4g}" if f['snapped'] else "-"
        L.append(f"| {f['index']} | {f['n_vertices']} | {f['center'][0]:.4f} | "
                 f"{f['center'][1]:.4f} | {f['length']:.4f} | "
                 f"{f['width']:.4f} | {f['angle']:+.4f} | {sn} | "
                 f"{f['iou']:.6f} | {f['dmax']:.3g} |")
    L.append("")
    return L

def _invocation_lines(args):
    """Markdown lines giving the program version and how it was called."""
    inv = args.get('invocation') or {}
    vi = args.get('version_info') or version_info()
    L = ["## Program and invocation\n"]
    v = f"scdc {vi['version']}, source fingerprint {vi['fingerprint']}"
    if 'git_commit' in vi:
        v += f", git commit {vi['git_commit']}" + (
            " with uncommitted changes" if vi['git_dirty'] else "")
    L.append(f"- Version: {v}")
    L.append(f"- Environment: Python {vi['python']}, numpy {vi['numpy']}, "
             f"scipy {vi['scipy']}, {vi['platform']}")
    if inv.get('cwd'):
        L.append(f"- Working directory: `{inv['cwd']}`")
    L.append("")
    if inv.get('process'):
        L.append("Command typed in the terminal:\n")
        L += ["```bash", inv['process'], "```", ""]
    if inv.get('equivalent'):
        L.append("Equivalent command with every option written explicitly, "
                 "defaults included (the complete list, with the origin of "
                 "each value, is in the appendix at the end of the report):\n")
        L += ["```bash", inv['equivalent'], "```", ""]
        if inv.get('unset'):
            L.append("Options not set (unused, or value derived by the "
                     "program as stated in the appendix): "
                     + ", ".join(f"`{o}`" for o in inv['unset']) + ".\n")
        if inv.get('flags_off'):
            L.append("Flags not given: " + ", ".join(
                f"`{o}`" for o in inv['flags_off']) + ".\n")
    if inv.get('api'):
        L.append("Called from Python, not from the terminal:\n")
        L += ["```python", inv['api'], "```", ""]
    return L


def _options_appendix(args):
    inv = args.get('invocation') or {}
    opts = inv.get('options')
    if not opts:
        return []
    L = ["## Appendix. Value of every option\n",
         "| option | value | origin |", "|---|---|---|"]
    for o, val, origin in opts:
        val = val.replace("|", "\\|")
        L.append(f"| `{o}` | {val} | {origin} |")
    L.append("")
    return L


def _field_lines(model, res, args):
    """Markdown lines of the response to the applied field."""
    fr = res.get('field')
    if fr is None:
        return []
    rings = model['rings']
    B_T = fr['Bext_T']
    L = ["## Applied perpendicular field B_ext\n"]
    L.append(f"Uniform field B_ext = {fr['Bext_uT']:g} uT = {B_T:.6g} T "
             "along +z, perpendicular to the plane of the GDS, applied after "
             "a zero-field cooldown (fluxoid n = 0 in every hole), with no "
             "injected current. The layer-3 and layer-4 pads carry no "
             "current in this computation, a DC current through their "
             "resistances to ground cannot be sustained.\n")
    L.append("For each layer-5 surface S of net area A (holes removed) and "
             "normal +z, Phi = Phi_applied + Phi_screening with "
             "Phi_applied = A B_ext and Phi_screening = Oint_dS A_K . dl the "
             "flux of the currents of the film. The flux concentration is "
             "C = Phi / (A B_ext). C = 1 without superconductor, C < 1 for a "
             "surface covered by screening metal, C > 1 where the flux "
             "expelled from the metal is pushed, typically slots and gaps "
             "open to the outside. A hole enclosed by a closed loop of metal "
             "keeps n = 0 and therefore C close to 0. Surfaces evaluated at "
             f"z = "
             f"{args.get('loop_z', 0.0):g} um.\n")
    if rings:
        L.append("| surface | x_c | y_c | area A (um^2) | A B_ext / Phi0 | "
                 "Phi / Phi0 | concentration Phi/(A B_ext) |")
        L.append("|---|---|---|---|---|---|---|")
        for i, r in enumerate(rings):
            L.append(f"| {r['name']} | {r['centroid'][0]:.3f} | "
                     f"{r['centroid'][1]:.3f} | {fr['area'][i]:.6g} | "
                     f"{fr['flux_applied'][i]/PHI0:.6g} | "
                     f"{fr['flux'][i]/PHI0:.6g} | "
                     f"{fr['concentration'][i]:.6g} |")
        L.append("")
    else:
        L.append("No flux surface in layer 5.\n")
    sol = fr['solution']
    it = sol.info.get('iterations', '-')
    rr = sol.info.get('residual', None)
    L.append(f"- Self energy of the screening currents E = {fr['energy']:.6e} J")
    L.append(f"- Magnetic moment of the film m_z = (1/2) Int (r x K).z dA = "
             f"Int K.A_ext dA / B_ext = {fr['m_z']*1e-12:.6e} A.m^2 "
             "(negative for a diamagnetic response)")
    if fr.get('check') is not None:
        L.append(f"- Consistency of the linear response, B_ext m_z / (-2 E) "
                 f"= {fr['check']:.6f} (1 expected without nonlinear "
                 "junctions)")
    L.append(f"- Solver: {it} CG iterations, relative residual "
             + (f"{rr:.2e}" if rr is not None else "-"))
    L.append("")
    return L


def write_report(path, model, res, sols, args, files):
    """Markdown report. The program version and the command line come
    first, then the mutual-inductance matrix, the eigenvalues of its
    inverse and, with --Bext, the flux concentrations."""
    asm = model['assembly']
    snames = list(model['source_names'])
    lnames = [r['name'] for r in model['rings']]
    M = res['M']
    I = model['current']
    fr = res.get('field')
    L = []
    L.append(f"# scdc report: {args.get('gds', '')}\n")
    head = f"Generated {datetime.datetime.now():%Y-%m-%d %H:%M}. "
    if snames:
        head += (f"Injected current I = {I*1e3:g} mA in each layer-3 polygon "
                 "in turn, returned through the layer-4 polygons. ")
    if fr is not None:
        head += (f"Applied field B_ext = {fr['Bext_uT']:g} uT along +z, "
                 "computed separately without injected current. ")
    L.append(head + "All lengths in um.\n")
    L += _invocation_lines(args)

    # ---------------------------------------------- matrix M
    L.append("## Mutual-inductance matrix M = Phi / I (pH)\n")
    if snames:
        L.append("Rows are the flux surfaces of layer 5, columns the "
                 "injection polygons of layer 3. M[i, j] is the flux through "
                 "surface i when the current I is injected in polygon j "
                 "alone, divided by I.\n")
    if M.size:
        L.append("| surface \\ source | " + " | ".join(snames) + " |")
        L.append("|---|" + "---|" * len(snames))
        for i, nm in enumerate(lnames):
            L.append(f"| {nm} | " + " | ".join(f"{m*1e12:.6g}" for m in M[i]) + " |")
        L.append("")
    elif not snames:
        L.append("No injection polygon in layer 3, M is not computed.\n")
    else:
        L.append("No flux surface in layer 5, M is empty.\n")

    # ---------------------------------------------- eigenvalues of M^-1
    if snames:
        L.append("## Eigenvalues of M^-1 (mA / Phi0)\n")
        L.append("Phi0 = h / 2e = 2.067833848e-15 Wb. Each eigenvalue "
                 "lambda_k of M^-1 (A/Wb) is expressed as lambda_k Phi0 in "
                 "mA, the current along the k-th eigen-direction producing "
                 "one flux quantum through the corresponding combination of "
                 "surfaces. Sorted by increasing modulus.\n")
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
                         "injection polygons are distinct objects), hence "
                         "complex eigenvalues can occur.\n")
        else:
            L.append(f"Not computed. {msg}.\n")
            if M.size and M.ndim == 2:
                sv = np.linalg.svd(M, compute_uv=False)
                good = sv > sv.max() * 1e-15
                L.append("Singular values of the pseudo-inverse M^+ "
                         "(mA/Phi0), which coincide with the moduli of the "
                         "eigenvalues of M^-1 when M is square and normal:\n")
                L.append("| k | 1/sigma_k (mA/Phi0) |")
                L.append("|---|---|")
                for k, s in enumerate(sv[good]):
                    L.append(f"| {k} | {PHI0 * 1e3 / s:.6g} |")
                L.append("")

    # ---------------------------------------------- applied field
    L += _field_lines(model, res, args)

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
    if snames:
        L.append(f"- Injected current I = {I*1e3:g} mA")
        gs = model.get('ground_shares')
        if gs is None:
            L.append(f"- Ground current shares (layer 4): "
                     f"{np.round(model['ground_share'], 6).tolist()}")
        else:
            L.append("- Ground current shares (layer 4), per source. A pad "
                     "not connected in DC to the source carries no return "
                     "current:")
            for nm, row in zip(snames, np.atleast_2d(gs)):
                L.append(f"  - {nm}: {np.round(row, 6).tolist()}")
    if fr is not None:
        L.append(f"- Applied field B_ext = {fr['Bext_uT']:g} uT along +z, "
                 "vector potential A_ext = (B_ext/2) z x (r - r_c) with r_c = "
                 f"({fr['center'][0]:.6g}, {fr['center'][1]:.6g}) um (the "
                 "result does not depend on r_c)")
    L.append(f"- Flux surfaces evaluated at z = {args.get('loop_z', 0.0):g} um")
    L.append(f"- Edge segment length: {args.get('seg_len'):g} um near the region "
             f"of interest")
    L.append(f"- Solver: {'dense reference' if args.get('dense') else 'pFFT + conjugate gradient'}"
             + ("" if args.get('dense') else f", near_cells = {args.get('near_cells')}"))
    if model.get('contacts'):
        L.append(f"- Bridges: {len(model['bridges'])}, contacts (feet): "
                 f"{len(model['contacts'])}, pillar inductances "
                 f"{'included' if np.any(model.get('pillar_L')) else 'disabled'}")
    if model.get('bridges'):
        L.append("- Bridge shape: " + (
            "closest rectangle to each layer-2 polygon (minimum area of the "
            "symmetric difference), axis snapping for tilts up to "
            f"{args.get('bridge_snap_angle', 0.5):g} deg"
            if model.get('bridge_fits') is not None
            else "layer-2 polygons as drawn"))
    L.append("")

    # ---------------------------------------------- bridge rectangles
    if model.get('bridge_fits'):
        L += _bridge_table(model['bridge_fits'],
                           args.get('bridge_snap_angle'))

    # ---------------------------------------------- mesh
    L.append("## Mesh\n")
    L.append(f"- Sheets: {len(asm.sheets)} ({', '.join(s.name for s in asm.sheets)})")
    L.append(f"- Triangles: {asm.n_tri}, nodes: {asm.n_nodes}")
    h = asm.h_tri
    L.append(f"- Triangle size sqrt(area): min {h.min():.3g}, median "
             f"{np.median(h):.3g}, max {h.max():.3g} um")
    L.append("")

    # ---------------------------------------------- sources
    if snames:
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
        cols = [f"Phi/Phi0 ({s})" for s in snames]
        if fr is not None:
            cols.append("Phi/Phi0 (B_ext)")
        L.append("| surface | x_c | y_c | area (um^2) | " +
                 " | ".join(cols) + " |")
        L.append("|---|---|---|---|" + "---|" * len(cols))
        for i, r in enumerate(model['rings']):
            vals = [f"{p/PHI0:.6g}" for p in res['flux'][i]]
            if fr is not None:
                vals.append(f"{fr['flux'][i]/PHI0:.6g}")
            L.append(f"| {r['name']} | {r['centroid'][0]:.3f} | "
                     f"{r['centroid'][1]:.3f} | {r['area']:.4g} | " +
                     " | ".join(vals) + " |")
        L.append("")
    else:
        L.append("None.\n")

    # ---------------------------------------------- pillars
    cases = list(zip(snames, sols))
    if fr is not None:
        cases.append(("B_ext", fr['solution']))
    if model.get('contacts') and cases and \
            all('contact_currents' in s.info for _, s in cases):
        L.append("## Pillar currents (uA, upward)\n")
        pL = model.get('pillar_L')
        L.append("| pillar | bridge | x_c | y_c | L_self (pH) | " +
                 " | ".join(nm for nm, _ in cases) + " |")
        L.append("|---|---|---|---|---|" + "---|" * len(cases))
        for k, c in enumerate(model['contacts']):
            Lk = pL[k, k] * 1e12 if pL is not None and pL.size else 0.0
            vals = [s.info['contact_currents'][k] * 1e6 for _, s in cases]
            L.append(f"| {k} | {getattr(c, 'bridge', -1)} | {c.center[0]:.2f} | "
                     f"{c.center[1]:.2f} | {Lk:.3f} | " +
                     " | ".join(f"{v:.4f}" for v in vals) + " |")
        L.append("")

    # ---------------------------------------------- junctions
    junctions = model.get('junctions', [])
    if junctions:
        IJ = res['junction_current']            # (n_sources, n_junctions)
        LJ = res['junction_LJ']                 # (n_sources, n_junctions)
        rows = [(s, IJ[j], LJ[j]) for j, s in enumerate(snames)]
        if fr is not None:
            rows.append(("B_ext", fr['junction_current'], fr['junction_LJ']))
        L.append("## Josephson junctions (layer 6)\n")
        L.append(f"Nonlinear iteration L_J(I) = L_J0 / sqrt(1 - (I/I_c)^2): "
                 f"{'on' if args.get('nonlinear') else 'off'}.\n")
        for jn in junctions:
            L.append(f"### Junction {jn['index']}\n")
            L.append(f"L_J0 = {jn['LJ0']*1e9:.4g} nH, I_c = {jn['Ic']*1e6:.4g} uA, "
                     f"E_J/h = {jn['EJ']/6.62607015e-34/1e9:.4g} GHz, "
                     f"F = L/L_square = {jn['F']:.4g}\n")
            L.append("| case | I_J (uA) | I_J / I_c | phase (rad) | L_J (nH) |")
            L.append("|---|---|---|---|---|")
            over = []
            for nm, ij, lj in rows:
                k = jn['index']
                x = float(np.clip(ij[k] / jn['Ic'], -1, 1))
                if abs(ij[k]) >= jn['Ic']:
                    over.append(nm)
                L.append(f"| {nm} | {ij[k]*1e6:.4f} | {x:.4f} | "
                         f"{np.arcsin(x):.5f} | {lj[k]*1e9:.4g} |")
            L.append("")
            if over:
                L.append("Warning: |I_J| >= I_c for " + ", ".join(over) +
                         ". No static solution exists with this junction in "
                         "the zero-voltage state, the values above are those "
                         "of the linear (or capped nonlinear) model and are "
                         "not physical. Reduce the current or the field.\n")

    # ---------------------------------------------- files
    L.append("## Output files\n")
    for f in files:
        L.append(f"- `{f}`")
    L.append("")
    L += _options_appendix(args)
    with open(path, "w") as f:
        f.write("\n".join(L))
    return path


# ----------------------------------------------------------------------
def solve_model(model, seg_len, out_prefix="scdc", loop_z=0.0, plot=True,
                show=False, dense=False, near_cells=4, near_hmax=None,
                grid=None, log_color=False, nonlinear=False, nl_tol=1e-4,
                nl_maxiter=20, backend="auto", thickness=None, verbose=True,
                report=True, report_args=None, Bext=None):
    """Solves the problem of `model` once per injection polygon, then, if
    Bext (uT) is given, once more for the uniform field Bext along +z with
    no injected current. Writes the outputs and returns (sols, res).
    `model` is the dictionary produced by geometry.build_model, or any
    dictionary with the same keys (this allows tests without the GDS tool
    chain). `terminal_sets` may be empty when Bext is given.

    res contains
        M        (n_surfaces, n_sources) mutual inductances in henry
        flux     (n_surfaces, n_sources) fluxes in weber
        energy, L_eff (n_sources,)
        junction_current, junction_LJ (n_sources, n_junctions)
        inv_eigenvalues (n,) complex, in mA/Phi0, or None
        field    None, or a dictionary with Bext_uT, Bext_T, K (n_tri, 2),
                 flux, flux_applied, area, concentration (n_surfaces,),
                 energy, m_z (A.um^2), check = Bext m_z / (-2 E),
                 junction_current, junction_LJ (n_junctions,), solution
    """
    asm = model['assembly']
    current = model['current']
    junctions = model.get('junctions', [])
    terminal_sets = model['terminal_sets']
    snames = list(model['source_names'])
    nsrc = len(terminal_sets)
    if nsrc == 0 and Bext is None:
        raise ValueError("nothing to compute, no injection polygon in "
                         "layer 3 and no applied field (--Bext)")
    report_args = dict(report_args or {})
    report_args.update(seg_len=seg_len, loop_z=loop_z, dense=dense,
                       near_cells=near_cells, nonlinear=nonlinear,
                       thickness=thickness, Bext=Bext)

    if verbose:
        print(f"  Lambda = {model['Lambda']:.4g} um, "
              f"L_square = {MU0*model['Lambda']*1e12:.4g} pH/square")
        if nsrc:
            gs = model.get('ground_shares')
            if gs is None:
                print(f"  ground current shares: "
                      f"{np.round(model['ground_share'], 6)}")
            else:
                for nm, row in zip(snames, np.atleast_2d(gs)):
                    print(f"  ground current shares for {nm}: "
                          f"{np.round(row, 6)}")
            print(f"  {nsrc} injection polygon(s): {', '.join(snames)}")
        if Bext is not None:
            print(f"  applied field B_ext = {Bext:g} uT along +z")

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

    def _solve(terminals, kernel=None, drive=None):
        if dense:
            s = solve_currents(asm, terminals, contacts, pillar_L=pL,
                               drive=drive, verbose=verbose)
            s.energy = total_energy(asm, s.K, contacts=contacts, pillar_L=pL)
            return s
        return solve_currents_fast(asm, terminals, contacts,
                                   near_cells=near_cells, near_hmax=near_hmax,
                                   grid=grid, kernel=kernel, pillar_L=pL,
                                   drive=drive, verbose=verbose)

    # the pFFT kernel depends on the geometry only, it is built once and
    # shared by all sources, the field case and all nonlinear iterations
    kernel = None if dense else FastKernel(asm, grid=grid, near_cells=near_cells,
                                           near_hmax=near_hmax,
                                           backend=get_backend(backend, verbose),
                                           verbose=verbose)

    def _solve_case(terminals, drive=None):
        """One computation, the junctions restarting from their
        small-current inductance. Returns (sol, I_J, L_J)."""
        for jn in junctions:
            jn['LJ'] = jn['LJ0']
            jn['Lambda'] = jn['LJ'] / (MU0 * jn['F'])
            asm.tri_Lambda[jn['tri']] = jn['Lambda']
        sol = _solve(terminals, kernel, drive)
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
                sol = _solve(terminals, kernel, drive)
                IJ = junction_currents(asm, sol.K, junctions)
        return sol, IJ, np.array([jn['LJ'] for jn in junctions])

    sols, Ks, energies, IJs, LJs = [], [], [], [], []
    for j, terminals in enumerate(terminal_sets):
        if verbose:
            print()
            print(f"  ---- source {j}: {snames[j]} ----")
        sol, IJ, LJ = _solve_case(terminals)
        sols.append(sol)
        Ks.append(sol.K)
        energies.append(sol.energy)
        IJs.append(IJ)
        LJs.append(LJ)
        if verbose:
            print(f"  energy E = {sol.energy:.6e} J, "
                  f"L = 2E/I^2 = {2*sol.energy/current**2*1e12:.4f} pH")

    Ks = np.array(Ks).reshape(nsrc, asm.n_tri, 2)
    energies = np.array(energies, dtype=float)
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
    if nsrc:
        w, cond, msg = inverse_eigenvalues(M)
    else:
        w, cond, msg = None, None, "no injection polygon in layer 3"
    lnames = [r['name'] for r in model['rings']]

    # ------------------------------------------------ applied field
    field = None
    if Bext is not None:
        if verbose:
            print()
            print(f"  ---- applied field B_ext = {Bext:g} uT ----")
        B_T = float(Bext) * 1e-6                     # tesla
        Bz = B_T * 1e-12                             # Wb/um^2
        lo, hi = asm.points.min(axis=0), asm.points.max(axis=0)
        center = 0.5 * (lo + hi)
        drive = external_field_drive(asm, Bz, center, contacts=contacts)
        sol_f, IJ_f, LJ_f = _solve_case([], drive)
        area = np.array([ring_area(r) for r in model['rings']], dtype=float)
        phi_scr = surface_fluxes(asm, sol_f.K, model['rings'], z=loop_z,
                                 seg=0.5 * seg_len) if nl else np.zeros(0)
        phi_app = Bz * area
        phi = phi_app + phi_scr
        with np.errstate(divide="ignore", invalid="ignore"):
            conc = np.where(phi_app != 0, phi / phi_app, np.nan)
        # m_z = Int K.A_ext dA / B_z, i.e. (1/2) Int (r x K).z dA including
        # the closure of the bridge loops at the feet, independent of r_c
        W = float(sol_f.info.get('drive_energy', np.nan))
        m_z = W / Bz if Bz != 0 else magnetic_moment_z(asm, sol_f.K, center)
        E_f = float(sol_f.energy)
        check = W / (-2.0 * E_f) if E_f > 0 and np.isfinite(W) else None
        if nonlinear and junctions:
            check = None
        field = dict(Bext_uT=float(Bext), Bext_T=B_T, center=center,
                     K=sol_f.K, flux=phi, flux_applied=phi_app,
                     flux_screening=phi_scr, area=area, concentration=conc,
                     energy=E_f, m_z=m_z, check=check,
                     junction_current=IJ_f, junction_LJ=LJ_f, solution=sol_f)
        if verbose:
            print(f"  self energy E = {E_f:.6e} J, m_z = {m_z*1e-12:.6e} "
                  "A.m^2" + (f", B m_z / (-2E) = {check:.6f}"
                              if check is not None else ""))

    if verbose:
        if nsrc:
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
        if field is not None and nl:
            print()
            print(f"  flux concentration Phi/(A B_ext), B_ext = {Bext:g} uT")
            wid = max([len(s) for s in lnames] + [8])
            print("  " + f"{'surface':{wid}s}" + "     area[um^2]   "
                  "A B/Phi0      Phi/Phi0   concentration")
            for i, nm in enumerate(lnames):
                print("  " + f"{nm:{wid}s}" + f" {field['area'][i]:14.6g} "
                      f"{field['flux_applied'][i]/PHI0:10.5g} "
                      f"{field['flux'][i]/PHI0:13.5g} "
                      f"{field['concentration'][i]:15.6g}")
        cases = list(zip(snames, sols))
        if field is not None:
            cases.append(("B_ext", field['solution']))
        if contacts and cases and \
                all('contact_currents' in s.info for _, s in cases):
            print()
            print("  pillar   bridge     x_c       y_c    L_self[pH]  " +
                  "  ".join(f"I[uA] {nm}" for nm, _ in cases))
            for k, c in enumerate(contacts):
                Lk = pL[k, k] * 1e12 if pL is not None and pL.size else 0.0
                print(f"  {k:6d} {getattr(c, 'bridge', -1):8d} "
                      f"{c.center[0]:9.2f} {c.center[1]:9.2f} {Lk:11.3f}  "
                      + "  ".join(f"{s.info['contact_currents'][k]*1e6:12.4f}"
                                  for _, s in cases))
        if junctions:
            rows = [(s, IJs[j], LJs[j]) for j, s in enumerate(snames)]
            if field is not None:
                rows.append(("B_ext", field['junction_current'],
                             field['junction_LJ']))
            print()
            print("  case            junction   I_J[uA]    I_J/I_c   phase[rad]   L_J[nH]")
            for nm, ij, lj in rows:
                for jn in junctions:
                    k = jn['index']
                    if abs(ij[k]) >= jn['Ic']:
                        print(f"  warning: junction {k} carries |I_J| >= I_c "
                              f"for {nm}, no static zero-voltage solution")
                    x = np.clip(ij[k] / jn['Ic'], -1, 1)
                    print(f"  {nm:14s} {k:8d} {ij[k]*1e6:10.4f} {x:10.4f} "
                          f"{np.arcsin(x):12.5f} {lj[k]*1e9:10.4g}")

    # ------------------------------------------------ files
    files = []
    fn = f"{out_prefix}_solution.npz"
    extra = {}
    if field is not None:
        extra = dict(K_field=field['K'], Bext_uT=field['Bext_uT'],
                     flux_field=field['flux'],
                     concentration=field['concentration'],
                     area_field=field['area'])
    np.savez_compressed(
        fn,
        points=asm.points, triangles=asm.triangles, tri_sheet=asm.tri_sheet,
        node_sheet=asm.node_sheet, K=Ks, area=asm.area,
        centroid=asm.centroid, tri_z=asm.tri_z, Lambda=model['Lambda'],
        current=current, energy=energies, L_eff=L_eff, flux=flux, M=M,
        source_names=np.array(snames, dtype=str),
        loop_names=np.array(lnames, dtype=str),
        thickness=thickness if thickness else 0.0,
        junction_current=IJs, version=__version__,
        overlays=np.array(json.dumps(model['overlays'])), **extra)
    files.append(fn)

    if nsrc:
        fn = f"{out_prefix}_mutual.csv"
        with open(fn, "w") as f:
            f.write("surface_index,surface_name,area_um2,xc_um,yc_um,"
                    + ",".join(f"M_pH_{_safe_name(s)}" for s in snames) + "\n")
            for i, r in enumerate(model['rings']):
                f.write(f"{i},{r['name']},{r['area']:.6g},{r['centroid'][0]:.6g},"
                        f"{r['centroid'][1]:.6g},"
                        + ",".join(f"{m*1e12:.8e}" for m in M[i]) + "\n")
        files.append(fn)

    if field is not None:
        fn = f"{out_prefix}_field.csv"
        with open(fn, "w") as f:
            f.write("surface_index,surface_name,area_um2,xc_um,yc_um,"
                    "Bext_uT,flux_applied_Phi0,flux_Phi0,concentration\n")
            for i, r in enumerate(model['rings']):
                f.write(f"{i},{r['name']},{field['area'][i]:.8g},"
                        f"{r['centroid'][0]:.6g},{r['centroid'][1]:.6g},"
                        f"{field['Bext_uT']:.8g},"
                        f"{field['flux_applied'][i]/PHI0:.8e},"
                        f"{field['flux'][i]/PHI0:.8e},"
                        f"{field['concentration'][i]:.8e}\n")
        files.append(fn)

    res = dict(M=M, flux=flux, energy=energies, L_eff=L_eff,
               junction_current=IJs, junction_LJ=LJs, inv_eigenvalues=w,
               cond=cond, inv_message=msg, source_names=snames,
               loop_names=lnames, field=field)

    if plot or show:
        title = f"$\\Lambda$ = {model['Lambda']:.3g} $\\mu$m"
        maps = [Ks[j] for j in range(nsrc)]
        names = list(snames)
        captions = [f"injection of I = {current*1e3:g} mA in {nm}"
                    for nm in snames]
        groups = [0] * nsrc
        tags = [f"{k}_{_safe_name(nm)}" for k, nm in enumerate(snames)]
        if field is not None:
            maps.append(field['K'])
            names.append("B_ext")
            captions.append(f"applied field B_ext = {field['Bext_uT']:g} uT "
                            "along +z")
            groups.append(1)
            tags.append("Bext")
        files += plot_solutions(model, np.array(maps), names, out_prefix,
                                thickness=thickness, title=title,
                                log_color=log_color, show=show,
                                verbose=verbose, captions=captions,
                                groups=groups, file_tags=tags)

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
        nl_tol=1e-4, nl_maxiter=20, backend="auto", verbose=True,
        Bext=None, invocation=None, **kw):
    """Builds the model from the GDS and solves it once per injection
    polygon, and once more for the applied field if Bext (uT, along +z) is
    given. Returns (model, sols, res) with sols a list of Solution, one per
    layer-3 polygon, and res the dictionary described in solve_model.

    invocation : dictionary describing the command line, filled by main().
    When run() is called directly from Python, the report gives the call
    with its arguments instead."""
    call = dict(seg_len=seg_len, thickness=thickness, lambda_L=lambda_L,
                L_square=L_square, current=current, out_prefix=out_prefix,
                loop_z=loop_z, plot=plot, show=show, dense=dense,
                near_cells=near_cells, near_hmax=near_hmax, grid=grid,
                log_color=log_color, nonlinear=nonlinear,
                mesh_only=mesh_only, nl_tol=nl_tol, nl_maxiter=nl_maxiter,
                backend=backend, Bext=Bext, **kw)
    if invocation is None:
        invocation = dict(
            cwd=os.getcwd(),
            api="run(" + ", ".join([repr(gds)] + [f"{k}={v!r}"
                                                  for k, v in call.items()])
                + ")")
    try:
        from .geometry import build_model
    except ImportError:
        from geometry import build_model

    if verbose:
        print(f"  scdc {__version__}")
    model = build_model(gds, thickness=thickness, lambda_L=lambda_L,
                        L_square=L_square, seg_len=seg_len, current=current,
                        verbose=verbose, require_sources=Bext is None, **kw)
    if mesh_only:
        mesh_report(model, f"{out_prefix}_mesh.png", near_cells=near_cells,
                    grid=grid, near_hmax=near_hmax)
        return model, None, None
    sols, res = solve_model(
        model, seg_len, out_prefix=out_prefix, loop_z=loop_z, plot=plot,
        show=show, dense=dense, near_cells=near_cells, near_hmax=near_hmax,
        grid=grid, log_color=log_color, nonlinear=nonlinear, nl_tol=nl_tol,
        nl_maxiter=nl_maxiter, backend=backend, thickness=thickness,
        verbose=verbose, Bext=Bext,
        report_args=dict(gds=gds,
                         bridge_snap_angle=kw.get('bridge_snap_angle', 0.5),
                         thickness=thickness, lambda_L=lambda_L,
                         L_square=L_square, invocation=invocation,
                         version_info=version_info()))
    return model, sols, res


# ----------------------------------------------------------------------
def _process_command():
    """Command line of the running process, as typed in the terminal (up
    to the quoting of the shell)."""
    orig = getattr(sys, "orig_argv", None)          # Python >= 3.10
    if orig:
        return shlex.join([os.path.basename(orig[0])] + list(orig[1:]))
    main_mod = sys.modules.get("__main__")
    spec = getattr(main_mod, "__spec__", None)
    exe = os.path.basename(sys.executable) or "python"
    if spec is not None and spec.name:              # python -m package.module
        return shlex.join([exe, "-m", spec.name] + sys.argv[1:])
    return shlex.join([exe] + sys.argv)


def _unset_meaning(a):
    """Value taken by the options left at None, keyed by option string."""
    seg = a.seg
    return {
        "--thickness": "not used" if a.Lsq is not None else "not set",
        "--lambda-london": "not used" if a.Lsq is not None else "not set",
        "--Lsq": "not set, Lambda from --thickness and --lambda-london",
        "--EJ": "not set", "--Ic": "not set", "--LJ": "not set",
        "--junction-file": "not set",
        "--bridge-Lsq": "not set, bridges as layer 1 unless "
                        "--bridge-thickness or --bridge-lambda",
        "--seg-far": f"not set, 5*seg = {5 * seg:g}",
        "--fine-radius": f"not set, 200*seg/1.5 = {200 * seg / 1.5:g}",
        "--seg-max": f"not set, 30*seg = {30 * seg:g}",
        "--bridge-thickness": "not set, = --thickness",
        "--bridge-lambda": "not set, = --lambda-london",
        "--pillar-height": f"not set, = --bridge-height = {a.bridge_height:g}",
        "--pillar-lambda": "not set, = lambda of the bridges",
        "--ground-resistances": "not set, 0.01 Ohm per layer-4 polygon",
        "--cell": "not set, first top-level cell",
        "--near-hmax": "not set, no cap",
        "--grid": "not set, median triangle size",
        "--min-seg": "not set, contours as drawn",
        "--pad-tol": "not set, pads as drawn",
        "--Bext": "not set, no applied field",
    }


def _describe_options(parser, ns, argv):
    """Equivalent command with every option explicit, list of
    (option, value, origin) and lists of unset options and absent flags."""
    # options given on the command line: parse again with every default
    # suppressed, abbreviations being resolved by argparse itself
    saved = [(act, act.default) for act in parser._actions]
    try:
        for act, _ in saved:
            act.default = argparse.SUPPRESS
        given = set(vars(parser.parse_args(argv)))
    finally:
        for act, d in saved:
            act.default = d
    # package layout (python -m scdc.run) or flat layout (python run.py)
    prog = ["python", "-m", f"{__package__}.run"] if __package__ else \
        ["python", "run.py"]
    meaning = _unset_meaning(ns)
    parts, table, unset, off = list(prog), [], [], []
    for act in parser._actions:
        if act.dest in ("help", "version") or act.dest is argparse.SUPPRESS:
            continue
        val = getattr(ns, act.dest, None)
        origin = "command line" if act.dest in given else "default"
        if not act.option_strings:                  # positional
            parts.append(str(val))
            table.append((act.dest, str(val), origin))
            continue
        opt = max(act.option_strings, key=len)
        if isinstance(act, argparse._StoreTrueAction):
            table.append((opt, "on" if val else "off", origin))
            if val:
                parts.append(opt)
            else:
                off.append(opt)
            continue
        if val is None:
            table.append((opt, meaning.get(opt, "not set"), origin))
            unset.append(opt)
            continue
        parts += [opt, str(val)]
        table.append((opt, str(val), origin))
    return shlex.join(parts), table, unset, off


def main(argv=None):
    p = argparse.ArgumentParser(
        description="DC current density and mutual inductances in a "
                    "superconducting circuit described by a GDS. Each "
                    "polygon of layer 3 is treated in turn as the single "
                    "injection point. With --Bext, the response to a "
                    "uniform field perpendicular to the plane and the flux "
                    "concentration in the layer-5 surfaces are computed as "
                    "well. All lengths in micrometres.")
    p.add_argument("--version", action="version",
                   version=f"scdc {__version__}")
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
    p.add_argument("--Bext", type=float, default=None,
                   help="uniform applied field along +z, perpendicular to "
                        "the plane of the GDS (uT). Adds a computation "
                        "without injected current, the current map of the "
                        "screening currents and the flux concentration "
                        "Phi/(A B_ext) of each layer-5 surface. Layers 3 "
                        "and 4 may then be empty")
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
    p.add_argument("--bridge-shape", type=str, default="rect",
                   choices=["rect", "exact"],
                   help="'rect' replaces each layer-2 polygon by its closest "
                        "rectangle (minimum area of the symmetric "
                        "difference), 'exact' keeps the polygons as drawn "
                        "(default rect)")
    p.add_argument("--bridge-snap-angle", type=float, default=0.5,
                   help="bridge rectangles tilted by at most this angle (deg) "
                        "with respect to the x or y axis are made exactly "
                        "axis-aligned, 0 disables (default 0.5)")
    argv = sys.argv[1:] if argv is None else list(argv)
    a = p.parse_args(argv)
    equivalent, table, unset, off = _describe_options(p, a, argv)
    invocation = dict(process=_process_command(), equivalent=equivalent,
                      options=table, unset=unset, flags_off=off,
                      cwd=os.getcwd())

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

    run(a.gds, a.seg, Bext=a.Bext, invocation=invocation,
        bridge_shape=a.bridge_shape, bridge_snap_angle=a.bridge_snap_angle,
        thickness=a.thickness, lambda_L=a.lambda_london,
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
