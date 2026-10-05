"""
geometry.py -- Lecture du GDS, booleens et maillage.

Dependances optionnelles (seulement pour cette couche) :
    gdstk     lecture GDSII
    shapely   booleens polygonaux
    triangle  mailleur de Shewchuk (Triangle)
"""

from __future__ import annotations

import numpy as np

try:
    from .core import Assembly, Sheet, effective_penetration_depth
    from .solver import Contact, Terminal, build_terminal_sets
except ImportError:
    from core import Assembly, Sheet, effective_penetration_depth
    from solver import Contact, Terminal, build_terminal_sets


# ----------------------------------------------------------------------
def read_gds_layers(path, layers, cell=None, scale=1.0):
    """Retourne {layer: shapely geometry} pour les layers demandes."""
    import gdstk
    from shapely.geometry import Polygon
    from shapely.ops import unary_union

    lib = gdstk.read_gds(path)
    if cell is None:
        cells = lib.top_level()
        if not cells:
            raise ValueError("no top-level cell in the GDS")
        c = cells[0]
    else:
        c = {x.name: x for x in lib.cells}[cell]

    polys = c.get_polygons(depth=None)
    out = {}
    for lay in layers:
        geoms = []
        for p in polys:
            if p.layer != lay:
                continue
            pts = np.asarray(p.points) * scale
            if len(pts) < 3:
                continue
            g = Polygon(pts)
            if not g.is_valid:
                g = g.buffer(0)
            if not g.is_empty:
                geoms.append(g)
        out[lay] = unary_union(geoms) if geoms else None
    return out


def read_gds_labels(path, layer, cell=None, scale=1.0):
    """Labels (x, y, texte) d'un layer."""
    import gdstk
    lib = gdstk.read_gds(path)
    c = lib.top_level()[0] if cell is None else {x.name: x for x in lib.cells}[cell]
    out = []
    for lab in c.get_labels(depth=None):
        if lab.layer == layer:
            out.append((lab.origin[0] * scale, lab.origin[1] * scale, lab.text))
    return out


def _label_names(polys, labels, prefix):
    """Name of each polygon: the text of a GDS label lying inside it, else
    prefix + index. Duplicate names are made unique with a numeric suffix."""
    from shapely.geometry import Point
    names, seen = [], {}
    for k, p in enumerate(polys):
        name = None
        for (x, y, text) in labels:
            if p.buffer(1e-9).contains(Point(x, y)):
                name = text.strip()
                break
        if not name:
            name = f"{prefix}{k}"
        if name in seen:
            seen[name] += 1
            name = f"{name}_{seen[name]}"
        else:
            seen[name] = 0
        names.append(name)
    return names


def _sort_polygons(polys):
    """Deterministic order (centroid x, then y), the GDS order being lost in
    the boolean union."""
    key = [(round(p.centroid.x, 6), round(p.centroid.y, 6)) for p in polys]
    order = sorted(range(len(polys)), key=lambda k: key[k])
    return [polys[k] for k in order]


_UNITS = {'ghz': 1e9, 'mhz': 1e6, 'khz': 1e3, 'hz': 1.0,
          'ua': 1e-6, 'na': 1e-9, 'ma': 1e-3, 'a': 1.0,
          'nh': 1e-9, 'ph': 1e-12, 'uh': 1e-6, 'h': 1.0,
          'kohm': 1e3, 'ohm': 1.0, 'mohm': 1e6,
          'uev': 1e-6, 'mev': 1e-3, 'ev': 1.0, 'k': 1.0}


def _parse_junction_text(text):
    """"EJ=25GHz" -> {"EJ_GHz": 25}, "Ic=30uA" -> {"Ic_uA": 30}, ..."""
    import re
    out = {}
    for key, val, unit in re.findall(
            r"([A-Za-z]+)\s*=\s*([0-9.eE+-]+)\s*([A-Za-z]*)", text):
        k, u = key.lower(), unit.lower()
        v = float(val)
        f = _UNITS.get(u, 1.0)
        if k == 'ej':
            out['EJ_GHz'] = v * f / 1e9 if u.endswith('hz') else v
        elif k == 'ic':
            out['Ic_uA'] = v * f * 1e6 if u.endswith('a') else v
        elif k == 'lj':
            out['LJ_nH'] = v * f * 1e9 if u.endswith('h') else v
        elif k == 'rn':
            out['Rn_Ohm'] = v * f
        elif k == 'delta':
            out['Delta_ueV'] = v * f * 1e6 if u.endswith('ev') else v
    return out


def _norm_params(p):
    """Cles utilisateur (unites pratiques) -> arguments de josephson_inductance."""
    q = {}
    if 'EJ_GHz' in p: q['EJ_GHz'] = float(p['EJ_GHz'])
    if 'EJ_J' in p: q['EJ_J'] = float(p['EJ_J'])
    if 'Ic_uA' in p: q['Ic_A'] = float(p['Ic_uA']) * 1e-6
    if 'Ic_A' in p: q['Ic_A'] = float(p['Ic_A'])
    if 'LJ_nH' in p: q['LJ_H'] = float(p['LJ_nH']) * 1e-9
    if 'LJ_H' in p: q['LJ_H'] = float(p['LJ_H'])
    if 'Rn_Ohm' in p: q['Rn_Ohm'] = float(p['Rn_Ohm'])
    if 'Delta_ueV' in p: q['Delta_ueV'] = float(p['Delta_ueV'])
    return q


def _as_polygon_list(geom):
    from shapely.geometry import Polygon, MultiPolygon, GeometryCollection
    if geom is None or geom.is_empty:
        return []
    if isinstance(geom, Polygon):
        return [geom]
    if isinstance(geom, (MultiPolygon, GeometryCollection)):
        return [g for g in geom.geoms if isinstance(g, Polygon) and not g.is_empty]
    return []


def _rings(geom):
    """Tous les contours (exterieurs et interieurs) d'une geometrie."""
    out = []
    for p in _as_polygon_list(geom):
        out.append(np.asarray(p.exterior.coords)[:-1])
        for r in p.interiors:
            out.append(np.asarray(r.coords)[:-1])
    return out


def _pslg(rings, tol=1e-9):
    """Assemble un graphe rectiligne planaire a partir de contours."""
    verts, segs = [], []
    for r in rings:
        r = np.asarray(r, dtype=float)
        if len(r) > 1 and np.allclose(r[0], r[-1]):
            r = r[:-1]
        if len(r) < 3:
            continue
        n0, m = len(verts), len(r)
        verts.extend(r.tolist())
        segs.extend([[n0 + i, n0 + (i + 1) % m] for i in range(m)])
    V = np.array(verts, dtype=float)
    S = np.array(segs, dtype=int)
    # fusion des sommets confondus
    key = np.round(V / max(tol, 1e-12)).astype(np.int64)
    _, first, inv = np.unique(key, axis=0, return_index=True,
                              return_inverse=True)
    inv = inv.ravel()
    V2 = V[first]
    S2 = inv[S]
    S2 = S2[S2[:, 0] != S2[:, 1]]
    S2 = np.unique(np.sort(S2, axis=1), axis=0)
    return V2, S2


def _hole_points(geom):
    """Un point dans chaque trou reel de la geometrie."""
    from shapely.geometry import Polygon
    pts = []
    for p in _as_polygon_list(geom):
        for r in p.interiors:
            h = Polygon(r).difference(geom)
            if h.is_empty:
                continue
            for q in _as_polygon_list(h):
                c = q.representative_point()
                pts.append([c.x, c.y])
    return np.array(pts) if pts else np.zeros((0, 2))


def _densify_lines(lines, spacing):
    """Subdivise chaque arete des polylignes a l'espacement local spacing(pts).

    lines : liste de tableaux (N,2), polylignes ouvertes ou fermees.
    """
    out = []
    for r in lines:
        r = np.asarray(r, dtype=float)
        if len(r) < 2:
            continue
        a, b = r[:-1], r[1:]
        L = np.linalg.norm(b - a, axis=1)
        keep = L > 0
        a, b, L = a[keep], b[keep], L[keep]
        if len(a) == 0:
            continue
        s = spacing(0.5 * (a + b))
        n = np.maximum(1, np.ceil(L / s)).astype(int)
        pts = []
        for ai, bi, ni in zip(a, b, n):
            t = np.arange(ni) / ni
            pts.append(ai + t[:, None] * (bi - ai))
        pts.append(b[-1:])
        out.append(np.vstack(pts))
    return out


def _noded_linework(geom, extra_rings, grid=1e-4):
    """Graphe planaire de tous les contours, calcule par GEOS.

    Les contours du metal et les contours imposes (contacts, pieds, jonctions)
    sont reunis en un ensemble de lignes puis "nodes" : toute intersection
    devient un sommet, tout recouvrement colineaire est fusionne. Les
    coordonnees sont prealablement arrondies sur une grille de `grid` um.
    Retourne une liste de polylignes ouvertes.
    """
    import shapely
    from shapely.geometry import LineString, MultiLineString
    from shapely.ops import unary_union
    lines = []
    for r in _rings(geom) + [np.asarray(x, dtype=float) for x in extra_rings]:
        if len(r) >= 3:
            lines.append(LineString(np.vstack([r, r[:1]])))
    ml = unary_union(lines)
    ml = shapely.set_precision(ml, grid)
    ml = unary_union(ml)                      # re-nodage apres arrondi
    if isinstance(ml, LineString):
        geoms = [ml]
    else:
        geoms = [x for x in getattr(ml, 'geoms', []) if isinstance(x, LineString)]
    return [np.asarray(x.coords) for x in geoms if len(x.coords) >= 2]


def _pslg_lines(lines, tol=1e-7):
    """PSLG (sommets, segments) a partir de polylignes ouvertes."""
    verts, segs = [], []
    for r in lines:
        n0, m = len(verts), len(r)
        verts.extend(r.tolist())
        segs.extend([[n0 + i, n0 + i + 1] for i in range(m - 1)])
    V = np.array(verts, dtype=float)
    S = np.array(segs, dtype=int)
    key = np.round(V / tol).astype(np.int64)
    _, first, inv = np.unique(key, axis=0, return_index=True,
                              return_inverse=True)
    inv = inv.ravel()
    V2 = V[first]
    S2 = inv[S]
    S2 = S2[S2[:, 0] != S2[:, 1]]
    S2 = np.unique(np.sort(S2, axis=1), axis=0)
    return V2, S2


def triangulate(geom, seg_len, extra_rings=(), min_angle=28.0, quiet=True,
                refine=(), spacing=None, seg_max=None, seg_far=None):
    """Maille une geometrie shapely avec Triangle, avec gradation.

    seg_len : longueur de reference des segments (um).
    spacing : fonction pts(N,2) -> espacement local des points de bord. Par
              defaut seg_len partout. Le maillage est fin sur les bords et
              Triangle le laisse grossir vers l'interieur.
    seg_far : espacement le plus grossier atteint par `spacing` (um), sert a
              la premiere passe de subdivision. Defaut = max de spacing sur
              les milieux d'aretes.
    seg_max : taille maximale des triangles interieurs (um), None = aucune.
    refine  : liste de (x, y, seg_local) imposant une taille de maille dans la
              region fermee contenant (x, y).
    """
    import triangle as tr
    if spacing is None:
        spacing = lambda p: np.full(len(p), seg_len)
    lines = _noded_linework(geom, extra_rings)

    # ---- passe 1 : subdivision uniforme a l'espacement grossier, pour que
    # le test d'appartenance a la zone fine soit fait par morceaux sur les
    # longues aretes
    if seg_far is None:
        mids = np.vstack([0.5 * (r[:-1] + r[1:]) for r in lines if len(r) > 1])
        seg_far = float(np.max(spacing(mids)))
    seg_far = max(float(seg_far), float(seg_len))
    lines = _densify_lines(lines, lambda p: np.full(len(p), seg_far))
    # ---- passe 2 : espacement local (seg_len dans la zone d'interet)
    lines = _densify_lines(lines, spacing)

    V, S = _pslg_lines(lines)
    H = _hole_points(geom)
    A = dict(vertices=V, segments=S)
    if len(H):
        A['holes'] = H
    flags = f"pq{min_angle:g}"
    if seg_max:
        flags += f"a{np.sqrt(3.0) / 4.0 * seg_max ** 2:.10g}"
    if refine:
        A['regions'] = np.array([[x, y, 1.0, np.sqrt(3.0) / 4.0 * s ** 2]
                                 for (x, y, s) in refine], dtype=float)
        flags += "a" if not seg_max else ""
        flags += "A"
    if quiet:
        flags += "Q"
    try:
        out = tr.triangulate(A, flags)
    except RuntimeError as e:
        np.savez("scdc_pslg_debug.npz", vertices=V, segments=S, holes=H)
        raise RuntimeError(
            "Triangle failed. The segment graph is saved in "
            "scdc_pslg_debug.npz for diagnosis. Usual causes are invalid GDS "
            "polygons (self-intersection), or contact contours crossing with "
            "vertices closer than 1e-4 um") from e
    return (np.asarray(out['vertices'], dtype=float),
            np.asarray(out['triangles'], dtype=int))


# In build_model, pass seg_far explicitly to the metal call (optional, the
# function recovers it from `spacing` otherwise):
#
#     Vm, Tm = triangulate(metal, seg_len, extra_rings=extra, refine=refine,
#                          spacing=spacing, seg_max=seg_max, seg_far=seg_far)



def contains_mask(geom, pts):
    import shapely
    return np.asarray(shapely.contains_xy(geom, pts[:, 0], pts[:, 1]))


# ----------------------------------------------------------------------
# Pieds des ponts et piliers
# ----------------------------------------------------------------------
def _select_feet(parts, mode="ends"):
    """Choix des pieds parmi les composantes connexes de pont ∩ metal.

    mode "ends" : un pont ne repose que sur ses deux extremites. On garde
    les deux composantes dont les centroides sont les plus eloignes, ce qui
    elimine les conducteurs simplement survoles (conducteur central d'une
    CPW, pistes croisees). Retourne (pieds_gardes, pieds_ignores).
    mode "all"  : toutes les composantes (comportement historique, vias
    multiples).
    """
    if mode == "all" or len(parts) <= 2:
        return list(parts), []
    c = np.array([p.centroid.coords[0] for p in parts])
    d = np.linalg.norm(c[:, None, :] - c[None, :, :], axis=2)
    i, j = np.unravel_index(int(np.argmax(d)), d.shape)
    keep = [parts[i], parts[j]]
    drop = [p for k, p in enumerate(parts) if k not in (i, j)]
    return keep, drop


# distance moyenne geometrique du contour d'un carre de cote a avec lui-meme
# (quadrature numerique, le courant supraconducteur circule en surface).
# Pour la section pleine on aurait 0.44705 a (Rosa).
_GMD_SQUARE_SHELL = 0.5823


def pillar_self_inductance(side, height, lambda_L=None, mu0=4.0e-13 * np.pi):
    """Inductance d'un pilier droit, section carree de cote `side`, longueur
    `height` (um), en henry.

    Partie magnetique, formule du conducteur droit avec distance moyenne
    geometrique R de la section (Rosa 1908, Grover 1946)

        L = (mu0 h / 2 pi) [ ln(2h/R) - 1 + R/h ],   R = 0.5823 a

    valable pour h >> a, l'erreur relative etant O((a/h)^2). Le courant
    circulant en surface, il n'y a pas de terme interne.

    Partie cinetique, couche de Meissner d'epaisseur lambda_L sur le
    perimetre 4a (valable pour a >> lambda_L)

        L_kin = mu0 lambda_L h / (4 a)
    """
    R = _GMD_SQUARE_SHELL * side
    Lm = mu0 * height / (2 * np.pi) * (np.log(2 * height / R) - 1.0 + R / height)
    Lk = 0.0 if not lambda_L else mu0 * lambda_L * height / (4 * side)
    return Lm + Lk


def pillar_mutual_inductance(dist, height, mu0=4.0e-13 * np.pi):
    """Mutuelle entre deux piliers paralleles de meme longueur h, axes a la
    distance d, courants comptes dans le meme sens (Grover, filaments)

        M = (mu0 h / 2 pi) [ ln(h/d + sqrt(1 + h^2/d^2)) - sqrt(1 + d^2/h^2) + d/h ]

    La distance des axes remplace la distance moyenne geometrique entre les
    deux sections, ce qui est exact a O((a/d)^2) pres.
    """
    d = np.asarray(dist, dtype=float)
    r = height / d
    return mu0 * height / (2 * np.pi) * (np.log(r + np.sqrt(1 + r * r))
                                         - np.sqrt(1 + 1 / (r * r)) + 1 / r)


def pillar_inductance_matrix(centers, side, height, lambda_L=None):
    """Matrice (n, n) des inductances propres et mutuelles des piliers.

    centers : (n, 2) positions des axes. L'energie des piliers vaut
    (1/2) I^T L I, I_k etant le courant montant dans le pilier k.

    Les courants verticaux ne se couplent pas aux nappes horizontales, le
    produit scalaire J.J' du noyau de Neumann etant nul, et ne produisent pas
    de B_z. Les piliers n'interviennent donc que par cette matrice.
    """
    centers = np.atleast_2d(np.asarray(centers, dtype=float))
    n = len(centers)
    L = np.zeros((n, n))
    if n == 0:
        return L
    d = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=2)
    off = ~np.eye(n, dtype=bool)
    L[off] = pillar_mutual_inductance(d[off], height)
    L[np.diag_indices(n)] = pillar_self_inductance(side, height, lambda_L)
    return L


# ----------------------------------------------------------------------
def _junction_geometry_factor(jn, V, T, verbose=True):
    """Resout Laplace dans le patch entre ses deux zones de contact.

    Contacts : noeuds du patch partages avec des triangles hors patch,
    regroupes par connexite le long du bord du patch. Les deux plus grands
    groupes sont les electrodes A (V=0) et B (V=1).
        G = Int |grad V|^2 dA,   F = L/L_carre = 1/G,
        I_J = Int K.grad V dA   (exact pour K conservatif dans le patch).
    Stocke jn['F'], jn['gradV'] (n_tri_patch, 2), jn['contacts'].
    """
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla
    tri = T[jn['tri']]
    if len(tri) == 0:
        raise ValueError(f"junction {jn['index']}: no triangle, mesh too "
                         "coarse or patch outside the metal")
    nodes = np.unique(tri)
    in_patch = np.zeros(len(V), bool)
    in_patch[nodes] = True
    tri_in = np.zeros(len(T), bool)
    tri_in[jn['tri']] = True
    # noeuds de contact : appartenant a un triangle hors patch
    outside = T[~tri_in]
    touched = np.zeros(len(V), bool)
    touched[np.unique(outside)] = True
    contact = nodes[touched[nodes]]
    # regroupement par connexite via les aretes du patch entre noeuds de contact
    e = np.vstack([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]])
    cmask = np.zeros(len(V), bool)
    cmask[contact] = True
    e = e[cmask[e[:, 0]] & cmask[e[:, 1]]]
    loc = -np.ones(len(V), int)
    loc[contact] = np.arange(len(contact))
    A = sp.coo_matrix((np.ones(len(e)), (loc[e[:, 0]], loc[e[:, 1]])),
                      shape=(len(contact), len(contact)))
    ng, lab = sp.csgraph.connected_components(A, directed=False)
    sizes = np.bincount(lab)
    order = np.argsort(-sizes)
    if ng < 2:
        raise ValueError(f"junction {jn['index']}: a single contact region, "
                         "the patch must connect two electrodes")
    if ng > 2 and verbose:
        print(f"  warning: junction {jn['index']} touches {ng} metal regions, "
              f"the two largest are taken as electrodes")
    A_nodes = contact[lab == order[0]]
    B_nodes = contact[lab == order[1]]
    # tri A/B par position pour un signe reproductible (A = plus a gauche/bas)
    if V[A_nodes].mean(axis=0).sum() > V[B_nodes].mean(axis=0).sum():
        A_nodes, B_nodes = B_nodes, A_nodes

    # rigidite locale
    p = V[tri]
    x0, y0 = p[:, 0, 0], p[:, 0, 1]
    x1, y1 = p[:, 1, 0], p[:, 1, 1]
    x2, y2 = p[:, 2, 0], p[:, 2, 1]
    a2 = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)
    grad = np.empty((len(tri), 3, 2))
    grad[:, 0, 0] = y1 - y2; grad[:, 0, 1] = x2 - x1
    grad[:, 1, 0] = y2 - y0; grad[:, 1, 1] = x0 - x2
    grad[:, 2, 0] = y0 - y1; grad[:, 2, 1] = x1 - x0
    grad /= a2[:, None, None]
    area = 0.5 * np.abs(a2)
    nl = -np.ones(len(V), int)
    nl[nodes] = np.arange(len(nodes))
    rows, cols, vals = [], [], []
    for i in range(3):
        for j in range(3):
            rows.append(nl[tri[:, i]]); cols.append(nl[tri[:, j]])
            vals.append(np.einsum('md,md->m', grad[:, i], grad[:, j]) * area)
    S = sp.coo_matrix((np.concatenate(vals), (np.concatenate(rows),
                                              np.concatenate(cols))),
                      shape=(len(nodes), len(nodes))).tolil()
    Vn = np.zeros(len(nodes))
    fixed = np.zeros(len(nodes), bool)
    fixed[nl[A_nodes]] = True
    fixed[nl[B_nodes]] = True
    Vn[nl[B_nodes]] = 1.0
    S = S.tocsr()
    free = ~fixed
    rhs = -S[free][:, fixed] @ Vn[fixed]
    Vn[free] = spla.spsolve(S[free][:, free].tocsc(), rhs) if free.any() else 0
    gV = np.einsum('mkd,mk->md', grad, Vn[nl[tri]])
    G = np.sum(area * np.sum(gV ** 2, axis=1))
    jn['F'] = 1.0 / G
    jn['gradV'] = gV
    jn['contacts'] = (A_nodes, B_nodes)


# ----------------------------------------------------------------------
# Airbridges as exact rectangles
# ----------------------------------------------------------------------
def _rectify_bridges(bridges, snap_deg=0.5, iou_warn=0.95, verbose=True):
    """Replaces each layer-2 polygon by its closest rectangle.

    The closest rectangle minimises the area of the symmetric difference
    with the polygon, see rectfit.py (moment tensor and minimum-area
    bounding rectangle for the orientation, then each edge placed where
    half of its length lies inside the polygon). A rectangle tilted by at
    most snap_deg degrees with respect to the x or y axis is made exactly
    axis-aligned, its centre and sides being kept.

    Returns (rectangles as shapely polygons, list of fit dictionaries).
    """
    from shapely.geometry import Polygon
    from scipy.spatial import cKDTree
    try:
        from .rectfit import fit_rectangle, rect_overlap_area
    except ImportError:
        from rectfit import fit_rectangle, rect_overlap_area
    rects, fits = [], []
    for bi, b in enumerate(bridges):
        rings = [np.asarray(b.exterior.coords)] + \
                [np.asarray(r.coords) for r in b.interiors]
        f = fit_rectangle(rings, snap_deg=snap_deg)
        f['index'] = bi
        fits.append(f)
        rects.append(Polygon(f['corners']))
        if verbose and f['iou'] < iou_warn:
            print(f"  warning: bridge {bi} at ({f['center'][0]:.1f}, "
                  f"{f['center'][1]:.1f}) is not close to a rectangle, "
                  f"IoU = {f['iou']:.4f}, largest vertex deviation "
                  f"{f['dmax']:.3g} um, check it or use --bridge-shape exact")
        if verbose and f['n_holes']:
            print(f"  warning: bridge {bi} has {f['n_holes']} hole(s), "
                  "filled by the rectangle")
    if len(fits) > 1:
        cen = np.array([f['center'] for f in fits])
        rmax = max(np.hypot(f['length'], f['width']) for f in fits)
        for i, j in sorted(cKDTree(cen).query_pairs(rmax)):
            a = rect_overlap_area(fits[i], fits[j])
            if a > 0 and verbose:
                print(f"  warning: the rectangles of bridges {i} and {j} "
                      f"overlap ({a:.3g} um2), their sheets are meshed "
                      "separately")
    if verbose and fits:
        nv = np.array([f['n_vertices'] for f in fits])
        iou = np.array([f['iou'] for f in fits])
        dm = np.array([f['dmax'] for f in fits])
        tilt = np.array([abs(f['angle'] - 90.0 * np.round(f['angle'] / 90.0))
                         for f in fits])
        nsnap = sum(f['snapped'] for f in fits)
        print(f"  layer 2: {len(fits)} polygon(s) replaced by their closest "
              f"rectangle, {nv.min()} to {nv.max()} vertices, IoU >= "
              f"{iou.min():.6f}, vertex deviation <= {dm.max():.3g} um, "
              f"{nsnap} snapped to the axes (tilt <= {snap_deg:g} deg)")
        tl = tilt > 1e-9
        if tl.any():
            print(f"  layer 2: {int(tl.sum())} rectangle(s) remain tilted "
                  f"with respect to the axes, {tilt[tl].min():.3g} to "
                  f"{tilt[tl].max():.3g} deg (--bridge-snap-angle to align "
                  "them)")
    return rects, fits

def build_model(gds_path, *,
                bridge_shape="rect", bridge_snap_angle=0.5,
                seg_len, thickness=None, lambda_L=None,
                L_square=None, seg_far=None, fine_radius=None, seg_max=None,
                layer_metal=1, layer_bridge=2, layer_source=3,
                layer_ground=4, layer_loop=5, layer_junction=6,
                junction_default=None, junction_file=None, min_seg=None,
                pad_tol=None, bridge_height=3.0, bridge_thickness=None,
                bridge_lambda=None, bridge_L_square=None,
                bridge_feet="ends", pillar_side=30.0, pillar_height=None,
                pillar_lambda=None, pillars=True,
                cell=None, scale=1.0,
                current=1e-3, ground_resistances=None, require_sources=True,
                verbose=True):
    """Construit l'assemblage, les terminaux et les contacts a partir du GDS.

    Le layer 3 peut contenir plusieurs polygones d'injection. Chacun donne
    un jeu de terminaux (`terminal_sets[j]`, injection de `current` dans le
    polygone j, retours par les polygones du layer 4), le calcul etant mene
    pour chaque polygone comme s'il etait seul. Les polygones des layers 3
    et 5 sont nommes par un label GDS place a l'interieur, sinon S0, S1 ...
    et L0, L1 ... (ordre par centroide croissant en x puis y).

    Le layer 1 est decrit soit par (thickness, lambda_L), soit par L_square
    en henry par carre. Les jonctions (layer 6) sont des rectangles ; chacune
    recoit une inductance L_J fournie, par ordre de priorite, par
      1. un label GDS place dans le rectangle (texte "EJ=25GHz", "Ic=30uA",
         "LJ=5nH", "Rn=8kOhm Delta=180ueV", separateurs espace ou virgule),
      2. le fichier JSON junction_file, liste d'objets {"x","y", + une des
         cles EJ_GHz, Ic_uA, LJ_nH, Rn_Ohm/Delta_ueV} rapportes au rectangle
         qui contient (x, y),
      3. junction_default, dictionnaire du meme type sans x, y.

    Maillage gradue : les bords sont discretises a seg_len a moins de
    fine_radius des surfaces du layer 5 et des jonctions, a seg_far au-dela
    (defaut 5*seg_len), et les triangles interieurs sont plafonnes a seg_max
    (defaut 30*seg_len).

    min_seg : tolerance (um) of the Douglas-Peucker simplification of the
    layer-1 contours, applied before noding, the neighbourhood of the
    junctions being kept exact.
    pad_tol : tolerance (um) of the simplification of the layer-3 and
    layer-4 polygons. It is applied to each pad polygon BEFORE its
    intersection with the metal, so that wherever the contact contour
    follows a metal edge it is that edge exactly and no near-tangent pair
    of lines reaches Triangle. None = contours kept as drawn.

    Ponts (layer 2). Le polygone est la vue de dessus du pont. Ses pieds sont
    les composantes connexes de pont ∩ metal retenues par `bridge_feet`
      "ends" : les deux composantes les plus eloignees l'une de l'autre, les
               autres (conducteur central d'une CPW, pistes survolees) sont
               ignorees et le pont les enjambe sans contact ;
      "all"  : toutes les composantes (vias multiples).
    Chaque pied retenu est relie au plan par un pilier vertical de section
    carree `pillar_side` et de hauteur `pillar_height` (defaut bridge_height),
    dont l'inductance propre et les mutuelles pilier-pilier sont retournees
    dans `pillar_L` (matrice n_contacts x n_contacts, courants montants).
    `pillars=False` desactive ce terme. `pillar_lambda` (defaut lambda_L du
    pont) sert a la petite contribution cinetique du pilier.

    Bridge shape. With bridge_shape="rect" (default) every connected
    component of layer 2 is replaced by its closest rectangle, the one
    minimising the area of the symmetric difference (rectfit.py), before
    the feet are computed. Rectangles tilted by at most bridge_snap_angle
    degrees with respect to the axes are made exactly axis-aligned.
    bridge_shape="exact" keeps the polygons as drawn. The fits are
    returned in `bridge_fits` (None in exact mode).

    require_sources : when False (applied-field computation, run.py
    --Bext), layers 3 and 4 may both be empty, terminal_sets is then an
    empty list. A layer-3 polygon still requires at least one layer-4 pad.

    Retourne un dictionnaire avec toutes les pieces du probleme.
    """
    import json
    from shapely.ops import unary_union
    from shapely.geometry import Point, LineString
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla
    seg_far = seg_far or 5.0 * seg_len
    seg_max = seg_max or 30.0 * seg_len
    fine_radius = 200.0 * seg_len / 1.5 if fine_radius is None else fine_radius
    try:
        from .core import sheet_inductance_to_Lambda, josephson_inductance, MU0
    except ImportError:
        from core import sheet_inductance_to_Lambda, josephson_inductance, MU0

    layers = [layer_metal, layer_bridge, layer_source, layer_ground,
              layer_loop, layer_junction]
    geo = read_gds_layers(gds_path, layers, cell=cell, scale=scale)
    labels = read_gds_labels(gds_path, layer_junction, cell=cell, scale=scale)
    src_labels = read_gds_labels(gds_path, layer_source, cell=cell, scale=scale)
    loop_labels = read_gds_labels(gds_path, layer_loop, cell=cell, scale=scale)

    if L_square is not None:
        Lam = sheet_inductance_to_Lambda(L_square)
    elif thickness is not None and lambda_L is not None:
        Lam = effective_penetration_depth(lambda_L, thickness)
    else:
        raise ValueError("give L_square, or thickness and lambda_L")

    import shapely
    metal = geo[layer_metal]
    # arrondi sur grille + reparation : les trous en "trou de serrure"
    # deviennent de vrais anneaux interieurs
    metal = shapely.set_precision(metal, 1e-4).buffer(0)
    metal = unary_union(_as_polygon_list(metal))
    if min_seg:
        # simplification de Douglas-Peucker du metal, hors voisinage des
        # jonctions (patchs de l'ordre du micron) qui reste exact. Les deux
        # morceaux sont decoupes par le meme polygone de protection, donc
        # leur frontiere commune est identique et l'union est propre.
        simp = metal.simplify(min_seg, preserve_topology=True).buffer(0)
        jr = _as_polygon_list(geo[layer_junction])
        if jr:
            protect = unary_union(jr).buffer(seg_len + 5.0 * min_seg)
            simp = unary_union(_as_polygon_list(simp.difference(protect))
                               + _as_polygon_list(metal.intersection(protect)))
        metal = unary_union(_as_polygon_list(simp.buffer(0)))
        if verbose:
            n0 = sum(len(r) for r in _rings(metal))
            print(f"  contours simplified to {min_seg} um, {n0} vertices")
    electrodes = metal
    if metal is None:
        raise ValueError(f"layer {layer_metal} is empty, no metal to mesh")
    src_geo = geo[layer_source]
    gnd_geo = geo[layer_ground]
    brg_geo = geo[layer_bridge]
    loop_geo = geo[layer_loop]

    src_regions = _sort_polygons(_as_polygon_list(src_geo))
    gnd_regions = _sort_polygons(_as_polygon_list(gnd_geo))
    bridges = _as_polygon_list(brg_geo)
    if bridge_shape not in ("rect", "exact"):
        raise ValueError("bridge_shape must be 'rect' or 'exact'")
    bridge_fits = None
    if bridges and bridge_shape == "rect":
        bridges, bridge_fits = _rectify_bridges(
            bridges, snap_deg=bridge_snap_angle, verbose=verbose)
    loops = _sort_polygons(_as_polygon_list(loop_geo))
    if not src_regions and require_sources:
        raise ValueError(f"layer {layer_source} is empty, at least one "
                         "injection polygon (disk, rectangle ...) is required "
                         "unless an applied field is given (--Bext)")
    if not gnd_regions and src_regions:
        raise ValueError(f"layer {layer_ground} is empty")
    source_names = _label_names(src_regions, src_labels, "S")
    loop_names = _label_names(loops, loop_labels, "L")
    if verbose:
        print(f"  {len(src_regions)} injection polygon(s) in layer "
              f"{layer_source}" + (f": {', '.join(source_names)}"
                                   if source_names else ""))
        if loops:
            print(f"  {len(loops)} flux surface(s) in layer {layer_loop}: "
                  f"{', '.join(loop_names)}")

    # effective contact regions (intersection with the metal). The pad
    # polygons are simplified first, the metal edge then bounds the contact
    # exactly where the pad overhangs it.
    def _simplify_pad(p, tol):
        if not tol:
            return p
        q = unary_union(_as_polygon_list(
            p.simplify(tol, preserve_topology=True).buffer(0)))
        return p if q.is_empty else q

    if pad_tol and verbose:
        n_before = sum(len(r) for g in src_regions + gnd_regions
                       for r in _rings(g))
    src_s = [_simplify_pad(s_, pad_tol) for s_ in src_regions]
    gnd_s = [_simplify_pad(g, pad_tol) for g in gnd_regions]
    if pad_tol and verbose:
        n_after = sum(len(r) for g in src_s + gnd_s for r in _rings(g))
        print(f"  layer-{layer_source} and layer-{layer_ground} contours "
              f"simplified to {pad_tol} um, {n_before} -> {n_after} vertices")
    src_c = [s_.intersection(metal) for s_ in src_s]
    for nm, c_ in zip(source_names, src_c):
        if c_.is_empty:
            raise ValueError(f"injection polygon '{nm}' does not overlap the "
                             f"metal of layer {layer_metal}")
    gnd_c = [g.intersection(metal) for g in gnd_s]
    feet = []            # (indice_du_pont, geometrie du pied)
    for bi, b in enumerate(bridges):
        parts = _as_polygon_list(b.intersection(metal))
        keep, drop = _select_feet(parts, bridge_feet)
        if drop and verbose:
            xy = ", ".join(f"({p.centroid.x:.1f}, {p.centroid.y:.1f})"
                           for p in drop)
            print(f"  bridge {bi}: {len(parts)} overlap regions, "
                  f"{len(drop)} flown over without contact at {xy}")
        if len(keep) < 2 and verbose:
            print(f"  warning: bridge {bi} has only {len(keep)} foot/feet")
        for f in keep:
            feet.append((bi, f))

    # ---------------------------------------------------------- jonctions
    # sorted like layers 3 and 5 (centroid x, then y), so that junction k
    # designates the same rectangle from one run to the next
    junction_rects = _sort_polygons(_as_polygon_list(geo[layer_junction]))
    junctions = []
    for k, rect in enumerate(junction_rects):
        # zone active : ce qui depasse des electrodes, sinon le rectangle
        active = rect.difference(electrodes)
        if active.is_empty or active.area < 1e-6 * rect.area:
            active = rect
        active = unary_union(_as_polygon_list(active))
        junctions.append(dict(index=k, geom=active, rect=rect, params=None))

    # parametres : labels, fichier, defaut
    for jn in junctions:
        for (x, y, text) in labels:
            if jn['rect'].buffer(1e-9).contains(Point(x, y)):
                jn['params'] = _parse_junction_text(text)
                break
    if junction_file:
        with open(junction_file) as f:
            entries = json.load(f)
        if isinstance(entries, dict):
            entries = entries.get("junctions", [])
        for e in entries:
            for jn in junctions:
                if jn['params'] is None and \
                        jn['rect'].buffer(1e-9).contains(Point(e['x'], e['y'])):
                    jn['params'] = {k: v for k, v in e.items()
                                    if k not in ('x', 'y')}
    for jn in junctions:
        if jn['params'] is None:
            if junction_default is None:
                b = jn['rect'].bounds
                raise ValueError(
                    f"junction {jn['index']} at x [{b[0]:.3f}, {b[2]:.3f}] "
                    f"y [{b[1]:.3f}, {b[3]:.3f}] um has no parameters (no "
                    f"layer-{layer_junction} TEXT label inside it, no "
                    "junction_file entry, no --EJ/--Ic/--LJ default)")
            jn['params'] = dict(junction_default)
        LJ, Ic, EJ = josephson_inductance(**_norm_params(jn['params']))
        jn.update(LJ=LJ, Ic=Ic, EJ=EJ, LJ0=LJ)

    if junctions:
        metal = unary_union([metal] + [jn['geom'] for jn in junctions])

    extra = []
    for s_ in src_c:
        extra += _rings(s_)
    for g_ in gnd_c:
        extra += _rings(g_)
    for _, f in feet:
        extra += _rings(f)
    refine = []
    for jn in junctions:
        extra += _rings(jn['geom'])
        c = jn['geom'].representative_point()
        refine.append((c.x, c.y, min(seg_len, np.sqrt(jn['geom'].area) / 4)))

    # zone d'interet : layer 5 et jonctions, bords fins a moins de fine_radius
    roi_parts = loops + [jn['geom'] for jn in junctions]
    if roi_parts:
        roi = unary_union(roi_parts).buffer(fine_radius)
        def spacing(p):
            m = contains_mask(roi, p)
            return np.where(m, seg_len, seg_far)
    else:
        spacing = lambda p: np.full(len(p), seg_len)

    if verbose:
        print(f"  meshing the metal (layer {layer_metal}): edges at {seg_len} um "
              f"near the region of interest, {seg_far} um elsewhere, interior "
              f"up to {seg_max} um ...")
    Vm, Tm = triangulate(metal, seg_len, extra_rings=extra, refine=refine,
                         spacing=spacing, seg_max=seg_max)
    Lam_tri = np.full(len(Tm), Lam)
    cent = Vm[Tm].mean(axis=1)
    if verbose:
        h = np.sqrt(0.5 * np.abs(np.cross(Vm[Tm[:, 1]] - Vm[Tm[:, 0]],
                                          Vm[Tm[:, 2]] - Vm[Tm[:, 0]])))
        print(f"  {len(Tm)} triangles, sizes {h.min():.3g} to {h.max():.3g} um, "
              f"median {np.median(h):.3g} um")

    # ----------------------------------------- facteur geometrique des jonctions
    for jn in junctions:
        m = contains_mask(jn['geom'].buffer(1e-9), cent)
        jn['tri'] = np.where(m)[0]
        _junction_geometry_factor(jn, Vm, Tm, verbose)
        jn['Lambda'] = jn['LJ'] / (MU0 * jn['F'])
        Lam_tri[m] = jn['Lambda']
        if verbose:
            cx, cy = jn['rect'].centroid.coords[0]
            print(f"  junction {jn['index']} at ({cx:.1f}, {cy:.1f}): "
                  f"L_J = {jn['LJ']*1e9:.4g} nH, "
                  f"I_c = {jn['Ic']*1e6:.4g} uA, "
                  f"E_J/h = {jn['EJ']/6.62607015e-34/1e9:.4g} GHz, "
                  f"F = L/L_square = {jn['F']:.4g}, "
                  f"Lambda_J = {jn['Lambda']:.3g} um")
    sheets = [Sheet(Vm, Tm, z=0.0, Lambda=Lam_tri, name="metal")]

    # ponts
    if bridge_L_square is not None:
        Lam_b = sheet_inductance_to_Lambda(bridge_L_square)
    elif bridge_thickness is not None or bridge_lambda is not None:
        bt = thickness if bridge_thickness is None else bridge_thickness
        bl = lambda_L if bridge_lambda is None else bridge_lambda
        Lam_b = effective_penetration_depth(bl, bt)
    else:
        Lam_b = Lam
    bridge_sheet_index = {}
    for bi, b in enumerate(bridges):
        rings_b = [f for (j, f) in feet if j == bi]
        if verbose:
            print(f"  meshing bridge {bi} ({len(rings_b)} foot/feet) ...")
        er = []
        for f in rings_b:
            er += _rings(f)
        Vb, Tb = triangulate(b, seg_len, extra_rings=er)
        bridge_sheet_index[bi] = len(sheets)
        sheets.append(Sheet(Vb, Tb, z=bridge_height, Lambda=Lam_b,
                            name=f"bridge{bi}"))

    asm = Assembly(sheets)
    off = asm.node_offset

    def nodes_in(geom, sheet_idx):
        s = asm.sheets[sheet_idx]
        m = contains_mask(geom.buffer(seg_len * 1e-6), s.points)
        return np.where(m)[0] + off[sheet_idx]

    # contacts verticaux
    contacts = []
    pillar_centers = []
    for bi, f in feet:
        si = bridge_sheet_index[bi]
        na, nb = nodes_in(f, 0), nodes_in(f, si)
        if len(na) == 0 or len(nb) == 0:
            print(f"  warning: foot of bridge {bi} not resolved by the mesh")
            continue
        c = Contact(na, nb, f"bridge{bi}")
        c.center = np.array(f.centroid.coords[0])
        c.bridge = bi
        contacts.append(c)
        pillar_centers.append(c.center)

    # piliers : parallelepipedes de base carree pillar_side, hauteur
    # pillar_height, axe au centroide du pied
    if pillars and contacts:
        ph = bridge_height if pillar_height is None else float(pillar_height)
        if pillar_lambda is None:
            pillar_lambda = lambda_L if bridge_lambda is None else bridge_lambda
        pillar_L = pillar_inductance_matrix(np.array(pillar_centers),
                                            pillar_side, ph, pillar_lambda)
        if verbose:
            print(f"  pillars: {len(contacts)}, side {pillar_side:g} um, "
                  f"height {ph:g} um, self inductance = "
                  f"{pillar_L[0, 0]*1e12:.2f} pH, max mutual = "
                  f"{(pillar_L[~np.eye(len(contacts), dtype=bool)].max()*1e12 if len(contacts) > 1 else 0):.2f} pH")
            if abs(ph - bridge_height) > 1e-9:
                print(f"  warning: pillar height {ph:g} um differs from the "
                      f"bridge altitude {bridge_height:g} um")
    else:
        pillar_L = np.zeros((len(contacts), len(contacts)))

    # terminaux, construits apres les contacts car les ponts participent a
    # la connexite DC
    if ground_resistances is None:
        ground_resistances = [0.01] * len(gnd_c)
    ground_resistances = np.asarray(ground_resistances, dtype=float)
    if len(ground_resistances) != len(gnd_c):
        raise ValueError("number of resistances differs from the number of "
                         f"polygons in layer {layer_ground}")
    gcond = 1.0 / ground_resistances
    # parts de reference si tous les plots etaient relies a la source, les
    # parts effectives par source sont dans ground_shares
    gshare = gcond / gcond.sum() if len(gcond) else np.zeros(0)
    gnd_nodes = [nodes_in(g, 0) for g in gnd_c]
    for k, nd in enumerate(gnd_nodes):
        if len(nd) == 0 and verbose:
            print(f"  warning: ground polygon gnd{k} contains no mesh node "
                  f"of layer {layer_metal}, it carries no current")
    sources, src_nodes = [], []
    for nm, c_, reg in zip(source_names, src_c, src_regions):
        nodes = nodes_in(c_, 0)
        if len(nodes) == 0:
            raise ValueError(f"injection polygon '{nm}' contains no mesh node, "
                             "refine the mesh (--seg) or enlarge the polygon")
        sources.append(dict(name=nm, geom=c_, rect=reg, nodes=nodes,
                            centroid=np.array(reg.centroid.coords[0]),
                            area=reg.area))
        src_nodes.append(nodes)
    terminal_sets, gshares = build_terminal_sets(
        asm, src_nodes, gnd_nodes, current, conductances=gcond,
        contacts=contacts, source_names=source_names, verbose=verbose)
    terminals = terminal_sets[0] if terminal_sets else []

    rings = []
    for nm, p in zip(loop_names, loops):
        rings.append(dict(name=nm,
                          exterior=np.asarray(p.exterior.coords)[:-1],
                          interiors=[np.asarray(r.coords)[:-1]
                                     for r in p.interiors],
                          area=p.area,
                          centroid=np.array(p.centroid.coords[0])))

    # contours a superposer sur les figures (listes de tableaux (N,2))
    overlays = dict(
        source=[r.tolist() for s_ in src_c for r in _rings(s_)],
        ground=[r.tolist() for g in gnd_c for r in _rings(g)],
        bridge=[np.asarray(b.exterior.coords)[:-1].tolist() for b in bridges],
        loop=[r['exterior'].tolist() for r in rings],
        junction=[r.tolist() for jn in junctions for r in _rings(jn['geom'])],
        metal=[r.tolist() for r in _rings(metal)],
    )

    return dict(assembly=asm,
                bridge_fits=bridge_fits, bridge_shape=bridge_shape,
                terminals=terminals, terminal_sets=terminal_sets,
                sources=sources, source_names=source_names,
                loop_names=loop_names, contacts=contacts,
                pillar_L=pillar_L, current=current,
                rings=rings, metal=metal, Lambda=Lam, ground_share=gshare,
                ground_shares=gshares,
                src_contacts=src_c, gnd_contacts=gnd_c, bridges=bridges,
                junctions=junctions, overlays=overlays)
