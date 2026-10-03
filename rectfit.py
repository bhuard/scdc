"""
rectfit.py -- Closest rectangle to a polygon, pure numpy.

Used by geometry.build_model to replace every layer-2 polygon (airbridge)
by an exact rectangle. A polygon is given as a list of rings, exterior
first and holes after, each an (N, 2) array, open or closed, in any
orientation.

Definition
----------
The closest rectangle R to a polygon P is the one minimising the area of
the symmetric difference

    |P xor R| = |P| + |R| - 2 |P n R| .

Method
------
1. Orientation, two candidates.
   a. Principal axes of the second-moment tensor of P, closed form, O(n).
      They coincide with the symmetry axes of any shape that has two
      orthogonal mirror axes (rectangle with rounded or chamfered corners,
      redundant collinear vertices, ...).
   b. Orientation of the minimum-area bounding rectangle (rotating
      calipers on the convex hull, O(n log n)), which remains defined when
      the tensor is isotropic (square-like shapes).
2. Initial rectangle in each frame (u, v), centred on the centroid, sides
   sqrt(12 S_uu) and sqrt(12 S_vv), S being the covariance of P per unit
   area. Exact for a rectangle whatever the number of vertices.
3. Refinement of the four edges at fixed orientation. With
   R = [u0, u1] x [v0, v1] and l(u) the length of the segment
   {u} x [v0, v1] lying inside P,

       d|P xor R| / du1 = (v1 - v0) - 2 l(u1) ,

   so that at the optimum each edge has exactly half of its length inside
   P. Each edge is moved to the global minimum of the 1D restriction of
   |P xor R| (cumulative integral of (v1 - v0) - 2 l sampled on a grid,
   then nested sampling down to `tol`), the edges being updated in turn
   until they no longer move. For a rectangle whose corners are rounded or
   chamfered with a size up to a quarter of the shorter side, the optimum
   is the unrounded rectangle exactly, whereas the moment-matched
   rectangle is slightly smaller.
4. The candidate with the smallest |P xor R| is kept.

Optionally the orientation is snapped to the x and y axes when the tilt is
below `snap_deg`, the centre and the sides being kept.

    python rectfit.py        # self-test
"""

from __future__ import annotations

import numpy as np


# ----------------------------------------------------------------------
def _signed_area(r):
    x, y = r[:, 0], r[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def _clean_rings(rings):
    """Open rings, float64, exterior counter-clockwise, holes clockwise.
    Consecutive duplicate vertices are removed."""
    out = []
    for k, r in enumerate(rings):
        r = np.asarray(r, dtype=float)
        if len(r) > 1 and np.allclose(r[0], r[-1]):
            r = r[:-1]
        if len(r) > 1:
            d = np.any(r != np.roll(r, -1, axis=0), axis=1)
            r = r[d]
        if len(r) < 3:
            if k == 0:
                raise ValueError("exterior ring with fewer than 3 vertices")
            continue
        a = _signed_area(r)
        if (k == 0) != (a > 0):
            r = r[::-1]
        out.append(r)
    return out


def polygon_moments(rings):
    """Area, centroid and covariance per unit area of a polygon with holes.

    Green's theorem on each ring, with c_i = x_i y_{i+1} - x_{i+1} y_i,

        A    = (1/2)  sum c_i
        S_x  = (1/6)  sum (x_i + x_{i+1}) c_i
        I_xx = (1/12) sum (x_i^2 + x_i x_{i+1} + x_{i+1}^2) c_i
        I_xy = (1/24) sum (x_i y_{i+1} + 2 x_i y_i + 2 x_{i+1} y_{i+1}
                           + x_{i+1} y_i) c_i

    computed relative to the mean exterior vertex to avoid cancellation on
    chips several millimetres across. Rings must be oriented as returned
    by _clean_rings.
    """
    o = rings[0].mean(axis=0)
    A = Sx = Sy = Ixx = Iyy = Ixy = 0.0
    for r in rings:
        x, y = (r - o).T
        x1, y1 = np.roll(x, -1), np.roll(y, -1)
        c = x * y1 - x1 * y
        A += c.sum() / 2.0
        Sx += ((x + x1) * c).sum() / 6.0
        Sy += ((y + y1) * c).sum() / 6.0
        Ixx += ((x * x + x * x1 + x1 * x1) * c).sum() / 12.0
        Iyy += ((y * y + y * y1 + y1 * y1) * c).sum() / 12.0
        Ixy += ((x * y1 + 2 * x * y + 2 * x1 * y1 + x1 * y) * c).sum() / 24.0
    if A <= 0:
        raise ValueError("polygon of zero or negative area")
    cx, cy = Sx / A, Sy / A
    S = np.array([[Ixx / A - cx * cx, Ixy / A - cx * cy],
                  [Ixy / A - cx * cy, Iyy / A - cy * cy]])
    return A, o + np.array([cx, cy]), S


def _min_area_rect_angle(pts):
    """Angle (rad) of one side of the minimum-area bounding rectangle."""
    from scipy.spatial import ConvexHull
    try:
        h = pts[ConvexHull(pts).vertices]
    except Exception:
        return None
    e = np.roll(h, -1, axis=0) - h
    phi = np.unique(np.mod(np.arctan2(e[:, 1], e[:, 0]), np.pi / 2))
    c, s = np.cos(phi)[:, None], np.sin(phi)[:, None]
    hc = h - h.mean(axis=0)
    u = hc[:, 0] * c + hc[:, 1] * s
    v = -hc[:, 0] * s + hc[:, 1] * c
    area = np.ptp(u, axis=1) * np.ptp(v, axis=1)
    return float(phi[np.argmin(area)])


# ----------------------------------------------------------------------
def _frame(theta):
    """Unit vectors (e1, e2) of the frame rotated by theta. Exact for
    multiples of pi/2, so that a snapped rectangle is exactly axis-aligned."""
    k = theta / (np.pi / 2)
    if abs(k - round(k)) < 1e-15:
        e1 = [(1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.0, -1.0)][int(round(k)) % 4]
        e1 = np.array(e1)
    else:
        e1 = np.array([np.cos(theta), np.sin(theta)])
    return e1, np.array([-e1[1], e1[0]])


def _edges(rings_uv):
    """All ring edges (a -> b) stacked, as (ua, va, ub, vb)."""
    a = np.vstack(rings_uv)
    b = np.vstack([np.roll(r, -1, axis=0) for r in rings_uv])
    return a[:, 0], a[:, 1], b[:, 0], b[:, 1]


def _inside_length(E, s, lo, hi):
    """Length of the segments {s_k} x [lo, hi] lying inside the polygon
    whose edges are E = (ua, va, ub, vb), even-odd rule (holes included).
    Half-open crossing test, so vertices lying on a line count once."""
    ua, va, ub, vb = E
    sel = (np.maximum(ua, ub) >= s.min()) & (np.minimum(ua, ub) <= s.max())
    ua, va, ub, vb = ua[sel], va[sel], ub[sel], vb[sel]
    if len(ua) == 0:
        return np.zeros(len(s))
    S = s[:, None]
    cross = (ua <= S) != (ub <= S)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (S - ua) / (ub - ua)
        v = np.where(cross, va + t * (vb - va), np.inf)
    v.sort(axis=1)
    if v.shape[1] % 2:
        v = np.hstack([v, np.full((len(s), 1), np.inf)])
    seg = np.minimum(v[:, 1::2], hi) - np.maximum(v[:, 0::2], lo)
    return np.sum(np.clip(seg, 0.0, None), axis=1)


def _line_min(E, a, b, lo, hi, side, tol, n=65):
    """Position t in [a, b] of an edge of the box minimising |P xor R|.

    side = +1 for the upper edge (u1), -1 for the lower edge (u0). The 1D
    objective is the integral of side * ((hi - lo) - 2 l(t)), minimised on
    a grid of n points, then on the bracket around the minimum, until the
    bracket is shorter than tol."""
    h = hi - lo
    while b - a > tol:
        s = np.linspace(a, b, n)
        g = side * (h - 2.0 * _inside_length(E, s, lo, hi))
        F = np.concatenate([[0.0], np.cumsum(0.5 * (g[1:] + g[:-1]) * np.diff(s))])
        k = int(np.argmin(F))
        a, b = s[max(k - 1, 0)], s[min(k + 1, n - 1)]
    return 0.5 * (a + b)


def _refine_box(rings_uv, box, tol, maxsweep=40):
    """Coordinate descent on the four edges of the box (u0, u1, v0, v1)."""
    Eu = _edges(rings_uv)
    Ev = (Eu[1], Eu[0], Eu[3], Eu[2])            # roles of u and v swapped
    allp = np.vstack(rings_uv)
    umin, vmin = allp.min(axis=0)
    umax, vmax = allp.max(axis=0)
    u0, u1, v0, v1 = box
    for _ in range(maxsweep):
        old = np.array([u0, u1, v0, v1])
        um = 0.5 * (u0 + u1)
        u1 = _line_min(Eu, um, umax, v0, v1, +1, tol)
        u0 = _line_min(Eu, umin, um, v0, v1, -1, tol)
        vm = 0.5 * (v0 + v1)
        v1 = _line_min(Ev, vm, vmax, u0, u1, +1, tol)
        v0 = _line_min(Ev, vmin, vm, u0, u1, -1, tol)
        if np.max(np.abs(np.array([u0, u1, v0, v1]) - old)) < 2 * tol:
            break
    return u0, u1, v0, v1


# ----------------------------------------------------------------------
def _clip_box(P, box):
    """Sutherland-Hodgman clipping of a ring (any shape) by an axis box.
    The signed area of the result is the signed area of P n box."""
    u0, u1, v0, v1 = box
    for d_fn in (lambda Q: Q[:, 0] - u0, lambda Q: u1 - Q[:, 0],
                 lambda Q: Q[:, 1] - v0, lambda Q: v1 - Q[:, 1]):
        if len(P) == 0:
            return P
        d = d_fn(P)
        dn = np.roll(d, -1)
        Pn = np.roll(P, -1, axis=0)
        ins = d >= 0
        cross = ins != (dn >= 0)
        with np.errstate(divide="ignore", invalid="ignore"):
            t = np.where(cross, d / (d - dn), 0.0)
        X = P + t[:, None] * (Pn - P)
        P = np.stack([P, X], axis=1)[np.stack([ins, cross], axis=1)]
    return P


def _score(rings_uv, box, area_P):
    """(|P n R|, |P xor R|, IoU, max distance of the vertices of P to the
    boundary of R)."""
    u0, u1, v0, v1 = box
    inter = sum(_signed_area(_clip_box(r, box)) if len(r) else 0.0
                for r in rings_uv)
    inter = max(inter, 0.0)
    area_R = (u1 - u0) * (v1 - v0)
    sym = area_P + area_R - 2.0 * inter
    iou = inter / (area_P + area_R - inter)
    p = np.vstack(rings_uv)
    u, v = p[:, 0], p[:, 1]
    inside = (u >= u0) & (u <= u1) & (v >= v0) & (v <= v1)
    din = np.minimum.reduce([u - u0, u1 - u, v - v0, v1 - v])
    dout = np.hypot(np.maximum.reduce([u0 - u, np.zeros_like(u), u - u1]),
                    np.maximum.reduce([v0 - v, np.zeros_like(v), v - v1]))
    dmax = float(np.max(np.where(inside, din, dout)))
    return inter, sym, iou, dmax


def _to_frame(rings, c, theta):
    e1, e2 = _frame(theta)
    return [np.column_stack([(r - c) @ e1, (r - c) @ e2]) for r in rings]


# ----------------------------------------------------------------------
def fit_rectangle(rings, refine=True, snap_deg=0.0, tol=None):
    """Closest rectangle to a polygon in the sense of |P xor R|.

    rings    : list of (N, 2) arrays, exterior first, holes after.
    refine   : False keeps the moment-matched rectangle (steps 1-2 only).
    snap_deg : if the tilt of the rectangle to the x or y axis is at most
               snap_deg degrees, it is rotated about its centre to be
               exactly axis-aligned.
    tol      : tolerance on the edge positions (um), default 1e-10 times
               the size of the polygon.

    Returns a dict with
        center (2,), angle (deg, long side, in (-90, 90]), length >= width,
        corners (4, 2) counter-clockwise, area_polygon, sym_diff (area of
        P xor R), iou, dmax (largest distance of a polygon vertex to the
        rectangle boundary), n_vertices, n_holes, snapped (bool),
        tilt_before_snap (deg).
    """
    R = _clean_rings(rings)
    A, c, S = polygon_moments(R)
    allp = np.vstack(R)
    size = float(np.max(np.ptp(allp, axis=0)))
    tol = 1e-10 * size if tol is None else float(tol)

    lam, vec = np.linalg.eigh(S)                   # ascending
    thetas = [float(np.arctan2(vec[1, 1], vec[0, 1]))]
    t_mabr = _min_area_rect_angle(R[0])
    if t_mabr is not None:
        # same frame modulo pi/2 -> one candidate only
        d = (t_mabr - thetas[0]) % (np.pi / 2)
        if min(d, np.pi / 2 - d) > 1e-12:
            thetas.append(t_mabr)

    best = None
    for th in thetas:
        Ruv = _to_frame(R, c, th)
        Suv = np.array([[np.cos(th), np.sin(th)], [-np.sin(th), np.cos(th)]])
        Cuv = Suv @ S @ Suv.T
        hu = 0.5 * np.sqrt(12.0 * max(Cuv[0, 0], 0.0))
        hv = 0.5 * np.sqrt(12.0 * max(Cuv[1, 1], 0.0))
        box = (-hu, hu, -hv, hv)
        sc = _score(Ruv, box, A)
        # an exact rectangle is already recovered by the moments
        if refine and sc[1] > 1e-12 * A:
            box = _refine_box(Ruv, box, tol)
            sc = _score(Ruv, box, A)
        if best is None or sc[1] < best[3][1] - 1e-12 * A:
            best = (th, box, Ruv, sc)
    th, (u0, u1, v0, v1), Ruv, sc = best

    # centre and long axis
    e1, e2 = _frame(th)
    center = c + 0.5 * (u0 + u1) * e1 + 0.5 * (v0 + v1) * e2
    lu, lv = u1 - u0, v1 - v0
    if lv > lu:
        th += np.pi / 2
        lu, lv = lv, lu
    th = (th + np.pi / 2) % np.pi - np.pi / 2       # [-90, 90) deg
    if th <= -np.pi / 2 + 1e-15:
        th += np.pi

    tilt = np.degrees(th) - 90.0 * np.round(np.degrees(th) / 90.0)
    snapped = False
    if snap_deg and 0.0 < abs(tilt) <= snap_deg:
        th = np.radians(90.0 * np.round(np.degrees(th) / 90.0))
        snapped = True
    box = (-lu / 2, lu / 2, -lv / 2, lv / 2)
    Ruv = _to_frame(R, center, th)
    inter, sym, iou, dmax = _score(Ruv, box, A)
    e1, e2 = _frame(th)
    corners = np.array([center + a * e1 + b * e2 for a, b in
                        [(-lu / 2, -lv / 2), (lu / 2, -lv / 2),
                         (lu / 2, lv / 2), (-lu / 2, lv / 2)]])
    return dict(center=center, angle=float(np.degrees(th)), length=float(lu),
                width=float(lv), corners=corners, area_polygon=float(A),
                sym_diff=float(sym), iou=float(iou), dmax=dmax,
                n_vertices=int(sum(len(r) for r in R)),
                n_holes=len(R) - 1, snapped=snapped,
                tilt_before_snap=float(tilt))


def rect_overlap_area(fa, fb):
    """Area of the intersection of two fitted rectangles."""
    d = np.linalg.norm(fa['center'] - fb['center'])
    if d > 0.5 * (np.hypot(fa['length'], fa['width'])
                  + np.hypot(fb['length'], fb['width'])):
        return 0.0
    th = np.radians(fa['angle'])
    Q = _to_frame([fb['corners']], fa['center'], th)[0]
    box = (-fa['length'] / 2, fa['length'] / 2,
           -fa['width'] / 2, fa['width'] / 2)
    return max(_signed_area(_clip_box(Q, box)), 0.0)


# ======================================================================
if __name__ == "__main__":
    import time
    rng = np.random.default_rng(3)

    def rect_ring(cx, cy, L, W, ang_deg, n_per_side=1):
        t = np.radians(ang_deg)
        e1 = np.array([np.cos(t), np.sin(t)]); e2 = np.array([-e1[1], e1[0]])
        cs = [(-L/2, -W/2), (L/2, -W/2), (L/2, W/2), (-L/2, W/2)]
        pts = []
        for k in range(4):
            a, b = np.array(cs[k]), np.array(cs[(k + 1) % 4])
            for s in np.arange(n_per_side) / n_per_side:
                p = a + s * (b - a)
                pts.append([cx, cy] + p[0] * e1 + p[1] * e2)
        return np.array(pts)

    def rounded_ring(cx, cy, L, W, r, ang_deg, n_arc=24, chamfer=False):
        t = np.radians(ang_deg)
        e1 = np.array([np.cos(t), np.sin(t)]); e2 = np.array([-e1[1], e1[0]])
        pts = []
        for (sx, sy, a0) in [(1, -1, -90), (1, 1, 0), (-1, 1, 90), (-1, -1, 180)]:
            ccx, ccy = sx * (L/2 - r), sy * (W/2 - r)
            m = 2 if chamfer else n_arc
            for a in np.radians(np.linspace(a0, a0 + 90, m)):
                q = (ccx + r * np.cos(a), ccy + r * np.sin(a))
                pts.append([cx, cy] + q[0] * e1 + q[1] * e2)
        return np.array(pts)

    def check(name, rings, L, W, ang, cx, cy, tol_len, tol_ang=1e-6, **kw):
        t0 = time.time()
        f = fit_rectangle(rings, **kw)
        dt = time.time() - t0
        da = (f['angle'] - ang + 90) % 180 - 90
        err = max(abs(f['length'] - L), abs(f['width'] - W),
                  np.hypot(*(f['center'] - [cx, cy])))
        ok = err < tol_len and abs(da) < tol_ang
        print(f"  {name:38s} n={f['n_vertices']:5d}  L={f['length']:.6f}  "
              f"W={f['width']:.6f}  angle={f['angle']:+.6f}  IoU={f['iou']:.6f}"
              f"  err={err:.1e}  {dt*1e3:.1f} ms  {'ok' if ok else 'FAILED'}")
        return ok, f

    print("rectfit self-test")
    fit_rectangle([rect_ring(0, 0, 2, 1, 0)])          # warm-up (scipy import)
    res = []
    r = rect_ring(1994.81, 8100.27, 200.0, 25.0, -81.1331)
    res.append(check("exact rectangle, 4 vertices", [r], 200, 25, -81.1331,
                     1994.81, 8100.27, 1e-6)[0])
    r = rect_ring(2442.31, 9098.86, 144.5, 25.0, 80.0983, n_per_side=500)
    res.append(check("rectangle, 2000 collinear vertices", [r], 144.5, 25,
                     80.0983, 2442.31, 9098.86, 1e-6)[0])
    r = rect_ring(500.0, 300.0, 200.0, 25.0, 12.0, n_per_side=200)
    r += rng.uniform(-5e-4, 5e-4, r.shape)
    res.append(check("rectangle + 0.5 nm jitter, 800 vertices", [r], 200,
                     25, 12.0, 500, 300, 2e-3, tol_ang=1e-3)[0])
    r = rounded_ring(100.0, 50.0, 200.0, 25.0, 5.0, 33.0)
    res.append(check("rounded corners r = 5 (W/5)", [r], 200, 25, 33.0,
                     100, 50, 1e-5)[0])
    r = rounded_ring(100.0, 50.0, 200.0, 25.0, 6.0, -20.0, chamfer=True)
    res.append(check("chamfered corners c = 6", [r], 200, 25, -20.0,
                     100, 50, 1e-5)[0])
    fm = fit_rectangle([rounded_ring(0, 0, 200.0, 25.0, 5.0, 0.0)],
                       refine=False)
    print(f"  (same rounded shape, moments only: L={fm['length']:.4f} "
          f"W={fm['width']:.4f}, IoU={fm['iou']:.6f})")
    r = rect_ring(10.0, 20.0, 50.0, 50.0, 30.0, n_per_side=7)
    r += rng.uniform(-1e-4, 1e-4, r.shape)
    res.append(check("square tilted 30 deg (isotropic tensor)", [r], 50, 50,
                     30.0, 10, 20, 5e-4, tol_ang=1e-3)[0])
    outer = rect_ring(0, 0, 200.0, 25.0, 45.0)
    hole = rect_ring(0, 0, 20.0, 5.0, 45.0)
    ok, f = check("rectangle with a hole", [outer, hole], 200, 25, 45.0,
                  0, 0, 1e-5)
    res.append(ok and f['n_holes'] == 1)
    r = rect_ring(0.0, 0.0, 200.0, 25.0, 0.4, n_per_side=3)
    ok, f = check("tilt 0.4 deg, snap_deg = 0.5", [r], 200, 25, 0.0, 0, 0,
                  1e-6, snap_deg=0.5)
    res.append(ok and f['snapped'] and np.all(np.diff(f['corners'][[0, 1], 1]) == 0))
    print(f"    snapped, tilt before snap {f['tilt_before_snap']:.4f} deg, "
          f"IoU after snap {f['iou']:.6f}, corners y "
          f"{f['corners'][0, 1]:.6f} {f['corners'][1, 1]:.6f}")
    # dog bone, not a rectangle, IoU must flag it
    db = np.array([[-100, -12.5], [-80, -12.5], [-80, -5], [80, -5],
                   [80, -12.5], [100, -12.5], [100, 12.5], [80, 12.5],
                   [80, 5], [-80, 5], [-80, 12.5], [-100, 12.5]], float)
    f = fit_rectangle([db])
    print(f"  dog bone 200 x 25, span 10: L={f['length']:.3f} "
          f"W={f['width']:.3f} IoU={f['iou']:.4f} dmax={f['dmax']:.3f}")
    res.append(f['iou'] < 0.9)
    fa = fit_rectangle([rect_ring(0, 0, 10, 2, 0)])
    fb = fit_rectangle([rect_ring(4, 0, 10, 2, 90)])
    ov = rect_overlap_area(fa, fb)
    print(f"  overlap of two crossed 10 x 2 rectangles: {ov:.6f} (exact 4)")
    res.append(abs(ov - 4.0) < 1e-6)
    print("all tests passed" if all(res) else f"SOME TESTS FAILED {res}")
