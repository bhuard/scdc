"""
viewer.py -- Visualisation des courants, une carte par polygone du layer 3.

    python -m scdc.viewer result_solution.npz              # interactive window
    python -m scdc.viewer result_solution.npz --png out.png # all sources
    python -m scdc.viewer result_solution.npz --png out.png --map 1

Interactive window, keys
    i   next current map (next injection polygon of layer 3)
    I   previous current map
    l   toggle linear / logarithmic colour scale of |K|
    S   recompute the streamlines on the visible area (lower-case s is
        the matplotlib shortcut for saving the figure)
    q   quiver (arrows) on the visible area
    c   clear streamlines and arrows
    mouse wheel or trackpad   zoom centred on the cursor (factor 1.25 per
                              step, progressive on a trackpad), matplotlib
                              tools for panning
A left click prints K, |K| and J = K/d at the clicked point.

The colour scale is common to all maps of the same group, the maps of the
injection polygons (same injected current in each source) forming one
group, so that they can be compared when cycling with i. The response to
the applied field B_ext (run.py --Bext), when present, is the last map and
has its own colour scale.
"""

from __future__ import annotations

import json
import sys
import numpy as np

OVERLAY_STYLE = dict(
    source=dict(color="#00e5ff", lw=1.6, ls="-"),
    ground=dict(color="#7CFC00", lw=1.6, ls="-"),
    bridge=dict(color="#ff9900", lw=1.2, ls="--"),
    loop=dict(color="#ff4fd8", lw=1.4, ls="-"),
    metal=dict(color="#888888", lw=0.4, ls="-"),
    junction=dict(color="#ffffff", lw=1.2, ls=":"),
)


# ----------------------------------------------------------------------
def nodal_from_tri(points, triangles, area, K):
    acc = np.zeros((len(points), 2))
    w = np.zeros(len(points))
    for c in range(3):
        np.add.at(acc, triangles[:, c], K * area[:, None])
        np.add.at(w, triangles[:, c], area)
    return acc / np.maximum(w, 1e-30)[:, None]


class CurrentView:
    """Map of |K| with streamlines for one sheet, holding one current map
    per injection polygon.

    K : (n_tri, 2) for a single map, or (n_maps, n_tri, 2).
    names : one label per map, shown in the title.
    captions : one title per map, default "injection in <name>".
    groups : one integer per map, the maps of a group share the colour
             scale, default a single group.
    """

    KEYS = ("i", "I", "l", "S", "q", "c")

    def __init__(self, points, triangles, K, area, overlays=None,
                 thickness=None, title=None, log_color=False, names=None,
                 captions=None, groups=None):
        import matplotlib.pyplot as plt
        import matplotlib.tri as mtri
        self.plt = plt
        # keys used here must not trigger matplotlib's own shortcuts
        # (q closes the figure, l toggles the log y-axis, c is "back")
        for key, val in list(plt.rcParams.items()):
            if key.startswith("keymap.") and isinstance(val, list):
                plt.rcParams[key] = [k for k in val if k not in self.KEYS]
        K = np.asarray(K, dtype=float)
        if K.ndim == 2:
            K = K[None]
        self.Ks = K                                   # (n_maps, n_tri, 2)
        self.n_maps = len(K)
        self.names = list(names) if names is not None else \
            [f"source {k}" for k in range(self.n_maps)]
        if len(self.names) != self.n_maps:
            raise ValueError("names must have one entry per current map")
        self.captions = list(captions) if captions is not None else \
            [f"injection in {nm}" for nm in self.names]
        self.groups = np.zeros(self.n_maps, dtype=int) if groups is None \
            else np.asarray(groups, dtype=int)
        if len(self.captions) != self.n_maps or len(self.groups) != self.n_maps:
            raise ValueError("captions and groups need one entry per map")
        self.pts, self.tri, self.area = points, triangles, area
        self.d = thickness
        self.base_title = title
        self.T = mtri.Triangulation(points[:, 0], points[:, 1], triangles)
        self.overlays = overlays or {}
        self.log = log_color
        self.artists = []
        self.imap = 0

        # colour scale common to the maps of each group
        mags = np.linalg.norm(self.Ks, axis=2)
        self.scales = {}
        for g in np.unique(self.groups):
            m = mags[self.groups == g]
            vmax = float(np.percentile(m, 99.7))
            vmax = vmax if vmax > 0 else max(float(m.max()), 1e-300)
            vmin = max(float(np.percentile(m, 0.5)), 1e-5 * vmax)
            self.scales[int(g)] = (vmin, vmax)

        x0, x1 = points[:, 0].min(), points[:, 0].max()
        y0, y1 = points[:, 1].min(), points[:, 1].max()
        asp = (y1 - y0) / max(x1 - x0, 1e-9)
        self.fig, self.ax = plt.subplots(figsize=(11, max(4, 10 * asp + 1)))
        self.mesh = None
        self.cb = None
        self._load_map(0)
        self._draw_background()
        for kind, rings in self.overlays.items():
            st = OVERLAY_STYLE.get(kind, dict(color="w", lw=1))
            for r in rings:
                r = np.asarray(r)
                if len(r) < 2:
                    continue
                p = np.vstack([r, r[:1]])
                self.ax.plot(p[:, 0], p[:, 1], zorder=6, **st)
        self.ax.set_aspect("equal")
        self.ax.set_xlim(x0, x1)
        self.ax.set_ylim(y0, y1)
        self.ax.set_xlabel(r"x ($\mu$m)")
        self.ax.set_ylabel(r"y ($\mu$m)")
        self._set_title()
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        self.fig.canvas.mpl_connect("button_press_event", self._on_click)
        self.fig.canvas.mpl_connect("scroll_event", self._on_scroll)
        self.zoom_base = 1.25          # factor per wheel step

    # ------------------------------------------------------------------
    @property
    def K(self):
        return self.Ks[self.imap]

    def _load_map(self, k):
        import matplotlib.tri as mtri
        self.imap = int(k) % self.n_maps
        self.vmin, self.vmax = self.scales[int(self.groups[self.imap])]
        self.mag = np.linalg.norm(self.K, axis=1)
        Kn = nodal_from_tri(self.pts, self.tri, self.area, self.K)
        self.ix = mtri.LinearTriInterpolator(self.T, Kn[:, 0])
        self.iy = mtri.LinearTriInterpolator(self.T, Kn[:, 1])

    def _set_title(self, extra=None):
        t = self.captions[self.imap]
        if self.n_maps > 1:
            t += f"  [{self.imap + 1}/{self.n_maps}, key i]"
        if self.base_title:
            t = self.base_title + "  --  " + t
        if extra:
            t += "\n" + extra
        self.ax.set_title(t, fontsize=9)

    def set_map(self, k, redraw_lines=True):
        """Switches to current map k (cyclic) and redraws the background."""
        self._load_map(k)
        self.clear_lines()
        self._draw_background()
        self._set_title()
        if redraw_lines:
            self.streamlines()
        self.fig.canvas.draw_idle()

    def _norm(self):
        from matplotlib.colors import LogNorm, Normalize
        return LogNorm(self.vmin, self.vmax) if self.log \
            else Normalize(0.0, self.vmax)

    def _draw_background(self):
        if self.mesh is not None:
            self.mesh.remove()
        self.mesh = self.ax.tripcolor(self.T, facecolors=self.mag,
                                      cmap="inferno", norm=self._norm(),
                                      shading="flat", rasterized=True)
        if self.cb is None:
            self.cb = self.fig.colorbar(self.mesh, ax=self.ax, fraction=0.035,
                                        pad=0.02)
        else:
            self.cb.update_normal(self.mesh)
        lab = r"$|K|$ (A/$\mu$m)"
        if self.log:
            lab += "  [log]"
        self.cb.set_label(lab)

    def _grid(self, n=500):
        x0, x1 = self.ax.get_xlim()
        y0, y1 = self.ax.get_ylim()
        ny = max(40, int(n * (y1 - y0) / max(x1 - x0, 1e-9)))
        xs = np.linspace(x0, x1, n)
        ys = np.linspace(y0, y1, ny)
        X, Y = np.meshgrid(xs, ys)
        U = self.ix(X, Y)
        V = self.iy(X, Y)
        bad = np.ma.getmaskarray(U) | np.ma.getmaskarray(V)
        U = np.ma.masked_array(np.ma.filled(U, 0.0), mask=bad)
        V = np.ma.masked_array(np.ma.filled(V, 0.0), mask=bad)
        return X, Y, U, V

    def streamlines(self, density=2.2):
        X, Y, U, V = self._grid()
        sp = np.ma.sqrt(U ** 2 + V ** 2)
        lw = 0.3 + 1.5 * np.clip(np.ma.filled(sp, 0.0) / self.vmax, 0, 1)
        # streamplot adds each arrow head to the axes as an individual
        # patch, the returned PatchCollection is not what is drawn, so the
        # new patches are tracked explicitly to be removable
        before = {id(p) for p in self.ax.patches}
        st = self.ax.streamplot(X, Y, U, V, color="white", density=density,
                                linewidth=lw, arrowsize=0.7)
        self.artists.append(st.lines)
        self.artists.append(st.arrows)
        self.artists += [p for p in self.ax.patches if id(p) not in before]

    def quiver(self, n=45):
        X, Y, U, V = self._grid(n)
        q = self.ax.quiver(X, Y, U, V, color="white", scale_units="width",
                           width=0.0025, alpha=0.9)
        self.artists.append(q)

    def clear_lines(self):
        for a in self.artists:
            try:
                a.remove()
            except Exception:
                pass
        self.artists = []

    # ------------------------------------------------------------------
    def _on_key(self, ev):
        if ev.key == "i":
            self.set_map(self.imap + 1)
            return
        elif ev.key == "I":
            self.set_map(self.imap - 1)
            return
        elif ev.key == "l":
            self.log = not self.log
            self._draw_background()
        elif ev.key == "S":
            self.streamlines()
        elif ev.key == "q":
            self.quiver()
        elif ev.key == "c":
            self.clear_lines()
        else:
            return
        self.fig.canvas.draw_idle()

    def _on_scroll(self, ev):
        """Zoom centred on the cursor. ev.step is +-1 per wheel step and may
        be fractional on a trackpad, hence the exponential."""
        if ev.inaxes != self.ax or ev.xdata is None:
            return
        step = getattr(ev, "step", 0.0)
        if not step:
            step = 1.0 if ev.button == "up" else -1.0
        f = self.zoom_base ** (-step)          # < 1 : zoom in
        x0, x1 = self.ax.get_xlim()
        y0, y1 = self.ax.get_ylim()
        xc, yc = ev.xdata, ev.ydata
        # lower bound on the width, about one mesh cell
        wmin = 3.0 * float(np.sqrt(np.median(self.area)))
        if (x1 - x0) * f < wmin and f < 1:
            return
        self.ax.set_xlim(xc + (x0 - xc) * f, xc + (x1 - xc) * f)
        self.ax.set_ylim(yc + (y0 - yc) * f, yc + (y1 - yc) * f)
        self.fig.canvas.draw_idle()

    def _on_click(self, ev):
        if ev.inaxes != self.ax or ev.button != 1 or \
                self.fig.canvas.toolbar.mode:
            return
        t = self.T.get_trifinder()(ev.xdata, ev.ydata)
        if t < 0:
            return
        k = self.K[t]
        msg = f"({ev.xdata:.2f}, {ev.ydata:.2f}) um   K = ({k[0]:.4e}, " \
              f"{k[1]:.4e}) A/um   |K| = {self.mag[t]:.4e} A/um"
        if self.d:
            msg += f"   J = {self.mag[t]/self.d:.4e} A/um^2 " \
                   f"= {self.mag[t]/self.d*1e8:.4e} A/cm^2"
        print(f"[{self.names[self.imap]}] " + msg)
        self._set_title(msg)
        self.fig.canvas.draw_idle()

    def save(self, filename, dpi=190, density=2.4):
        self.streamlines(density)
        self.fig.tight_layout()
        self.fig.savefig(filename, dpi=dpi)
        return filename

    def show(self):
        self.streamlines()
        self.fig.tight_layout()
        self.plt.show()


# ----------------------------------------------------------------------
def load_npz(path):
    """Loads a *_solution.npz file. K is returned as (n_maps, n_tri, 2),
    files written before the multi-source version (K of shape (n_tri, 2))
    are accepted."""
    d = np.load(path, allow_pickle=False)
    ov = json.loads(str(d["overlays"])) if "overlays" in d else {}
    sel = d["tri_sheet"] == 0
    tri = d["triangles"][sel]
    K = d["K"]
    if K.ndim == 2:
        K = K[None]
    K = K[:, sel]
    if "source_names" in d:
        names = [str(s) for s in d["source_names"]]
    else:
        names = [f"source {k}" for k in range(len(K))]
    cur = float(d["current"])
    captions = [f"injection of I = {cur*1e3:g} mA in {nm}" for nm in names]
    groups = [0] * len(names)
    if "K_field" in d and d["K_field"].size:
        K = np.concatenate([K, d["K_field"][sel][None]], axis=0)
        names.append("B_ext")
        captions.append(f"applied field B_ext = {float(d['Bext_uT']):g} uT "
                        "along +z")
        groups.append(1)
    return dict(points=d["points"], triangles=tri, K=K, names=names,
                captions=captions, groups=groups,
                area=d["area"][sel], overlays=ov,
                thickness=float(d["thickness"]) if "thickness" in d else None,
                Lambda=float(d["Lambda"]), current=float(d["current"]))


def main(argv=None):
    import argparse
    import re
    p = argparse.ArgumentParser(
        description="Interactive viewer of the current maps, key i cycles "
                    "through the injection polygons of layer 3.")
    p.add_argument("npz")
    p.add_argument("--png", default=None,
                   help="write PNG files instead of opening the window, one "
                        "per source (suffix _<k>_<name> added when several)")
    p.add_argument("--map", type=int, default=None,
                   help="index of the single map to write with --png")
    p.add_argument("--log", action="store_true")
    a = p.parse_args(argv)
    if a.png:
        import matplotlib
        matplotlib.use("Agg")
    m = load_npz(a.npz)
    n = np.max(m["triangles"]) + 1
    v = CurrentView(m["points"][:n], m["triangles"], m["K"], m["area"],
                    overlays=m["overlays"], thickness=m["thickness"],
                    title=f"$\\Lambda$ = {m['Lambda']:.3g} $\\mu$m",
                    log_color=a.log, names=m["names"],
                    captions=m["captions"], groups=m["groups"])
    if a.png:
        idx = range(v.n_maps) if a.map is None else [a.map % v.n_maps]
        for k in idx:
            v.set_map(k, redraw_lines=False)
            if v.n_maps > 1 and a.map is None:
                stem, ext = (a.png.rsplit(".", 1) + ["png"])[:2]
                nm = re.sub(r"[^A-Za-z0-9_.-]+", "_", v.names[k]).strip("_")
                fn = f"{stem}_{k}_{nm}.{ext}"
            else:
                fn = a.png
            v.save(fn)
            v.clear_lines()
            print("written:", fn)
    else:
        v.show()


if __name__ == "__main__":
    main()
