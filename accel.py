"""
accel.py -- Backends pour les parties couteuses de fast.py.

Trois implementations d'une meme interface.

  numpy  : numpy + scipy.fft multithread (workers = nombre de coeurs), et
           decoupage en blocs traites par un pool de threads pour les
           calculs de paires (numpy libere le GIL sur les ufuncs).
           C'est le backend de reference, teste, en double precision.
  mlx    : Apple MLX (GPU Metal, memoire unifiee). Simple precision.
  torch  : PyTorch, device 'mps' sur Apple Silicon (ou 'cuda'). Simple
           precision sur MPS.

Les backends GPU sont experimentaux : ils n'ont pas pu etre executes sur la
machine qui a ecrit ce code. Un autotest compare leurs resultats a numpy sur
un echantillon au demarrage et retombe sur numpy en cas d'ecart. Toute
l'arithmetique GPU utilise des coordonnees relatives (centre des triangles
en float64 cote CPU, ecarts en float32) pour eviter la perte de precision de
la simple precision sur des puces de plusieurs millimetres.

Choix : variable d'environnement SCDC_BACKEND=numpy|mlx|torch|auto, ou
argument backend= des fonctions de fast.py.
"""

from __future__ import annotations

import os
import warnings
import numpy as np
from concurrent.futures import ThreadPoolExecutor

N_WORKERS = max(1, os.cpu_count() or 1)


# ======================================================================
class NumpyBackend:
    name = "numpy"
    dtype = np.float64

    def __init__(self, workers=None):
        self.workers = workers or N_WORKERS
        import scipy.fft as sfft
        self.sfft = sfft

    # ---------------------------------------------------- utilitaires
    def _pmap(self, fn, n, chunk):
        """Applique fn(s, e) sur des blocs [s, e) en parallele (threads)."""
        bounds = [(s, min(s + chunk, n)) for s in range(0, n, chunk)]
        if self.workers == 1 or len(bounds) == 1:
            return [fn(s, e) for s, e in bounds]
        with ThreadPoolExecutor(self.workers) as ex:
            return list(ex.map(lambda b: fn(*b), bounds))

    # ---------------------------------------------------- paires exactes
    def exact_pairs(self, cen, qrel, z, area, I, J, chunk=300000):
        """G_ij = A_i A_j <1/|r-r'|> (quadrature 3x3), diagonale analytique.

        cen  : (M,3) centres (x, y, z) float64
        qrel : (M,3,2) points de quadrature relatifs au centre
        """
        out = np.empty(len(I))

        def work(s, e):
            i, j = I[s:e], J[s:e]
            dc = cen[i] - cen[j]                      # (n,3) float64
            dz2 = dc[:, 2] ** 2
            acc = np.zeros(e - s)
            for p in range(3):
                for q in range(3):
                    dx = dc[:, 0] + qrel[i, p, 0] - qrel[j, q, 0]
                    dy = dc[:, 1] + qrel[i, p, 1] - qrel[j, q, 1]
                    acc += 1.0 / np.sqrt(np.maximum(dx * dx + dy * dy + dz2,
                                                    1e-24))
            g = area[i] * area[j] * acc / 9.0
            d = i == j
            g[d] = (16.0 * np.pi / 3.0) * (area[i[d]] / np.pi) ** 1.5
            out[s:e] = g

        self._pmap(work, len(I), chunk)
        return out

    # ---------------------------------------------------- paires grille
    def grid_pairs(self, fp_i, fp_j, fp_w, plane, tab, R, I, J, chunk=150000):
        """(S K S^T)_ij par table du noyau grille. tab : (npl,npl,2R+1,2R+1)."""
        npl = tab.shape[0]
        W = 2 * R + 1
        tflat = tab.ravel()
        out = np.empty(len(I))

        def work(s, e):
            i, j = I[s:e], J[s:e]
            base = (plane[i] * npl + plane[j]) * (W * W)      # (n,)
            acc = np.zeros(e - s)
            fij, fjj, fwj = fp_i[j], fp_j[j], fp_w[j]         # (n,12)
            for a in range(12):
                di = np.clip(fp_i[i, a][:, None] - fij + R, 0, W - 1)
                dj = np.clip(fp_j[i, a][:, None] - fjj + R, 0, W - 1)
                idx = base[:, None] + di * W + dj
                acc += fp_w[i, a] * np.einsum('pk,pk->p', fwj,
                                              np.take(tflat, idx))
            out[s:e] = acc

        self._pmap(work, len(I), chunk)
        return out

    # ---------------------------------------------------- FFT
    def prepare_kernel(self, K):
        """Precalcule la transformee d'un noyau reel (NX,NY)."""
        return self.sfft.rfft2(K, workers=self.workers)

    def conv(self, Q, Khat, NX, NY):
        """Convolution lineaire de Q (nx,ny) par le noyau, retour (nx,ny)."""
        nx, ny = Q.shape
        F = self.sfft.rfft2(Q, s=(NX, NY), workers=self.workers)
        out = self.sfft.irfft2(F * Khat, s=(NX, NY), workers=self.workers)
        return out[:nx, :ny]

    def to_host(self, x):
        return np.asarray(x)


# ======================================================================
class MLXBackend:
    """Apple MLX. Toute l'arithmetique en float32 sur le GPU."""
    name = "mlx"
    dtype = np.float32

    def __init__(self):
        import mlx.core as mx
        self.mx = mx
        self._cache = {}

    def _dev(self, key, arr, dtype=None):
        """Uploads arr once. The host array is kept in the cache so that the
        identity test stays valid, an id() being reusable once the array is
        freed."""
        ent = self._cache.get(key)
        if ent is None or ent[0] is not arr:
            dev = self.mx.array(np.ascontiguousarray(
                arr.astype(dtype or np.float32)))
            ent = (arr, dev)
            self._cache[key] = ent
        return ent[1]

    def exact_pairs(self, cen, qrel, z, area, I, J, chunk=2000000):
        mx = self.mx
        qr = self._dev('qrel', qrel.reshape(len(qrel), 6))     # (M,6)
        ar = self._dev('area', area)
        out = np.empty(len(I))
        for s in range(0, len(I), chunk):
            i, j = I[s:s + chunk], J[s:s + chunk]
            dc = (cen[i] - cen[j]).astype(np.float32)          # CPU float64
            dcx = mx.array(dc[:, 0]); dcy = mx.array(dc[:, 1])
            dz2 = mx.array(dc[:, 2] ** 2)
            mi, mj = mx.array(i.astype(np.int32)), mx.array(j.astype(np.int32))
            qi = mx.take(qr, mi, axis=0); qj = mx.take(qr, mj, axis=0)
            acc = mx.zeros((len(i),), dtype=mx.float32)
            for p in range(3):
                for q in range(3):
                    dx = dcx + qi[:, 2 * p] - qj[:, 2 * q]
                    dy = dcy + qi[:, 2 * p + 1] - qj[:, 2 * q + 1]
                    acc = acc + mx.rsqrt(mx.maximum(dx * dx + dy * dy + dz2,
                                                    1e-24))
            g = mx.take(ar, mi) * mx.take(ar, mj) * acc / 9.0
            mx.eval(g)
            g = np.array(g, dtype=np.float64)
            d = i == j
            g[d] = (16.0 * np.pi / 3.0) * (area[i[d]] / np.pi) ** 1.5
            out[s:s + chunk] = g
        return out

    def grid_pairs(self, fp_i, fp_j, fp_w, plane, tab, R, I, J, chunk=1000000):
        mx = self.mx
        npl = tab.shape[0]
        W = 2 * R + 1
        tflat = mx.array(tab.ravel().astype(np.float32))
        FI = self._dev('fpi', fp_i, np.int32); FJ = self._dev('fpj', fp_j, np.int32)
        FW = self._dev('fpw', fp_w)
        PL = self._dev('plane', plane, np.int32)
        out = np.empty(len(I))
        for s in range(0, len(I), chunk):
            i = mx.array(I[s:s + chunk].astype(np.int32))
            j = mx.array(J[s:s + chunk].astype(np.int32))
            base = (mx.take(PL, i) * npl + mx.take(PL, j)) * (W * W)
            fii, fji, fwi = mx.take(FI, i, axis=0), mx.take(FJ, i, axis=0), mx.take(FW, i, axis=0)
            fij, fjj, fwj = mx.take(FI, j, axis=0), mx.take(FJ, j, axis=0), mx.take(FW, j, axis=0)
            acc = mx.zeros((len(I[s:s + chunk]),), dtype=mx.float32)
            for a in range(12):
                di = mx.clip(fii[:, a:a + 1] - fij + R, 0, W - 1)
                dj = mx.clip(fji[:, a:a + 1] - fjj + R, 0, W - 1)
                idx = base[:, None] + di * W + dj
                acc = acc + fwi[:, a] * mx.sum(fwj * mx.take(tflat, idx), axis=1)
            mx.eval(acc)
            out[s:s + chunk] = np.array(acc, dtype=np.float64)
        return out

    def prepare_kernel(self, K):
        mx = self.mx
        return mx.fft.rfft2(mx.array(K.astype(np.float32)))

    def conv(self, Q, Khat, NX, NY):
        mx = self.mx
        nx, ny = Q.shape
        F = mx.fft.rfft2(mx.array(Q.astype(np.float32)), s=(NX, NY))
        out = mx.fft.irfft2(F * Khat, s=(NX, NY))[:nx, :ny]
        mx.eval(out)
        return np.array(out, dtype=np.float64)

    def to_host(self, x):
        return np.array(x, dtype=np.float64)


# ======================================================================
class TorchBackend:
    """PyTorch, device mps (Apple) ou cuda. float32 sur mps."""
    name = "torch"
    dtype = np.float32

    def __init__(self, device=None):
        import torch
        self.t = torch
        if device is None:
            if torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        self.device = torch.device(device)
        self.fdt = torch.float32 if device != "cpu" else torch.float64
        self.dtype = np.float32 if device != "cpu" else np.float64
        self._cache = {}

    def _dev(self, key, arr, dtype=None):
        k = (key, id(arr))
        if k not in self._cache:
            self._cache[k] = self.t.as_tensor(
                np.ascontiguousarray(arr), device=self.device,
                dtype=dtype or self.fdt)
        return self._cache[k]

    def exact_pairs(self, cen, qrel, z, area, I, J, chunk=2000000):
        t = self.t
        qr = self._dev('qrel', qrel.reshape(len(qrel), 6))
        ar = self._dev('area', area)
        out = np.empty(len(I))
        for s in range(0, len(I), chunk):
            i, j = I[s:s + chunk], J[s:s + chunk]
            dc = t.as_tensor(cen[i] - cen[j], device=self.device, dtype=self.fdt)
            ti = t.as_tensor(i, device=self.device); tj = t.as_tensor(j, device=self.device)
            qi, qj = qr[ti], qr[tj]
            dz2 = dc[:, 2] ** 2
            acc = t.zeros(len(i), device=self.device, dtype=self.fdt)
            for p in range(3):
                for q in range(3):
                    dx = dc[:, 0] + qi[:, 2 * p] - qj[:, 2 * q]
                    dy = dc[:, 1] + qi[:, 2 * p + 1] - qj[:, 2 * q + 1]
                    acc += t.rsqrt(t.clamp(dx * dx + dy * dy + dz2, min=1e-24))
            g = (ar[ti] * ar[tj] * acc / 9.0).cpu().numpy().astype(np.float64)
            d = i == j
            g[d] = (16.0 * np.pi / 3.0) * (area[i[d]] / np.pi) ** 1.5
            out[s:s + chunk] = g
        return out

    def grid_pairs(self, fp_i, fp_j, fp_w, plane, tab, R, I, J, chunk=1000000):
        t = self.t
        npl = tab.shape[0]
        W = 2 * R + 1
        tflat = t.as_tensor(tab.ravel(), device=self.device, dtype=self.fdt)
        FI = self._dev('fpi', fp_i, t.int64); FJ = self._dev('fpj', fp_j, t.int64)
        FW = self._dev('fpw', fp_w)
        PL = self._dev('plane', plane, t.int64)
        out = np.empty(len(I))
        for s in range(0, len(I), chunk):
            i = t.as_tensor(I[s:s + chunk], device=self.device)
            j = t.as_tensor(J[s:s + chunk], device=self.device)
            base = (PL[i] * npl + PL[j]) * (W * W)
            fii, fji, fwi = FI[i], FJ[i], FW[i]
            fij, fjj, fwj = FI[j], FJ[j], FW[j]
            acc = t.zeros(len(i), device=self.device, dtype=self.fdt)
            for a in range(12):
                di = t.clamp(fii[:, a:a + 1] - fij + R, 0, W - 1)
                dj = t.clamp(fji[:, a:a + 1] - fjj + R, 0, W - 1)
                idx = base[:, None] + di * W + dj
                acc += fwi[:, a] * (fwj * tflat[idx]).sum(dim=1)
            out[s:s + chunk] = acc.cpu().numpy().astype(np.float64)
        return out

    def prepare_kernel(self, K):
        t = self.t
        return t.fft.rfft2(t.as_tensor(K, device=self.device, dtype=self.fdt))

    def conv(self, Q, Khat, NX, NY):
        t = self.t
        nx, ny = Q.shape
        F = t.fft.rfft2(t.as_tensor(Q, device=self.device, dtype=self.fdt),
                        s=(NX, NY))
        out = t.fft.irfft2(F * Khat, s=(NX, NY))[:nx, :ny]
        return out.cpu().numpy().astype(np.float64)

    def to_host(self, x):
        return x.cpu().numpy().astype(np.float64)


# ======================================================================
def _selftest(bk, ref, verbose=True):
    """Compare bk a ref sur un petit probleme aleatoire. Retourne True si ok."""
    rng = np.random.default_rng(0)
    M = 400
    cen = np.column_stack([rng.uniform(0, 3000, M), rng.uniform(0, 3000, M),
                           rng.choice([0.0, 3.0], M)])
    qrel = rng.normal(0, 0.3, (M, 3, 2))
    area = rng.uniform(0.2, 1.0, M)
    I = np.repeat(np.arange(M), 5)
    J = (I + rng.integers(0, 5, len(I))) % M
    # rendre les paires proches pour que les deux calculs soient sensibles
    cen[J[:, None].ravel()] = cen[I] + rng.normal(0, 2.0, (len(I), 3)) * [1, 1, 0]
    g1 = ref.exact_pairs(cen, qrel, cen[:, 2], area, I, J)
    g2 = bk.exact_pairs(cen, qrel, cen[:, 2], area, I, J)
    e1 = np.max(np.abs(g1 - g2) / np.abs(g1))
    R = 6; W = 2 * R + 1; npl = 2
    tab = rng.uniform(0.5, 2.0, (npl, npl, W, W))
    fp_i = rng.integers(10, 40, (M, 12)); fp_j = rng.integers(10, 40, (M, 12))
    fp_w = rng.uniform(0, 1, (M, 12))
    plane = (cen[:, 2] > 1).astype(int)
    h1 = ref.grid_pairs(fp_i, fp_j, fp_w, plane, tab, R, I, J)
    h2 = bk.grid_pairs(fp_i, fp_j, fp_w, plane, tab, R, I, J)
    e2 = np.max(np.abs(h1 - h2) / np.abs(h1))
    NX, NY = 64, 48
    K = rng.uniform(0.1, 1, (NX, NY)); Q = rng.normal(0, 1, (32, 24))
    c1 = ref.conv(Q, ref.prepare_kernel(K), NX, NY)
    c2 = bk.conv(Q, bk.prepare_kernel(K), NX, NY)
    e3 = np.max(np.abs(c1 - c2)) / np.max(np.abs(c1))
    tol = 1e-4 if bk.dtype == np.float32 else 1e-10
    ok = e1 < tol and e2 < tol and e3 < tol
    if verbose:
        print(f"  backend self-test {bk.name}: exact pairs {e1:.1e}, "
              f"grid pairs {e2:.1e}, FFT {e3:.1e} -> "
              f"{'ok' if ok else 'FAILED, falling back to numpy'}")
    return ok


def get_backend(name="auto", verbose=True):
    """Retourne un backend valide. name : auto | numpy | mlx | torch."""
    name = (name or os.environ.get("SCDC_BACKEND", "auto")).lower()
    ref = NumpyBackend()
    if name == "numpy":
        return ref
    candidates = []
    if name in ("auto", "mlx"):
        candidates.append("mlx")
    if name in ("auto", "torch"):
        candidates.append("torch")
    for c in candidates:
        try:
            bk = MLXBackend() if c == "mlx" else TorchBackend()
            if c == "torch" and bk.device.type == "cpu" and name == "auto":
                continue
            if _selftest(bk, ref, verbose):
                return bk
        except Exception as e:           # module absent, device absent ...
            if name != "auto" and verbose:
                warnings.warn(f"backend {c} unavailable ({e}), falling back to numpy")
    return ref
