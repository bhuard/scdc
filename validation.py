"""
validation.py -- Comparaison du noyau physique a des resultats analytiques.
Ne necessite ni gdstk, ni shapely, ni triangle.

    python validation.py
"""

import numpy as np
from scipy.special import ellipk, ellipe

try:
    from scdc.core import (Assembly, Sheet, MU0, PHI0, build_dof_map,
                           effective_penetration_depth, flux_through_polygon)
    from scdc.solver import Terminal, Contact, solve_currents, total_energy
except ImportError:                      # fichiers a plat, sans package
    from core import (Assembly, Sheet, MU0, PHI0, build_dof_map,
                      effective_penetration_depth, flux_through_polygon)
    from solver import Terminal, Contact, solve_currents, total_energy


# ---------------------------------------------------------------- maillages
def rect_mesh(x0, x1, y0, y1, nx, ny):
    xs = np.linspace(x0, x1, nx + 1)
    ys = np.linspace(y0, y1, ny + 1)
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    pts = np.column_stack([X.ravel(), Y.ravel()])
    idx = lambda i, j: i * (ny + 1) + j
    t = []
    for i in range(nx):
        for j in range(ny):
            a, b, c, d = idx(i, j), idx(i+1, j), idx(i+1, j+1), idx(i, j+1)
            t += [[a, b, c], [a, c, d]] if (i + j) % 2 == 0 else [[a, b, d], [b, c, d]]
    return pts, np.array(t)


def annulus_mesh(r_in, r_out, nr, nt):
    rs = np.linspace(r_in, r_out, nr + 1)
    ts = np.linspace(0, 2 * np.pi, nt, endpoint=False)
    pts = np.array([[r * np.cos(t), r * np.sin(t)] for r in rs for t in ts])
    idx = lambda i, j: i * nt + (j % nt)
    t = []
    for i in range(nr):
        for j in range(nt):
            a, b, c, d = idx(i, j), idx(i+1, j), idx(i+1, j+1), idx(i, j+1)
            t += [[a, b, c], [a, c, d]]
    return pts, np.array(t)


def maxwell_M(a, b, z):
    """Mutuelle entre deux spires filiformes coaxiales."""
    k2 = 4 * a * b / ((a + b) ** 2 + z ** 2)
    k = np.sqrt(k2)
    return MU0 * np.sqrt(a * b) * ((2/k - k) * ellipk(k2) - (2/k) * ellipe(k2))


line = "=" * 72

# ====================================================== 1 : potentiel vecteur
print(line)
print("1. Mutuelle nappe annulaire / spire coaxiale, contre formule de Maxwell")
print(line)
R, w = 20.0, 0.4
pts, tris = annulus_mesh(R - w/2, R + w/2, 2, 720)
asm = Assembly([Sheet(pts, tris, 0.0, 1.0)])
c = asm.centroid
r = np.linalg.norm(c, axis=1)
K = (1.0 / w) * np.column_stack([-c[:, 1]/r, c[:, 0]/r])
for b_rad, zoff in [(5.0, 0.0), (10.0, 0.0), (5.0, 8.0), (30.0, 4.0)]:
    th = np.linspace(0, 2*np.pi, 721)[:-1]
    ring = np.column_stack([b_rad*np.cos(th), b_rad*np.sin(th)])
    Mn = flux_through_polygon(asm, K, ring, z=zoff, seg=0.15)
    Mt = maxwell_M(R, b_rad, zoff)
    print(f"   b={b_rad:5.1f} z={zoff:4.1f}   M_num={Mn*1e12:9.4f} pH   "
          f"M_th={Mt*1e12:9.4f} pH   ecart {abs(Mn-Mt)/abs(Mt)*100:.3f} %")

# ====================================================== 2 : piste, Lambda grand
print()
print(line)
print("2. Inductance cinetique d'une piste, limite Lambda grand")
print(line)
W, L = 4.0, 40.0
Lam = effective_penetration_depth(10.0, 0.05)
pts, tris = rect_mesh(0, L, -W/2, W/2, 80, 10)
asm = Assembly([Sheet(pts, tris, 0.0, Lam)])
I = 1e-3
src = np.where(pts[:, 0] < 0.51)[0]
snk = np.where(pts[:, 0] > L - 0.51)[0]
sol = solve_currents(asm, [Terminal(src, I), Terminal(snk, -I)], verbose=False)
Lsim = 2 * total_energy(asm, sol.K) / I**2
Lth = MU0 * Lam * L / W
print(f"   Lambda = {Lam:.4g} um")
print(f"   L_simule = {Lsim*1e12:.1f} pH   mu0*Lambda*l/W = {Lth*1e12:.1f} pH"
      f"   ecart {abs(Lsim-Lth)/Lth*100:.2f} %")
mid = np.abs(asm.centroid[:, 0] - L/2) < 1.0
k = np.linalg.norm(sol.K, axis=1)[mid]
print(f"   |K| au centre = {k.mean():.6e} +- {k.std():.1e} A/um   "
      f"(I/W = {I/W:.6e})")

# ====================================================== 3 : profil Meissner
print()
print(line)
print("3. Conservation du courant et profil (I/pi)/sqrt((W/2)^2-y^2), Lambda -> 0")
print(line)
W, L = 4.0, 30.0
pts, tris = rect_mesh(0, L, -W/2, W/2, 120, 30)
src = np.where(pts[:, 0] < 0.26)[0]
snk = np.where(pts[:, 0] > L - 0.26)[0]
for Lam in (1e-4, 0.02, 0.5, 20.0):
    asm = Assembly([Sheet(pts, tris, 0.0, Lam)])
    s = solve_currents(asm, [Terminal(src, I), Terminal(snk, -I)], verbose=False)
    xa, xb = L/2 - 1.0, L/2 + 1.0
    sel = (asm.centroid[:, 0] > xa) & (asm.centroid[:, 0] < xb)
    Itr = np.sum(s.K[sel, 0] * asm.area[sel]) / (xb - xa)
    y, kx = asm.centroid[sel, 1], s.K[sel, 0]
    m = np.abs(y) < 0.8 * W/2
    ideal = (I/np.pi) / np.sqrt((W/2)**2 - y[m]**2)
    rms = np.sqrt(np.mean((kx[m] - ideal)**2)) / np.mean(ideal)
    print(f"   Lambda={Lam:8.4g} um   I_transporte={Itr*1e3:.6f} mA   "
          f"ecart au profil ideal {rms*100:6.2f} %")

# ====================================================== 4 : airbridge
print()
print(line)
print("4. Airbridge shuntant un plan continu, conservation du courant")
print(line)
Lam = effective_penetration_depth(0.09, 0.1)
p0, t0 = rect_mesh(0, 24, -2, 2, 96, 16)
pb, tb = rect_mesh(8, 16, -1, 1, 32, 8)
asm = Assembly([Sheet(p0, t0, 0.0, Lam, "metal"),
                Sheet(pb, tb, 3.0, Lam, "bridge")])
o = asm.node_offset
n1, n2 = asm.sheets[0].points, asm.sheets[1].points
ct = [Contact(np.where((n1[:, 0] >= 7.99) & (n1[:, 0] <= 9.01)
                       & (abs(n1[:, 1]) <= 1.01))[0],
              np.where(n2[:, 0] <= 9.01)[0] + o[1]),
      Contact(np.where((n1[:, 0] >= 14.99) & (n1[:, 0] <= 16.01)
                       & (abs(n1[:, 1]) <= 1.01))[0],
              np.where(n2[:, 0] >= 14.99)[0] + o[1])]
src = np.where(n1[:, 0] < 0.26)[0]
snk = np.where(n1[:, 0] > 23.74)[0]
sol = solve_currents(asm, [Terminal(src, I), Terminal(snk, -I)], ct,
                     verbose=False)


def cut(K, x, sheet):
    t = asm.tri_offset[sheet]
    n = len(asm.sheets[sheet].triangles)
    cc, aa = asm.centroid[t:t+n], asm.area[t:t+n]
    sel = (cc[:, 0] > x - 0.5) & (cc[:, 0] < x + 0.5)
    return np.sum(K[t:t+n][sel, 0] * aa[sel])


Ip, Ib = cut(sol.K, 12.0, 0), cut(sol.K, 12.0, 1)
print(f"   plan  {Ip*1e3:.5f} mA   pont {Ib*1e3:.5f} mA   "
      f"total {(Ip+Ib)*1e3:.5f} mA")

# ====================================================== 5 : fluxoide
print()
print(line)
print("5. Fluxoide impose n = 1 dans un anneau 6-10 um")
print(line)
pa, ta = annulus_mesh(6.0, 10.0, 10, 200)
asm = Assembly([Sheet(pa, ta, 0.0, Lam)])
nd, ndof, holes, loops = build_dof_map(asm)
sol = solve_currents(asm, [], [], fluxoid={holes[0][1]: 1}, verbose=False)
E = total_energy(asm, sol.K)
Ic = sol.info['g'][holes[0][0]]
Lring = PHI0 / abs(Ic)
Rm, a_eq = 8.0, 1.0
Lth = MU0 * Rm * (np.log(8 * Rm / a_eq) - 2.0)
print(f"   courant persistant {Ic*1e6:.4f} uA")
print(f"   fluxoide {sol.info['fluxoids'][holes[0][0]]/PHI0:.6f} Phi0   "
      f"coherence 2E/(I*Phi0) = {2*E/Ic/PHI0:.6f}")
print(f"   L = Phi0/I = {Lring*1e12:.3f} pH   "
      f"mu0*R*(ln(8R/a)-2) = {Lth*1e12:.3f} pH")

# ====================================================== 6 : rapide vs dense
print()
print(line)
print("6. Solveur pFFT + gradient conjugue contre assemblage dense")
print(line)
try:
    from scdc.fast import solve_currents_fast
except ImportError:
    from fast import solve_currents_fast
W, L = 4.0, 30.0
pts, tris = rect_mesh(0, L, -W/2, W/2, 120, 24)
src = np.where(pts[:, 0] < 0.26)[0]
snk = np.where(pts[:, 0] > L - 0.26)[0]
for Lam in (1e-3, 0.3):
    asm = Assembly([Sheet(pts, tris, 0.0, Lam)])
    terms = [Terminal(src, I), Terminal(snk, -I)]
    sd = solve_currents(asm, terms, verbose=False)
    Ed = total_energy(asm, sd.K)
    sf = solve_currents_fast(asm, terms, verbose=False)
    err = np.linalg.norm(sf.K - sd.K) / np.linalg.norm(sd.K)
    print(f"   Lambda={Lam:6.3g}  |dK|/|K| = {err:.2e}   dE/E = "
          f"{abs(sf.energy-Ed)/Ed*100:.3f} %   "
          f"{sf.info['iterations']} iterations CG")

# ====================================================== 7 : jonction
print()
print(line)
print("7. Patch jonction dans une piste, L_tot - L_piste contre L_J")
print(line)
try:
    from scdc.core import josephson_inductance
    from scdc.fast import FastKernel
except ImportError:
    from core import josephson_inductance
    from fast import FastKernel
LJ, Ic, EJ = josephson_inductance(EJ_GHz=20.0)
print(f"   E_J/h = 20 GHz  ->  I_c = {Ic*1e9:.2f} nA, L_J = {LJ*1e9:.3f} nH")
W, L = 4.0, 30.0
pts, tris = rect_mesh(0, L, -W/2, W/2, 150, 20)
Lam_f = effective_penetration_depth(0.09, 0.1)
src = np.where(pts[:, 0] < 0.21)[0]
snk = np.where(pts[:, 0] > L - 0.21)[0]
asm0 = Assembly([Sheet(pts, tris, 0.0, Lam_f)])
FK = FastKernel(asm0, verbose=False)
s0 = solve_currents_fast(asm0, [Terminal(src, I), Terminal(snk, -I)],
                         kernel=FK, verbose=False)
L0 = 2 * s0.energy / I**2
cen = pts[tris].mean(axis=1)
for ell, LJn in [(1.0, 0.5e-9), (1.0, 5e-9)]:
    inJ = np.abs(cen[:, 0] - L/2) < ell/2
    Lam = np.full(len(tris), Lam_f)
    Lam[inJ] = LJn * W / ell / MU0
    asm = Assembly([Sheet(pts, tris, 0.0, Lam)])
    s = solve_currents_fast(asm, [Terminal(src, I), Terminal(snk, -I)],
                            kernel=FK, verbose=False)
    Lt = 2 * s.energy / I**2
    print(f"   L_J = {LJn*1e9:4.1f} nH : L_tot - L_piste = {(Lt-L0)*1e9:.5f} nH"
          f"   ecart {(Lt-L0-LJn)/LJn*100:+.4f} %")

# ====================================================== 8 to 11 : applied field
try:
    from scdc.core import external_field_drive, magnetic_moment_z
except ImportError:
    from core import external_field_drive, magnetic_moment_z
from scipy.spatial import Delaunay


def disk_mesh(R, nr, grade=1.0, c=(0.0, 0.0)):
    """Disk of radius R, nr rings of nodes, spacing graded towards the edge
    for grade > 1."""
    s = np.linspace(0, 1, nr + 1)
    rs = R * (1 - (1 - s) ** grade)
    dr = np.diff(rs)
    p = [[0.0, 0.0]]
    for k, r in enumerate(rs[1:]):
        nt = max(8, int(np.ceil(2 * np.pi * r / dr[k])))
        t = np.linspace(0, 2 * np.pi, nt, endpoint=False)
        p += list(np.column_stack([r * np.cos(t), r * np.sin(t)]))
    p = np.array(p) + np.asarray(c)
    return p, Delaunay(p).simplices


Bz = 1e-6 * 1e-12              # 1 uT in Wb/um^2

print()
print(line)
print("8. Disk in a perpendicular field, kinetic limit Lambda >> R, and gauge")
print(line)
# K = -A_ext / (mu0 Lambda) in the London gauge, m_z = -pi B R^4 / (8 mu0 Lambda)
R = 10.0
for c in [(0.0, 0.0), (37.0, -21.0)]:
    pd, td = disk_mesh(R, 14, c=c)
    for Lam in (500.0, 5e4):
        asm = Assembly([Sheet(pd, td, 0.0, Lam)])
        s = solve_currents(asm, [], drive=external_field_drive(asm, Bz),
                           verbose=False)
        m = magnetic_moment_z(asm, s.K, center=c)
        # same moment computed with the discrete r^2 of the mesh
        r2 = np.sum((asm.centroid - np.asarray(c)) ** 2, axis=1)
        mk = -Bz / (4 * MU0 * Lam) * np.sum(asm.area * r2)
        E = total_energy(asm, s.K)
        print(f"   centre ({c[0]:5.1f}, {c[1]:5.1f}) Lambda = {Lam:7.0f} um   "
              f"m / m_kin = {m/mk:.5f}   B m / (-2E) = {Bz*m/(-2*E):.7f}")

print()
print(line)
print("9. Disk in a perpendicular field, Meissner limit, m = -8 B R^3 / (3 mu0)")
print(line)
pd, td = disk_mesh(R, 16, grade=2.0)
mM = -8 * Bz * R ** 3 / (3 * MU0)
for Lam in (0.1, 0.02):
    asm = Assembly([Sheet(pd, td, 0.0, Lam)])
    s = solve_currents_fast(asm, [], drive=external_field_drive(asm, Bz),
                            verbose=False)
    m = magnetic_moment_z(asm, s.K)
    x = Lam / R
    print(f"   Lambda/R = {x:.3f}   m / m_Meissner = {m/mM:.4f}   "
          f"1 - m/m_M = {1 - m/mM:.4f}   (Lambda/R) ln(R/Lambda) = "
          f"{x*np.log(1/x):.4f}")

print()
print(line)
print("10. Ring 6-10 um, reciprocity I_B = -B m_1 / Phi0 between the field")
print("    response (n = 0) and the fluxoid state n = 1")
print(line)
asm = Assembly([Sheet(pa, ta, 0.0, effective_penetration_depth(0.09, 0.1))])
nd, ndof, holes, loops = build_dof_map(asm)
s1 = solve_currents(asm, [], fluxoid={holes[0][1]: 1}, verbose=False)
sB = solve_currents(asm, [], drive=external_field_drive(asm, Bz), verbose=False)
I1 = s1.info['g'][holes[0][0]]
IB = sB.info['g'][holes[0][0]]
m1 = magnetic_moment_z(asm, s1.K)
EB = total_energy(asm, sB.K)
print(f"   I_B = {IB*1e9:.6f} nA   -B m_1 / Phi0 = {-Bz*m1/PHI0*1e9:.6f} nA")
print(f"   effective area |m_1 / I_1| = {abs(m1/I1):.3f} um^2, between "
      f"pi r_in^2 = {np.pi*36:.3f} and pi r_out^2 = {np.pi*100:.3f} um^2")
print(f"   B m_B / (-2E) = {Bz*magnetic_moment_z(asm, sB.K)/(-2*EB):.7f}   "
      f"fluxoid of the hole {sB.info['fluxoids'][holes[0][0]]/PHI0:.2e} Phi0")

print()
print(line)
print("11. Loop closed by an airbridge with pillars in a field, dense against")
print("    pFFT and independence on the gauge centre")
print(line)
Lam = effective_penetration_depth(0.09, 0.1)
p0, t0 = rect_mesh(0, 24, 0, 16, 48, 32)
cc = p0[t0].mean(axis=1)
t0 = t0[~((np.abs(cc[:, 0] - 12) < 2) & (cc[:, 1] < 12))]     # U shape
u = np.unique(t0)
rm = -np.ones(len(p0), int)
rm[u] = np.arange(len(u))
p0, t0 = p0[u], rm[t0]
pb, tb = rect_mesh(6, 18, 4, 8, 24, 8)
asm = Assembly([Sheet(p0, t0, 0.0, Lam, "metal"),
                Sheet(pb, tb, 3.0, Lam, "bridge")])
o = asm.node_offset
n1, n2 = asm.sheets[0].points, asm.sheets[1].points
inA = lambda q, x0, x1: (q[:, 0] >= x0 - 1e-9) & (q[:, 0] <= x1 + 1e-9) & \
    (q[:, 1] >= 4 - 1e-9) & (q[:, 1] <= 8 + 1e-9)
ct = [Contact(np.where(inA(n1, 6, 8))[0], np.where(inA(n2, 6, 8))[0] + o[1]),
      Contact(np.where(inA(n1, 16, 18))[0], np.where(inA(n2, 16, 18))[0] + o[1])]
pLm = np.array([[20e-12, 5e-12], [5e-12, 20e-12]])
out = {}
for key, cen in [("dense", None), ("dense, centre moved", (500.0, -300.0)),
                 ("pFFT", None)]:
    dr = external_field_drive(asm, Bz, center=cen, contacts=ct)
    if key.startswith("dense"):
        s = solve_currents(asm, [], ct, pillar_L=pLm, drive=dr, verbose=False)
        s.energy = total_energy(asm, s.K, contacts=ct, pillar_L=pLm)
    else:
        s = solve_currents_fast(asm, [], ct, pillar_L=pLm, drive=dr,
                                verbose=False)
    out[key] = s
    print(f"   {key:20s} pillar currents {s.info['contact_currents']*1e9} nA"
          f"   E = {s.energy:.6e} J")
ref = out["dense"].K
for key in ("dense, centre moved", "pFFT"):
    print(f"   |K - K_dense| / |K_dense| ({key}) = "
          f"{np.linalg.norm(out[key].K - ref)/np.linalg.norm(ref):.2e}")
