# scdc — DC current density and mutual inductances in a superconducting circuit

Input, a GDS. Output, a current map for each injection polygon of layer 3, the
mutual-inductance matrix `M_ij = Φ_i / I_j` relating the flux through each
layer-5 surface to the current injected into each layer-3 polygon, and a
Markdown report. All produced files and all program messages are in English.

All lengths are in micrometres, currents in amperes, fluxes in webers.

---

## 1. Layer convention

| layer | content | role |
|---|---|---|
| 1 | all metallic structures | superconducting film, plane `z = 0` |
| 2 | top view of the airbridges | raised sheets at `z = h` |
| 3 | one or several polygons, disks or rectangles | injection of the current `I`, one computation per polygon |
| 4 | polygons | returns to ground, each through `R = 0.01 Ω`, only those connected in DC to the source carry current |
| 5 | surfaces | contours where `Φ` is computed, then `M = Φ/I` |
| 6 | rectangles | Josephson junctions, patches with a large kinetic inductance |

The polygons of layers 3 and 5 can carry a GDS label, a TEXT object placed
inside the polygon. That text is used as the name in the report, in the matrix
`M`, in the file names and in the map titles. Failing that, the name is `S0`,
`S1`, … for the sources and `L0`, `L1`, … for the surfaces, in order of
increasing centroid `x` then `y`.

### Several sources and the matrix `M`

Each polygon `j` of layer 3 gives rise to an independent computation, as if it
were alone, with the current `I` injected into that polygon and returned by the
layer-4 polygons that are connected to it in DC (distribution
`I_k = I (1/R_k)/Σ_{l∈C} 1/R_l` over the pads `k` of the galvanic cluster `C`
of the source, see §2.2). The other layer-3 polygons are then nothing but
metal. The pFFT kernel depends on the geometry
only and is built once. One obtains

```
M_ij = Φ_i^{(j)} / I ,     i = layer-5 surface,  j = layer-3 polygon
```

If `M` is square and regular, the report also gives the eigenvalues `λ_k` of
`M⁻¹`, expressed in `mA/Φ0` as `λ_k Φ0 × 10³` with `Φ0 = h/2e`. The eigenvalue
`λ_k Φ0` is the current, along the `k`-th eigendirection, that produces one flux
quantum in the corresponding combination of surfaces. Since `M` is not symmetric
in general (surfaces and sources are distinct objects), complex eigenvalues are
possible and are then printed with their imaginary part. If `M` is not square
the report says so and gives instead the inverses of the singular values of `M`,
which coincide with the moduli of the eigenvalues when `M` is square and normal.

---

## 2. Physical model

### 2.1 Energy

Each film is treated as a current sheet `K` (A/µm) at altitude `z_s`. The total
energy is the sum of a kinetic term and a magnetic term

```
E[K] = (µ0 Λ / 2) ∫ |K|² dA  +  (µ0 / 8π) ∫∫ K(r)·K(r') / |r − r'| dA dA'
```

with the effective penetration depth

```
Λ = λ coth(d/λ)      →  λ²/d  when d ≪ λ,   →  λ  when d ≫ λ
```

The corresponding Pearl length is `2Λ`. The two requested parameters, thickness
`d` and London length `λ`, enter only through `Λ`, which is the signature of the
two-dimensional London model.

The kinetic inductance per square is `L_□ = µ0 Λ`, that is `µ0 λ²/d` for a thin
film, or `ħ R_□ / (π Δ)` in the dirty limit. It can be given directly with
`--Lsq` (in pH per square) instead of `d` and `λ`, which is usually the measured
quantity.

### 2.2 Why an energy minimisation

In the established DC regime, `∂(Λ J)/∂t = E = 0` in the superconductor. The
film is therefore **equipotential** and the network is purely inductive. Two
consequences follow.

- The film is equipotential on each galvanic cluster, a cluster being a set
  of metal pieces joined by the mesh or by bridge feet. The layer-4 pads of the
  cluster `C` that contains the source are therefore all at the same potential
  `V_C`, and `I_k = V_C / R_k`. With identical resistances `R_k = 0.01 Ω` to
  ground, the current splits in **exactly equal** shares, `I_k = I/N_C`, `N_C`
  being the number of pads in `C`. More generally the code uses

  ```
  I_k = I (1/R_k) / Σ_{l∈C} (1/R_l)    for k in C,
  I_k = 0                              otherwise,
  ```

  so that the common value of `R` does not influence the result. Only an
  imbalance between resistances would. A pad lying on a piece of metal with no
  DC path to the source carries no net current. That piece still carries
  screening currents of zero net value, which the model computes. The shares
  are thus computed separately for each layer-3 polygon and printed per source
  in the report. The computation stops with an explicit message if no pad is
  connected to a source, the current then having no return path, or if a
  source overlaps several disjoint pieces, the split between them being then
  undefined. A pad overlapping several disjoint pieces only contributes, for
  each source, through its part on the cluster of that source.
- The internal distribution is not fixed by the DC regime alone, it depends on
  history. For a zero-field cooldown followed by a ramp of the current from 0 to
  `I`, the distribution reached is the unique minimum of `E` under the
  constraints of conservation and of fluxoid quantisation with `n = 0` in each
  hole. This is the solution that is computed.

Stationarity of `E` returns the London equation integrated over the thickness

```
µ0 Λ K(r) + A(r) = − (ħ/2e) ∇θ
```

and, around each hole `C_k`,

```
∮_{C_k} (µ0 Λ K + A)·dl = n_k Φ0 ,   n_k = 0 by default
```

The parameter `fluxoid={hole_index: n}` allows trapped flux to be imposed.

### 2.3 Discretisation

`K` is constant per triangle, and decomposed as

```
K = K_seed  +  Σ_b α_b F_b  +  ẑ × ∇g
```

- `K_seed` is any admissible distribution carrying the injection at the
  terminals, obtained from an auxiliary resistive problem `∇·(σ∇V) = 0`.
- `F_b` are the loop fields associated with the airbridges. Their number is the
  cycle rank of the graph whose vertices are the connected components of metal
  and of bridges, and whose edges are the bridge feet. A bridge that
  short-circuits two points of the same plane creates a degree of freedom. A
  bridge that is the only path between two disjoint planes does not.
- `g` is a nodal P1 stream function, constant on each contour, with one free
  constant per hole and the gauge `g = 0` on the outer contour of each connected
  component.

The energy becomes a quadratic form and the minimum is obtained from a dense
linear system of size `ndof(g) + number of loops`.

The kernel `G_{TT'} = ∫_T ∫_{T'} dA dA' / |r − r'|` is evaluated with a
three-point quadrature per triangle off the diagonal. The diagonal term uses the
exact value for the equivalent disk of area `A`,

```
∫∫ dA dA' / |r − r'| = (16π/3) a³ ,   a = √(A/π)
```

which follows from the mean inverse distance `⟨1/r⟩ = 16/(3πa)` between two
points of a disk.

### 2.4 Josephson junctions

For `I ≪ I_c` a junction is a linear inductance

```
L_J = Φ0 / (2π I_c) = (Φ0/2π)² / E_J ,      E_J = Φ0 I_c / (2π)
```

A rectangle of length `ℓ` along the current and width `w`, with sheet inductance
`L_□^J`, has `L = L_□^J ℓ / w` if the current in it is uniform, which is ensured
as soon as the kinetic term dominates. Each layer-6 rectangle therefore receives

```
Λ_J = L_J w / (µ0 ℓ)
```

In practice the patch is not always a clean rectangle, its ends may land on
oblique sides or overhang the electrodes. The code therefore assumes nothing
about its shape. The active zone is the part of the polygon that sticks out of
the electrodes (the whole polygon if it lies entirely on metal). After meshing,
the patch nodes shared with the outside metal are grouped into contact zones,
the two largest are the electrodes A and B, and Laplace's equation is solved in
the patch with `V = 0` on A and `V = 1` on B. Then

```
G = ∫ |∇V|² dA ,     F = L / L_□ = 1/G ,     Λ_J = L_J / (µ0 F) ,
I_J = ∫_J K · ∇V dA
```

For a rectangle `ℓ × w` one recovers `F = ℓ/w` exactly, and `I_J` is exact for
any conservative `K` in the patch. If the patch touches more than two metal
zones, a warning says so.

The rectangle must be the only connection between its two electrodes. A patch
narrower than the track it crosses is short-circuited by the surrounding film,
which is the physical behaviour of an inductance in parallel with a
superconductor, and the validation test shows it.

The parameters are provided, in order of priority, by a GDS label in the
rectangle (`EJ=20GHz`, `Ic=40nA`, `LJ=8nH`, `Rn=8kOhm Delta=180ueV`). A GDS TEXT
object is required, in KLayout a Text object, and not a text converted into
polygons, which is only a drawing. Failing that, by a JSON file
`--junction-file` of the form `[{"x": 100.5, "y": 60, "EJ_GHz": 20}, ...]`, or by
a default value `--EJ`, `--Ic`, `--LJ`. The Ambegaokar-Baratoff relation
`I_c R_n = πΔ/(2e)` is used for `Rn`.

`--nonlinear` iterates `L_J(I) = L_J0 / √(1 − (I/I_c)²)` until convergence on the
current of each junction. The pFFT kernel depends on the geometry only, it is
built once and reused. The result gives `I_J`, `I_J/I_c` and the phase
`φ = arcsin(I_J/I_c)`.

The range of validity is the one you describe, `I_J` small compared with `I_c`,
no DC voltage across the junction. The model contains neither the junction
capacitance nor its dynamics, which is consistent with a DC computation.

### 2.5 Bridges, feet and pillars

Layer 2 is the top view of the bridge. Its feet are the connected components of
`bridge ∩ metal`. A bridge that spans a CPW covers three of them, the two ground
planes and the centre conductor. By default (`--bridge-feet ends`) only the two
components furthest apart are feet, the others are flown over without contact.
`--bridge-feet all` restores the behaviour where every overlap region is a
contact (multiple vias).

**Bridges as rectangles.** By default (`--bridge-shape rect`) every connected
component `P` of layer 2 is replaced, before the feet are computed, by the
rectangle `R` closest to it in the sense of the area of the symmetric
difference

```
|P △ R| = |P| + |R| − 2 |P ∩ R|
```

(`rectfit.py`, pure numpy). The orientation is taken from two candidates, the
principal axes of the second-moment tensor of `P` (closed form, `O(n)`, exact
for any shape with two orthogonal mirror axes) and the minimum-area bounding
rectangle (rotating calipers, `O(n log n)`, defined also for square-like
shapes). The initial rectangle is centred on the centroid with sides
`√(12 σ_u²)` and `√(12 σ_v²)`, exact for a rectangle whatever its number of
vertices. Each edge is then moved at fixed orientation to the minimum of the
1D restriction of `|P △ R|`. Since

```
∂|P △ R| / ∂u₁ = (v₁ − v₀) − 2 ℓ(u₁) ,
```

with `ℓ(u)` the length of the edge `{u} × [v₀, v₁]` inside `P`, each edge of the
optimum has exactly half of its length inside `P`. For a rectangle whose
corners are rounded or chamfered by up to a quarter of the shorter side, the
result is the unrounded rectangle. A rectangle tilted by at most
`--bridge-snap-angle` degrees (default 0.5) with respect to an axis is rotated
about its centre to be exactly axis-aligned, which avoids nearly parallel
segments between bridge and metal contours in the mesher. The report lists,
for each bridge, the number of vertices, centre, sides, angle, `IoU` and the
largest distance of a vertex of `P` to `∂R`, and a warning is printed when
`IoU < 0.95`. `--bridge-shape exact` keeps the polygons as drawn.

Each foot is connected to the plane by a vertical pillar, a parallelepiped with
a square base of side `a` (`--pillar-side`, default 30 µm) and height `h`
(`--pillar-height`, default `--bridge-height`). A vertical current produces no
`B_z` and does not couple to the horizontal sheets, the scalar product `J·J'` of
the Neumann kernel being zero. The pillars therefore enter only through their
matrix of self and mutual inductances `L_p`, and the energy receives the term

```
E_p = (1/2) Iᵀ L_p I ,     I_k = ∮ current flowing up pillar k
```

Self-inductance of a straight conductor with a surface current, `R = 0.5823 a`
being the geometric mean distance of the square contour with itself
(Rosa 1908, Grover 1946)

```
L = (µ0 h / 2π) [ ln(2h/R) − 1 + R/h ]  +  µ0 λ h / (4a)
```

Mutual inductance between two parallel pillars whose axes are a distance `d`
apart

```
M = (µ0 h / 2π) [ ln(h/d + √(1 + h²/d²)) − √(1 + d²/h²) + d/h ]
```

For `a = 30 µm`, `h = 200 µm`, `L = 88.9 pH` and `M(100 µm) = 33.0 pH`, hence
`L₁ + L₂ − 2M = 111.8 pH` for the round trip of a bridge. These formulas are
valid for `h ≫ a` and `d ≫ a`, the relative error being `O((a/h)²)` and
`O((a/d)²)`.

The current `I_k` is the weak divergence of `K` summed over the nodes of the
foot (`solver.contact_current_operator`), a linear functional of `K` that
depends only on the seed and on the loop fields. The term `Dᵀ L_p D` has rank
`n_contacts`, it is added to the conjugate-gradient operator and to the loop
block of the preconditioner. `run.py` prints the current and the
self-inductance of each pillar. The test with two artificial pillars recovers
`L₁ + L₂ − 2M` to 10⁻⁴ by identification on a lumped-element model.
`--no-pillars` returns to perfect contacts.

### 2.6 Flux

`Φ` is obtained from Stokes' theorem, `Φ = ∮_{∂S} A·dl`, rather than from
`∫_S B_z dA`. The vector potential of a current sheet is continuous everywhere,
including in the plane of the film, whereas `B_z` is discontinuous there. This
formulation therefore remains valid even for a layer-5 surface lying in the
plane of the metal.

```
A(r) = (µ0/4π) Σ_T A_T K_T / |r − r_T|
```

with quadrature refinement for nearby triangles. Any holes in the layer-5
polygons are handled by the orientation of the interior contours. The normal is
`+ẑ` and the outer contour is traversed counter-clockwise.

---

## 3. Installation

```bash
pip install numpy scipy matplotlib shapely gdstk triangle
```

`triangle` is the Python interface to Shewchuk's mesher. The physical kernel
(`core.py`, `solver.py`) depends only on numpy and scipy, which allows it to be
tested without the GDS chain.

## 4. Usage

Two directory layouts are accepted.

**Flat**, all the `.py` files in the same folder, the scripts are launched
directly.

```bash
cd folder_containing_the_py_files
python make_example_gds.py example.gds
python run.py example.gds --thickness 0.1 --lambda-london 0.09 --seg 1.5 \
              --current 1e-3 --bridge-height 3.0 --out result
python validation.py
```

**As a package**, the `.py` files in a `scdc/` subfolder containing
`__init__.py`, launched from the parent folder.

```
project/
  scdc/
    __init__.py  core.py  solver.py  geometry.py  run.py  make_example_gds.py
  validation.py
```

```bash
cd project
python -m scdc.make_example_gds example.gds
python -m scdc.run example.gds --thickness 0.1 --lambda-london 0.09 --seg 1.5 \
       --current 1e-3 --bridge-height 3.0 --out result
python -m scdc.run example.gds --Lsq 0.14 --seg 1.5 --EJ 20 --nonlinear
python validation.py
```

The error `No module named 'scdc'` means that the `scdc/` folder does not exist,
or that the command is not launched from its parent. In that case use the flat
form.

Outputs (prefix `--out`, here `result`).

- `result_report.md`, a Markdown report starting with the matrix `M` in pH and
  the eigenvalues of `M⁻¹` in mA/Φ0, then parameters (including the ground
  current shares per source), mesh, energy and
  inductance `2E/I²` per source, flux `Φ/Φ0` per surface and per source, pillar
  currents and junction currents.
- `result_current_<k>_<name>.png`, a map of `|K|` with streamlines for each
  layer-3 polygon.
- `result_mutual.csv`, one row per layer-5 surface, one column
  `M_pH_<source>` per layer-3 polygon.
- `result_solution.npz`, the complete mesh, `K` of shape
  `(n_sources, n_tri, 2)`, `M`, fluxes and the names of the sources and
  surfaces.

As a library.

```python
from scdc.run import run
model, sols, res = run("example.gds", thickness=0.1, lambda_L=0.09,
                       seg_len=1.5, current=1e-3)
print(res["M"] * 1e12)            # (n_surfaces, n_sources) matrix in pH
print(res["inv_eigenvalues"])     # eigenvalues of M^-1 in mA/Phi0
J = sols[0].current_density(0.1)  # A/µm² for the first source
```

`run.solve_model(model, ...)` accepts a dictionary built by hand (mesh,
`terminal_sets`, `rings`, …) and allows the loop over sources and the report to
be tested without the GDS chain. The terminal sets can be built with
`solver.build_terminal_sets(asm, source_nodes, ground_nodes, current,
conductances=..., contacts=...)`, which restricts the returns of each source to
the pads of its galvanic cluster (`solver.dc_clusters`), as `build_model`
does.

Useful options.

| option | effect |
|---|---|
| `--seg` | length of the mesh segments, controls cost and accuracy |
| `--bridge-height` | altitude of the airbridges |
| `--bridge-thickness`, `--bridge-lambda` | distinct properties for the bridges |
| `--bridge-feet` | `ends` (default, two extreme feet) or `all` |
| `--bridge-shape` | `rect` (default, closest rectangle to each layer-2 polygon) or `exact` |
| `--bridge-snap-angle` | tilt (deg) below which a bridge rectangle is made axis-aligned, default 0.5 |
| `--pillar-side`, `--pillar-height`, `--pillar-lambda`, `--no-pillars` | bridge pillars |
| `--loop-z` | altitude of the layer-5 surfaces |
| `--ground-resistances` | list, useful only when the `R_k` differ |
| `--scale` | conversion to micrometres if the GDS is not in µm |
| `--Lsq` | sheet inductance of layer 1 in pH, replaces `--thickness --lambda-london` |
| `--EJ`, `--Ic`, `--LJ`, `--junction-file` | parameters of the layer-6 junctions |
| `--nonlinear` | iteration on `L_J(I)` |
| `--show`, `--log` | interactive window, logarithmic scale |
| `--dense` | dense reference assembly |
| `--near-cells` | pFFT precorrection radius, default 4 |
| `--backend` | `numpy`, `mlx`, `torch` or `auto` |
| `--seg-far`, `--fine-radius`, `--seg-max` | mesh grading |

### Graded mesh

On a 10 mm chip, layer 1 can total more than one metre of perimeter, and a
uniform mesh at 1.5 µm would exceed `10^8` triangles. The mesh is therefore
graded. The edges are discretised at `--seg` within `--fine-radius` of the
layer-5 surfaces and of the junctions, at `--seg-far` beyond that (default
`5*seg`), and Triangle lets the elements grow towards the interior up to
`--seg-max` (default `30*seg`). Since the current is concentrated on the edges,
that is where resolution matters. The interior triangles, where `K` is nearly
zero because of screening, can be large.

The pFFT near field adapts to this grading. A pair of triangles is treated
exactly if its distance is smaller than `near_cells` times the maximum of the
grid cell and of the sizes of the two triangles. On a mesh with a ratio of 21
between the largest and the smallest element, the deviation from the dense
assembly is 0.04 % in energy.

### Cost and fast method

By default the computation uses `fast.py`, a method of moments accelerated in
the manner of FastHenry. FastHenry accelerates the matrix-vector product with a
multipole expansion. Here, the films being planar, the precorrected FFT of
Phillips and White is used instead, better suited to sheets. The principle.

1. The sources `A_T K_T` of each triangle are projected onto a regular grid
   through their three quadrature points and bilinear weights, a sparse
   operator `S`. The kernel `1/r` is a convolution on the grid, hence an FFT,
   and the return to the triangles uses `S^T`. The operator `S K S^T` is
   symmetric.
2. For pairs of triangles closer than `near_cells` cells, the grid value is
   subtracted and the exact value added. This correction is a sparse matrix.
3. The energy is minimised by conjugate gradient. The preconditioner is the
   sparse LU factorisation of the local part of the operator (kinetic term and
   near field), the analogue of FastHenry's sparsified matrix.

Cost per iteration `O(N log N)`, memory `O(N)`, typically 10 to 40 iterations.
On a laptop, 30 000 triangles take 7 s and 120 000 triangles 30 s, where the
dense assembly would require 115 GB. The deviation from the dense reference is
of the order of 0.1 % on the energy, adjustable with `--near-cells`.

The option `--dense` keeps the direct assembly of `solver.py` for checks on
small meshes.

Refining near the edges remains what changes the result the most, since `K`
diverges as `1/√s` at a distance `s` from an edge when `Λ ≪ W`.

### Acceleration on Apple Silicon

The profile at 120 000 triangles is, in order of magnitude, 20 s of kernel
construction (gathering of near pairs), 5 s of preconditioner factorisation and
8 s of conjugate gradient dominated by the FFTs. Three levels, by decreasing
confidence.

1. **Multithreaded numpy, the default.** The FFTs go through `scipy.fft` with
   `workers` equal to the number of cores, and the pair computations are split
   into blocks handled by a thread pool, numpy releasing the GIL on these
   operations. The dense products use Accelerate, Apple's BLAS, which is the one
   in the numpy and scipy wheels for macOS arm64 since numpy 2.0. Check with
   `python -c "import numpy; numpy.show_config()"`, the output must contain
   `accelerate`.
2. **CHOLMOD for the preconditioner.** The local matrix is symmetric positive
   definite, a multithreaded Cholesky factorisation replaces SuperLU when
   `scikit-sparse` is present.
   ```bash
   brew install suite-sparse
   pip install scikit-sparse
   ```
3. **Metal GPU, experimental.** `accel.py` contains two backends, MLX
   (`pip install mlx`) and PyTorch on the `mps` device, for the near pairs and
   the FFTs. They work in single precision with coordinates relative to the
   centre of each triangle, which preserves accuracy on chips several
   millimetres across. These backends could not be executed on the machine that
   wrote this code. A self-test compares their results with numpy at startup on
   a sample, exact pairs, grid pairs and FFT, and the computation falls back to
   numpy in case of discrepancy or error. The conjugate-gradient tolerance
   becomes `3e-6` in single precision.

```bash
python -m scdc.run example.gds --Lsq 0.14 --seg 1.5 --backend mlx
SCDC_BACKEND=torch python -m scdc.run ...
```

What to expect. Kernel construction and the FFTs can gain an order of magnitude
on the GPU. The sparse products and the preconditioner solve stay on the CPU, so
the overall gain on a large structure will rather be a factor of 2 to 4, to be
measured with `--backend numpy` against `--backend mlx` on the same GDS. If the
self-test fails, the message indicates which part differs, and I will be able to
correct it with that feedback.

### Visualisation

`run.py` writes a map of `|K|` with streamlines. To explore, the interactive
window opens with `--show`, or later from the saved file.

```bash
python -m scdc.viewer result_solution.npz            # interactive
python -m scdc.viewer result_solution.npz --png zoom.png --log
```

Keys in the window. `i` moves to the current map of the next injection polygon
(`I` for the previous one), the title giving the name of the source and its
rank, `l` toggles linear and logarithmic, `S` (upper case, since `s` is the
matplotlib shortcut for saving) recomputes the streamlines on the visible area
after a zoom, `q` displays arrows, `c` clears. A left click prints `K`, `|K|` and
`J = K/d` at the clicked point. The colour scale is common to all maps, the
injected current being the same for each source. With `--png`, one image per
source is written (`--map k` to write only one).

---

## 5. Validation

`validation.py` compares the code with analytical results.

| test | result |
|---|---|
| mutual inductance annular sheet / coaxial loop, Maxwell formula | deviation 0.001 % to 0.006 % |
| current conservation across a cut | 1.000000 mA for 1 mA imposed |
| profile `K(y)` of a track with `Λ → 0` compared with `(I/π)/√((W/2)² − y²)` | RMS deviation 1.6 % |
| kinetic inductance of a track, `Λ` large, compared with `µ0 Λ ℓ/W` | deviation 2 % |
| imposed fluxoid `n = 1` in a ring, consistency `2E = Φ0 I` | exact to 10⁻⁶ |
| junction patch `L_J = 0.5` and `5 nH` in a track, `L_tot − L_track` | deviation 2×10⁻⁵ |
| pFFT solver against dense assembly | 0.1 % on `E`, 10⁻³ on `K` |
| ring 6–10 µm, `L = Φ0/I = 23.1 pH` against `µ0 R [ln(8R/a) − 2] ≈ 21.7 pH` | consistent |

```bash
python validation.py
```

---

## 6. Limitations, and why agreement with experiment stays imperfect

This code solves the 2D London model properly. It does not address the usual
causes of discrepancy between simulation and measurement, which are worth
knowing.

1. **Sheet model.** The formulation assumes `d` small compared with the lateral
   dimensions. For `d ≳ λ` the current distribution across the thickness is no
   longer uniform, `Λ = λ coth(d/λ)` is only a first-order correction, and a
   volume extraction of the FastHenry or TetraHenry type is more faithful.
2. **Effective `λ`.** The relevant value is `λ_eff = λ_L √(1 + ξ0/ℓ)` for a dirty
   film, and depends on the deposition, the temperature and the field. In
   practice this is the first source of discrepancy on the kinetic inductance,
   often 10 % to 30 %.
3. **Edges and real geometry.** The GDS is not the fabricated circuit. An
   under-etch or over-etch of a few tens of nanometres strongly modifies the
   edge current as soon as `Λ ≪ W`, and therefore the inductance.
4. **Airbridges.** They are modelled as planar sheets at `z = h` connected to
   layer 1 by straight pillars with a lumped inductance (§2.5). The real
   curvature of the bridge, the contact resistance and the current distribution
   over the pillar cross-section near the feet are neglected.
5. **Injection contacts.** The vertical current profile in the disks of layers 3
   and 4 is imposed by the auxiliary resistive problem and is not relaxed. This
   affects the current crowding inside the pad, and very little the far field.
   A discussion of this point can be found in Khapaev and Kupriyanov
   (arXiv:1412.3231).
6. **Trapped vortices.** A trapped Abrikosov vortex locally modifies `K` and adds
   one flux quantum. The code accepts `n_k ≠ 0` for the holes of the layout, not
   for vortices placed arbitrarily in the metal.
7. **Nonlinearity.** `Λ(J)` grows when `J` approaches `J_c`, which is not taken
   into account. At 1 mA in a track a few microns wide, the correction generally
   stays below one percent, but it becomes dominant in constrictions.
8. **Chip-edge effects.** A finite ground plane, the package, and couplings
   outside the layout are not described.

---

## 7. References

Two-dimensional London model and screening currents.

- J. Pearl, *Current distribution in superconducting films carrying quantized
  fluxoids*, Appl. Phys. Lett. **5**, 65 (1964).
  <https://doi.org/10.1063/1.1754056>
- E. H. Brandt and M. Indenbom, *Type-II-superconductor strip with current in a
  perpendicular magnetic field*, Phys. Rev. B **48**, 12893 (1993).
  <https://doi.org/10.1103/PhysRevB.48.12893>
- R. Meservey and P. M. Tedrow, *Measurements of the kinetic inductance of
  superconducting linear structures*, J. Appl. Phys. **40**, 2028 (1969).
  <https://doi.org/10.1063/1.1657905>
- J. R. Clem and K. K. Berggren, *Geometry-dependent critical currents in
  superconducting nanocircuits*, Phys. Rev. B **84**, 174510 (2011).
  <https://arxiv.org/abs/1109.4881>

Inductance extraction and existing codes.

- L. Bishop-Van Horn and K. A. Moler, *SuperScreen*, Comput. Phys. Commun.
  **280**, 108464 (2022). <https://arxiv.org/abs/2203.13388>
- M. M. Khapaev, *Inductance extraction of multilayer finite-thickness
  superconductor circuits*, IEEE Trans. Microw. Theory Tech. **49**, 217 (2001).
- M. M. Khapaev, A. Yu. Kidiyarova-Shevchenko, P. Magnelind, M. Yu. Kupriyanov,
  *3D-MLSI*, IEEE Trans. Appl. Supercond. **11**, 1090 (2001).
  <https://ieeexplore.ieee.org/document/919537/>
- M. M. Khapaev and M. Yu. Kupriyanov, *Inductance extraction of superconductor
  structures with internal current sources*, <https://arxiv.org/abs/1412.3231>
- M. Kamon, M. J. Tsuk, J. K. White, *FastHenry, a multipole-accelerated 3-D
  inductance extraction program*, IEEE Trans. Microw. Theory Tech. **42**,
  1750 (1994). <https://doi.org/10.1109/22.310584>
- J. R. Phillips and J. K. White, *A precorrected-FFT method for electrostatic
  analysis of complicated 3-D structures*, IEEE Trans. Comput.-Aided Des.
  Integr. Circuits Syst. **16**, 1059 (1997).
  <https://doi.org/10.1109/43.662670>
