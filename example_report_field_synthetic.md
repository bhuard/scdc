# scdc report: synthetic_washer.gds

Generated 2026-10-03 22:35. Injected current I = 1 mA in each layer-3 polygon in turn, returned through the layer-4 polygons. Applied field B_ext = 10 uT along +z, computed separately without injected current. All lengths in um.

## Program and invocation

- Version: scdc 1.0.0, source fingerprint 9b37d3b7f51a
- Environment: Python 3.12.3, numpy 2.4.4, scipy 1.17.1, Linux-6.18.44-fc-v64-x86_64-with-glibc2.39
- Working directory: `/home/claude/pkg`

Command typed in the terminal:

```bash
python -m scdc.run synthetic_washer.gds --seg 1 --Lsq 0.14 --Bext 10 --out ex/example_field
```

Equivalent command with every option written explicitly, defaults included (the complete list, with the origin of each value, is in the appendix at the end of the report):

```bash
python -m scdc.run synthetic_washer.gds --Lsq 0.14 --layer-junction 6 --seg 1.0 --current 0.001 --Bext 10.0 --bridge-height 3.0 --bridge-feet ends --pillar-side 30.0 --loop-z 0.0 --layer-metal 1 --layer-bridge 2 --layer-source 3 --layer-ground 4 --layer-loop 5 --scale 1.0 --out ex/example_field --backend auto --near-cells 4 --bridge-shape rect --bridge-snap-angle 0.5
```

Options not set (unused, or value derived by the program as stated in the appendix): `--thickness`, `--lambda-london`, `--EJ`, `--Ic`, `--LJ`, `--junction-file`, `--bridge-Lsq`, `--seg-far`, `--fine-radius`, `--seg-max`, `--bridge-thickness`, `--bridge-lambda`, `--pillar-height`, `--pillar-lambda`, `--ground-resistances`, `--cell`, `--near-hmax`, `--grid`, `--min-seg`, `--pad-tol`.

Flags not given: `--nonlinear`, `--no-pillars`, `--no-plot`, `--show`, `--log`, `--dense`, `--mesh-only`.

## Mutual-inductance matrix M = Phi / I (pH)

Rows are the flux surfaces of layer 5, columns the injection polygons of layer 3. M[i, j] is the flux through surface i when the current I is injected in polygon j alone, divided by I.

| surface \ source | IN |
|---|---|
| HOLE | -7.36233 |
| FILM | -0.0118316 |
| OUTSIDE | 0.136397 |
| ALL | -4.98035 |

## Eigenvalues of M^-1 (mA / Phi0)

Phi0 = h / 2e = 2.067833848e-15 Wb. Each eigenvalue lambda_k of M^-1 (A/Wb) is expressed as lambda_k Phi0 in mA, the current along the k-th eigen-direction producing one flux quantum through the corresponding combination of surfaces. Sorted by increasing modulus.

Not computed. M is 4 x 1, not square, so M^-1 is not defined.

Singular values of the pseudo-inverse M^+ (mA/Phi0), which coincide with the moduli of the eigenvalues of M^-1 when M is square and normal:

| k | 1/sigma_k (mA/Phi0) |
|---|---|
| 0 | 0.232611 |

## Applied perpendicular field B_ext

Uniform field B_ext = 10 uT = 1e-05 T along +z, perpendicular to the plane of the GDS, applied after a zero-field cooldown (fluxoid n = 0 in every hole), with no injected current. The layer-3 and layer-4 pads carry no current in this computation, a DC current through their resistances to ground cannot be sustained.

For each layer-5 surface S of net area A (holes removed) and normal +z, Phi = Phi_applied + Phi_screening with Phi_applied = A B_ext and Phi_screening = Oint_dS A_K . dl the flux of the currents of the film. The flux concentration is C = Phi / (A B_ext). C = 1 without superconductor, C < 1 for a surface covered by screening metal, C > 1 where the flux expelled from the metal is pushed, typically slots and gaps open to the outside. A hole enclosed by a closed loop of metal keeps n = 0 and therefore C close to 0. Surfaces evaluated at z = 0 um.

| surface | x_c | y_c | area A (um^2) | A B_ext / Phi0 | Phi / Phi0 | concentration Phi/(A B_ext) |
|---|---|---|---|---|---|---|
| HOLE | 0.000 | 0.000 | 144 | 0.696381 | 1.45535 | 2.08988 |
| FILM | 13.000 | 0.000 | 36 | 0.174095 | 0.0086356 | 0.0496027 |
| OUTSIDE | 30.000 | 0.000 | 100 | 0.483598 | 0.528737 | 1.09334 |
| ALL | 0.000 | 0.000 | 3600 | 17.4095 | 16.1447 | 0.927351 |

- Self energy of the screening currents E = 5.564981e-19 J
- Magnetic moment of the film m_z = (1/2) Int (r x K).z dA = Int K.A_ext dA / B_ext = -1.112996e-13 A.m^2 (negative for a diamagnetic response)
- Consistency of the linear response, B_ext m_z / (-2 E) = 1.000000 (1 expected without nonlinear junctions)
- Solver: 17 CG iterations, relative residual 8.71e-09

## Model parameters

- GDS file: `synthetic_washer.gds`
- Layer-1 sheet inductance: 0.14 pH/square
- Effective penetration depth Lambda = 0.1119 um, L_square = mu0 Lambda = 0.1406 pH/square
- Injected current I = 1 mA
- Ground current shares (layer 4), per source. A pad not connected in DC to the source carries no return current:
  - IN: [1.0]
- Applied field B_ext = 10 uT along +z, vector potential A_ext = (B_ext/2) z x (r - r_c) with r_c = (0, 0) um (the result does not depend on r_c)
- Flux surfaces evaluated at z = 0 um
- Edge segment length: 1 um near the region of interest
- Solver: pFFT + conjugate gradient, near_cells = 4

## Mesh

- Sheets: 1 (metal)
- Triangles: 2856, nodes: 1545
- Triangle size sqrt(area): min 0.707, median 0.707, max 0.707 um

## Injection polygons (layer 3)

| source | x_c | y_c | area (um^2) | mesh nodes | energy E (J) | L = 2E/I^2 (pH) | CG iterations | residual |
|---|---|---|---|---|---|---|---|---|
| IN | -19.500 | 0.000 | 40 | 82 | 8.479169e-18 | 16.9583 | 18 | 2.25e-09 |

## Flux surfaces (layer 5)

| surface | x_c | y_c | area (um^2) | Phi/Phi0 (IN) | Phi/Phi0 (B_ext) |
|---|---|---|---|---|---|
| HOLE | 0.000 | 0.000 | 144 | -3.56041 | 1.45535 |
| FILM | 13.000 | 0.000 | 36 | -0.00572172 | 0.0086356 |
| OUTSIDE | 30.000 | 0.000 | 100 | 0.0659615 | 0.528737 |
| ALL | 0.000 | 0.000 | 3600 | -2.40849 | 16.1447 |

## Output files

- `ex/example_field_solution.npz`
- `ex/example_field_mutual.csv`
- `ex/example_field_field.csv`
- `ex/example_field_current_0_IN.png`
- `ex/example_field_current_Bext.png`
- `ex/example_field_report.md`

## Appendix. Value of every option

| option | value | origin |
|---|---|---|
| `gds` | synthetic_washer.gds | command line |
| `--thickness` | not used | default |
| `--lambda-london` | not used | default |
| `--Lsq` | 0.14 | command line |
| `--layer-junction` | 6 | default |
| `--EJ` | not set | default |
| `--Ic` | not set | default |
| `--LJ` | not set | default |
| `--junction-file` | not set | default |
| `--nonlinear` | off | default |
| `--bridge-Lsq` | not set, bridges as layer 1 unless --bridge-thickness or --bridge-lambda | default |
| `--seg` | 1.0 | command line |
| `--seg-far` | not set, 5*seg = 5 | default |
| `--fine-radius` | not set, 200*seg/1.5 = 133.333 | default |
| `--seg-max` | not set, 30*seg = 30 | default |
| `--current` | 0.001 | default |
| `--Bext` | 10.0 | command line |
| `--bridge-height` | 3.0 | default |
| `--bridge-thickness` | not set, = --thickness | default |
| `--bridge-lambda` | not set, = --lambda-london | default |
| `--bridge-feet` | ends | default |
| `--pillar-side` | 30.0 | default |
| `--pillar-height` | not set, = --bridge-height = 3 | default |
| `--pillar-lambda` | not set, = lambda of the bridges | default |
| `--no-pillars` | off | default |
| `--loop-z` | 0.0 | default |
| `--ground-resistances` | not set, 0.01 Ohm per layer-4 polygon | default |
| `--layer-metal` | 1 | default |
| `--layer-bridge` | 2 | default |
| `--layer-source` | 3 | default |
| `--layer-ground` | 4 | default |
| `--layer-loop` | 5 | default |
| `--cell` | not set, first top-level cell | default |
| `--scale` | 1.0 | default |
| `--out` | ex/example_field | command line |
| `--no-plot` | off | default |
| `--show` | off | default |
| `--log` | off | default |
| `--dense` | off | default |
| `--backend` | auto | default |
| `--near-cells` | 4 | default |
| `--near-hmax` | not set, no cap | default |
| `--grid` | not set, median triangle size | default |
| `--mesh-only` | off | default |
| `--min-seg` | not set, contours as drawn | default |
| `--pad-tol` | not set, pads as drawn | default |
| `--bridge-shape` | rect | default |
| `--bridge-snap-angle` | 0.5 | default |
