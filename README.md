# JMS: Joint Minimum Support focusing inversion of gravity and magnetic data

Code accompanying the article

> Z. Azimi and A. Afshar, *Joint Minimum Support: simultaneous focusing inversion of gravity and magnetic data with a shared-support stabilizer, a fair benchmark, and application to the Sangan iron-ore deposit, Iran* (submitted to *Computers & Geosciences*).

The repository contains a self-contained Python implementation of 3D gravity and magnetic inversion on a topography-draped prism mesh, with six stabilizers that are compared under identical conditions:

| Name | Stabilizer | Reference |
|---|---|---|
| `L2` | smooth (minimum L2 norm with depth weighting) | Li and Oldenburg (1996) |
| `PTSS` | power-type structural self-constrained joint inversion | Ming et al. (2024) |
| `L1` | L1 norm | Farquharson (2008) |
| `MS` | minimum support | Last and Kubik (1983); Portniaguine and Zhdanov (1999) |
| `MGS` | minimum gradient support | Portniaguine and Zhdanov (1999) |
| **`JMS`** | **Joint Minimum Support (proposed)**: density and magnetization are solved simultaneously and the minimum-support penalty of each property is relaxed where the other property is non-zero | this work |

All methods use the same solver (bound-constrained L-BFGS-B with beta continuation), depth weighting, bounds, error model, base-level estimation and stopping rule (the same chi-squared misfit), so that only the stabilizer differs.

## Installation

Python 3.10 or newer is required. In the repository folder:

```
pip install -r requirements.txt
```

The code was tested with Python 3.13, numpy 2.5, scipy 1.18, pandas 3.0 and matplotlib 3.11 on Linux, and runs on Windows and macOS as well (pure Python, no compiled extensions).

## Quick start (runs in seconds to a minute)

```
python examples/minimal_example.py                 # one block, smooth L2 versus JMS (a few seconds)
python examples/quick_test_published_models.py     # all six methods on the two test models of Ming et al. (2024), about 40 s
```

The second script writes `outputs/table_paper_tests_dice.csv`, which should match `results/tables/table_paper_tests_dice.csv`.

## Reproducing all results of the article

Open `Focusing_Joint_Inversion.ipynb` in Jupyter and run all cells, or run the equivalent script:

```
python run_all.py
```

**Data availability.** The synthetic benchmarking data and code are fully available here: the notebook builds every synthetic model on the public station geometry `data/survey_geometry.csv` and reproduces all synthetic tables and figures of the article exactly. The field data from the Sangan iron-ore deposit and the borehole logs are available from the corresponding author upon reasonable request, owing to industrial and institutional data restrictions. Without them, Part C (field data) is skipped automatically; with them in `data/` (see `data/README.md`), the notebook reproduces the field-data results as well.

The notebook computes the kernel matrices of the 25 m draped mesh (once; a few minutes and about 3 GB of memory, cached in `outputs/`), runs every inversion (several hours on a laptop the first time; every result is cached in `outputs/results_*.pkl`, so later runs take minutes), and writes every table (`outputs/table_*.csv`, `outputs/tables.xlsx`) and figure (`outputs/*.png`). The tables in `results/tables/` are the reference results of the article; a complete run reproduces them exactly (the random models use a fixed seed).

Notebook sections:

| Part | Content |
|---|---|
| 1-5 | settings, library, data and error estimation, draped mesh and kernels, common tools |
| A | training model: selection of kappa (JMS) and lambda (PTSS) with a pre-declared criterion |
| B | held-out test model (main accuracy result) |
| B2 | spurious density in light magnetic bodies and the one-way coupling variant |
| A2 | sensitivity of JMS to eps |
| D | ten random models and paired Wilcoxon tests |
| C | Sangan field data: data fit, sections, depth slices, borehole validation, depth to the top of the ore (requires the field data) |

## Repository contents

```
jms_lib.py                          library: mesh, kernels, solver, stabilizers, JMS, PTSS, metrics
Focusing_Joint_Inversion.ipynb      notebook that reproduces the article
run_all.py                          the same code as a plain script
build_notebook.py                   generates the notebook and run_all.py from one source
examples/minimal_example.py         minimal use of the library
examples/quick_test_published_models.py   test on the published synthetic models
data/survey_geometry.csv            station geometry of the Sangan survey (no data values)
data/README.md                      data formats and data availability
docs/USER_GUIDE.md                  inputs, outputs, options and expected behaviour
docs/TUTORIAL.md                    how to apply the methods to your own data
results/tables/                     reference tables of the article
```

## Documentation

* [User guide](docs/USER_GUIDE.md): functions, inputs, outputs, options, run times and expected behaviour.
* [Tutorial](docs/TUTORIAL.md): step-by-step use on synthetic data and on your own survey.

## License and citation

The code is released under the MIT license (see `LICENSE`). If you use it, please cite the article above (see `CITATION.cff`).

## Contact

Ahmad Afshar (corresponding author), a.afshar@aut.ac.ir
