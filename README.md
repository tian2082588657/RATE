# TeRed + RATE — replication package

Code and result artefacts for a measurement study of **provenance-graph
reduction** and **Reduction-Aware Topology Encoding (RATE)**.

The study asks what graph reduction does to the structural signals a
provenance-based detector consumes, and what it costs. It is evaluated on
**DARPA TC E5 CADETS** (122 Avro audit windows), with **StreamSpot** used for
the end-to-end pipeline smoke test. Four reduction operators are implemented
(`identity`, `cpr`, `nodemerge`, `tered` — the TeRed template collapse), and
the topological channels are `none`, `dual_naive` (edge-count degrees),
`rate_single` (one merged channel) and `rate` (dual-channel, mass-sum
degrees).

This repository contains the source code, the run scripts and the aggregated
result files that back the paper's tables. The large intermediate data cache
is **not** included (see [Data and cache](#data-and-cache)).

---

## Layout

| Path | Contents |
|---|---|
| `rate_core.py` | Canonical graph structure, fingerprinting, cache helpers |
| `run_pipeline.py` | End-to-end pipeline (StreamSpot entry point) |
| `adapters/` | Dataset loaders: `darpa_tc.py` (E5 Avro), `streamspot.py` |
| `reduction/` | `base.py` (identity + invariants), `cpr.py`, `nodemerge.py`, `tered.py` (template collapse), `template_mining.py` |
| `features/` | `rate.py` (topology encodings + semantic one-hot), `tape.py`, `v4extras.py` (three structural columns) |
| `models/` | `detector.py` (BenignEnsemble, 5th-percentile cosine radius), `detector_graphsage.py`, `torch_net.py` |
| `train/` | Optional torch training path |
| `eval/` | `metrics.py` — node-level and alert-level metrics |
| `scripts/` | Main experiment drivers (e3–e7) |
| `scripts_ablation/` | Ablation, cost, baseline and protocol-matched scripts (e6d–e10) |
| `tests/` | Pipeline invariants (degree-mass identity, node-map sanity, shared context) |
| `configs/` | Example configuration |
| `results/` | Aggregated CSVs, protocol-matched baseline outputs, plots, short reports |
| `v4_validation_results/` | Per-partition validation outputs for the v4 feature set |
| `*.sh` | Batch launchers used for the full runs |

`results/v3_final/` holds the per-partition aggregates that the paper's main
tables are built from (`e6_alert_full_v3.csv`, `e6d_ablation_v3.csv`,
`e6d_ratestar_v3.csv`, `e7_origunit.csv`), plus the generated LaTeX table body
`paper_tables_v3.tex`. `scripts_ablation/gen_paper_tex.py` regenerates that
file from the CSV directory.

## Environment

```bash
pip install -r requirements.txt          # numpy, networkx — enough for the full pipeline
# optional, only for the GraphSAGE baseline / torch training path:
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Python 3.10+ is assumed. The detection pipeline itself is NumPy/NetworkX only;
`torch` is required solely by `models/detector_graphsage.py` and `train/`.

## Data and cache

Two datasets are needed for the full study. Neither is redistributed here;
both are publicly available from their original providers.

| Dataset | Expected location | Format |
|---|---|---|
| DARPA TC E5 CADETS | `dataset/darpae5/cadets/` | `ta1-cadets-1-e5-official-2.bin.<N>.gz`, Avro container, 122 windows |
| StreamSpot | `dataset/streamspot/all.tar.gz` | scenes / threatrace splits |
| Ground truth | `ground_truth/` | PIDSMaker node-level CSVs (per attack window) |

The scripts default to absolute author paths such as
`/TeRed+RATE/dataset/darpae5/cadets` and `/TeRed+RATE/code`. **Override them
with `--data-dir` (and adjust the `cd` lines in the `*.sh` launchers)** before
running anywhere else.

`cache/` is **excluded from version control** (see `.gitignore`). It holds the
parsed graphs (`cache/darpa/parsed/`, `cache/streamspot/parsed/`), the mined
template libraries (`cache/darpa/e5_templates/`) and the reduced graphs
(`cache/darpa/e5_f1_reduce/`) — roughly 850 MB in total, with individual
parsed JSONL files up to ~290 MB, which exceeds GitHub's file-size limit. It is
regenerated automatically on the first run; the parsing step takes about
5–8 minutes per large window (287k nodes / 9.33M edges for `bin.116`).

## Quick start (StreamSpot smoke test)

```bash
# TeRed reduction + RATE dual-channel + benign one-class detection
python run_pipeline.py --operator tered --encoding rate --gids "0-6,500-501"

# same reduced graph, edge-count degrees instead of mass sums
python run_pipeline.py --operator tered --encoding dual_naive --gids "0-6,500-501"

# no reduction
python run_pipeline.py --operator identity --encoding rate --gids "0-6,500-501"
```

Each run appends one row to `results/` and caches intermediates under `cache/`.

## Main DARPA pipeline

The canonical configuration used throughout the paper is:

```
--semantic onehot     # omit one-hot type block if set to anything else
--tape-base 10000
--v4-scale raw
--share-k -1          # TeRed: keep all matched instances (no per-template cap)
--detector benign
```

and the alert-aggregation hyper-parameters are `--top-k 100 --bfs-q 75
--min-cluster 3 --max-cluster 200 --max-alerts 20`. These are the defaults of
the drivers, so an explicit command is only needed to change them.

Stage order:

1. **Node-level, per attack window** — `scripts/e5_f1_eval.py`
   (or `run_v4_full_k-1.sh` for the five-window batch).
2. **Alert-level, shared-context pipeline** — `scripts/e6_alert_eval.py`,
   with the ablation replay in `scripts_ablation/e6d_ablation_replay.py`
   (pass `--configs 'TeRed+RATE*'` explicitly to get the fourth configuration;
   the default filter drops it).
3. **Original-unit evaluation** — `scripts/e7_origunit_eval.py` maps reduced
   nodes back to original entities through `ReductionResult.node_map`, so that
   coverage and recall are measured on a unit that is comparable across
   configurations.
4. **Baselines** — `scripts_ablation/e9_baseline_protocol_matched.py` runs the
   published detectors under the same protocol; `e10_paper_method_matched.py`
   runs this paper's four configurations under the identical pipeline.

Protocol-matched baseline outputs are in `results/ablation_full/` and
`results/{kairos,flash,threatrace}_full38/`; the accompanying report is
`results/ablation_full/PROTOCOL_MATCHED_2026-09-13.md`.

## Notes on cost

The reference reduction implementation is an **offline batch** stage, not a
streaming filter:

- On the largest windows it is slower than real time. Measured per-window
  reduction time is 1.6–6.4× the window's own time span, and the largest
  partition did not finish within an 8-hour budget.
- The reported saving (≈20% fewer edges, ≈23% less scoring time) applies to
  the **scoring stage only**; reduction itself costs more than scoring with
  this implementation.
- `scripts_ablation/e6h3_index_headroom.py` bounds what a signature index can
  buy: it prunes about 61% of candidate comparisons, for an upper bound of
  roughly 2.6× on matching speed.

## Templates and licence

A `LICENSE` file and a `CITATION.cff` are not yet included — the authors
intend to add both before the repository is cited in a publication. Until a
licence is added, all rights are reserved by default.

## Contact

Issues and questions about reproduction are welcome through the GitHub issue
tracker.
