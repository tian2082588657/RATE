# Protocol-Matched Baseline Reproduction (E9 + E10) — 2026-09-13

This report closes the *"protocol-matched published baseline missing"* gap
that the paper carried through three review rounds (DeepSeek
Q1/Q3/Q4 critique). It does so by running every detector on the **same**
test partitions, with the **same** alert-aggregation pipeline, the **same**
fixed budget and the **same** node-level ground truth (`node2attacks`).

## 1. Setup

| item | value |
|---|---|
| Dataset | DARPA TC E5 CADETS, PIDSMaker 38-bin reconstruction |
| Test windows | 154 (05-16: 83 + 05-17: 71), each ≈15 min |
| Test ground truth | `node2attacks` from Kairos epoch-3 score pkl (87 positive events across 33 windows; identical across all three baseline pkl files) |
| Train windows | 60 (05-08: 31, 05-09: 16, 05-11: 0*, 05-12: 13) — benign period, used by paper method only; published detectors ship their own training |
| Alert pipeline | top-k=100 seeds → threshold=P75 → BFS expansion (min_c=3, max_c=200) → aggregate (0.7·max+0.3·mean) → top-max_alerts=20 |
| Score files | Kairos `/scores_model_epoch_3.pkl`; FLASH `thr_flash_loss/pr_dir/scores_model_epoch_3.pkl`; ThreaTrace `/scores_model_epoch_3.pkl` |

\* 05-11 directory has no windows in this PIDSMaker build.

## 2. Paper-method scoring (E10)

Paper method is the same BenignEnsemble one-class cosine detector used
in the main paper, fitted on the benign windows of the same test capture
so that the detector sees the same training data the published detectors
were originally trained on (PIDSMaker publishes their checkpoints). The
graph is the PIDSMaker nx.MultiDiGraph re-emitted as a `CanonicalGraph`
with `μ ≡ 1` (no collapse reduction), so `rate ≡ dual_naive` per the
INV-1 invariant of `features/rate`. The four encodings are the four
topology-coding switch states of the main paper (Section 5.1) without
the reduction operator:

| internal switch | public name | meaning |
|---|---|---|
| `none` | **identity** | semantic one-hot only (paper's "TeRed+identity" without reduction) |
| `dual_naive` | **count** | semantic + per-direction count tape (paper's "TeRed+count" without reduction) |
| `rate` | **rate** | semantic + per-direction μ tape (paper's "TeRed+RATE" without reduction) — collapses to count when μ=1 |
| `rate_ratio` | **rate_ratio** | semantic + count tape + μ-ratio tape (paper's A5b rescue encoding) |

Fitting cost (60 windows, ≈1.17 M nodes total): 16/18/18/20 s per encoding
in order. Test cost (154 windows, 4.71 M nodes): 73 s for all four
encodings after graphs are loaded once.

## 3. Results (E9 ∪ E10)

All seven rows are on the same 154 test windows, same GT, same alert
pipeline, same budget. The only thing that varies is the source of the
per-node score.

| Detector (epoch 3) | source | alerts | hit | P | cov | F1 | alerts/win |
|---|---|---:|---:|---:|---:|---:|---:|
| Kairos              | E9 | 2606 | 23 | **0.0088** | **0.4253** | **0.0173** | 16.92 |
| FLASH               | E9 | 1845 | 11 | 0.0060 | 0.2299 | 0.0116 | 11.98 |
| ThreaTrace          | E9 | 1871 |  0 | 0.0000 | 0.0000 | 0.0000 | 12.15 |
| Paper — identity    | E10 | 2300 |  9 | 0.0039 | 0.1954 | 0.0077 | 14.94 |
| Paper — count       | E10 | 2757 | 13 | 0.0047 | 0.2759 | 0.0093 | 17.90 |
| Paper — rate        | E10 | 2757 | 13 | 0.0047 | 0.2759 | 0.0093 | 17.90 |
| Paper — rate_ratio  | E10 | 2728 | 13 | **0.0048** | **0.2989** | **0.0094** | 17.71 |

Robustness check (Kairos epoch 11, FLASH epoch 11, ThreaTrace deterministic):
Kairos P=0.0086 cov=0.4138 F1=0.0169; FLASH P=0.0052 cov=0.2184 F1=0.0102;
ThreaTrace P=0.0000 cov=0.0000 F1=0.0000 — ranking identical to epoch 3.

## 4. Reading the table

**(a) No detector exceeds F1=0.02.** The ceiling is the alert-aggregation
pipeline, not the node scorer. The BFS expansion threshold (P75) and the
hard budget of 20 alerts per window cap what any node-level ranker can
deliver under this GT definition. This is the same ceiling the paper's
own ablation A1 measures: removing BFS expansion drops alert F1 from
0.483 to 0.046. Every row of this table sits below 0.02 because every
row uses the strict pipeline.

**(b) Kairos is the strongest scorer of the seven.** It produces the
highest P (0.0088) and the highest cov (0.4253), about 1.8× the paper
method on P and 1.4× on cov. This is consistent with Kairos being the
best-trained published detector on this capture (TGN with a published
checkpoint), and is consistent with the paper's positioning that the
*scorer* quality is not where it claims novelty — its novelty is in the
reduction-and-encoding side, which is *not* what's being tested here.

**(c) The paper's `rate_ratio` A5b encoding still wins among paper-method
rows.** +22 % F1 (0.0094 vs 0.0077), +53 % cov (0.2989 vs 0.1954) over
identity — the same direction and a similar magnitude as the seven-file
A5b measurement in the main paper. This is a within-system cross-check,
not a published-baseline comparison.

**(d) rate ≡ count.** Both yield identical numbers to four decimals:
same alerts (2757), same hits (13), same cov (24). This is the expected
behaviour under μ ≡ 1 and is the positive control for INV-1
(`features/rate` invariant). It also shows that the μ signal only
manifests after reduction; without reduction it is literally absent.

**(e) ThreaTrace emits no hits.** Its node ranks are usable for AUC
(0.36 on this capture) but its top-decile scores are below the P75
threshold inside the alert window, so every cluster it produces has zero
positives. This is a threshold-mismatch effect, not a ranking collapse:
under its native threshold the detector flags 77 % of nodes (Section
4.6.1 of the paper), but 77 % is also the false-positive regime that
the strict pipeline is designed to suppress.

## 5. What this changes in the paper

- The "joint-first missing experiment" of Discussion (Section 7) listed
  (i) a clean false-alarm rate and (ii) a protocol-matched published
  baseline. This report closes (ii) for the three detectors whose
  score-pkl checkpoints were available inside PIDSMaker. (i) remains
  open — the benign-replay capture is not available to us.
- The Discussion paragraph "Single dataset, and protocol-mismatched
  baselines" must be reworded from "we treat them as a reproduction
  note and make no numerical comparison" to "we treat them as a
  reproduction note for their native protocols and *additionally* as
  numerical rows of Table X under a single protocol" — with Table X
  being this protocol-matched table.
- The Section-1 sentence claiming "no protocol-matched comparison
  against a published system is included" is now false. It is replaced
  by the protocol-matched row set in Table X.
- The abstract and Section 7 positioning paragraph need a half-sentence
  change: the paper is still a measurement study of reduction, but its
  encoding claim is now also cross-checked against three published
  detectors on a common protocol (and Kairos outranks the paper
  method, as it should, while `rate_ratio > count = rate > identity`
  is preserved within the paper method).

## 6. What this does NOT change

- The paper's headline F1@5/10/20 numbers (Table `tab:main`) come from
  the paper's own partition on the paper's own reduced graph. They are
  not changed by E9/E10, because E9/E10 use PIDSMaker's own graph with
  no reduction.
- The reduction story (cost down, detection flat) is unaffected; the
  protocol-matched table deliberately uses no reduction, so it does not
  test the reduction claim at all.
- The "false-alarm rate" gap is still open. The protocol-matched table
  shows alert precision at a fixed budget; it does not show a false
  alarm per day.

## 7. Reproducibility

Scripts:
- `scripts_ablation/e9_baseline_protocol_matched.py` — three published
  detectors, single graph pipeline, GT = `node2attacks` from Kairos
  epoch-3 pkl.
- `scripts_ablation/e10_paper_method_matched.py` — paper method,
  four encodings, BenignEnsemble fit on 05-08/09/11/12 benign windows.

Outputs (in `results/ablation_full/`):
- `e9_smoke.json` — Kairos/FLASH/ThreaTrace epoch 3, 154 windows.
- `e9_baseline_matched_e11.json` — same systems, epoch 11.
- `e10_paper_matched.json` — paper method, four encodings, 154 windows.

Server commit: `~/paper_code/` carries the deployed scripts and the
minimal paper modules (`rate_core.py`, `features/{__init__,tape,rate}.py`,
`models/{__init__,detector}.py`).

---

## Addendum — 2026-09-13 16:50 (`问题-0913-2.md`)

DeepSeek's second review of the protocol-matched table identified six hard issues plus several recommendations. All eleven are addressed in the paper; the script and JSON outputs are unchanged.

**Fix #1 (naming conflict).** The four paper rows in Table 15 originally used the same names as Table 4's three configurations (`identity`, `count`, `rate`, `rate_ratio`) but meant different things. Renamed to `no_topo`, `count`, `μ`, `count+ratio`. Caption explicitly says that Table 15's `no_topo` is *no topology encoding, only semantic one-hot* and is **not** the same as Table 4's `identity` (dual-channel count).

**Fix #2 (bold rule).** Caption states the rule explicitly: "Bold marks the best row within each block (published detectors vs. paper-method configurations), not the global maximum."

**Fix #3 (ref29 mismatch).** `ref29` is "High Fidelity Data Reduction" (Xu et al., CCS 2016), not CPR. The sentence that used it as a CPR citation was rewritten to match `ref29`'s actual content. *(Subsequently tightened in the 第三轮补救 — see below.)*

**Fix #4 (THREATRACE gap).** Added a paragraph after Table 15 explaining why THREATRACE's Table 14 87.1% recall and Table 15 zero-hit row are not in conflict: the native row reaches 87.1% only by flagging 77% of nodes (≈2.6 M positives per window); the protocol-matched pipeline caps selection at 20 alerts per window with a P75 BFS threshold, and THREATRACE's per-node scores do not surface any positive under that selection.

**Fix #5 (Limitations title).** Renamed from "Single dataset, and protocol-mismatched baselines" to "Single dataset, and the scope of the protocol-matched comparison".

**Fix #6 (cross-table comparability).** Caption states explicitly that the numbers are not comparable to Table 4 (different partition, different positive definition, different graph).

**Fix #7 (μ≡1 disclosure).** Caption and Abstract both state that Table 15 uses the unreduced PIDSMaker graph and therefore does not retest the reduction claim.

**Fix #8 (Abstract redundancy).** Merged the two false-alarm sentences into one.

**Fix #9 (positive definition).** Caption states the positive is an "event" (87 in 33 windows), distinct from Table 4's "entity-partition pair" of 136.

**Fix #10 (INV-1 jargon).** Replaced with reader-friendly explanation: "invariant INV-1 of `features/rate`, `rate` = Σμ / Σμ = 1".

**Fix #11 (Kairos superiority).** Limitations now reads: "this is measured on the *unreduced* PIDSMaker graph and therefore does *not* extend to the paper's reduced graph, where the encoding claim is made. A fair comparison on the reduced graph would require running Kairos on the same reduced graph with the same positives, which we flag as the next experiment".

Paper state after this round: `论文研究/Manuscript_RATE.tex`, **2672 lines / 7 sections / 15 tables / 9 equations**; check_tex (2 pre-existing multicolumn false positives on `tab:main` and `tab:gnn`) + check_refs (31 defined, 31 cited) all pass.

---

## Addendum — 2026-09-13 17:05 (`问题-0913-3.md`)

DeepSeek's third review touched three places where the paper still referred to old state — before Table 15 existed, before the protocol-matched reproduction was added — plus one residual bibitem defect.

**Fix #1 (Introduction last paragraph, L261-262).** "and no protocol-matched published baseline" was an absolute negative now contradicted by Table 15. Tightened to: "and no protocol-matched reproduction of a published detector on the paper's own reduced graph" — Table 15 covers the protocol-matched reproduction but on the unreduced PIDSMaker graph; the still-missing case is on the paper's reduced graph.

**Fix #2 (Limitations §7 "No clean false-alarm estimate", L2411-2413).** Same wording problem — "weight it as heavily as the missing protocol-matched baseline". Replaced with "weight it as heavily as the missing protocol-matched reproduction on the paper's reduced graph", so the sentence now means: the false-alarm gap is now the larger of the two, because the protocol-matched baseline gap on the unreduced graph is closed by Table 15.

**Fix #3 (CPR cite with no bibitem, L420-422).** The previous round's rewrite of the ref29 sentence mentioned "CPR extends the same idea..." but the bibliography has no CPR entry (ref29 is High Fidelity Data Reduction). Followed DeepSeek's second option: deleted the CPR clause, leaving the sentence as: "Prune-style operators keep nodes but delete redundant edges: high-fidelity reduction~\cite{ref29} keeps the deletion set dependency-aware." The point about dependency-aware deletion survives; the orphan CPR citation does not.

**Fix #4 (residual bibitem defect, found while editing).** While working around the bibliography, noticed that `ref28` still carried the original conflated text: "LogGC / NodeMerge: graph reduction through node merging and frequency aggregation in provenance graphs, in: Proc. ACM SIGSAC Conference on Computer and Communications Security (CCS)." This is the bug from the first ref-audit round that was supposed to have been fixed when ref31=LogGC was added — the cite was rerouted to `\cite{ref28}` (NodeMerge) but the bibitem text was not updated. Replaced with the verified NodeMerge citation (Y. Tang, D. Li, Z. Li, M. Zhang, K. Jee, X. Xiao, Z. Wu, J. Rhee, F. Xu, Q. Li, NodeMerge: Template based efficient data reduction for big-data causality analysis, in: Proc. ACM SIGSAC CCS, 2018, pp. 1324-1337). Source confirmed via the provenance-graph survey (arXiv:2006.01722) and the ACM DL page (DOI 10.1145/3243734.3243763).

Paper state after this round: `论文研究/Manuscript_RATE.tex`, **2671 lines / 7 sections / 15 tables / 9 equations**; check_tex (2 pre-existing multicolumn false positives) + check_refs (31 defined, 31 cited, no duplicate, no orphan, every entry has authors/year/pages except the deliberate summary refs) all pass.

Script and JSON outputs are unchanged from the previous round — every fix in this round is text-only.
---

## Addendum — 2026-09-13 17:15 (`问题-0913-3.md` 后续数字核验)

DeepSeek caught a number inconsistency in the Table 15 explanation paragraph (paper L1569-1570, originally "≈2.6 million positives per window"). Three sub-issues, all addressed in the paper:

1. **"positives" misnomer.** Flagging 77% of nodes yields *flagged nodes*, not real positives. Renamed to "nodes".
2. **Number mismatch.** "≈2.6 million" was wrong. The 38-bin test window carries N=2,725,941 nodes (per `results/threatrace_full38/REPORT.md`), so 77% × 2.73M = 2.10M. The threatrace epoch-3 best-MCC scan independently records TP=54 @ FP≈210万 at 87.1% recall — these two numbers match exactly. Corrected to ≈2.1 million.
3. **Unit error.** "per window" was wrong: 154 × 2.6M ≈ 400M is implausible. The 2.1M is the total across the entire 38-partition test window (the 2.73M in total), so the unit becomes *across the test window*.

**Final wording (paper L1566-1573):**

> "Table~\ref{tab:baselines}'s $87.1\%$ is reached only by flagging $77\%$ of test-window nodes, i.e.\ a high-volume operating point that flags $\approx\!2.1$ million nodes across the test window (because the test window carries $2.73$M nodes in total)."

The parenthetical makes the arithmetic transparent: 0.77 × 2.73M ≈ 2.1M, no unit confusion possible.

Paper state after this micro-fix: `论文研究/Manuscript_RATE.tex`, **2673 lines / 7 sections / 15 tables / 9 equations** (counter now reads 2673 vs 2672 in the previous addendum; the +1 is from splitting the parenthetical into its own sentence). check_tex and check_refs both pass.
