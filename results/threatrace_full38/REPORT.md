# THREATRACE @ CADETS_E5 38-bin 完整基线报告（2026-09-11）

## TL;DR
- THREATRACE 全流程（construction 复用 63a95c7e hash、word2vec 特征化 + 节点类型分类训练）跑通。
- 官方阈值 1.5 退化为"全标"：tp=62 但 fp=272 万（precision 2e-5），指标无意义。
- 阈值扫描揭示**更本质的问题：threatrace_score 排序能力接近无效甚至反向**（AUC 0.36~0.38 < 0.5）：
  - top-10 万节点（3.7% 数据）只捞到 **1 个** TP；
  - best-MCC 点要标 **200 万+ 节点** 才能达到 recall 87%（epoch3: TP=54 @ FP≈210 万）；
  - 两攻击在官方阈值下均可触发（percent_detected_attacks=1.0），但毫无定位能力。
- 结论：THREATRACE 的节点类型混淆分数在 CADETS_E5 38-bin 上**不可用**，与 Flash（recall 9.7%）同为真弱基线，且排序比 Flash 更差。

## 各 epoch 汇总（N=2,725,941，正例 62）
| epoch | 官方阈值下 | best-MCC 点 |
|---|---|---|
| 0 | R=21.0% @ 159 万 FP | R=100% @ 272 万 FP（全标） |
| 1 | R=24.2% @ 183 万 FP | R=100% @ 272 万 FP（全标） |
| 3 | R=82.3% @ 200 万 FP | R=87.1% @ 210 万 FP |
| 5 | R=90.3% @ 262 万 FP | R=87.1% @ 210 万 FP |
| 7 | R=90.3% @ 262 万 FP | R=82.3% @ 194 万 FP |
| 9 | R=90.3% @ 267 万 FP | R=87.1% @ 210 万 FP |
| 11 | R=90.3% @ 267 万 FP | R=87.1% @ 210 万 FP |

- 官方阈值随 val loss 单调升高（5.9→8.2），阈值本身合理，但分数分布上恶意节点并不集中在高分段 → 排序无效。
- score 语义：log(次高类概率/最高类概率)，衡量"类型混淆度"；在本数据集上恶意节点并不比良性节点更混淆。

## 对比速览（CADETS_E5 38-bin，正例 62）
| 系统 | 最佳可达 recall | 排序质量 |
|---|---|---|
| Kairos（TGN） | 75.8% @ 最佳 MCC | 好（AUC 高） |
| Flash（loss-based） | 9.7% @ 2.56 万 FP | 弱 |
| **ThreaTrace（type-confusion）** | **87.1% 但需标 77% 节点** | **≈反向（AUC 0.36）** |

## 产物
- `threshold_scan.json` / `pr_curve.png` / `recall_at_topk.png` / `thr_scan_threatrace.log`
- `threatrace_eval1.log`：完整流水线日志
- 服务器侧：`~/thr_scan_threatrace/`
