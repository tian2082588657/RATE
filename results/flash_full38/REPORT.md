# FLASH @ CADETS_E5 38-bin 完整基线报告（2026-09-10）

## TL;DR
- FLASH 全流程（12 epoch）跑通，**官方评估指标无效**：框架 bug 导致 `flash_score` 恒为 0（详见下）。
- 用真实 loss 分数重建后，FLASH 是**真弱基线**：最佳 MCC 点 recall 仅 **9.7%（6/62）**，2.56 万 FP；top-10 万节点也只有 7 个 TP（11.3%）。两攻击在 best-MCC 阈值下均有节点命中（att=[0,1]）。
- 与 Kairos 不同（Kairos 是"排序好但官方阈值拖累"），FLASH 是**排序能力本身不足**——不是阈值/评估口径问题。

## 框架 Bug：flash_score 恒为 0（重要发现）
- 位置：`pidsmaker/objectives/predict_node_type.py` 返回 `"out": F.log_softmax(h, dim=1)`（全负值），而 `pidsmaker/detection/training_methods/inference_loop.py` 的 Flash 分支直接用原始 out 算置信度：
  `conf = (top1 - top2) / (top1 + eps)`。
  log_softmax 下 top1<0，分母为负 → conf 为大负值 → 被 `max(conf, 0)` 截为 0。
- 实测：全部 154 个 test CSV、45.9 万行 flash_score 全为 0；官方评估 tp=0/fn=62、auc=0.5 全部无效。**该 bug 对任何数据集都会触发**，论文复现 PIDSMaker-Flash 需注意（修法：对 out 取 softmax 后再算 conf）。
- 修复脚本：`rebuild_flash_scores.py`（用 edge_losses CSV 的 loss 列重建 node 分数，max 聚合，零重训成本）。

## 阈值扫描结果（loss-based 重建分数，N=2,725,941，正例 62）
| 指标 | epoch0~11（各 epoch 一致） |
|---|---|
| 官方阈值（max_val_loss≈9.2~9.3） | R=9.68%, P=0.0002, TP=6 |
| best-F1 / best-MCC | thr≈22, TP=6, FP=25,603, R=9.68% |
| top-50,000 | TP=6 (9.68%) |
| top-100,000 | TP=7 (11.29%) |
| TPR@FPR=1e-5~1e-3 | 0% |
| AUC | ≈0.5（文件内 json 有精确值） |

- 12 个 epoch 结果几乎完全一致 → 模型早熟，训练更久无收益。

## 对比速览（CADETS_E5 38-bin）
| 系统 | 最佳可达 recall | 备注 |
|---|---|---|
| Kairos（TGN） | 75.8% @ 最佳 MCC 阈值 | 排序好，官方阈值拖累 |
| Flash（loss-based） | **9.7%** @ best MCC | 排序能力不足；flash_score 另有框架 bug |

## 产物
- `threshold_scan.json` / `pr_curve.png` / `recall_at_topk.png` / `scan.log`：阈值扫描输出
- `rebuild_flash_scores.py`：分数重建脚本
- `flash_eval1.log`：完整流水线日志
- 服务器侧：`~/thr_flash_loss/`（重建分数与扫描）、`~/thr_scan_flash/`（无效的全零分数扫描，仅留档）
