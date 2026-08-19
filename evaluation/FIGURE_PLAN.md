# Ballast figure plan

## 已生成：robustness figure

`../figures/fig_robustness.pdf` / `.png` 是两联图；原始表位于
`../results/phi_sweep.csv` 与 `../results/snapshot_robustness.csv`：

- 左图固定同一网络与随机种子，扫 `phi = 0.1,...,0.5`；重点看 Ballast
  pooled curve 是否稳定，并与 per-channel control 和三条 baseline 比较。
- 右图固定 `phi=0.3`，比较 block 600000、640000、677167 三个历史快照；
  用 paired lines 同时展示 pooled、per-channel 及其差值，避免只报告一个
  “有利快照”。

复现命令：

```bash
cd evaluation
.venv/bin/python3 run_phi_sweep.py --paper --jobs 4
.venv/bin/python3 run_snapshot_robustness.py --paper --jobs 4
.venv/bin/python3 plot_results.py
```

## 建议继续补的图（按优先级）

1. **协议时序图（正文）**：`u -> witness quorum -> w_i -> F_chan/contract`。
   把 receipt 必须先于 forwarding、claim 的 drawId/HTLC binding、repay 的
   counterparty authorization 画在同一张图。这是审稿人理解新安全边界最
   关键的一图。
2. **攻击与修复两联图（安全分析）**：左边画同版本 `cmt_A/cmt_B` 分叉；
   右边画 `[60,0] -> [0,60]` sequential reset。用红叉分别标出 unique
   receipt 与 transition/decrease authorization 的拦截点。
3. **完整实现成本拆分（完成新电路后）**：stacked bars 分为 vector
   transition proving、receipt/threshold verification、channel verification、
   on-chain claim。不要把当前 scalar-kernel 数字标成完整协议。
4. **拓扑解释图（可放附录）**：横轴 node degree 或 flow diversity，纵轴
   pooled/per-channel capital ratio，附置信区间，用来解释哪些节点获得
   multiplexing gain，而不仅仅报告全网均值。

风格上建议沿用 Shaduf/Shaduf++ 的“机制时序图 + 端到端性能图”，同时像
Horcrux 一样把正常路径与争议路径分开画。所有 robustness 曲线应标明快照、
随机种子数、支付数和置信区间。
