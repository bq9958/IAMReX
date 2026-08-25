# ls2d-baseline — 单核 C++ 两相流 Level-Set 教学基线：规划（v0，2026-08-25）

目标：从 IAMReX 中抽出「变密度投影法 + 最简单 Level Set（含 Sussman 重初始化 + 体积修正）」，
重写为一份不依赖 AMReX 的单核 2D C++ 代码，英文详注、与 Almgren 1998 / Sussman 1999 逐式对照，
并用可验证的算例校验。之后它是 task1（多相流）与 task2（流固耦合）的共同 baseline。

## 0. 基准来源
- 算法：Sussman, Almgren, Bell, Colella, Howell, Welcome, JCP 148 (1999) §2–3（单网格部分）
- 投影/对流细节：Almgren, Bell, Colella, Howell, Welcome, JCP 142 (1998) §2–3
- 实现参考：IAMReX `Source/NavierStokes.cpp::advance_semistaggered_twophase_ls`、
  `NavierStokesBase.cpp::{reinit, reinitialization_sussman, rk_first_reinit, rk_second_reinit, mass_fix, phi_to_sgn0}`、
  `NS_LS.cpp::{phi_to_heavi, heavi_to_rhoormu}`、`prob/prob_init.cpp::{init_rsv, set_rsv_vel, init_RayleighTaylor_LS, init_BreakingWave}`

## 1. 范围（v1 做什么 / 不做什么）
做：2D 笛卡尔均匀网格；cell-centered u,φ,ρ,μ；node-centered p（与 IAMReX/paper 一致）；
    MAC 投影 + 近似 nodal 投影；unsplit Godunov-PLM 对流（含横向项）；Crank–Nicolson 粘性半隐式；
    Heaviside 光滑密度/粘度；Sussman 重初始化（ENO2 + RK2/Heun + 体积修正）；重力；
    周期 / slip wall / no-slip / outflow 边界；线性求解器 = PCG（可选 GMG）。
不做（v1）：AMR、subcycling、3D、EB、粒子、GPU。表面张力（paper 的 M 项）放 v2 —— 破波/RT 不需要，
    但 paper §5 的气泡算例需要，所以留接口。

## 2. 代码结构（目标 ~2500 行，单可执行文件 + 头文件）
```
ls2d-baseline/
├── PLAN.md                      本文件
├── docs/
│   ├── algorithm.md             逐式对照表：paper 公式号 ↔ 函数名 ↔ IAMReX 函数名
│   ├── deviations.md            IAMReX 与 paper 的差异 / 疑似 bug 清单（§5）
│   └── validation.md            算例、指标、结果表
├── src/
│   ├── main.cpp                 读 inputs，时间推进循环，输出
│   ├── grid.h                   Field2D（带 ghost cell 的二维数组）、Geometry、索引约定
│   ├── params.h                 输入参数（沿用 IAMReX 键名：ns.epsilon, ns.number_of_reinit ...）
│   ├── bc.h / bc.cpp            ghost cell 填充（periodic/slip/noslip/outflow）
│   ├── levelset.h / .cpp        H_eps (paper 50), rho(φ), mu(φ), sgn (57), redistance (51–65), volume fix (66–71)
│   ├── godunov.h / .cpp         PLM 斜率(4th-order limited)、预测到面 (20–23)、横向项、迎风 (25)
│   ├── mac.h / .cpp             MAC 投影 (27–28)、面速度、对流通量 [U·∇φ], [(U·∇)U]
│   ├── viscous.h / .cpp         L 算子 (paper §3.3)、CN 半隐式解 (16)
│   ├── projection.h / .cpp      nodal 近似投影 (17, §3.4)：D, G 算子，变系数 Poisson
│   ├── linsolve.h / .cpp        PCG (+Jacobi)；可选 GMG
│   ├── io.h / .cpp              CSV/VTK 输出，诊断（体积 (81)、|∇φ| 误差、动能、界面位置）
│   └── problems.cpp             RSV / Zalesak / RT / TaylorGreen / bubble 初始化（破波已删除）
├── tests/                       每个算例一个 inputs 文件 + 期望值 json
├── scripts/                     python: 画图、与 IAMReX plotfile 对比、收敛阶
└── CMakeLists.txt               -O2, 无外部依赖
```
风格：每个函数头部写 (a) 对应 paper 公式号 (b) 输入/输出场及其位置(cell/face/node) (c) ghost 需求。
循环全部显式 for(i,j)，不做模板魔法；数组访问用 `phi(i,j)`。

## 3. 时间推进（对照 Sussman 1999 §3.1）
给定 U^n, φ^n, ∇p^{n-1/2}：
1. Godunov 预测 U^{n+1/2}_face、φ^{n+1/2}_face → MAC 投影 (27) → U^ADV
2. φ^{n+1} = φ^n − Δt [U·∇φ]^{n+1/2}（非守恒形式，IAMReX `do_cons_phi=0` 默认）
3. φ^{n+1/2}=(φ^n+φ^{n+1})/2 → ρ^{n+1/2}, μ^{n+1/2} (13–15)；ρ^{n+1}=ρ(φ^{n+1})
4. CN 粘性半隐式求 U* (16)
5. nodal 近似投影 (17) → U^{n+1}, p^{n+1/2}
6. 重初始化 φ^{n+1}：RK2×N 次，每次后体积修正 (§3.5)
7. 时间步 (§3.1.1)
初始化：先投影 U^0，再迭代若干步求 p^{1/2} (§3.6)。

## 4. 验证矩阵（全 2D；真值 = 解析解或文献，**不以 IAMReX 自带算例结果为真值**——RSV / RT_LS 教程从未与文献校验过）
| # | 算例 | 检验什么 | 真值 / 判据 |
|---|---|---|---|
| V1 | 圆盘静止重初始化 | redistance 收敛到距离函数、零等值线不动 | 解析距离函数：band 内 max||∇φ|−1| < 1e-2；面积 (81) 变化 < 1e-4；零等值线位移 < 1e-3Δx 量级 |
| V2 | Zalesak 槽盘（Zalesak 1979 设置，Rider–Kothe 1998 参数） | 对流 + 重初始化几何保持 | 一圈后与初始形状的 L1 误差随网格 ≥1 阶收敛；面积损失 @128² < 1%；与 Rider–Kothe / Enright 2002 表中 LS 结果同量级 |
| V3 | 单涡 RSV（Rider–Kothe 1998, T=8 反转；LeVeque 1996） | 强拉伸 + 反转 | t=T 与初值的 L1 误差、面积损失；对照 Rider–Kothe 表 / Enright et al. 2002 表 1 的 level-set 列 |
| V4 | Taylor–Green 衰减涡（解析解） | 投影 + 粘性 + 对流精度 | u, p 的 L2 误差二阶收敛 |
| V5 | 静水平衡 + 重力（ρ 比 1000:1） | 变密度投影伪速度 | max|u| 不随时间增长，量级 < 1e-10·√(gL) |
| V6 | RT 不稳定性 | 全耦合两相 | 单模 RT：Tryggvason 1988 (Atwood 0.5, Re 3000) / Guermond–Quartapelle 2000 / Popinet Gerris 基准的尖峰与气泡位置 vs t；线性阶段增长率对照解析 Chandrasekhar |
| V7 | 破波 | — | 取消（太重）；仅保留初始化代码作示例 |
| V8 (v2) | 上升气泡 Hysing et al. 2009 2D 基准 case 1/2 | 表面张力 | 质心、上升速度、圆度 vs 基准数据（公开表格） |

IAMReX 的角色：用同参数跑 `amr.max_level=0`、`fixed_dt`，导出场（`Tools/plt_to_numpyArray` 或 yt）与 ls2d 逐点比——
目的是**定位**两者差异出在哪一步（对流 / 投影 / reinit），从而发现任一方的 bug；不是把 IAMReX 当真值。
发现的 bug 两边同时修（IAMReX 走 PR）。

## 5. 已发现的 IAMReX ↔ paper 差异（待逐条确认；确认为 bug 的顺手修回 IAMReX）
D1 **ENO 迎风分支**：paper (62) 第三种情况（w_L<0, w_R>0，膨胀）取 0；IAMReX `rk_*_reinit` 取 (dxp+dxm)/2。
    影响：|∇d| 在膨胀区被平均而非置零，可能改变远离界面处收敛，界面附近影响小。需数值验证。
D2 **符号函数**：paper 用冻结的 S_Δx(φ^0)=2(H−1/2) (57) 既选迎风也做系数；IAMReX 迎风选用 sgn0（冻结），
    但更新系数用 `phi/sqrt(phi²+(|∇φ|·2Δτ)²)`（Peng 1999 式，随当前 φ 变）。两者都合理，但与 paper 不同。
D3 **RK2 形式**：paper (54)-(55) 印刷形式 d^{k+1}=d^{(1)}+Δτ/2(L^k+L^{(1)}) 不自洽；IAMReX 实现等价于标准 Heun
    d^{k+1}=d^k+Δτ/2(L^k+L^{(1)})（推导：phi_new = phi1 − ½Δτ·s·(G1−G0)）。以 Heun 为准，文档里说明。
D4 **mass_fix 的 δ 函数**：paper (68) H'_ε(d^0) 用 d^0；IAMReX 分支条件用当前 `phi`，cos 内用 `phi_ori` —— 条件与取值不一致，
    |φ| 跨 ε 边界的格点会错。疑似 bug，修法：两处都用 phi_ori。
D5 **九点积分权重**：已确认 = Sussman–Fatemi 1999 式 (4.8)：∫g ≈ h²/24 (16 g_ij + Σ 8 邻点)，IAMReX 与 ls2d 的 16/1 权重正确。
D6 **符号翻转钳制**：IAMReX 在 |φ|≤ε 内若一步 RK 使 φ 变号则 `phi2 = 0.1*phi`，bulk 变号直接 Abort。paper 无此项。
    教学版保留但注明为工程保护，并统计触发次数。
D7 **粘度时间层**：IAMReX 文档写 μ(φ^n) 与 μ(φ^{n+1})，代码 viscn/viscnp1 都用 μ(φ^{n+1/2})；paper (16) 也用 n+1/2 → 代码对，文档改。
D8 **ε 取值**：paper 符号函数用 S_Δx（ε=Δx），Heaviside 用 ε=αΔx；IAMReX 两者统一 ε=epsilon·Δx（默认 2）。
D9 **表面张力**：IAMReX LS 分支无 M 项；paper 有。v2 补。
D10 `levelset_diffcomp` 为空函数（conservative LS 路径未实现）——不在本次范围，记录。

## 6. 里程碑（状态 2026-08-25：M1–M7 全部完成；破波验证取消）
M1 (文档) algorithm.md 逐式对照表 + deviations.md 定稿，确认 D1–D8 各自处理方式 —— 需要你拍板
M2 (代码) grid/bc/levelset/io + V1、V2（纯对流用给定速度场，先写简单二阶迎风，验证 reinit 独立正确）
M3 (代码) godunov/mac + V2/V3 对照文献表；与 IAMReX 逐点比定位差异
M4 (代码) viscous/projection/linsolve + V4 Taylor–Green 二阶收敛
M5 (代码) 全耦合 + V5/V6 对照文献 → 定位并修 IAMReX bug（走 PR）
M6 整理 README、可读性审查；冻结为 baseline v1（2D only）——破波验证取消
M7 (v2) 表面张力 + V8

## 7. 待讨论 / 需要你决定
Q1 D1 的处理：教学版按 paper 取 0，还是按 IAMReX 取平均？（我建议：按 paper，加开关复现 IAMReX）
Q2 D2：教学版用冻结 sgn0（paper）还是 Peng 式？（建议 paper 为默认，Peng 为选项）
Q3 线性求解器：PCG 够用（128²–256² 秒级）还是一开始就写 GMG？（建议 PCG 先行，MG 作 v2）
Q4 Godunov：完整 unsplit + 横向项（忠实 paper，代码量大），还是先用 RK2-MOL + PLM 迎风把整链打通再替换？
    （建议：先 MOL 打通 M2–M5，再实现 Godunov 做 side-by-side —— 这本身也是很好的教学对比）
Q5 压力放 node（paper/IAMReX）还是 cell（更常见教材写法）？（建议 node，忠实来源）
Q6 输出格式：VTK legacy（ParaView 直接开）+ CSV 诊断；要不要顺带写 AMReX plotfile 读取脚本以便和 IAMReX 逐点比？

## 8. 待办（2026-08-25 晚）
- ~~nodal Q1 多重网格~~ 已修：限制算子应为延拓转置（权重和 4），原来用了 1/16 → 迭代 77→5。
- ~~体积约束 δ 带宽 → IAMReX 第二个 PR~~ 已提交：`ns.epsilon_massfix` + `mass.txt` 相体积列（RSV +18.3% → +4.5%，IAMReX PR #78）。
- ~~教学版最后一遍可读性审查~~ 完成：约 87 条函数头注释补齐，nodal_project 的 2 处调试打印删除，冒烟测试数值逐位不变。
- ~~v2：表面张力 + Hysing V8~~ 完成：静止液滴 Δp 误差 0.8%；Hysing case 1 80×160 三个基准量误差 ≤0.3%。
