# 9D-MLP RKPM 权重模型交付包

本文件夹可以单独复制给同学使用。它包含一个已训练的 **9D-MLP checkpoint**、完整的十项二次一致性投影代码和推理入口。输入为标记点在中心网格单元内的相位及三轴相邻单元宽度；输出为完整 `3×3×3` stencil 的 27 个权重。

## 文件

| 文件 | 作用 |
|---|---|
| `nonuniform_compact9_projected_targeted.pt` | 已训练参数和训练元数据，seed `20261302` |
| `compact_geometry.py` | 9D 网络、动态十项投影、27 点几何重建；复制自原 benchmark |
| `geometry_fixed_constraints.py` | 相位右端和几何固定形式的十项矩阵；复制自原 benchmark |
| `cached_projection.py` | 几何模板缓存、FP32 投影和一次残差修正；仅增加 CPU 运行兼容判断 |
| `inference.py` | 对外使用的 `RKPM9D` 类，封装加载、输入检查和两种投影方式 |
| `example_inference.py` | 两个样本的可运行示例，同时检查十项残差 |
| `requirements.txt` | Python 依赖 |

原始 checkpoint 来自 `rkpm_ml_benchmark/Nonuniform_Quadratic_C64/nonuniform_compact9_projected_targeted.pt`，复制后 SHA-256 与原件一致：`14B503F218F679FABF16DF35A1F3FF307FF7273883E6797731752771A15A99C5`。本包不含训练数据，运行推理不需要原 benchmark 文件夹。

## 快速运行

建议 Python 3.10 及以上。验证环境为 Python 中的 NumPy `2.0.1`、PyTorch `2.5.1`；GPU 测试设备为 NVIDIA GeForce RTX 4060 Laptop GPU。按本机 CUDA 环境安装 PyTorch，再安装其余依赖：

```powershell
python -m pip install -r requirements.txt
python example_inference.py --device cpu --projection cached
```

如已装好 CUDA 版 PyTorch，也可运行：

```powershell
python example_inference.py --device cuda --projection cached
```

示例输出应包含 `weights shape: (2, 27)`，权重和接近 1，`maximum absolute 10C residual` 约在 `10⁻⁷` 量级。最后几位随设备和浮点运算顺序变化。

## 在自己的脚本中调用

从本文件夹运行下面的代码，或将本文件夹加入 Python 模块搜索路径：

```python
import numpy as np
from inference import RKPM9D

# N 个标记点；三轴依次为 x、y、z；每轴依次为左邻、中心、右邻宽度。
widths = np.ones((2, 3, 3), dtype=np.float64)
widths[1] = [[0.8, 1.0, 1.25],
             [1.1, 1.0, 0.9],
             [0.7, 1.0, 1.4]]

# 标记点在各自中心单元中的位置；0 是左侧边界，值须小于 1。
phase = np.array([[0.2, 0.5, 0.8],
                  [0.35, 0.6, 0.25]], dtype=np.float32)

predictor = RKPM9D(device="cpu")  # 也可用 "cuda"
predictor.prepare(widths, projection="cached")
weights = predictor.predict(phase)  # NumPy float32 数组，形状 (2,27)
```

`widths` 可以使用物理长度，模型内部使用左右邻单元与中心单元的宽度比；所有宽度必须为正。`phase` 的形状为 `(N,3)`，每项均在 `[0,1)`。**同一行的 `widths[n]`、`phase[n]` 和输出 `weights[n]` 必须属于同一个标记点。**

输出的 27 列按 `k` 外层、`j` 中层、`i` 内层排列，`i` 变化最快：第 0 列对应偏移 `(-1,-1,-1)`，第 13 列对应中心 `(0,0,0)`，第 26 列对应 `(1,1,1)`。调用者须自行定位中心单元，并按此顺序把权重与 27 个网格索引配对。模型只返回权重，不返回网格索引。

## 两种投影方式与缓存使用

- `projection="cached"`：默认方式。`prepare()` 从六个宽度比建立几何模板的 $A_G$ 和 $K_G$ 缓存；`predict()` 做网络前向、缓存投影和一次残差修正。**同一批标记点、相同顺序、相同局部网格宽度**下，标记点只在各自中心单元内移动时可重复调用 `predict(new_phase)`，复用缓存。
- `projection="dynamic"`：每次根据九维输入重建粒子中心坐标，逐标记点构造 $A_pA_p^T$ 并求解十维线性系统；便于核对公式，但大批量反复推理通常更慢。

网格宽度变化、标记点批次增减或顺序变化后，重新调用 `prepare(new_widths)`。若标记点跨越中心单元，还须重新定位中心单元、生成新的 27 个索引和相位，然后重新 `prepare()`；不能只替换相位继续使用旧映射。`prepare()` 本身会构建缓存，耗时不包含在原 benchmark 报告的稳态推理时间中。

十项约束使用基底 `[1,x,y,z,xy,yz,zx,x²,y²,z²]`。缓存形式将相对标记点的约束 $A_pw=e_0$ 改写为相对中心单元的 $A_Gw=b(phase-0.5)$。FP32 运算下约束残差通常很小，但不应期待逐位等于零。更完整的公式推导在原项目的 `RKPM_MLP_计算说明/9D-MLP_映射项与十项投影推导.md` 中；本交付包独立运行不依赖该文档。

## 模型来源和已测精度

网络结构为 `9 → 128 → 128 → 64 → 27`，隐层使用 GELU，共 **27,803 个可训练参数**。训练时，网络原始 27 维输出先经动态十项投影，再与 production NumPy RKPM 权重计算 MSE；checkpoint 中 `projected=True` 表示该训练约定，**不能把原始 27 维网络输出直接当成最终权重**。

原项目使用相同 checkpoint 的 100k 非均匀网格 holdout、FP32 几何缓存加一次残差修正，报告权重 RMSE `2.9005e-3`、P99.9 绝对误差 `2.4503e-2`、最大绝对误差 `1.5394e-1`，参考是 production NumPy RKPM 标签。该评估中的十项残差最大值约为 `2.77e-7`（零阶）、`1.71e-7`（一阶）和 `2.39e-7`（二阶）。这些是原项目数据集上的权重级结果，并非新网格或完整流固耦合仿真的误差保证。原始记录在 `rkpm_ml_benchmark/Nonuniform_Quadratic_C64/compact_geometry_accuracy.json`。

## 适用范围

本模型针对**轴对齐、结构化正交网格中的完整 27 点 stencil**。训练数据包含均匀、近均匀、中等非均匀和 2:1 AMR 宽度比；实验也测试了更难的有效几何。输入没有单独提供支持尺度或绝对网格尺寸，因而应沿用训练数据的 RKPM 支持尺度策略和可由相位、宽度比确定的局部几何关系。边界截断 stencil、非正交网格或超出已测比例范围的配置需要另行验证。

这是可以单独复制的**推理交付包**。本工作区 IAMReX 的 `--solver ml` 分支现已使用同一 checkpoint，模型文件复制在 `IAMReX/Tools/RKPM_weight/models/9d_mlp.pt`；旧 Transolver 由 `--solver transolver` 选择。IAMReX 当前统一网格只给模型提供六个比例均为 1 的输入，并保留 C++ 原有的 float32 权重回传格式。跨中心单元运动仍需要扩展协议以更新 stencil 索引。
