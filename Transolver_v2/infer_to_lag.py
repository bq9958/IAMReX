#!/usr/bin/env python3
"""
infer_to_lag.py —— 用 TransolverSlim 给每个 marker 重算铺展权重，
替换 rkpm_mappings.lag 中的 w 值（i/j/k/Vcell/eps 与文本结构原样保留）。

本阶段目标：跑通"模型输出 → .lag → IAMReX"这条管线，不评估权重精度。
用法:
    /opt/miniconda3/envs/jupyter_env/bin/python infer_to_lag.py
"""
import argparse
import ast
import os
import re
import sys

import numpy as np
import torch

# 让 transolver_slim 可导入（与脚本同目录）
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from IAMReX.Transolver_v2.transolver_slim import TransolverSlim  # noqa: E402


# ---------- 1. 文件解析 ----------
def load_id(path):
    """rkpm_mappings.id：合法 Python 字典字面量 {marker_id: (x, y, z)}"""
    with open(path) as f:
        return ast.literal_eval(f.read())


def load_lag(path):
    """rkpm_mappings.lag：正则解析（与 verify_rkpm_alignment.py 同思路）。

    返回 {marker_id: [(i, j, k, w, Vcell, eps), ...]}，每个 marker 27 项。
    """
    txt = open(path).read()
    out = {}
    for bid, body in re.findall(r'(\d+)\s*:\s*\[(.*?)\]', txt, re.S):
        rows = []
        for e in re.findall(r'\{([^}]*)\}', body):
            kv = dict(re.findall(r'"(\w+)"\s*:\s*([-\d.eE]+)', e))
            rows.append((int(kv['i']), int(kv['j']), int(kv['k']),
                         float(kv['w']), float(kv['Vcell']), float(kv['eps'])))
        out[int(bid)] = rows
    return out


def parse_inputs(path):
    """AMReX inputs 文件：key = value，# 行内/整行注释。"""
    params = {}
    for line in open(path):
        line = line.split('#', 1)[0].strip()
        if not line or '=' not in line:
            continue
        k, _, v = line.partition('=')
        params[k.strip()] = v.strip()
    return params


# ---------- 2. 模型 ----------
def load_model(model_dir):
    """从 model_best.pt 载入（文件自带 cfg 超参，无需手写）。"""
    ck = torch.load(os.path.join(model_dir, 'model_best.pt'), map_location='cpu')
    model = TransolverSlim(**ck['cfg'])
    model.load_state_dict(ck['state_dict'])
    model.eval()
    return model


def load_norm_stats(model_dir):
    s = np.load(os.path.join(model_dir, 'norm_stats.npz'))
    return s['xm'], s['xs'], s['ym'], s['ys']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model',  default='out_inv')
    ap.add_argument('--id',     default='../IAMReX/Tools/RKPM_weight/rkpm_mappings.id')
    ap.add_argument('--lag',    default='../IAMReX/Tools/RKPM_weight/rkpm_mappings.lag')
    ap.add_argument('--inputs', default='../IAMReX/Tools/RKPM_weight/inputs.3d.flow_past_ellipsoid')
    ap.add_argument('--out',    default='rkpm_mappings_transolver.lag')
    args = ap.parse_args()

    model = load_model(args.model)
    xm, xs, ym, ys = load_norm_stats(args.model)

    markers = load_id(args.id)
    lag = load_lag(args.lag)
    print(f'markers: {len(markers)} (id), {len(lag)} (lag)')

    # 一致性检查：每个 marker 必须有恰好 27 项
    bad = [mid for mid, rows in lag.items() if len(rows) != 27]
    if bad:
        print(f'[warn] 有 {len(bad)} 个 marker 不是 27 项: {bad[:5]} ...')

    # ---------- 3. 网格参数 ----------
    p = parse_inputs(args.inputs)
    plo = np.array([float(x) for x in p['geometry.prob_lo'].split()])
    phi = np.array([float(x) for x in p['geometry.prob_hi'].split()])
    ncell = np.array([float(x) for x in p['amr.n_cell'].split()])
    maxlev = int(p['amr.max_level'].split()[0])
    dx = (phi - plo) / (ncell * (2 ** maxlev))
    print(f'dx = {dx.tolist()}')   # 本算例应为 [0.0029296875, 0.0029296875, 0.0029296875]

    # ---------- 4. 构造输入 [N, 27, 3] ----------
    ids = sorted(lag.keys())
    X = np.zeros((len(ids), 27, 3), dtype=np.float32)
    for n, mid in enumerate(ids):
        mpos = np.array(markers[mid], dtype=np.float64)   # marker 世界坐标
        for c, (i, j, k, w, Vc, eps) in enumerate(lag[mid]):
            center = (np.array([i, j, k], dtype=np.float64) + 0.5) * dx  # 单元中心
            X[n, c] = (center - mpos) / dx
            # TODO(精度阶段): rel_hat 的符号/缩放必须与训练数据生成脚本口径核对

    # ---------- 5. 推理 ----------
    Xn = (X - xm) / xs
    bx = torch.tensor(Xn, dtype=torch.float32)
    with torch.no_grad():
        pred = model(fx=bx, embedding=bx)        # [N, 27, 1]
    W = pred.numpy()[..., 0] * ys + ym           # 反标准化

    # ---------- 6. 防崩溃保险：每 marker 强制 Σw=1 ----------
    W = W / W.sum(axis=1, keepdims=True)
    print(f'w range: {W.min():.6f} ~ {W.max():.6f}')
    print(f'max |sum-1| after renorm: {np.abs(W.sum(axis=1) - 1).max():.3e}')

    # ---------- 7. 写回：只替换 "w": 旧值，其余字符逐字节保留 ----------
    txt = open(args.lag).read()
    n_w_orig = len(re.findall(r'"w"\s*:\s*[-+0-9.eE]+', txt))

    wtable = {mid: W[n] for n, mid in enumerate(ids)}
    flat = []
    for bid, body in re.findall(r'(\d+)\s*:\s*\[(.*?)\]', txt, re.S):
        flat.extend(wtable[int(bid)].tolist())   # 按文件出现顺序排平
    if len(flat) != n_w_orig:
        print(f'[error] 新权重数 {len(flat)} != 文件原 w 数 {n_w_orig}，中止')
        return

    it = iter(flat)
    new_txt = re.sub(r'"w"\s*:\s*[-+0-9.eE]+',
                     lambda m: f'"w": {next(it):.16g}', txt)
    open(args.out, 'w').write(new_txt)
    print(f'written: {args.out}  (替换 {n_w_orig} 个 w)')


if __name__ == '__main__':
    main()