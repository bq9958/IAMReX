import os, sys, re
import numpy as np
import torch
from mpi4py import MPI
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from transolver_slim import TransolverSlim

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(HERE, "out_inv")
LAG_PATH  = os.path.join(HERE, "..", "..", "run", "RKPM_transolver", "rkpm_mappings.lag")
INPUTS    = os.path.join(HERE, "..", "..", "run", "RKPM_transolver", "inputs.3d.flow_past_ellipsoid")

# 
def load_lag_skeleton(path):
    txt = open(path).read()
    out = {}
    for bid, body in re.findall(r'(\d+)\s*:\s*\[(.*?)\]', txt, re.S):
        rows = []
        for e in re.findall(r'\{([^}]*)\}', body):
            kv = dict(re.findall(r'"(\w+)"\s*:\s*([-\d.eE]+)', e))
            rows.append((int(kv['i']), int(kv['j']), int(kv['k'])))
        out[int(bid)] = rows
    return out

def parse_inputs(path):
    p = {}
    for line in open(path):
        line = line.split('#', 1)[0].strip()
        if not line or '=' not in line: continue
        k, _, v = line.partition('=')
        p[k.strip()] = v.strip()
    return p

def main():
    world = MPI.COMM_WORLD
    rank, size = world.Get_rank(), world.Get_size()
    appnum = world.Get_attr(MPI.APPNUM)
    print(f"[server] world rank {rank}/{size}, appnum={appnum}", flush=True)
    assert appnum == 1, "本程序必须以 mpiexec 冒号语法里的第二段启动"

    # 与 C++ 侧 amrex::MPMD::Initialize 对齐：先 Allgather(appnum) 再 Split
    appnum_buf = np.array([appnum], dtype=np.int32)
    world.Allgather([appnum_buf, MPI.INT], [np.empty(size, dtype=np.int32), MPI.INT])
    local = world.Split(color=appnum)
    if rank != size - 1:
        return

    # 加载模型 + 骨架 + 网格参数
    ck = torch.load(os.path.join(MODEL_DIR, 'model_best.pt'), map_location='cpu')
    model = TransolverSlim(**ck['cfg'])
    model.load_state_dict(ck['state_dict'])
    model.eval()
    s = np.load(os.path.join(MODEL_DIR, 'norm_stats.npz'))
    xm, xs, ym, ys = s['xm'], s['xs'], s['ym'], s['ys']
    lag = load_lag_skeleton(LAG_PATH)
    p = parse_inputs(INPUTS)
    plo = np.array([float(x) for x in p['geometry.prob_lo'].split()])
    phi = np.array([float(x) for x in p['geometry.prob_hi'].split()])
    ncell = np.array([float(x) for x in p['amr.n_cell'].split()])
    maxlev = int(p['amr.max_level'].split()[0])
    dx = (phi - plo) / (ncell * (2 ** maxlev))
    ids = sorted(lag.keys())
    print(f"[server] 已加载, dx={dx.tolist()}, markers={len(ids)}", flush=True)

    cfd_root = 0
    while True:
        t0 = time.time()
        nm = np.empty(1, dtype=np.int32)
        world.Recv([nm, MPI.INT], source=cfd_root, tag=200)
        Nm = int(nm[0])
        if nm == 0:
            print("[server] 收到关闭信号，退出", flush=True)
            break

        pos = np.empty(Nm * 3, dtype=np.float32)
        world.Recv([pos, MPI.FLOAT], source=cfd_root, tag=201)
        pos = pos.reshape(Nm, 3)
        t1 = time.time()
        print(f"[server] 收到 {Nm} 个 marker 位置", flush=True)

        # 构造 rel_hat [Nm,27,3]
        X = np.zeros((Nm, 27, 3), dtype=np.float32)
        for n, mid in enumerate(ids):
            for c, (i, j, k) in enumerate(lag[mid]):
                center = (np.array([i, j, k], dtype=np.float64) + 0.5) * dx
                X[n, c] = (center - pos[n]) / dx
        t2 = time.time()

        # 前向 + 反标准化 + 每 marker Σw=1
        Xn = (X - xm) / xs
        bx = torch.tensor(Xn, dtype=torch.float32)
        with torch.no_grad():
            pred = model(fx=bx, embedding=bx)
        W = pred.numpy()[..., 0] * ys + ym
        W = W / W.sum(axis=1, keepdims=True)
        t3 = time.time()

        world.Send([W.astype(np.float32).ravel(), MPI.FLOAT], dest=cfd_root, tag=202)
        t4 = time.time()
        print(f"[server] 已回发 {Nm} markers 权重, sum={float(W.sum()):.6f}", flush=True)

        print(f"[server] 收位置={t1-t0:.4f}s  算rel_hat={t2-t1:.4f}s  前向={t3-t2:.4f}s  发权重={t4-t3:.4f}s", flush=True)

if __name__ == '__main__':
    main()