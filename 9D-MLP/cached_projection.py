"""Precomputed geometry-fixed A/K caches for nonuniform 3x3x3 stencils."""
from __future__ import annotations
from dataclasses import dataclass
import time
import numpy as np
import torch
from geometry_fixed_constraints import projection_cache_from_ratios, project_cached

OFFSETS_NP=np.asarray([(i,j,k) for k in (-1,0,1) for j in (-1,0,1) for i in (-1,0,1)],dtype=np.float32)

@dataclass
class GeometryCache:
    mode: str
    precision: str
    ids: torch.Tensor
    A: torch.Tensor
    K: torch.Tensor
    signature_count: int
    cell_count: int
    build_seconds: float
    storage_bytes: int
    ratios_unique: np.ndarray

    @classmethod
    def build(cls,widths: np.ndarray,device: torch.device,precision: str="fp32",
              mode: str="template",chunk: int=4096):
        if mode not in ("template","per_cell"): raise ValueError(mode)
        if precision not in ("fp32","fp64"): raise ValueError(precision)
        w=np.asarray(widths,dtype=np.float64)
        signatures=np.stack((w[:,:,0]/w[:,:,1],w[:,:,2]/w[:,:,1]),axis=-1).reshape(len(w),6)
        if mode=="template": unique,ids=np.unique(signatures,axis=0,return_inverse=True)
        else: unique=signatures.copy(); ids=np.arange(len(w),dtype=np.int64)
        dtype=torch.float64 if precision=="fp64" else torch.float32
        offsets=torch.as_tensor(OFFSETS_NP,device=device,dtype=dtype)
        as_,ks_ = [],[]; start=time.perf_counter()
        for st in range(0,len(unique),chunk):
            ratio=torch.as_tensor(unique[st:st+chunk],device=device,dtype=dtype)
            A,K=projection_cache_from_ratios(ratio,offsets,dtype)
            as_.append(A);ks_.append(K)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        build=time.perf_counter()-start
        A=torch.cat(as_); K=torch.cat(ks_)
        ids_t=torch.as_tensor(ids,device=device,dtype=torch.long)
        size=A.numel()*A.element_size()+K.numel()*K.element_size()+ids_t.numel()*ids_t.element_size()
        return cls(mode,precision,ids_t,A,K,len(unique),len(w),build,size,unique)

    def apply(self,raw: torch.Tensor,phase: torch.Tensor,correction: bool=False,
              output_fp32: bool=True):
        idx=self.ids
        A=self.A[idx];K=self.K[idx]
        dtype=A.dtype
        w=project_cached(raw.to(dtype),phase.to(dtype),A,K,correction)
        return w.float() if output_fp32 else w

    def memory(self):
        return {"A_bytes":self.A.numel()*self.A.element_size(),
                "K_bytes":self.K.numel()*self.K.element_size(),
                "id_bytes":self.ids.numel()*self.ids.element_size(),
                "total_bytes":self.storage_bytes}
