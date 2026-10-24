"""Geometry-fixed coordinates and reusable polynomial constraint operators."""
from __future__ import annotations

import numpy as np
import torch


def basis_torch(x: torch.Tensor) -> torch.Tensor:
    a,b,c=x.unbind(-1); one=torch.ones_like(a)
    return torch.stack((one,a,b,c,a*b,b*c,c*a,a*a,b*b,c*c),dim=-1)


def geometry_matrix(eta: torch.Tensor) -> torch.Tensor:
    """Return A_G [N,10,27], where eta is central-cell-centered geometry."""
    return basis_torch(eta).transpose(1,2).contiguous()


def phase_rhs(phase: torch.Tensor) -> torch.Tensor:
    """b(delta), delta=phase-0.5, ordered like the production quadratic basis."""
    return basis_torch(phase-0.5)


def signature_numpy(widths: np.ndarray) -> tuple[np.ndarray,np.ndarray]:
    """Six axis-neighbor/center width ratios; phase deliberately excluded."""
    w=np.asarray(widths)
    ratio=np.stack((w[:,:,0]/w[:,:,1],w[:,:,2]/w[:,:,1]),axis=-1).reshape(len(w),6)
    return ratio, np.unique(ratio,axis=0,return_inverse=True)


def eta_from_ratios_torch(ratios: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
    """Reconstruct 27 cell-center coordinates from [left/center,right/center] ratios."""
    n=len(ratios); axes=[]
    for axis in range(3):
        left=ratios[:,2*axis]; right=ratios[:,2*axis+1]
        off=offsets[:,axis][None,:]
        val=torch.where(off<0,-.5*(1+left[:,None]),
              torch.where(off>0,.5*(1+right[:,None]),torch.zeros((n,27),device=ratios.device,dtype=ratios.dtype)))
        axes.append(val)
    return torch.stack(axes,dim=-1)


def projection_cache_from_ratios(ratios: torch.Tensor, offsets: torch.Tensor,
                                 dtype: torch.dtype):
    """Stable solve-based K=A.T(AA.T)^-1; no explicit inverse."""
    eta=eta_from_ratios_torch(ratios.to(dtype),offsets.to(dtype))
    A=geometry_matrix(eta)
    gram=A@A.transpose(1,2)
    eye=torch.eye(10,device=A.device,dtype=A.dtype).expand(len(A),10,10)
    solved=torch.linalg.solve(gram,eye)
    K=A.transpose(1,2)@solved
    return A,K


def project_cached(raw: torch.Tensor, phase: torch.Tensor, A: torch.Tensor,
                   K: torch.Tensor, correction: bool=False) -> torch.Tensor:
    b=phase_rhs(phase)
    w=raw + (K@(b-(A@raw.unsqueeze(-1)).squeeze(-1)).unsqueeze(-1)).squeeze(-1)
    if correction:
        residual=b-(A@w.unsqueeze(-1)).squeeze(-1)
        w=w+(K@residual.unsqueeze(-1)).squeeze(-1)
    return w


def residual10(weights: torch.Tensor, phase: torch.Tensor, A: torch.Tensor):
    return (A@weights.unsqueeze(-1)).squeeze(-1)-phase_rhs(phase)
