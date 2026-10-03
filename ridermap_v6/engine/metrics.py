from __future__ import annotations
import numpy as np

def derived(pts):
    # pts order: head,chest,lS,rS,lE,rE,lT,rT
    head,chest,ls,rs,le,re,lt,rt=[pts[:,i] for i in range(8)]
    shoulder_mid=(ls+rs)/2; torso_mid=(lt+rt)/2
    # Screen-space metrics; RWB remains a proxy, not a physical CoG measurement.
    cg=(0.11*head+0.31*chest+0.13*ls+0.13*rs+0.08*le+0.08*re+0.08*lt+0.08*rt)
    sw=np.linalg.norm(ls-rs,axis=1); th=np.linalg.norm(shoulder_mid-torso_mid,axis=1)
    depth=(0.55*sw+0.45*th); base=np.median(depth[:min(len(depth),20)]) if len(depth) else 1
    depth_ratio=depth/(base+1e-9)
    rwb=0.65 + (depth_ratio-1.0)*0.28
    cgy_rel=cg[0,1]-cg[:,1] if len(cg) else np.array([])
    return cg,rwb,cgy_rel,depth_ratio
