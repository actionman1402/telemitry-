from __future__ import annotations
import numpy as np

PAIRS=((2,3),(4,5),(6,7))

def _scale_frame(p):
    sw=np.linalg.norm(p[2]-p[3])
    torso=np.linalg.norm(((p[6]+p[7])*0.5)-((p[2]+p[3])*0.5))
    arm=max(np.linalg.norm(p[2]-p[4]),np.linalg.norm(p[3]-p[5]))
    return float(max(sw,torso,arm*0.65,0.035))

def _semantic_match(a,pred):
    a=a.copy()
    for i,j in PAIRS:
        keep=np.linalg.norm(a[i]-pred[i])+np.linalg.norm(a[j]-pred[j])
        swap=np.linalg.norm(a[j]-pred[i])+np.linalg.norm(a[i]-pred[j])
        if swap+1e-6 < keep*0.90:a[[i,j]]=a[[j,i]]
    return a

def _interp_bad(values,good):
    n=len(values);x=np.arange(n);out=values.copy();gi=np.flatnonzero(good)
    if len(gi)==0:return out
    if len(gi)==1:out[:]=values[gi[0]];return out
    for d in range(values.shape[1]):
        out[:,d]=np.interp(x,gi,values[gi,d],left=values[gi[0],d],right=values[gi[-1],d])
    return out

def _zero_phase_ema(x,alpha):
    y=x.copy()
    for i in range(1,len(y)):y[i]=alpha*y[i]+(1-alpha)*y[i-1]
    z=y.copy()
    for i in range(len(z)-2,-1,-1):z[i]=alpha*z[i]+(1-alpha)*z[i+1]
    return z

def _rolling_median(v,window=21):
    n=len(v);o=np.empty_like(v);h=window//2
    for i in range(n):
        a=max(0,i-h);b=min(n,i+h+1);o[i]=np.nanmedian(v[a:b])
    return o

def stabilize(raw_pts: np.ndarray, raw_q: np.ndarray, anchors: dict[int,tuple[np.ndarray,np.ndarray]], timestamps: np.ndarray):
    raw=np.asarray(raw_pts,float);q=np.asarray(raw_q,float).copy();n=len(raw)
    if n==0:return raw_pts,raw_q
    corrected=raw.copy();scales=np.array([_scale_frame(p) for p in raw],float)
    per=[[] for _ in range(8)];anchor_quality=np.zeros(n,float);last_good=None
    for fi,(ap,aq) in sorted(anchors.items()):
        if not (0<=fi<n):continue
        aq=np.asarray(aq,float);a=_semantic_match(np.asarray(ap,float),raw[fi]);good=aq>=0.22
        if good.sum()<4:continue
        s=max(scales[fi],0.04);med=float(np.median(np.linalg.norm(a[good]-raw[fi,good],axis=1)))
        if med>max(0.15,s*1.25):continue
        if last_good is not None:
            lfi,la=last_good;dt=max(1,fi-lfi)
            c0=np.mean(la[[1,2,3,6,7]],axis=0);c1=np.mean(a[[1,2,3,6,7]],axis=0)
            if np.linalg.norm(c1-c0)>max(0.20,s*(0.50+0.12*dt)):continue
        delta=a-raw[fi];medvec=np.median(delta[good],axis=0)
        spread=float(np.median(np.linalg.norm(delta[good]-medvec,axis=1)));whole=spread<s*0.22;accepted=0
        for j in range(8):
            if aq[j]<0.25:continue
            mag=float(np.linalg.norm(delta[j]));lim=max(0.035,s*(1.05 if whole else (0.58 if j in (4,5) else 0.42)))
            if mag<=lim:per[j].append((fi,delta[j].copy(),float(aq[j])));accepted+=1
        if accepted>=4:
            anchor_quality[fi]=float(np.median(aq[aq>=0.25]));last_good=(fi,a.copy())
    xs=np.arange(n,dtype=float)
    for j,arr in enumerate(per):
        if not arr:continue
        f=np.array([v[0] for v in arr],float);d=np.array([v[1] for v in arr],float)
        if len(d)>=3:
            keep=np.ones(len(d),bool)
            for k in range(1,len(d)-1):
                pred=(d[k-1]+d[k+1])*0.5;s=max(scales[int(f[k])],0.04)
                if np.linalg.norm(d[k]-pred)>s*(0.55 if j in (4,5) else 0.40):keep[k]=False
            if keep.sum()>=2:f,d=f[keep],d[keep]
        cx=np.interp(xs,f,d[:,0],left=d[0,0],right=d[-1,0]);cy=np.interp(xs,f,d[:,1],left=d[0,1],right=d[-1,1])
        corrected[:,j]+=np.c_[cx,cy]
    raw_step=np.linalg.norm(np.diff(raw,axis=0,prepend=raw[:1]),axis=2)
    for j in range(8):
        local_scale=np.maximum(scales,0.04);medstep=_rolling_median(raw_step[:,j],15)
        spike=raw_step[:,j] > np.maximum(0.028,local_scale*(0.28 if j in (4,5) else 0.20))
        low=q[:,j]<0.22;frozen=raw_step[:,j]<8e-5
        body_center=np.mean(raw[:,[1,2,3,6,7]],axis=1);body_step=np.linalg.norm(np.diff(body_center,axis=0,prepend=body_center[:1]),axis=1)
        frozen &= body_step>4e-4
        good=np.isfinite(corrected[:,j]).all(axis=1)&~spike&~(low&frozen)
        good |= (low & ~spike & ~frozen & (raw_step[:,j] < np.maximum(0.015,medstep*3+0.004)))
        if good.sum()>=2:corrected[:,j]=_interp_bad(corrected[:,j],good)
    center=np.mean(corrected[:,[1,2,3,6,7]],axis=1);center_s=_zero_phase_ema(center,0.74)
    rel=corrected-center[:,None,:]
    for j in range(8):rel[:,j]=_zero_phase_ema(rel[:,j],0.61 if j in (4,5) else 0.56)
    corrected=center_s[:,None,:]+rel
    def limit_pass(arr,forward=True):
        rng=range(1,n) if forward else range(n-2,-1,-1);prev=lambda i:i-1 if forward else i+1
        centers=np.mean(arr[:,[1,2,3,6,7]],axis=1)
        for i in rng:
            k=prev(i);s=max(scales[i],0.04)
            for j in range(8):
                r0=arr[k,j]-centers[k];r1=arr[i,j]-centers[i];d=r1-r0;lim=s*(0.18 if j in (4,5) else 0.115);dn=float(np.linalg.norm(d))
                if dn>lim:arr[i,j]=centers[i]+r0+d*lim/(dn+1e-9)
        return arr
    corrected=limit_pass(corrected,True);corrected=limit_pass(corrected,False)
    def constrain_pair(a,b,lo=0.62,hi=1.55):
        nonlocal corrected
        d=corrected[:,b]-corrected[:,a];L=np.linalg.norm(d,axis=1);target=_rolling_median(L,31)
        for i,l in enumerate(L):
            t=max(target[i],0.012);cl=float(np.clip(l,t*lo,t*hi))
            if l>1e-8 and abs(cl-l)>1e-9:
                c=(corrected[i,a]+corrected[i,b])*0.5;v=d[i]/l
                corrected[i,a]=c-v*cl*0.5;corrected[i,b]=c+v*cl*0.5
    constrain_pair(2,3,0.58,1.62)
    for a,b in ((2,4),(3,5)):
        d=corrected[:,b]-corrected[:,a];L=np.linalg.norm(d,axis=1);target=_rolling_median(L,31)
        for i,l in enumerate(L):
            t=max(target[i],0.015);cl=float(np.clip(l,t*0.66,t*1.48))
            if l>1e-8:corrected[i,b]=corrected[i,a]+d[i]*(cl/l)
    shoulder_mid=(corrected[:,2]+corrected[:,3])*0.5;torso_mid=(corrected[:,6]+corrected[:,7])*0.5
    chest_geom=shoulder_mid*0.66+torso_mid*0.34;corrected[:,1]=0.28*corrected[:,1]+0.72*chest_geom
    corrected=np.clip(corrected,0,1)
    median_jump=np.median(raw_step,axis=1);bad=np.clip((median_jump-0.018)/0.055,0,1);q*=1-bad[:,None]*0.72
    if anchor_quality.max()>0:
        aq=anchor_quality.copy();inds=np.flatnonzero(aq>0)
        if len(inds):
            interp=np.interp(np.arange(n),inds,aq[inds],left=aq[inds[0]],right=aq[inds[-1]]);q=np.maximum(q,interp[:,None]*0.45)
    return corrected.astype(np.float32),np.clip(q,0,1).astype(np.float32)
