from __future__ import annotations
import numpy as np, cv2
from dataclasses import dataclass

NAMES=['head','chest','lShoulder','rShoulder','lElbow','rElbow','lTorso','rTorso']
BONES=((0,1),(2,3),(2,4),(3,5),(2,6),(3,7),(6,7))

@dataclass
class TrackSample:
    frame: int
    time_s: float
    pts: np.ndarray
    quality: np.ndarray
    global_quality: float

class SparseSkeletonTracker:
    def __init__(self, first_bgr: np.ndarray, initial_pts_norm: np.ndarray):
        self.h,self.w=first_bgr.shape[:2]
        self.prev=cv2.cvtColor(first_bgr,cv2.COLOR_BGR2GRAY)
        self.pts=np.clip(initial_pts_norm.astype(np.float32).copy(),0,1)
        self.velocity=np.zeros_like(self.pts,np.float32)
        self.features=None; self.owner=None; self.age=0; self.force_seed=False
        self.body_scale=max(self._body_scale(self.pts),0.05);self._seed(self.prev)

    @staticmethod
    def _body_scale(p):
        sw=np.linalg.norm(p[2]-p[3]);torso=np.linalg.norm(((p[6]+p[7])*0.5)-((p[2]+p[3])*0.5))
        arm=max(np.linalg.norm(p[2]-p[4]),np.linalg.norm(p[3]-p[5]))
        return float(max(sw,torso,arm*0.65,0.025))

    def _body_mask(self, shape):
        h,w=shape;mask=np.zeros((h,w),np.uint8);P=(self.pts*np.array([w,h],np.float32)).astype(np.int32)
        rad=max(9,int(min(w,h)*0.045))
        for a,b in BONES:cv2.line(mask,tuple(P[a]),tuple(P[b]),255,max(8,int(rad*1.25)),cv2.LINE_AA)
        for p in P:cv2.circle(mask,tuple(p),rad,255,-1,cv2.LINE_AA)
        k=max(3,int(rad*0.35)|1);return cv2.dilate(mask,np.ones((k,k),np.uint8),iterations=1)

    def _seed(self, gray):
        feats=[];owners=[];mask=self._body_mask(gray.shape)
        q=cv2.goodFeaturesToTrack(gray,maxCorners=90,qualityLevel=0.012,minDistance=5,blockSize=5,mask=mask,useHarrisDetector=False)
        if q is not None:
            q=q.reshape(-1,2);feats.extend(q.tolist());owners.extend([-1]*len(q))
        rad=max(8,int(min(self.w,self.h)*0.030))
        for j,(nx,ny) in enumerate(self.pts):
            x=int(nx*self.w);y=int(ny*self.h);x1=max(0,x-rad);x2=min(self.w,x+rad+1);y1=max(0,y-rad);y2=min(self.h,y+rad+1)
            roi=gray[y1:y2,x1:x2]
            if roi.size:
                lq=cv2.goodFeaturesToTrack(roi,maxCorners=5,qualityLevel=0.018,minDistance=4,blockSize=5,useHarrisDetector=False)
                if lq is not None:
                    lq=lq.reshape(-1,2);lq[:,0]+=x1;lq[:,1]+=y1;feats.extend(lq.tolist());owners.extend([j]*len(lq))
            feats.append([float(x),float(y)]);owners.append(j)
        if not feats:feats=[[float(self.pts[1,0]*self.w),float(self.pts[1,1]*self.h)]];owners=[1]
        self.features=np.asarray(feats,np.float32).reshape(-1,1,2);self.owner=np.asarray(owners,np.int16);self.age=0;self.force_seed=False

    def _norm(self,pix):
        a=pix.astype(np.float32).copy();a[:,0]/=self.w;a[:,1]/=self.h;return a

    def _apply_affine(self,pts_norm,M):
        if M is None:return pts_norm.copy()
        p=pts_norm*np.array([self.w,self.h],np.float32);o=cv2.transform(p.reshape(-1,1,2),M).reshape(-1,2);return self._norm(o)

    def relock(self,anchor_pts,anchor_q,reference_pts):
        a=np.asarray(anchor_pts,np.float32);r=np.asarray(reference_pts,np.float32);q=np.asarray(anchor_q,np.float32)
        valid=np.isfinite(a).all(axis=1)&np.isfinite(r).all(axis=1)&(q>=0.28)
        if valid.sum()<4:return False
        scale=max(self._body_scale(r),self.body_scale,0.04);med_shift=float(np.median(np.linalg.norm(a[valid]-r[valid],axis=1)))
        if med_shift>max(0.14,scale*1.20):return False
        src=(r[valid]*1000).astype(np.float32);dst=(a[valid]*1000).astype(np.float32)
        M,_=cv2.estimateAffinePartial2D(src,dst,method=cv2.RANSAC,ransacReprojThreshold=35,maxIters=200,confidence=0.99)
        if M is not None:
            cur=cv2.transform((self.pts*1000).reshape(-1,1,2),M).reshape(-1,2)/1000.0
            fit=cv2.transform((r*1000).reshape(-1,1,2),M).reshape(-1,2)/1000.0
        else:
            d=np.median(a[valid]-r[valid],axis=0);cur=self.pts+d;fit=r+d
        residual=a-fit
        for j in range(8):
            if not valid[j]:continue
            lim=scale*(0.20 if j in (4,5) else 0.14);rr=residual[j];rn=float(np.linalg.norm(rr))
            if rn>lim:rr*=lim/(rn+1e-9)
            cur[j]+=rr*(0.55+0.25*float(np.clip(q[j],0,1)))
        self.pts=np.clip(0.15*self.pts+0.85*cur,0,1).astype(np.float32);self.velocity*=0.55;self.force_seed=True;self._seed(self.prev);return True

    def step(self,bgr,frame_i,time_s):
        gray=cv2.cvtColor(bgr,cv2.COLOR_BGR2GRAY)
        if self.force_seed or self.features is None or len(self.features)<12:self._seed(self.prev)
        nxt,st,_=cv2.calcOpticalFlowPyrLK(self.prev,gray,self.features,None,winSize=(17,17),maxLevel=2,criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,12,0.02),flags=0,minEigThreshold=1e-4)
        if nxt is None:
            pred=np.clip(self.pts+self.velocity*0.65,0,1);self.velocity*=0.75;self.pts=pred;self.prev=gray;self._seed(gray)
            return TrackSample(frame_i,time_s,pred.copy(),np.zeros(8,np.float32),0.0)
        back,bst,_=cv2.calcOpticalFlowPyrLK(gray,self.prev,nxt,None,winSize=(17,17),maxLevel=2,criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,10,0.02),flags=0,minEigThreshold=1e-4)
        old=self.features.reshape(-1,2);new=nxt.reshape(-1,2);bk=back.reshape(-1,2);fb=np.linalg.norm(bk-old,axis=1)
        valid=(st.reshape(-1)>0)&(bst.reshape(-1)>0)&np.isfinite(fb)&(fb<1.6)
        velpx=np.linalg.norm(new-old,axis=1)/max(self.w,self.h);valid &= velpx < max(0.035,self.body_scale*0.48)
        gidx=valid&(self.owner<0)
        if gidx.sum()<7:gidx=valid
        M=None;inl=None
        if gidx.sum()>=6:M,inl=cv2.estimateAffinePartial2D(old[gidx],new[gidx],method=cv2.RANSAC,ransacReprojThreshold=2.0,maxIters=300,confidence=0.995,refineIters=4)
        old_pts=self.pts.copy()
        if M is not None:
            base=self._apply_affine(self.pts,M);inlier_ratio=float(np.mean(inl)) if inl is not None and len(inl) else 0.0
            global_q=float(np.clip(inlier_ratio*np.exp(-np.median(fb[gidx])/1.2),0,1))
        else:
            d=np.median((new-old)[gidx],axis=0) if gidx.any() else np.array([0,0],np.float32)
            dn=np.array([d[0]/self.w,d[1]/self.h],np.float32);base=np.clip(self.pts+dn+self.velocity*0.15,0,1);global_q=0.12 if gidx.any() else 0.0
        out=base.copy();qual=np.full(8,global_q*0.45,np.float32)
        if M is not None:pred_feat=cv2.transform(old.reshape(-1,1,2),M).reshape(-1,2);feat_res=new-pred_feat
        else:
            gd=(base-old_pts)*np.array([self.w,self.h],np.float32);feat_res=np.zeros_like(new)
            for j in range(8):feat_res[self.owner==j]=((new-old)[self.owner==j]-gd[j])
        for j in range(8):
            idx=valid&(self.owner==j)
            if idx.sum()>=2:
                rr=np.median(feat_res[idx],axis=0);rrn=np.array([rr[0]/self.w,rr[1]/self.h],np.float32)
                lim=max(0.0035,self.body_scale*(0.11 if j in (4,5) else 0.075));rn=float(np.linalg.norm(rrn))
                if rn>lim:rrn*=lim/(rn+1e-9)
                out[j]=base[j]+rrn;fbe=float(np.median(fb[idx]));count=min(1.0,idx.sum()/4.0)
                qual[j]=float(np.clip(0.25*global_q+0.75*count*np.exp(-fbe/1.0),0,1))
        oldc=np.mean(old_pts[[1,2,3,6,7]],axis=0);newc=np.mean(out[[1,2,3,6,7]],axis=0);oldrel=old_pts-oldc;newrel=out-newc;relstep=newrel-oldrel
        for j in range(8):
            lim=max(0.004,self.body_scale*(0.18 if j in (4,5) else 0.11));rn=float(np.linalg.norm(relstep[j]))
            if rn>lim:newrel[j]=oldrel[j]+relstep[j]*lim/(rn+1e-9)
        out=np.clip(newc+newrel,0,1);delta=out-old_pts;self.velocity=(self.velocity*0.72+delta*0.28).astype(np.float32)
        self.pts=out.astype(np.float32);ns=self._body_scale(self.pts);self.body_scale=float(np.clip(0.97*self.body_scale+0.03*ns,0.025,0.75))
        keep=valid
        if keep.sum()>=20:self.features=new[keep].astype(np.float32).reshape(-1,1,2);self.owner=self.owner[keep]
        else:self.features=None;self.owner=None
        self.prev=gray;self.age+=1
        if self.age>=7 or self.features is None or len(self.features)<24:self._seed(gray)
        return TrackSample(frame_i,time_s,self.pts.copy(),qual,float(np.clip(np.median(qual),0,1)))
