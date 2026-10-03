from __future__ import annotations
from pathlib import Path
import numpy as np
from .pose_onnx import PoseEstimator as OnnxPoseEstimator, PoseResult, KP_NOSE, KP_LSH, KP_RSH, KP_LEL, KP_REL, KP_LHIP, KP_RHIP

class PoseEstimator:
    """Uses ONNX/DirectML when available; falls back to an existing Ultralytics .pt model."""
    def __init__(self, model_path: str, input_size: int=384):
        self.model_path=str(model_path);self.size=int(input_size)
        if Path(model_path).suffix.lower()=='.onnx':
            self.impl=OnnxPoseEstimator(model_path,input_size)
            self.provider=self.impl.provider
            self.kind='ONNX'
        else:
            from ultralytics import YOLO
            self.model=YOLO(model_path)
            try:
                import torch
                self.device=0 if torch.cuda.is_available() else 'cpu'
            except Exception:self.device='cpu'
            self.provider='PyTorch CUDA' if self.device==0 else 'PyTorch CPU'
            self.kind='PT'

    @staticmethod
    def _semantic(pts,qs):
        lsh,rsh=pts[KP_LSH],pts[KP_RSH];lhip,rhip=pts[KP_LHIP],pts[KP_RHIP]
        shoulder=(lsh+rsh)*0.5;hip=(lhip+rhip)*0.5
        chest=shoulder*0.64+hip*0.36;ltor=lsh*0.42+lhip*0.58;rtor=rsh*0.42+rhip*0.58
        op=np.vstack([pts[KP_NOSE],chest,lsh,rsh,pts[KP_LEL],pts[KP_REL],ltor,rtor]).astype(np.float32)
        oq=np.array([qs[KP_NOSE],min(qs[KP_LSH],qs[KP_RSH],qs[KP_LHIP],qs[KP_RHIP]),qs[KP_LSH],qs[KP_RSH],qs[KP_LEL],qs[KP_REL],min(qs[KP_LSH],qs[KP_LHIP]),min(qs[KP_RSH],qs[KP_RHIP])],np.float32)
        return op,oq

    @staticmethod
    def _scale(p):
        return float(max(np.linalg.norm(p[2]-p[3]),np.linalg.norm(((p[6]+p[7])*0.5)-((p[2]+p[3])*0.5)),0.06))

    def infer(self,frame,roi=None,expected=None):
        if hasattr(self,'impl'):return self.impl.infer(frame,roi,expected)
        H,W=frame.shape[:2]
        if roi:
            x1,y1,x2,y2=map(int,roi);x1=max(0,x1);y1=max(0,y1);x2=min(W,x2);y2=min(H,y2)
            crop=frame[y1:y2,x1:x2];ox,oy=x1,y1
        else:crop=frame;ox=oy=0
        if crop.size==0:return None
        rr=self.model.predict(crop,imgsz=self.size,conf=0.08,verbose=False,device=self.device,max_det=8)
        if not rr:return None
        r=rr[0]
        if r.keypoints is None or r.boxes is None or len(r.boxes)==0:return None
        xy=r.keypoints.xy.cpu().numpy();
        if getattr(r.keypoints,'conf',None) is not None:qall=r.keypoints.conf.cpu().numpy()
        else:qall=np.ones((len(xy),17),np.float32)*0.5
        conf=r.boxes.conf.cpu().numpy();boxes=r.boxes.xyxy.cpu().numpy()
        best=None;best_score=-1e9
        for i in range(len(xy)):
            pts=np.asarray(xy[i],np.float32);pts[:,0]=(pts[:,0]+ox)/W;pts[:,1]=(pts[:,1]+oy)/H
            qs=np.clip(np.asarray(qall[i],np.float32),0,1);sem,semq=self._semantic(pts,qs);score=float(conf[i])
            if expected is not None and np.isfinite(expected).all():
                e=np.asarray(expected,float);good=semq>0.16;s=max(self._scale(e),0.06)
                if good.sum()>=3:score-=0.42*min(3.0,float(np.median(np.linalg.norm(sem[good]-e[good],axis=1)))/s)
            if score>best_score:
                b=boxes[i];box=((b[0]+ox)/W,(b[1]+oy)/H,(b[2]+ox)/W,(b[3]+oy)/H)
                best=PoseResult(np.clip(sem,0,1),semq,float(conf[i]),box);best_score=score
        return best
