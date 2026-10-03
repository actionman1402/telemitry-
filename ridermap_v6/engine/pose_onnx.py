from __future__ import annotations
import numpy as np, cv2
from dataclasses import dataclass

KP_NOSE=0; KP_LSH=5; KP_RSH=6; KP_LEL=7; KP_REL=8; KP_LHIP=11; KP_RHIP=12

@dataclass
class PoseResult:
    points: np.ndarray
    quality: np.ndarray
    person_conf: float
    box: tuple[float,float,float,float]

class PoseEstimator:
    def __init__(self, model_path: str, input_size: int=384):
        import onnxruntime as ort
        available=ort.get_available_providers()
        self.model_path=model_path
        self.session=None
        self.provider='CPUExecutionProvider'
        self.provider_error=''

        # Provider setup is deliberately defensive. Some Windows/VM/Parallels
        # systems advertise DirectML but fail while ONNX Runtime builds the
        # DirectML graph (0x80004005). That must never stop the app opening.
        attempts=[]
        if 'CUDAExecutionProvider' in available:
            attempts.append(('CUDAExecutionProvider',['CUDAExecutionProvider','CPUExecutionProvider']))
        if 'DmlExecutionProvider' in available:
            attempts.append(('DmlExecutionProvider',['DmlExecutionProvider','CPUExecutionProvider']))
        attempts.append(('CPUExecutionProvider',['CPUExecutionProvider']))

        errors=[]
        for label,providers in attempts:
            try:
                opts=ort.SessionOptions()
                opts.graph_optimization_level=ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                opts.intra_op_num_threads=0
                opts.inter_op_num_threads=1
                if label=='DmlExecutionProvider':
                    # Required/recommended for DirectML stability.
                    opts.enable_mem_pattern=False
                    opts.execution_mode=ort.ExecutionMode.ORT_SEQUENTIAL
                self.session=ort.InferenceSession(
                    model_path,
                    sess_options=opts,
                    providers=providers
                )
                actual=self.session.get_providers()
                self.provider=actual[0] if actual else label
                break
            except Exception as e:
                errors.append(f'{label}: {e}')
                self.session=None

        if self.session is None:
            raise RuntimeError('Could not initialise any ONNX Runtime provider. ' + ' | '.join(errors))

        if errors:
            self.provider_error=' | '.join(errors)

        self.input_name=self.session.get_inputs()[0].name
        shape=self.session.get_inputs()[0].shape
        self.size=int(shape[-1]) if isinstance(shape[-1],int) else int(input_size)

    @staticmethod
    def _letterbox(img,size):
        h,w=img.shape[:2]; r=min(size/w,size/h)
        nw,nh=max(1,int(round(w*r))),max(1,int(round(h*r)))
        im=cv2.resize(img,(nw,nh),interpolation=cv2.INTER_LINEAR)
        canvas=np.full((size,size,3),114,dtype=np.uint8)
        dx=(size-nw)//2;dy=(size-nh)//2;canvas[dy:dy+nh,dx:dx+nw]=im
        return canvas,r,dx,dy

    @staticmethod
    def _semantic(pts,qs):
        lsh,rsh=pts[KP_LSH],pts[KP_RSH]; lhip,rhip=pts[KP_LHIP],pts[KP_RHIP]
        shoulder_mid=(lsh+rsh)*0.5; hip_mid=(lhip+rhip)*0.5
        chest=shoulder_mid*0.64+hip_mid*0.36
        ltor=lsh*0.42+lhip*0.58; rtor=rsh*0.42+rhip*0.58
        outpts=np.vstack([pts[KP_NOSE],chest,lsh,rsh,pts[KP_LEL],pts[KP_REL],ltor,rtor]).astype(np.float32)
        outq=np.array([
            qs[KP_NOSE],min(qs[KP_LSH],qs[KP_RSH],qs[KP_LHIP],qs[KP_RHIP]),
            qs[KP_LSH],qs[KP_RSH],qs[KP_LEL],qs[KP_REL],min(qs[KP_LSH],qs[KP_LHIP]),min(qs[KP_RSH],qs[KP_RHIP])
        ],np.float32)
        return outpts,outq

    @staticmethod
    def _scale(p):
        return float(max(np.linalg.norm(p[2]-p[3]),np.linalg.norm(((p[6]+p[7])*0.5)-((p[2]+p[3])*0.5)),0.06))

    def infer(self, frame: np.ndarray, roi: tuple[int,int,int,int]|None=None, expected: np.ndarray|None=None) -> PoseResult|None:
        H,W=frame.shape[:2]
        if roi:
            x1,y1,x2,y2=map(int,roi);x1=max(0,x1);y1=max(0,y1);x2=min(W,x2);y2=min(H,y2)
            crop=frame[y1:y2,x1:x2];ox,oy=x1,y1
        else:
            crop=frame;ox=oy=0
        if crop.size==0:return None
        inp,r,dx,dy=self._letterbox(crop,self.size)
        blob=cv2.cvtColor(inp,cv2.COLOR_BGR2RGB).astype(np.float32)*(1/255.0)
        blob=np.transpose(blob,(2,0,1))[None,...]
        out=self.session.run(None,{self.input_name:blob})[0]
        a=np.asarray(out)
        if a.ndim==3:a=a[0]
        if a.ndim!=2:return None
        if a.shape[0] < a.shape[1] and a.shape[0] <= 128:a=a.T
        if a.shape[1] < 56:return None
        conf=a[:,4].astype(float)
        if np.nanmax(conf)>1.2: conf=1/(1+np.exp(-conf))
        cand=np.flatnonzero(conf>=0.08)
        if not len(cand):return None
        cand=cand[np.argsort(conf[cand])[-min(24,len(cand)):]]
        best=None;best_score=-1e9
        for idx in cand:
            row=a[idx];pc=float(conf[idx]);cx,cy,bw,bh=map(float,row[:4])
            raw=row[5:56].reshape(17,3);qs=raw[:,2].astype(np.float32)
            if np.nanmax(qs)>1.2:qs=1/(1+np.exp(-qs))
            qs=np.clip(qs,0,1)
            pts=np.empty((17,2),np.float32)
            pts[:,0]=((raw[:,0]-dx)/r+ox)/W;pts[:,1]=((raw[:,1]-dy)/r+oy)/H
            sem,semq=self._semantic(pts,qs)
            score=pc
            if expected is not None and np.isfinite(expected).all():
                e=np.asarray(expected,float);s=max(self._scale(e),0.06)
                good=semq>0.16
                if good.sum()>=3:
                    dist=float(np.median(np.linalg.norm(sem[good]-e[good],axis=1)))
                    score-=0.42*min(3.0,dist/s)
                bc=np.array([((cx-dx)/r+ox)/W,((cy-dy)/r+oy)/H])
                ec=np.mean(e[[1,2,3,6,7]],axis=0)
                score-=0.22*min(3.0,float(np.linalg.norm(bc-ec))/s)
            if score>best_score:
                x1n=((cx-bw/2-dx)/r+ox)/W;y1n=((cy-bh/2-dy)/r+oy)/H
                x2n=((cx+bw/2-dx)/r+ox)/W;y2n=((cy+bh/2-dy)/r+oy)/H
                best=PoseResult(np.clip(sem,0,1),semq,pc,(x1n,y1n,x2n,y2n));best_score=score
        return best if best is not None and best.person_conf>=0.10 else None
