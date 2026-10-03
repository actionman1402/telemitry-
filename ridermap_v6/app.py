from __future__ import annotations
import os, sys, csv, time, threading, queue, traceback, shutil
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np, cv2
from PIL import Image, ImageTk

from engine.ffmpeg_decode import probe_video, FFmpegFrameReader
from engine.pose_backend import PoseEstimator
from engine.tracker import SparseSkeletonTracker, NAMES, BONES
from engine.stabilizer import stabilize
from engine.metrics import derived

APP_NAME='Rider Mapping Pro V6'
BG='#0a0f1a'; CARD='#111827'; CARD2='#172033'; BORDER='#263247'; TEXT='#f4f7fb'; MUTED='#94a3b8'
ACCENT='#36a3ff'; ACCENT2='#63d7b0'; WARN='#ffb454'; BAD='#ff667a'; GRID='#263247'

def resource_path(*parts):
    base=Path(getattr(sys,'_MEIPASS',Path(__file__).resolve().parent))
    return str(base.joinpath(*parts))

def _candidate_roots():
    roots=[Path(resource_path()),Path.cwd()]
    lad=os.environ.get('LOCALAPPDATA')
    if lad:
        roots += [Path(lad)/'RiderMappingProV5Build',Path(lad)/'RiderMappingProV5Runtime',Path(lad)/'Programs'/'RiderMappingProV5',Path(lad)/'RiderMappingProV6']
    return roots

def bundled_or_path(name):
    direct=Path(resource_path('runtime',name))
    if direct.exists():return str(direct)
    for root in _candidate_roots():
        for rel in (name,Path('runtime')/name,Path('ffmpeg')/name,Path('bin')/name):
            p=root/rel
            if p.exists():return str(p)
    x=shutil.which(name)
    if x:return x
    raise FileNotFoundError(f'{name} was not found. Install FFmpeg or place {name} beside Rider Mapping.')

def find_model(mode):
    env=os.environ.get('RIDER_MAPPING_MODEL')
    if env and Path(env).exists():return env
    order={
        'Turbo':['yolo11n-pose.onnx','yolo11n-pose.pt','yolo11m-pose.onnx','yolo11m-pose.pt'],
        'Balanced':['yolo11n-pose.onnx','yolo11m-pose.onnx','yolo11n-pose.pt','yolo11m-pose.pt'],
        'Precision':['yolo11m-pose.onnx','yolo11n-pose.onnx','yolo11x-pose.onnx','yolo11m-pose.pt','yolo11n-pose.pt','yolo11x-pose.pt']
    }[mode]
    roots=_candidate_roots()+[Path.home()/'.cache'/'ultralytics']
    for n in order:
        for root in roots:
            for rel in (Path('models')/n,Path(n)):
                p=root/rel
                if p.exists():return str(p)
    raise FileNotFoundError('No bundled pose model found.')

def body_roi(frame,pts,pad_factor=0.55):
    h,w=frame.shape[:2];p=np.asarray(pts,float);xy=p*np.array([w,h])
    x1,y1=np.min(xy,axis=0);x2,y2=np.max(xy,axis=0);span=max(x2-x1,y2-y1,70);pad=max(32,span*pad_factor)
    return (int(x1-pad),int(y1-pad),int(x2+pad),int(y2+pad))

class AnchorWorker:
    def __init__(self,pose,max_pending=6):
        self.pose=pose;self.q=queue.Queue(maxsize=max_pending);self.results={};self.lock=threading.Lock();self.polled=set()
        self.thread=threading.Thread(target=self._run,daemon=True);self.thread.start()
    def submit(self,idx,frame,raw_pts):
        h,w=frame.shape[:2];scale=min(1.0,720.0/max(w,h))
        if scale<1:sm=cv2.resize(frame,(max(2,int(w*scale)//2*2),max(2,int(h*scale)//2*2)),interpolation=cv2.INTER_AREA)
        else:sm=frame.copy()
        roi=body_roi(sm,raw_pts,0.62);item=(idx,sm,roi,np.asarray(raw_pts,np.float32).copy())
        try:self.q.put_nowait(item)
        except queue.Full:
            try:self.q.get_nowait();self.q.task_done()
            except queue.Empty:pass
            try:self.q.put_nowait(item)
            except queue.Full:pass
    def _run(self):
        while True:
            item=self.q.get()
            if item is None:self.q.task_done();break
            idx,frame,roi,expected=item
            try:
                r=self.pose.infer(frame,roi,expected)
                if r is None:r=self.pose.infer(frame,None,expected)
                if r is not None:
                    with self.lock:self.results[idx]=(r.points,r.quality,float(r.person_conf))
            except Exception:pass
            finally:self.q.task_done()
    def poll(self):
        out=[]
        with self.lock:
            for k in sorted(self.results):
                if k not in self.polled:out.append((k,self.results[k]));self.polled.add(k)
        return out
    def finish(self):
        self.q.join();self.q.put(None);self.thread.join(timeout=10)
        with self.lock:return dict(self.results)
    @property
    def pending(self):return self.q.qsize()

def resample_uniform(ts,pts,q,frame_ids,fps,anchor_indices):
    ts=np.asarray(ts,float);pts=np.asarray(pts,float);q=np.asarray(q,float);frame_ids=np.asarray(frame_ids,int)
    if len(ts)<2 or fps<=0:return ts,pts,q,frame_ids,np.ones(len(ts),np.uint8),np.array([1 if i in anchor_indices else 0 for i in range(len(ts))],np.uint8)
    step=1.0/fps;end=ts[-1];grid=np.arange(0.0,end+step*0.25,step,dtype=float)
    op=np.empty((len(grid),8,2),np.float32);oq=np.empty((len(grid),8),np.float32)
    for j in range(8):
        for d in range(2):op[:,j,d]=np.interp(grid,ts,pts[:,j,d])
        oq[:,j]=np.interp(grid,ts,q[:,j])
    pos=np.searchsorted(ts,grid);pos=np.clip(pos,0,len(ts)-1);prev=np.clip(pos-1,0,len(ts)-1)
    choose=np.where(np.abs(ts[prev]-grid)<=np.abs(ts[pos]-grid),prev,pos);near=np.abs(ts[choose]-grid)<=step*0.22
    src=np.where(near,frame_ids[choose],-1);present=near.astype(np.uint8);ats=np.array([ts[i] for i in anchor_indices if 0<=i<len(ts)],float)
    ai=np.min(np.abs(grid[:,None]-ats[None,:]),axis=1)<=step*0.35 if len(ats) else np.zeros(len(grid),bool)
    return grid,op,oq,src,present,ai.astype(np.uint8)

class App(tk.Tk):
    def __init__(self):
        super().__init__();self.title(APP_NAME);self.geometry('1380x860');self.minsize(1120,700);self.configure(bg=BG)
        self.option_add('*Font','Segoe UI 10');cv2.setUseOptimized(True)
        try:cv2.setNumThreads(max(1,min(8,(os.cpu_count() or 4)-1)))
        except Exception:pass
        self.video='';self.running=False;self.cancel_flag=False;self.preview_photo=None
        self.mode=tk.StringVar(value='Balanced');self.resample=tk.BooleanVar(value=True)
        self.status=tk.StringVar(value='Ready');self.stage=tk.StringVar(value='Open a riding clip to begin')
        self.fps_var=tk.StringVar(value='—');self.frames_var=tk.StringVar(value='—');self.ai_var=tk.StringVar(value='—');self.quality_var=tk.StringVar(value='—')
        self.source_var=tk.StringVar(value='No video selected');self.output_var=tk.StringVar(value='No output yet');self._build()

    def _card(self,parent):return tk.Frame(parent,bg=CARD,highlightbackground=BORDER,highlightthickness=1,bd=0)
    def _label(self,parent,text='',var=None,size=10,bold=False,fg=TEXT,bg=CARD,**kw):
        return tk.Label(parent,text=text,textvariable=var,font=('Segoe UI',size,'bold' if bold else 'normal'),fg=fg,bg=bg,**kw)
    def _button(self,parent,text,command,accent=False,width=None):
        bg=ACCENT if accent else CARD2;active='#58b4ff' if accent else '#22304a'
        return tk.Button(parent,text=text,command=command,bg=bg,fg='white',activebackground=active,activeforeground='white',relief='flat',bd=0,cursor='hand2',padx=16,pady=10,font=('Segoe UI',10,'bold'),width=width)

    def _build(self):
        st=ttk.Style(self);st.theme_use('clam');st.configure('Blue.Horizontal.TProgressbar',troughcolor=CARD2,background=ACCENT,bordercolor=CARD2,lightcolor=ACCENT,darkcolor=ACCENT)
        header=tk.Frame(self,bg=BG,height=72);header.pack(fill='x',padx=24,pady=(18,8));header.pack_propagate(False)
        left=tk.Frame(header,bg=BG);left.pack(side='left',fill='y');self._label(left,'RIDER MAPPING',size=19,bold=True,bg=BG).pack(anchor='w')
        self._label(left,'Pro V6  •  semantic rider tracking',size=9,fg=MUTED,bg=BG).pack(anchor='w',pady=(2,0))
        tk.Label(header,textvariable=self.status,bg='#12263d',fg='#7cc5ff',font=('Segoe UI',9,'bold'),padx=12,pady=7).pack(side='right',pady=12)
        toolbar=self._card(self);toolbar.pack(fill='x',padx=24,pady=(0,14));inner=tk.Frame(toolbar,bg=CARD);inner.pack(fill='x',padx=14,pady=12)
        self._button(inner,'Open video',self.open_video).pack(side='left');tk.Label(inner,textvariable=self.source_var,bg=CARD,fg=MUTED,anchor='w').pack(side='left',fill='x',expand=True,padx=14)
        self.mode_wrap=tk.Frame(inner,bg=CARD);self.mode_wrap.pack(side='left',padx=8);self.mode_buttons={}
        for m in ('Turbo','Balanced','Precision'):
            b=tk.Button(self.mode_wrap,text=m,command=lambda x=m:self.set_mode(x),relief='flat',bd=0,padx=11,pady=8,cursor='hand2',font=('Segoe UI',9,'bold'));b.pack(side='left',padx=1);self.mode_buttons[m]=b
        self.set_mode('Balanced');self.run_btn=self._button(inner,'Analyse',self.start,accent=True);self.run_btn.pack(side='left',padx=(8,4));self.cancel_btn=self._button(inner,'Cancel',self.cancel);self.cancel_btn.pack(side='left');self.cancel_btn.config(state='disabled')
        main=tk.Frame(self,bg=BG);main.pack(fill='both',expand=True,padx=24,pady=(0,12));main.grid_columnconfigure(0,weight=3);main.grid_columnconfigure(1,weight=1,minsize=350);main.grid_rowconfigure(0,weight=1)
        preview_card=self._card(main);preview_card.grid(row=0,column=0,sticky='nsew',padx=(0,12));phead=tk.Frame(preview_card,bg=CARD);phead.pack(fill='x',padx=16,pady=(14,10))
        self._label(phead,'Tracking preview',size=12,bold=True).pack(side='left');self._label(phead,'confidence colours: green / amber / red',size=8,fg=MUTED).pack(side='right')
        self.preview=tk.Label(preview_card,text='Open a video to begin\n\nV6 analyses a reduced working frame for speed,\nthen exports normalised coordinates.',bg='#050912',fg=MUTED,anchor='center',font=('Segoe UI',11));self.preview.pack(fill='both',expand=True,padx=14,pady=(0,14))
        side=tk.Frame(main,bg=BG);side.grid(row=0,column=1,sticky='nsew')
        settings=self._card(side);settings.pack(fill='x',pady=(0,10));self._label(settings,'Analysis',size=12,bold=True).pack(anchor='w',padx=15,pady=(14,8))
        tk.Label(settings,text='Fast optical tracking on every decoded picture.\nAI pose runs in parallel and re-locks drift.',justify='left',bg=CARD,fg=MUTED,font=('Segoe UI',9)).pack(anchor='w',padx=15)
        tk.Checkbutton(settings,text='Resample output to exact source FPS grid',variable=self.resample,bg=CARD,fg=TEXT,selectcolor=CARD2,activebackground=CARD,activeforeground=TEXT,font=('Segoe UI',9),highlightthickness=0).pack(anchor='w',padx=12,pady=(10,14))
        live=self._card(side);live.pack(fill='x',pady=(0,10));self._label(live,'Live performance',size=12,bold=True).pack(anchor='w',padx=15,pady=(14,10));grid=tk.Frame(live,bg=CARD);grid.pack(fill='x',padx=15,pady=(0,14))
        for i,(name,var) in enumerate([('Processing',self.fps_var),('Decoded',self.frames_var),('AI anchors',self.ai_var),('Quality',self.quality_var)]):
            cell=tk.Frame(grid,bg=CARD2);cell.grid(row=i//2,column=i%2,sticky='nsew',padx=(0 if i%2==0 else 5,5 if i%2==0 else 0),pady=3);grid.grid_columnconfigure(i%2,weight=1)
            self._label(cell,name,size=8,fg=MUTED,bg=CARD2).pack(anchor='w',padx=9,pady=(7,0));self._label(cell,var=var,size=12,bold=True,bg=CARD2).pack(anchor='w',padx=9,pady=(1,7))
        mapcard=self._card(side);mapcard.pack(fill='both',expand=True,pady=(0,10));self._label(mapcard,'Rider CG trace',size=12,bold=True).pack(anchor='w',padx=15,pady=(14,8))
        self.map_canvas=tk.Canvas(mapcard,bg='#080d16',height=220,highlightthickness=0);self.map_canvas.pack(fill='both',expand=True,padx=12,pady=(0,12))
        output=self._card(side);output.pack(fill='x');self._label(output,'Output',size=10,bold=True).pack(anchor='w',padx=15,pady=(12,4));self._label(output,var=self.output_var,size=8,fg=MUTED,wraplength=320,justify='left').pack(anchor='w',padx=15,pady=(0,12))
        footer=self._card(self);footer.pack(fill='x',padx=24,pady=(0,18));ftop=tk.Frame(footer,bg=CARD);ftop.pack(fill='x',padx=14,pady=(10,6))
        self._label(ftop,var=self.stage,size=9,fg=MUTED).pack(side='left');self.perf=tk.StringVar(value='');self._label(ftop,var=self.perf,size=9,fg=MUTED).pack(side='right')
        self.progress=ttk.Progressbar(footer,style='Blue.Horizontal.TProgressbar',mode='determinate');self.progress.pack(fill='x',padx=14,pady=(0,12))

    def set_mode(self,mode):
        self.mode.set(mode)
        for m,b in self.mode_buttons.items():b.config(bg=ACCENT if m==mode else CARD2,fg='white',activebackground='#58b4ff' if m==mode else '#22304a')
    def open_video(self):
        p=filedialog.askopenfilename(filetypes=[('Video','*.mov *.mp4 *.m4v *.avi *.mkv *.mts'),('All files','*.*')])
        if p:self.video=p;self.source_var.set(Path(p).name);self.stage.set('Ready to analyse');self.status.set('Ready')
    def cancel(self):self.cancel_flag=True;self.status.set('Cancelling');self.stage.set('Stopping after the current decoded frame…')
    def ui(self,fn,*a,**kw):self.after(0,lambda:fn(*a,**kw))
    def start(self):
        if self.running:return
        if not self.video:return messagebox.showinfo(APP_NAME,'Choose a video first.')
        self.running=True;self.cancel_flag=False;self.run_btn.config(state='disabled');self.cancel_btn.config(state='normal');self.output_var.set('Working…');self.map_canvas.delete('all')
        threading.Thread(target=self.analyse,daemon=True).start()

    def analyse(self):
        t0=time.perf_counter();worker=None
        try:
            mode=self.mode.get();cfg={'Turbo':dict(max_side=840,anchor_every=7,preview_every=36,input_size=320),'Balanced':dict(max_side=1040,anchor_every=5,preview_every=28,input_size=384),'Precision':dict(max_side=1280,anchor_every=3,preview_every=22,input_size=512)}[mode]
            try:import av as _av;ffmpeg='';ffprobe=''
            except Exception:ffmpeg=bundled_or_path('ffmpeg.exe' if os.name=='nt' else 'ffmpeg');ffprobe=bundled_or_path('ffprobe.exe' if os.name=='nt' else 'ffprobe')
            self.ui(self.status.set,'Loading');self.ui(self.stage.set,'Opening video metadata…');meta=probe_video(ffprobe,self.video)
            expected=meta.frame_count or len(meta.timestamps) or int(round(meta.duration*(meta.avg_fps or meta.nominal_fps or 0)));model=find_model(mode)
            self.ui(self.stage.set,f'Loading pose model: {Path(model).name}');pose=PoseEstimator(model,cfg['input_size'])
            reader_obj=FFmpegFrameReader(ffmpeg,self.video,meta,cfg['max_side']);reader=iter(reader_obj);self.ui(self.progress.configure,maximum=max(1,expected),value=0)
            self.ui(self.stage.set,f'{meta.decoder.upper()} decode • {meta.width}×{meta.height} → {reader_obj.width}×{reader_obj.height} working frames')
            try:fi,t,first=next(reader)
            except StopIteration:raise RuntimeError('Video decoder returned no frames.')
            seed=pose.infer(first,None,None)
            if seed is None:raise RuntimeError('Could not find the rider on the first frame. Start the clip with the rider clearly visible.')
            tracker=SparseSkeletonTracker(first,seed.points);samples_pts=[seed.points.copy()];samples_q=[seed.quality.copy()];times=[t];frame_ids=[fi]
            worker=AnchorWorker(pose,max_pending=6);worker.results[0]=(seed.points.copy(),seed.quality.copy(),float(seed.person_conf));worker.polled.add(0)
            accepted_relocks=0;dec_count=1;last_ui=time.perf_counter();last_q=float(np.median(seed.quality));self.ui(self.show_preview,first,seed.points,seed.quality)
            for fi,t,frame in reader:
                if self.cancel_flag:break
                s=tracker.step(frame,fi,t);samples_pts.append(s.pts.copy());samples_q.append(s.quality.copy());times.append(t);frame_ids.append(fi);dec_count+=1
                if dec_count%cfg['anchor_every']==0:worker.submit(dec_count-1,frame,s.pts)
                for ai,(ap,aq,pc) in worker.poll():
                    if 0<=ai<len(samples_pts) and tracker.relock(ap,aq,samples_pts[ai]):accepted_relocks+=1
                last_q=float(np.median(s.quality))
                if dec_count%cfg['preview_every']==0:self.ui(self.show_preview,frame,s.pts,s.quality)
                now=time.perf_counter()
                if now-last_ui>0.18:
                    rate=dec_count/max(now-t0,1e-6);self.ui(self.progress.configure,value=min(dec_count,max(1,expected)));self.ui(self.status.set,'Tracking')
                    self.ui(self.stage.set,f'Frame {dec_count:,}'+(f' / {expected:,}' if expected else '')+f'  •  {reader_obj.width}×{reader_obj.height} work image')
                    self.ui(self.fps_var.set,f'{rate:.1f} fps');self.ui(self.frames_var.set,f'{dec_count:,}');self.ui(self.ai_var.set,f'{accepted_relocks} live / {len(worker.results)} ready')
                    self.ui(self.quality_var.set,f'{last_q*100:.0f}%');self.ui(self.perf.set,f'{pose.provider}  •  AI queue {worker.pending}');last_ui=now
            if self.cancel_flag:
                if worker:worker.finish()
                self.ui(self.status.set,'Cancelled');self.ui(self.stage.set,'Analysis cancelled');return
            self.ui(self.status.set,'Finishing');self.ui(self.stage.set,'Finishing AI anchors and repairing trajectory…');anchors_full=worker.finish()
            anchors={i:(v[0],v[1]) for i,v in anchors_full.items()};raw=np.asarray(samples_pts,np.float32);rq=np.asarray(samples_q,np.float32);ts=np.asarray(times,float)
            clean,cq=stabilize(raw,rq,anchors,ts);source_ids=np.asarray(frame_ids,int);source_present=np.ones(len(clean),np.uint8);anchor_flags=np.array([1 if i in anchors else 0 for i in range(len(clean))],np.uint8);out_ts=ts
            if self.resample.get():
                fps=meta.nominal_fps or meta.avg_fps
                if fps>0:out_ts,clean,cq,source_ids,source_present,anchor_flags=resample_uniform(ts,clean,cq,frame_ids,fps,set(anchors))
            cg,rwb,cgy_rel,depth=derived(clean);out=self.export_csv(out_ts,meta,clean,cq,cg,rwb,cgy_rel,depth,source_ids,source_present,anchor_flags);elapsed=time.perf_counter()-t0
            cleand=np.linalg.norm(np.diff(clean,axis=0),axis=2) if len(clean)>1 else np.zeros((0,8));qmed=float(np.median(cq)) if cq.size else 0
            self.ui(self.progress.configure,value=max(1,expected));self.ui(self.status.set,'Complete');self.ui(self.stage.set,f'Complete • {len(raw):,} decoded pictures • {len(clean):,} output rows')
            self.ui(self.fps_var.set,f'{len(raw)/max(elapsed,1e-9):.1f} fps');self.ui(self.frames_var.set,f'{len(clean):,}');self.ui(self.ai_var.set,f'{len(anchors)} anchors');self.ui(self.quality_var.set,f'{qmed*100:.0f}%')
            self.ui(self.perf.set,f'{elapsed:.1f}s total  •  p99 step {np.quantile(cleand,.99) if cleand.size else 0:.4f}');self.ui(self.output_var.set,out);self.ui(self.draw_map,cg,rwb)
        except Exception as e:
            tb=traceback.format_exc();self.ui(messagebox.showerror,APP_NAME,f'{e}\n\n{tb[-1800:]}');self.ui(self.status.set,'Failed');self.ui(self.stage.set,'Analysis failed')
        finally:
            self.running=False;self.ui(self.run_btn.config,state='normal');self.ui(self.cancel_btn.config,state='disabled')

    def export_csv(self,ts,meta,pts,q,cg,rwb,cgy_rel,depth,source_ids,source_present,anchor_flags):
        src=Path(self.video);out=src.with_name(src.stem+'_rider_mapping_v6.csv');fields=['sample','source_frame','source_present','time_s','source_nominal_fps','source_avg_fps','rwb','cgx','cgy','cgy_rel','depth_ratio','quality','pose_anchor']
        for n in NAMES:fields += [n+'_x',n+'_y',n+'_quality']
        with open(out,'w',newline='',encoding='utf-8') as f:
            w=csv.writer(f);w.writerow(fields)
            for i in range(len(pts)):
                row=[i,int(source_ids[i]),int(source_present[i]),f'{ts[i]:.9f}',f'{meta.nominal_fps:.6f}',f'{meta.avg_fps:.6f}',f'{rwb[i]:.8f}',f'{cg[i,0]:.8f}',f'{cg[i,1]:.8f}',f'{cgy_rel[i]:.8f}',f'{depth[i]:.8f}',f'{float(np.median(q[i])):.6f}',int(anchor_flags[i])]
                for j in range(8):row += [f'{pts[i,j,0]:.8f}',f'{pts[i,j,1]:.8f}',f'{q[i,j]:.6f}']
                w.writerow(row)
        return str(out)

    def show_preview(self,frame,pts,q):
        im=frame.copy();h,w=im.shape[:2];P=(pts*np.array([w,h])).astype(int)
        for a,b in BONES:cv2.line(im,tuple(P[a]),tuple(P[b]),(225,235,245),2,cv2.LINE_AA)
        for j,p in enumerate(P):
            qq=float(q[j]) if j<len(q) else 0;col=(92,215,99) if qq>=0.55 else ((84,180,255) if qq>=0.28 else (122,102,255))
            cv2.circle(im,tuple(p),7,col,-1,cv2.LINE_AA);cv2.circle(im,tuple(p),9,(15,20,30),1,cv2.LINE_AA);cv2.putText(im,NAMES[j].replace('Shoulder','Sh').replace('Elbow','El'),(p[0]+9,p[1]-7),cv2.FONT_HERSHEY_SIMPLEX,0.38,(245,245,245),1,cv2.LINE_AA)
        maxw=max(640,min(980,self.preview.winfo_width()-12));maxh=max(420,min(680,self.preview.winfo_height()-12));scale=min(1.0,maxw/w,maxh/h)
        if scale<1:im=cv2.resize(im,(max(2,int(w*scale)),max(2,int(h*scale))),interpolation=cv2.INTER_AREA)
        rgb=cv2.cvtColor(im,cv2.COLOR_BGR2RGB);photo=ImageTk.PhotoImage(Image.fromarray(rgb));self.preview_photo=photo;self.preview.config(image=photo,text='')

    def draw_map(self,cg,rwb):
        c=self.map_canvas;c.delete('all');c.update_idletasks();W=max(300,c.winfo_width());H=max(180,c.winfo_height());pad=22
        if len(cg)<2:return
        x=cg[:,0];y=cg[:,1];xmin,xmax=np.quantile(x,[.01,.99]);ymin,ymax=np.quantile(y,[.01,.99]);dx=max(xmax-xmin,.02);dy=max(ymax-ymin,.02);xmin-=dx*.12;xmax+=dx*.12;ymin-=dy*.12;ymax+=dy*.12
        c.create_rectangle(pad,pad,W-pad,H-pad,outline=GRID)
        for i in range(1,len(x)):
            x1=pad+(x[i-1]-xmin)/(xmax-xmin)*(W-2*pad);x2=pad+(x[i]-xmin)/(xmax-xmin)*(W-2*pad);y1=pad+(y[i-1]-ymin)/(ymax-ymin)*(H-2*pad);y2=pad+(y[i]-ymin)/(ymax-ymin)*(H-2*pad)
            c.create_line(x1,y1,x2,y2,fill=ACCENT,width=1)
        xx=pad+(x[-1]-xmin)/(xmax-xmin)*(W-2*pad);yy=pad+(y[-1]-ymin)/(ymax-ymin)*(H-2*pad);c.create_oval(xx-4,yy-4,xx+4,yy+4,fill=WARN,outline='')
        c.create_text(pad+3,10,anchor='w',fill=MUTED,font=('Segoe UI',8),text=f'RWB {rwb.min():.3f} — {rwb.max():.3f}')

if __name__=='__main__':App().mainloop()
