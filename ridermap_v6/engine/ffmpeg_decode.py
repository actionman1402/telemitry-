from __future__ import annotations
import json, subprocess
from dataclasses import dataclass
from typing import Iterator, Optional
import numpy as np

@dataclass
class VideoMeta:
    width:int; height:int; timestamps:list[float]; nominal_fps:float; avg_fps:float; duration:float
    frame_count:int=0; decoder:str='ffmpeg'


def _ratio(v):
    if not v or str(v) in {'0/0','N/A'}:return 0.0
    try:
        if hasattr(v,'numerator'):return float(v)
        a,b=str(v).split('/');return float(a)/float(b) if float(b) else 0.0
    except Exception:
        try:return float(v)
        except Exception:return 0.0


def _probe_av(video):
    import av
    c=av.open(video)
    try:
        s=c.streams.video[0]
        avg=_ratio(getattr(s,'average_rate',None));nom=_ratio(getattr(s,'base_rate',None)) or avg
        dur=0.0
        if getattr(s,'duration',None) is not None and getattr(s,'time_base',None) is not None:
            try:dur=float(s.duration*s.time_base)
            except Exception:pass
        if not dur and getattr(c,'duration',None):
            try:dur=float(c.duration)/1_000_000.0
            except Exception:pass
        return VideoMeta(int(s.width),int(s.height),[],nom,avg,dur,int(getattr(s,'frames',0) or 0),'pyav')
    finally:c.close()


def probe_video(ffprobe,video):
    try:return _probe_av(video)
    except Exception:pass
    cmd=[ffprobe,'-v','error','-select_streams','v:0','-show_entries','stream=width,height,nb_frames,r_frame_rate,avg_frame_rate,duration:format=duration:frame=best_effort_timestamp_time','-of','json',video]
    raw=subprocess.check_output(cmd,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0));d=json.loads(raw);s=d['streams'][0]
    ts=[]
    for f in d.get('frames',[]):
        v=f.get('best_effort_timestamp_time')
        if v not in (None,'N/A'):
            try:ts.append(float(v))
            except Exception:pass
    if ts:
        t0=ts[0];ts=[t-t0 for t in ts]
    dur=0.0
    for v in (s.get('duration'),d.get('format',{}).get('duration')):
        try:
            if v not in (None,'N/A'):dur=max(dur,float(v))
        except Exception:pass
    if ts:dur=max(dur,ts[-1])
    try:fc=int(s.get('nb_frames') or 0)
    except Exception:fc=0
    return VideoMeta(int(s['width']),int(s['height']),ts,_ratio(s.get('r_frame_rate')),_ratio(s.get('avg_frame_rate')),dur,fc,'ffmpeg')


def working_size(meta,max_side):
    w,h=meta.width,meta.height
    if max_side<=0 or max(w,h)<=max_side:return w,h
    r=max_side/max(w,h);return max(2,int(round(w*r/2))*2),max(2,int(round(h*r/2))*2)


class FFmpegFrameReader:
    def __init__(self,ffmpeg,video,meta,max_side=960):
        self.ffmpeg=ffmpeg;self.video=video;self.meta=meta;self.width,self.height=working_size(meta,max_side);self.proc=None

    def __iter__(self):
        if self.meta.decoder=='pyav':
            yield from self._iter_av();return
        yield from self._iter_ffmpeg()

    def _iter_av(self):
        import av
        c=av.open(self.video);s=c.streams.video[0]
        try:
            s.thread_type='AUTO'
            try:s.thread_count=0
            except Exception:pass
            i=0;t0=None
            for fr in c.decode(s):
                if fr.pts is not None:
                    try:t=float(fr.pts*fr.time_base)
                    except Exception:t=i/(self.meta.avg_fps or self.meta.nominal_fps or 1)
                else:t=i/(self.meta.avg_fps or self.meta.nominal_fps or 1)
                if t0 is None:t0=t
                t-=t0
                if (self.width,self.height)!=(fr.width,fr.height):
                    fr=fr.reformat(width=self.width,height=self.height,format='bgr24')
                    arr=fr.to_ndarray()
                else:arr=fr.to_ndarray(format='bgr24')
                yield i,t,arr;i+=1
        finally:c.close()

    def _iter_ffmpeg(self):
        w,h=self.width,self.height;frame_bytes=w*h*3;vf=[]
        if (w,h)!=(self.meta.width,self.meta.height):vf=['-vf',f'scale={w}:{h}:flags=fast_bilinear']
        cmd=[self.ffmpeg,'-hide_banner','-loglevel','error','-threads','0','-i',self.video,'-map','0:v:0','-an','-sn','-dn',*vf,'-fps_mode','passthrough','-f','rawvideo','-pix_fmt','bgr24','pipe:1']
        self.proc=subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,bufsize=max(frame_bytes*2,1024*1024),creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0));i=0
        try:
            while True:
                b=self.proc.stdout.read(frame_bytes)
                if len(b)<frame_bytes:break
                arr=np.frombuffer(b,dtype=np.uint8).reshape((h,w,3))
                if i<len(self.meta.timestamps):t=self.meta.timestamps[i]
                else:t=i/(self.meta.avg_fps or self.meta.nominal_fps or 1)
                yield i,t,arr;i+=1
        finally:
            if self.proc:
                try:self.proc.terminate()
                except Exception:pass
