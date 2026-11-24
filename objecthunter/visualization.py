from __future__ import annotations

import cv2
import numpy as np
from .types import Lifecycle


COLORS={Lifecycle.TRACKED:(60,220,60), Lifecycle.OCCLUDED:(0,220,255), Lifecycle.DORMANT:(140,140,140), Lifecycle.TENTATIVE:(255,180,0)}

def draw_dashboard(frame, detections, objects, metrics, world_bounds=(0,1,0,0.7), size=600):
    rgb=frame.copy()
    for d in detections:
        x1,y1,x2,y2=map(int,d.bbox); cv2.rectangle(rgb,(x1,y1),(x2,y2),(80,220,80),2); cv2.putText(rgb,f"{d.class_name} {d.confidence:.2f}",(x1,max(15,y1-5)),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1)
        if d.mask is not None and d.mask.shape[:2]==rgb.shape[:2]: rgb[d.mask>0]=(rgb[d.mask>0]*.65+np.array([40,200,40])*.35).astype(np.uint8)
    panel=np.zeros((size,size,3),np.uint8); panel[:]=(30,35,40); xmin,xmax,ymin,ymax=world_bounds
    def pix(p): return int((p[0]-xmin)/(xmax-xmin)*size), int(size-(p[1]-ymin)/(ymax-ymin)*size)
    for o in objects:
        if o.lifecycle==Lifecycle.DORMANT and o.last_reliable_position is not None: p=o.last_reliable_position
        else: p=o.position
        if np.any(np.isnan(p)): continue
        color=COLORS[o.lifecycle]; q=pix(p); cv2.circle(panel,q,8,color,-1); cv2.putText(panel,f"#{o.object_id}",(q[0]+8,q[1]),cv2.FONT_HERSHEY_SIMPLEX,.5,color,1)
        if o.lifecycle != Lifecycle.DORMANT: cv2.arrowedLine(panel,q,pix(p+o.velocity*.2),color,2)
        if len(o.trajectory)>1:
            pts=np.array([pix(t[1:]) for t in o.trajectory[-60:]],np.int32); cv2.polylines(panel,[pts],False,color,1)
    h=max(rgb.shape[0],size); w=rgb.shape[1]+size; out=np.zeros((h,w,3),np.uint8); out[:rgb.shape[0],:rgb.shape[1]]=rgb; out[:size,rgb.shape[1]:]=panel
    lines=[f"FPS {metrics.get('fps',0):.1f}",f"Latency {metrics.get('latency_ms',0):.1f} ms",f"GPU {metrics.get('gpu_memory_mb',0):.0f} MB",f"Tracked {sum(o.lifecycle==Lifecycle.TRACKED for o in objects)}",f"Occluded {sum(o.lifecycle==Lifecycle.OCCLUDED for o in objects)}",f"Dormant {sum(o.lifecycle==Lifecycle.DORMANT for o in objects)}"]
    if metrics.get('target_position') is not None:
        x,y=metrics['target_position']; lines.append(f"Target {x:.3f}, {y:.3f} {metrics.get('target_unit','px')}")
    for i,line in enumerate(lines): cv2.putText(out,line,(10,25+i*22),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),1)
    return out
