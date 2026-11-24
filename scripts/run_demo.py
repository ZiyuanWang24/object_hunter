from __future__ import annotations
import argparse, time, cv2, yaml, os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from objecthunter.camera import LatestFrameCamera
from objecthunter.detector import YOLOSegDetector
from objecthunter.embedder import DINOv2Embedder
from objecthunter.geometry import HomographyCalibrator
from objecthunter.pipeline import PerceptionPipeline
from objecthunter.tracker import MultiObjectTracker
from objecthunter.visualization import draw_dashboard

def main():
 p=argparse.ArgumentParser(); p.add_argument('--camera',type=int); p.add_argument('--source'); p.add_argument('--config',default='configs/default.yaml'); p.add_argument('--device',default='auto'); p.add_argument('--single-target',action='store_true',help='Track only one object: the first matching detection'); p.add_argument('--target-class',help='In single-target mode, select this semantic class, e.g. cup'); a=p.parse_args(); cfg=yaml.safe_load(open(a.config))
 target_class=a.target_class
 if a.single_target and not target_class:
  try: target_class=input('What item should I look for? (for example: cup, bottle, book): ').strip() or None
  except EOFError: target_class=None
  if target_class: print(f"Looking for: {target_class}")
 source=a.source if a.source else (a.camera if a.camera is not None else cfg['camera']['source'])
 device=a.device if a.device!='auto' else cfg['detector']['device']
 if str(device).startswith('cuda'):
  try:
   import torch
   if not torch.cuda.is_available(): print('CUDA requested but unavailable; falling back to CPU'); device='cpu'
  except ImportError: device='cpu'
 detector=YOLOSegDetector(cfg['detector']['model'],cfg['detector']['confidence'],cfg['detector']['iou'],cfg['detector']['imgsz'],device,cfg['detector']['classes'])
 embedder=DINOv2Embedder(cfg['embedder']['model'],cfg['embedder']['device']); geometry=None
 path=cfg['geometry']['calibration_path']
 if os.path.exists(path): geometry=HomographyCalibrator.load(path); print(f'Loaded calibration {path}')
 tracker=MultiObjectTracker(**cfg['tracker']); pipe=PerceptionPipeline(detector,tracker,geometry,embedder,cfg['embedder']['refresh_interval']); cap=LatestFrameCamera(source,cfg['camera']['width'],cfg['camera']['height']).start()
 paused=False; writer=None; recording=False; target_id=None; last_report=0.0
 try:
  while cap.running:
   item=cap.read()
   if item is None: continue
   ts,frame=item
   if not paused:
    detection_filter=None
    if a.single_target:
     def detection_filter(items):
      target=tracker.memory.objects.get(target_id) if target_id is not None else None
      wanted=target.class_name if target is not None else target_class
      candidates=[d for d in items if wanted is None or d.class_name.lower()==wanted.lower()]
      return candidates[:1]
    detections,objects=pipe.process(frame,ts,detection_filter)
    if a.single_target and target_id is None:
     created=[o for o in objects if o.last_observed_at==ts and o.last_detection is not None]
     if created: target_id=created[-1].object_id
    if a.single_target and target_id is not None:
     target=tracker.memory.objects.get(target_id)
     objects=[target] if target is not None else []
     if target is not None and ts-last_report>=0.5:
      pos=target.last_reliable_position if target.lifecycle.value=='DORMANT' and target.last_reliable_position is not None else target.position
      unit='m' if geometry is not None else 'px'; state='last-known' if target.lifecycle.value=='DORMANT' else target.lifecycle.value.lower()
      pipe.metrics['target_position']=(float(pos[0]),float(pos[1])); pipe.metrics['target_unit']=unit
      print(f'Target #{target.object_id} {target.class_name}: {state} location=({pos[0]:.3f}, {pos[1]:.3f}) {unit}',flush=True); last_report=ts
   bounds=(0,cfg['geometry']['world_width'],0,cfg['geometry']['world_height']) if geometry is not None else (0,frame.shape[1],0,frame.shape[0])
   view=draw_dashboard(frame,detections if not paused else [],objects if not paused else list(tracker.memory.objects.values()),pipe.metrics,bounds,cfg['visualization']['map_size']); cv2.imshow('ObjectHunter',view)
   if recording and writer is not None: writer.write(view)
   key=cv2.waitKey(1)&0xff
   if key==ord('q'): break
   if key==ord('p'): paused=not paused
   if key==ord('r'): tracker.memory.reset(); tracker.filters.clear(); target_id=None
   if key==ord('m'): os.makedirs('outputs',exist_ok=True); tracker.memory.export_json('outputs/world_model.json'); print('Saved outputs/world_model.json')
   if key==ord('s'): os.makedirs('outputs',exist_ok=True); cv2.imwrite('outputs/dashboard.png',view)
   if key==ord('v'):
    os.makedirs('outputs',exist_ok=True)
    if not recording:
     writer=cv2.VideoWriter('outputs/dashboard.mp4',cv2.VideoWriter_fourcc(*'mp4v'),cfg['visualization']['record_fps'],(view.shape[1],view.shape[0])); recording=writer.isOpened(); print('Recording started' if recording else 'Could not start recording')
    else:
     recording=False
     if writer: writer.release(); writer=None
     print('Recording stopped')
 finally:
  cap.stop(); cv2.destroyAllWindows()
  if writer is not None: writer.release()
if __name__=='__main__': main()
