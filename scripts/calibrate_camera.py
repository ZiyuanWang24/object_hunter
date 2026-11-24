from __future__ import annotations
import argparse, cv2, numpy as np, os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from objecthunter.geometry import HomographyCalibrator

def main():
 p=argparse.ArgumentParser(); p.add_argument('--camera',type=int,default=0); p.add_argument('--output',default='calibration/tabletop.json'); p.add_argument('--width',type=int,default=1280); p.add_argument('--height',type=int,default=720); p.add_argument('--world-width',type=float,default=1.0); p.add_argument('--world-height',type=float,default=.7); p.add_argument('--points',nargs=4,metavar='X,Y',help='Optional four pixel points, clockwise, for headless/manual calibration'); a=p.parse_args()
 print(f'Opening camera {a.camera}...',flush=True)
 cap=cv2.VideoCapture(a.camera,cv2.CAP_DSHOW)
 if not cap.isOpened(): cap=cv2.VideoCapture(a.camera)
 ok,frame=cap.read(); cap.release()
 if not ok: raise SystemExit('Could not read camera frame. Check camera index and permissions.')
 print(f'Captured frame: {frame.shape[1]}x{frame.shape[0]}',flush=True)
 os.makedirs('outputs',exist_ok=True); cv2.imwrite('outputs/calibration_frame.png',frame)
 def parse_point(value):
  try: x,y=(float(v.strip()) for v in value.split(',')); return (x,y)
  except Exception as e: raise SystemExit(f'Invalid point {value!r}; use X,Y such as 120,80') from e
 points=[parse_point(v) for v in a.points] if a.points else []
 if not points:
  print("A window named 'ObjectHunter Calibration' should now be visible.",flush=True)
  print('Click four tabletop corners clockwise. Press Q to cancel.',flush=True)
 def cb(event,x,y,flags,param):
  if event==cv2.EVENT_LBUTTONDOWN: points.append((x,y)); print(f'point {len(points)}: pixel=({x},{y})',flush=True)
 if not points:
  title='ObjectHunter Calibration'; cv2.namedWindow(title,cv2.WINDOW_NORMAL); cv2.resizeWindow(title,min(1280,frame.shape[1]),min(800,frame.shape[0])); cv2.setWindowProperty(title,cv2.WND_PROP_TOPMOST,1); cv2.setWindowProperty(title,cv2.WND_PROP_TOPMOST,0); cv2.setMouseCallback(title,cb)
 while len(points)<4:
  view=frame.copy()
  for i,p in enumerate(points): cv2.circle(view,p,6,(0,255,0),-1); cv2.putText(view,str(i+1),p,0,.8,(0,255,0),2)
  if not a.points: cv2.imshow(title,view)
  if cv2.waitKey(30)&0xff==ord('q'): return
 cv2.destroyAllWindows(); world=np.array([[0,0],[a.world_width,0],[a.world_width,a.world_height],[0,a.world_height]],np.float32)
 cal=HomographyCalibrator.from_points(points,world,(frame.shape[1],frame.shape[0])); cal.save(a.output); print(f'Saved {a.output}; RMS error {cal.rms_error:.4f} m')
if __name__=='__main__': main()
