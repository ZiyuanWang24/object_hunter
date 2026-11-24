from __future__ import annotations
import argparse, csv, json, time
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from objecthunter.kalman import ConstantVelocityKalman

def main():
 p=argparse.ArgumentParser(); p.add_argument('--config',default='configs/default.yaml'); p.add_argument('--output',default='outputs/evaluation.json'); a=p.parse_args(); Path(a.output).parent.mkdir(exist_ok=True)
 # Deterministic smoke benchmark for the non-neural tracking core. Dataset evaluation can be added with annotations.
 f=ConstantVelocityKalman([0,0]); t=time.perf_counter()
 for i in range(100): f.predict(.05); f.update([i*.001,0])
 result={'benchmark':'synthetic_kalman_smoke','frames':100,'elapsed_s':time.perf_counter()-t,'metrics_not_available_without_annotations':['IDF1','ID switches','re-identification success rate','localization error']}
 Path(a.output).write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2))
if __name__=='__main__': main()
