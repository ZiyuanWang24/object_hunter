import numpy as np
from objecthunter.association import cosine_similarity,bbox_iou,associate
from objecthunter.types import Detection,TrackRecord
from objecthunter.kalman import ConstantVelocityKalman

def view(record):
 class V: pass
 v=V(); v.record=record; v.kalman=ConstantVelocityKalman(record.position); v.class_name=record.class_name; v.gallery=record.gallery; v.last_detection=record.last_detection; return v

def test_similarity_iou():
 assert cosine_similarity([1,0],[1,0])==1; assert bbox_iou((0,0,10,10),(5,5,15,15))>0

def test_valid_and_invalid_assignments():
 r=TrackRecord(1,'cup',0,0,np.array([0.,0.]),np.zeros(4),np.eye(4),last_detection=Detection(0,'cup',1,(0,0,10,10))); r.gallery=[np.array([1.,0.])]
 d=Detection(0,'cup',1,(0,0,10,10),embedding=np.array([1.,0.]),world_position=np.array([0.,0.])); m,ut,ud,_=associate([d],[view(r)]); assert m==[(0,0)]
 bad=Detection(0,'person',1,(0,0,10,10),embedding=np.array([1.,0.]),world_position=np.array([0.,0.])); m,_,ud,_=associate([bad],[view(r)]); assert m==[] and ud==[0]

def test_empty_and_missing_embeddings():
 assert associate([],[],)[0]==[]
 r=TrackRecord(1,'cup',0,0,np.array([0.,0.]),np.zeros(4),np.eye(4)); m,_,_,_=associate([Detection(0,'cup',1,(0,0,10,10),world_position=np.array([0.,0.]))],[view(r)]); assert m==[(0,0)]
