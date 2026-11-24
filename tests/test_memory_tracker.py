import numpy as np
from objecthunter.tracker import MultiObjectTracker
from objecthunter.types import Detection,Lifecycle

def det(t,x,emb=(1.,0.)): return Detection(t,'cup',.9,(x,0,x+10,10),world_position=np.array([x,0.]),embedding=np.array(emb))

def test_confirmation_and_occlusion():
 tr=MultiObjectTracker(confirm_hits=3,dormant_timeout_s=1)
 for i in range(3): objs=tr.update([det(i,i)],i)
 assert objs[0].lifecycle==Lifecycle.TRACKED
 tr.update([],3.5); assert objs[0].lifecycle==Lifecycle.DORMANT

def test_reidentification_preserves_id():
 tr=MultiObjectTracker(confirm_hits=1,dormant_timeout_s=.5,reid_threshold=.7)
 o=tr.update([det(0,0)],0)[0]; oid=o.object_id; tr.update([],1); o2=tr.update([det(2,5)],2)[0]; assert o2.object_id==oid and o2.reidentification_count==1

def test_ambiguous_identity_creates_new_object():
 tr=MultiObjectTracker(confirm_hits=1, dormant_timeout_s=.5, reid_threshold=.7, ambiguity_margin=.2)
 tr.update([det(0,0,(1,0)), det(0,20,(1,0))],0); tr.update([],2); tr.update([det(3,2,(1,0))],3)
 assert len(tr.memory.objects)==3
