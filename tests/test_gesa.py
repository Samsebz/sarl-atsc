from types import SimpleNamespace
from sarl_atsc.gesa import GESAPlusPlus

def test_gesa_state_shape():
    lanes=[[[0,0],[100,0],[100,100],[0,100]],[[100,0],[200,0],[200,100],[100,100]]]
    g=GESAPlusPlus(lanes); d=SimpleNamespace(track_id=1,center=(50,50),bbox=(40,40,60,60)); state,lane_data=g.update([d],(100,200,3),fps=30)
    assert state.shape==(8,); assert len(lane_data)==2
