from sarl_atsc.safety import SafetyConstraintLayer

def test_min_green_blocks_switch():
    s=SafetyConstraintLayer(min_green_seconds=10); r=s.validate(1,0,2,[0,0,0,0]); assert r.overridden; assert r.approved_action==0

def test_approved_action():
    s=SafetyConstraintLayer(min_green_seconds=10); r=s.validate(0,0,20,[0,0,0,0]); assert not r.overridden
