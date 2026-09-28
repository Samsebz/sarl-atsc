from dataclasses import dataclass

@dataclass
class SafetyResult:
    proposed_action: int
    approved_action: int
    overridden: bool
    reason: str

class SafetyConstraintLayer:
    """Rule-based safety layer for signal phase validation."""

    def __init__(self, min_green_seconds=10, max_red_seconds=90,
                 prohibited_transitions=None, action_size=4):
        self.min_green_seconds = float(min_green_seconds)
        self.max_red_seconds = float(max_red_seconds)
        self.prohibited_transitions = prohibited_transitions or set()
        self.action_size = action_size

    def validate(self, proposed_action, current_phase, green_elapsed,
                 red_elapsed_by_phase):
        proposed_action = int(proposed_action)
        if green_elapsed < self.min_green_seconds and proposed_action != current_phase:
            return SafetyResult(proposed_action, current_phase, True, "minimum_green_time")
        if (current_phase, proposed_action) in self.prohibited_transitions:
            return SafetyResult(proposed_action, current_phase, True, "prohibited_transition")
        overdue = [p for p, elapsed in enumerate(red_elapsed_by_phase)
                    if elapsed >= self.max_red_seconds]
        if overdue and proposed_action not in overdue:
            return SafetyResult(proposed_action, overdue[0], True, "maximum_red_time")
        return SafetyResult(proposed_action, proposed_action, False, "approved")
