class SignalController:
    """Software signal-controller abstraction for simulation/demo."""

    PHASE_NAMES = {
        0: "NORTH_SOUTH_GREEN",
        1: "EAST_WEST_GREEN",
        2: "NORTH_SOUTH_LEFT",
        3: "EAST_WEST_LEFT",
    }

    def __init__(self, phase_count=4):
        self.phase_count = phase_count
        self.current_phase = 0

    def set_phase(self, phase):
        phase = int(phase) % self.phase_count
        self.current_phase = phase
        return self.PHASE_NAMES.get(phase, f"PHASE_{phase}")

    def state(self):
        return {
            "phase": self.current_phase,
            "phase_name": self.PHASE_NAMES.get(
                self.current_phase, f"PHASE_{self.current_phase}"
            ),
        }
