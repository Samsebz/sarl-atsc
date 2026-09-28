import math

class CentroidTracker:
    """Small dependency-free tracker for research/demo use."""

    def __init__(self, max_distance=80, max_missing=10):
        self.max_distance = max_distance
        self.max_missing = max_missing
        self.next_id = 0
        self.objects = {}
        self.missing = {}

    def update(self, detections):
        if not detections:
            for k in list(self.missing):
                self.missing[k] += 1
                if self.missing[k] > self.max_missing:
                    self.objects.pop(k, None)
                    self.missing.pop(k, None)
            return detections

        old_items = list(self.objects.items())
        assigned_old = set()

        for det in detections:
            best_id = None
            best_dist = float("inf")
            for obj_id, old_center in old_items:
                if obj_id in assigned_old:
                    continue
                dist = math.hypot(
                    det.center[0] - old_center[0],
                    det.center[1] - old_center[1]
                )
                if dist < best_dist and dist <= self.max_distance:
                    best_dist = dist
                    best_id = obj_id

            if best_id is None:
                best_id = self.next_id
                self.next_id += 1

            det.track_id = best_id
            self.objects[best_id] = det.center
            self.missing[best_id] = 0
            assigned_old.add(best_id)

        for obj_id in list(self.missing):
            if obj_id not in assigned_old:
                self.missing[obj_id] += 1
                if self.missing[obj_id] > self.max_missing:
                    self.objects.pop(obj_id, None)
                    self.missing.pop(obj_id, None)

        return detections
