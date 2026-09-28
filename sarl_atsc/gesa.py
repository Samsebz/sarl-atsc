from collections import defaultdict, deque
import numpy as np
import cv2

class GESAPlusPlus:
    """GESA++-style per-lane queue, density, flow and occupancy state."""
    def __init__(self, lane_polygons, history_frames=10, speed_threshold_kmh=2.0):
        self.lane_polygons = lane_polygons
        self.history_frames = history_frames
        self.speed_threshold_kmh = speed_threshold_kmh
        self.history = defaultdict(lambda: deque(maxlen=history_frames))
        self.prev_centers = {}

    @staticmethod
    def _inside(point, polygon):
        poly = np.asarray(polygon, dtype=np.int32)
        return cv2.pointPolygonTest(poly, point, False) >= 0

    def _lane_for(self, center):
        for i, poly in enumerate(self.lane_polygons):
            if self._inside(center, poly): return i
        return None

    def update(self, detections, frame_shape, fps=30.0):
        lane_data = []
        current_ids = {}
        lane_tracks = defaultdict(list)
        for d in detections:
            lane = self._lane_for(d.center)
            if lane is None: continue
            lane_tracks[lane].append(d); current_ids[d.track_id] = d.center
        for lane_idx, polygon in enumerate(self.lane_polygons):
            dets = lane_tracks[lane_idx]; queue = 0; occupied_area = 0.0
            for d in dets:
                prev = self.prev_centers.get(d.track_id); speed_kmh = 0.0
                if prev is not None:
                    px_per_frame = ((d.center[0]-prev[0])**2 + (d.center[1]-prev[1])**2)**0.5
                    speed_kmh = px_per_frame * fps * 0.05 * 3.6
                if speed_kmh < self.speed_threshold_kmh: queue += 1
                x1,y1,x2,y2 = d.bbox
                occupied_area += max(0,x2-x1) * max(0,y2-y1)
            poly = np.asarray(polygon); lane_area = max(1.0, float(cv2.contourArea(poly)))
            density = len(dets) / max(1.0, lane_area / 10000.0)
            occupancy = min(1.0, occupied_area / lane_area)
            flow = 0
            for d in dets:
                prev = self.prev_centers.get(d.track_id)
                if prev is not None and abs(d.center[0]-prev[0])+abs(d.center[1]-prev[1]) > 3:
                    flow += 1
            self.history[lane_idx].append((queue,density,flow,occupancy))
            arr = np.asarray(self.history[lane_idx], dtype=np.float32)
            lane_data.append([float(arr[:,0].mean()),float(arr[:,1].mean()),
                              float(arr[:,2].mean()),float(arr[:,3].mean())])
        self.prev_centers = current_ids
        while len(lane_data) < len(self.lane_polygons): lane_data.append([0.0,0.0,0.0,0.0])
        raw = np.asarray(lane_data, dtype=np.float32); state = raw.copy()
        for i in range(len(state)):
            state[i,0] = min(state[i,0]/30.0,1.0); state[i,1] = min(state[i,1]/20.0,1.0)
            state[i,2] = min(state[i,2]/20.0,1.0); state[i,3] = np.clip(state[i,3],0.0,1.0)
        return state.reshape(-1), lane_data
