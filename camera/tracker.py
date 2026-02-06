"""Simple object tracking using Kalman filter (DeepSort-style tracking).

This implements a simplified version of the DeepSort tracking algorithm:
- Kalman filter for motion prediction
- Hungarian algorithm (linear_sum_assignment) for optimal matching
- IoU-based cost matrix for association
- Track lifecycle management (tentative -> confirmed -> deleted)
"""
import numpy as np
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment
from typing import List, Tuple, Dict
from collections import defaultdict


class Track:
    """Represents a single tracked object."""
    
    _id_counter = 0
    
    def __init__(self, initial_bbox: Tuple[int, int, int, int]):
        Track._id_counter += 1
        self.id = Track._id_counter
        self.bbox = initial_bbox
        self.hits = 1
        self.age = 0
        self.time_since_update = 0
        self.state = "tentative"
        
        self.kf = self._init_kalman(initial_bbox)
        self.history = [initial_bbox]
        
    def _init_kalman(self, bbox: Tuple[int, int, int, int]) -> KalmanFilter:
        """Initialize Kalman filter for tracking."""
        kf = KalmanFilter(dim_x=7, dim_z=4)
        
        kf.F = np.array([
            [1, 0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1]
        ])
        
        kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0]
        ])
        
        kf.R *= 10.0
        kf.P[4:, 4:] *= 1000.0
        kf.Q[-1, -1] *= 0.01
        kf.Q[4:, 4:] *= 0.01
        
        x, y, w, h = bbox
        kf.x[:4] = np.array([[x + w/2], [y + h/2], [w], [h]])
        
        return kf
    
    def predict(self) -> Tuple[int, int, int, int]:
        """Predict next position."""
        self.kf.predict()
        self.age += 1
        self.time_since_update += 1
        
        cx, cy, w, h = self.kf.x[:4].flatten()
        x = int(cx - w/2)
        y = int(cy - h/2)
        
        return (x, y, int(w), int(h))
    
    def update(self, bbox: Tuple[int, int, int, int]):
        """Update track with new detection."""
        self.hits += 1
        self.time_since_update = 0
        
        x, y, w, h = bbox
        measurement = np.array([[x + w/2], [y + h/2], [w], [h]])
        self.kf.update(measurement)
        
        self.bbox = bbox
        self.history.append(bbox)
        
        if self.hits >= 3:
            self.state = "confirmed"
    
    def get_bbox(self) -> Tuple[int, int, int, int]:
        """Get current bounding box."""
        cx, cy, w, h = self.kf.x[:4].flatten()
        x = int(cx - w/2)
        y = int(cy - h/2)
        return (x, y, int(w), int(h))


class SimpleTracker:
    """Simple multi-object tracker inspired by DeepSort."""
    
    def __init__(self, max_age=30, min_hits=3, iou_threshold=0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.tracks: List[Track] = []
        self.frame_count = 0
        
    def _iou(self, bbox1: Tuple, bbox2: Tuple) -> float:
        """Calculate IoU between two bounding boxes."""
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        
        xi1 = max(x1, x2)
        yi1 = max(y1, y2)
        xi2 = min(x1 + w1, x2 + w2)
        yi2 = min(y1 + h1, y2 + h2)
        
        inter_width = max(0, xi2 - xi1)
        inter_height = max(0, yi2 - yi1)
        inter_area = inter_width * inter_height
        
        area1 = w1 * h1
        area2 = w2 * h2
        union_area = area1 + area2 - inter_area
        
        if union_area == 0:
            return 0
        return inter_area / union_area
    
    def _match_detections_to_tracks(self, detections: List[Tuple]) -> Tuple[List, List, List]:
        """Match detections to existing tracks using Hungarian algorithm.
        
        Uses scipy's linear_sum_assignment for optimal bipartite matching,
        similar to the association step in DeepSort.
        """
        if len(self.tracks) == 0:
            return [], list(range(len(detections))), []
        
        if len(detections) == 0:
            return [], [], list(range(len(self.tracks)))
        
        cost_matrix = np.zeros((len(detections), len(self.tracks)))
        
        for d, det in enumerate(detections):
            for t, track in enumerate(self.tracks):
                iou = self._iou(det, track.get_bbox())
                cost_matrix[d, t] = 1 - iou
        
        row_indices, col_indices = linear_sum_assignment(cost_matrix)
        
        matches = []
        unmatched_detections = list(range(len(detections)))
        unmatched_tracks = list(range(len(self.tracks)))
        
        for d, t in zip(row_indices, col_indices):
            iou = 1 - cost_matrix[d, t]
            if iou >= self.iou_threshold:
                matches.append((d, t))
                if d in unmatched_detections:
                    unmatched_detections.remove(d)
                if t in unmatched_tracks:
                    unmatched_tracks.remove(t)
        
        return matches, unmatched_detections, unmatched_tracks
    
    def update(self, detections: List[Tuple[int, int, int, int]]) -> List[Dict]:
        """Update tracker with new detections."""
        self.frame_count += 1
        
        for track in self.tracks:
            track.predict()
        
        matches, unmatched_dets, unmatched_tracks = self._match_detections_to_tracks(detections)
        
        for d, t in matches:
            self.tracks[t].update(detections[d])
        
        for d in unmatched_dets:
            self.tracks.append(Track(detections[d]))
        
        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]
        
        results = []
        for track in self.tracks:
            if track.state == "confirmed" or self.frame_count <= self.min_hits:
                x, y, w, h = track.get_bbox()
                results.append({
                    "id": track.id,
                    "bbox": (x, y, w, h),
                    "hits": track.hits,
                    "age": track.age
                })
        
        return results
    
    def get_active_tracks(self) -> int:
        """Get number of active confirmed tracks."""
        return sum(1 for t in self.tracks if t.state == "confirmed")
    
    def reset(self):
        """Reset tracker state."""
        self.tracks = []
        self.frame_count = 0
        Track._id_counter = 0


if __name__ == "__main__":
    tracker = SimpleTracker()
    
    detections_frame1 = [(100, 100, 50, 100), (200, 150, 45, 95)]
    detections_frame2 = [(105, 102, 50, 100), (198, 155, 45, 95)]
    detections_frame3 = [(110, 105, 50, 100), (195, 160, 45, 95), (300, 200, 40, 90)]
    
    print("Frame 1:")
    results = tracker.update(detections_frame1)
    for r in results:
        print(f"  Track {r['id']}: bbox={r['bbox']}, hits={r['hits']}")
    
    print("\nFrame 2:")
    results = tracker.update(detections_frame2)
    for r in results:
        print(f"  Track {r['id']}: bbox={r['bbox']}, hits={r['hits']}")
    
    print("\nFrame 3:")
    results = tracker.update(detections_frame3)
    for r in results:
        print(f"  Track {r['id']}: bbox={r['bbox']}, hits={r['hits']}")
    
    print(f"\nActive tracks: {tracker.get_active_tracks()}")
