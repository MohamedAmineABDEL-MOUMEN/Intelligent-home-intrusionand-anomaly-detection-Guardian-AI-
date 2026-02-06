"""Video frame simulator for testing without actual camera."""
import numpy as np
import cv2
from typing import Generator, Tuple, List, Dict
import random


class VideoSimulator:
    """Simulates video frames with optional person presence."""
    
    def __init__(self, width=640, height=480, fps=15):
        self.width = width
        self.height = height
        self.fps = fps
        self.frame_count = 0
        
        self.persons = []
        self.background = self._create_background()
        
    def _create_background(self) -> np.ndarray:
        """Create a simple room background."""
        bg = np.ones((self.height, self.width, 3), dtype=np.uint8) * 200
        
        cv2.rectangle(bg, (0, self.height//2), (self.width, self.height), (150, 120, 100), -1)
        
        cv2.rectangle(bg, (50, 100), (150, 300), (100, 80, 60), -1)
        cv2.rectangle(bg, (60, 110), (90, 140), (180, 220, 255), -1)
        cv2.rectangle(bg, (110, 110), (140, 140), (180, 220, 255), -1)
        
        cv2.rectangle(bg, (300, 150), (500, 350), (80, 60, 40), -1)
        cv2.rectangle(bg, (320, 250), (480, 350), (60, 40, 30), -1)
        
        return bg
    
    def _draw_person(self, frame: np.ndarray, x: int, y: int, scale: float = 1.0) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        """Draw a simple person figure on the frame."""
        height = int(120 * scale)
        width = int(40 * scale)
        
        person_color = (random.randint(50, 150), random.randint(50, 150), random.randint(100, 200))
        
        head_radius = int(15 * scale)
        cv2.circle(frame, (x, y), head_radius, (200, 180, 160), -1)
        
        body_top = y + head_radius
        body_bottom = y + int(60 * scale)
        cv2.rectangle(frame, (x - int(15 * scale), body_top), 
                     (x + int(15 * scale), body_bottom), person_color, -1)
        
        leg_bottom = y + height - head_radius
        cv2.rectangle(frame, (x - int(12 * scale), body_bottom), 
                     (x - int(3 * scale), leg_bottom), (50, 50, 100), -1)
        cv2.rectangle(frame, (x + int(3 * scale), body_bottom), 
                     (x + int(12 * scale), leg_bottom), (50, 50, 100), -1)
        
        bbox = (x - width//2, y - head_radius, width, height)
        
        return frame, bbox
    
    def add_intruder(self):
        """Add an intruder to the scene."""
        x = random.randint(100, self.width - 100)
        y = random.randint(150, self.height - 150)
        scale = random.uniform(0.8, 1.2)
        velocity = (random.randint(-5, 5), random.randint(-2, 2))
        
        self.persons.append({
            "x": x, "y": y, "scale": scale,
            "velocity": velocity,
            "is_intruder": True
        })
    
    def remove_intruders(self):
        """Remove all intruders from the scene."""
        self.persons = [p for p in self.persons if not p.get("is_intruder", False)]
    
    def _update_persons(self):
        """Update person positions."""
        for person in self.persons:
            vx, vy = person["velocity"]
            person["x"] += vx
            person["y"] += vy
            
            if person["x"] < 50 or person["x"] > self.width - 50:
                person["velocity"] = (-vx, vy)
                person["x"] = max(50, min(self.width - 50, person["x"]))
            if person["y"] < 100 or person["y"] > self.height - 100:
                person["velocity"] = (vx, -vy)
                person["y"] = max(100, min(self.height - 100, person["y"]))
    
    def get_frame(self, include_intruder=False) -> Tuple[np.ndarray, List[Dict]]:
        """Generate a single video frame."""
        self.frame_count += 1
        
        frame = self.background.copy()
        
        if include_intruder and len(self.persons) == 0:
            self.add_intruder()
        
        self._update_persons()
        
        detections = []
        for i, person in enumerate(self.persons):
            frame, bbox = self._draw_person(frame, person["x"], person["y"], person["scale"])
            detections.append({
                "class": "person",
                "confidence": random.uniform(0.7, 0.99),
                "bbox": bbox,
                "is_intruder": person.get("is_intruder", False)
            })
        
        noise = np.random.randint(-5, 5, frame.shape, dtype=np.int16)
        frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        return frame, detections
    
    def generate_video_stream(self, n_frames=100, intrusion_frames=None) -> Generator:
        """Generate a stream of video frames."""
        if intrusion_frames is None:
            intrusion_frames = list(range(30, 60))
        
        for i in range(n_frames):
            include_intruder = i in intrusion_frames
            
            if include_intruder and len(self.persons) == 0:
                self.add_intruder()
            elif not include_intruder and len(self.persons) > 0:
                if i > max(intrusion_frames):
                    self.remove_intruders()
            
            frame, detections = self.get_frame(include_intruder=False)
            yield frame, detections, i
    
    def save_frame(self, frame: np.ndarray, path: str):
        """Save a frame to disk."""
        cv2.imwrite(path, frame)


if __name__ == "__main__":
    import os
    os.makedirs("outputs", exist_ok=True)
    
    simulator = VideoSimulator()
    
    frame, detections = simulator.get_frame(include_intruder=False)
    simulator.save_frame(frame, "outputs/test_frame_empty.png")
    print(f"Empty frame saved. Detections: {len(detections)}")
    
    frame, detections = simulator.get_frame(include_intruder=True)
    simulator.save_frame(frame, "outputs/test_frame_intruder.png")
    print(f"Intruder frame saved. Detections: {len(detections)}")
    for det in detections:
        print(f"  - {det['class']}: conf={det['confidence']:.2f}, bbox={det['bbox']}")
