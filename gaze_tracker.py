#!/usr/bin/env python3
"""
Lightweight Eye Tracking Service
- Standalone or Electron-integratable via HTTP API
- Features: calibration phase, gaze tracking with visual trail/dot, camera selection
"""

import sys
import os
import threading
from collections import deque
from typing import Optional
import cv2
import numpy as np
import pygame

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'EyeGestures'))
from eyeGestures.utils import VideoCapture
from eyeGestures import EyeGestures_v3

try:
    from flask import Flask, jsonify, request
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False


class GazeTracker:
    def __init__(self, trail_length: int = 30):
        self.trail_length = trail_length
        self.trail: deque = deque(maxlen=trail_length)
        
        self.camera_index = 0
        self.running = False
        self.calibrating = True
        self.calibration_points = 10
        self.calibration_count = 0
        
        self.screen_width = 0
        self.screen_height = 0
        
        self.gestures: Optional[EyeGestures_v3] = None
        self.cap: Optional[VideoCapture] = None
        self.prev_calib_point = (0, 0)
        
    def set_camera(self, index: int):
        self.camera_index = index
        if self.cap:
            self.cap = VideoCapture(index)
        return index
    
    def get_available_cameras(self) -> list:
        available = []
        for i in range(3):  # Check first 3 indices
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                available.append(i)
                cap.release()
        return available
    
    def get_state(self) -> dict:
        return {
            "camera_index": self.camera_index,
            "calibrating": self.calibrating,
            "calibration_progress": f"{self.calibration_count}/{self.calibration_points}",
            "running": self.running
        }
    
    def run(self, api_port: Optional[int] = None):
        pygame.init()
        pygame.font.init()
        
        screen_info = pygame.display.Info()
        self.screen_width = screen_info.current_w
        self.screen_height = screen_info.current_h
        
        screen = pygame.display.set_mode((self.screen_width, self.screen_height), pygame.FULLSCREEN)
        pygame.display.set_caption("Eye Tracker")
        
        font = pygame.font.Font(None, 36)
        bold_font = pygame.font.Font(None, 64)
        bold_font.set_bold(True)
        
        self.gestures = EyeGestures_v3()
        self.cap = VideoCapture(self.camera_index)
        
        # Setup calibration map - 4x3 grid (12 points, use 10)
        x = np.array([0.15, 0.5, 0.85])
        y = np.array([0.15, 0.5, 0.85])
        xx, yy = np.meshgrid(x, y)
        calibration_map = np.column_stack([xx.ravel(), yy.ravel()])
        np.random.shuffle(calibration_map)
        self.gestures.uploadCalibrationMap(calibration_map, context="tracker")
        self.gestures.setFixation(1.0)
        
        if api_port and FLASK_AVAILABLE:
            api_thread = threading.Thread(target=self._run_api, args=(api_port,), daemon=True)
            api_thread.start()
        
        clock = pygame.time.Clock()
        self.running = True
        self.calibrating = True
        
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q and pygame.key.get_mods() & pygame.KMOD_CTRL:
                        self.running = False
                    elif event.key == pygame.K_c:
                        cameras = self.get_available_cameras()
                        if cameras:
                            idx = cameras.index(self.camera_index) if self.camera_index in cameras else -1
                            self.set_camera(cameras[(idx + 1) % len(cameras)])
                    elif event.key == pygame.K_SPACE:
                        self.calibrating = True
                        self.calibration_count = 0
                        self.prev_calib_point = (0, 0)
                        self.trail.clear()
                        self.gestures = EyeGestures_v3()
                        self.gestures.uploadCalibrationMap(calibration_map, context="tracker")
                        self.gestures.setFixation(1.0)
            
            ret, frame = self.cap.read()
            if not ret or frame is None:
                screen.fill((30, 30, 30))
                text = font.render("Waiting for camera...", True, (200, 200, 200))
                screen.blit(text, (self.screen_width // 2 - 100, self.screen_height // 2))
                pygame.display.flip()
                clock.tick(30)
                continue
            
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = np.flip(frame, axis=1)
            
            calibrate = self.calibration_count < self.calibration_points
            gaze_event, calib_event = self.gestures.step(
                frame, calibrate, self.screen_width, self.screen_height, context="tracker"
            )
            
            screen.fill((20, 20, 25))
            
            if gaze_event is None:
                # Show "looking for face" message
                text = font.render("Looking for face... Position yourself in front of camera", True, (200, 150, 100))
                screen.blit(text, (self.screen_width // 2 - 250, self.screen_height // 2))
                pygame.display.flip()
                clock.tick(30)
                continue
            
            if calibrate:
                # Calibration phase
                if calib_event:
                    px, py = int(calib_event.point[0]), int(calib_event.point[1])
                    if px != self.prev_calib_point[0] or py != self.prev_calib_point[1]:
                        self.calibration_count += 1
                        self.prev_calib_point = (px, py)
                    
                    # Draw calibration target
                    pygame.draw.circle(screen, (60, 60, 80), (px, py), calib_event.acceptance_radius + 20)
                    pygame.draw.circle(screen, (100, 50, 255), (px, py), calib_event.acceptance_radius)
                    pygame.draw.circle(screen, (255, 255, 255), (px, py), 8)
                    
                    # Progress text
                    text = bold_font.render(f"{self.calibration_count}/{self.calibration_points}", True, (255, 255, 255))
                    text_rect = text.get_rect(center=(px, py - calib_event.acceptance_radius - 40))
                    screen.blit(text, text_rect)
                    
                    # Instructions
                    inst = font.render("Look at the circle until it moves", True, (150, 150, 150))
                    screen.blit(inst, (self.screen_width // 2 - 150, 50))
            else:
                # Tracking phase
                self.calibrating = False
                x, y = int(gaze_event.point[0]), int(gaze_event.point[1])
                
                # Add to trail
                self.trail.append((x, y, gaze_event.fixation))
                
                # Draw trail
                for i, (tx, ty, fix) in enumerate(self.trail):
                    alpha = int((i / len(self.trail)) * 200)
                    radius = int((i / len(self.trail)) * 15) + 3
                    if fix:
                        color = (alpha, 50, 50)  # Red trail for fixation
                    else:
                        color = (50, alpha, 50)  # Green trail for movement
                    pygame.draw.circle(screen, color, (tx, ty), radius)
                
                # Draw main gaze dot
                if gaze_event.fixation:
                    pygame.draw.circle(screen, (255, 80, 80), (x, y), 25)
                    pygame.draw.circle(screen, (255, 200, 200), (x, y), 12)
                else:
                    pygame.draw.circle(screen, (80, 255, 80), (x, y), 20)
                    pygame.draw.circle(screen, (200, 255, 200), (x, y), 8)
            
            # Status bar
            status = f"[C] Camera: {self.camera_index}  [Space] Recalibrate  [Ctrl+Q] Quit"
            if not calibrate:
                status = f"Fixation: {'Yes' if gaze_event.fixation else 'No'}  |  " + status
            text = font.render(status, True, (120, 120, 120))
            screen.blit(text, (10, self.screen_height - 40))
            
            pygame.display.flip()
            clock.tick(60)
        
        pygame.quit()
    
    def _run_api(self, port: int):
        if not FLASK_AVAILABLE:
            return
        
        app = Flask(__name__)
        
        @app.route('/status', methods=['GET'])
        def status():
            return jsonify(self.get_state())
        
        @app.route('/camera', methods=['GET'])
        def get_cameras():
            return jsonify({"cameras": self.get_available_cameras(), "current": self.camera_index})
        
        @app.route('/camera', methods=['POST'])
        def set_camera():
            data = request.get_json() or {}
            idx = data.get('index', 0)
            self.set_camera(idx)
            return jsonify({"camera_index": self.camera_index})
        
        @app.route('/recalibrate', methods=['POST'])
        def recalibrate():
            self.calibrating = True
            self.calibration_count = 0
            return jsonify({"success": True})
        
        @app.route('/quit', methods=['POST'])
        def quit_app():
            self.running = False
            return jsonify({"success": True})
        
        app.run(host='127.0.0.1', port=port, threaded=True, use_reloader=False)


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Eye Tracking Service')
    parser.add_argument('--api-port', type=int, default=None, help='Enable HTTP API on specified port')
    parser.add_argument('--camera', type=int, default=0, help='Camera index (default: 0)')
    parser.add_argument('--trail', type=int, default=30, help='Trail length (default: 30)')
    args = parser.parse_args()
    
    tracker = GazeTracker(trail_length=args.trail)
    tracker.camera_index = args.camera
    tracker.run(api_port=args.api_port)


if __name__ == '__main__':
    main()
