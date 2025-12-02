#!/usr/bin/env python3
"""Transparent click-through gaze overlay with heatmap for macOS."""

import math
import signal
import sys
import os
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Optional, List, Tuple
import numpy as np
import cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'EyeGestures'))
from eyeGestures.utils import VideoCapture
from eyeGestures import EyeGestures_v3

import Quartz
from AppKit import (
    NSApplication, NSApp, NSWindow, NSView, NSColor, NSScreen,
    NSBorderlessWindowMask, NSBackingStoreBuffered,
    NSApplicationActivationPolicyAccessory, NSGraphicsContext
)
from Foundation import NSObject, NSTimer
from PyObjCTools import AppHelper
import objc


# Configuration
CALIBRATION_POINTS = 9  # 3x3 grid
CALIBRATION_GRID = [0.15, 0.5, 0.85]
SAMPLES_PER_POINT = 6  # ~54 total samples
HEATMAP_DECAY = 0.99
HEATMAP_KERNEL_SIZE = 81
HEATMAP_SIGMA = 30
PEAK_MIN_DISTANCE = 100
PEAK_THRESHOLD = 0.15
TRAIL_MAX_AGE = 0.4
TRAIL_MAX_LENGTH = 20
FPS = 60

# Smoothing - we do our own since library's is broken
EMA_ALPHA = 0.15  # Low = more smoothing (we're the only smoother now)
VELOCITY_THRESHOLD = 300  # Tight threshold for jump rejection
LOST_TRACKING_FRAMES = 45  # More frames before giving up
MIN_DETECTION_CONFIDENCE = 0.3  # Lower for distance tolerance
MIN_TRACKING_CONFIDENCE = 0.3

# Global for cleanup and recalibration
_app_delegate = None
_recalibrate_requested = False


class GazeSmoother:
    """Multi-stage smoothing: median filter + EMA + velocity clamping."""
    
    def __init__(self, alpha: float = EMA_ALPHA, velocity_thresh: float = VELOCITY_THRESHOLD):
        self.alpha = alpha
        self.velocity_thresh = velocity_thresh
        self.x: Optional[float] = None
        self.y: Optional[float] = None
        self.vx: float = 0.0  # Velocity for momentum
        self.vy: float = 0.0
        self.lost_frames = 0
        self.history: deque = deque(maxlen=5)  # For median filter
    
    def update(self, raw_x: float, raw_y: float) -> Tuple[int, int]:
        """Apply multi-stage smoothing."""
        # Stage 1: Median filter to reject spikes
        self.history.append((raw_x, raw_y))
        if len(self.history) >= 3:
            xs = sorted([p[0] for p in self.history])
            ys = sorted([p[1] for p in self.history])
            median_x = xs[len(xs) // 2]
            median_y = ys[len(ys) // 2]
        else:
            median_x, median_y = raw_x, raw_y
        
        if self.x is None:
            self.x, self.y = median_x, median_y
            self.lost_frames = 0
            return (int(median_x), int(median_y))
        
        # Stage 2: Velocity-based outlier detection
        dx = median_x - self.x
        dy = median_y - self.y
        velocity = (dx**2 + dy**2) ** 0.5
        
        # Clamp velocity - don't allow huge jumps
        if velocity > self.velocity_thresh:
            # Scale down the movement
            scale = self.velocity_thresh / velocity
            dx *= scale * 0.3  # Extra damping for outliers
            dy *= scale * 0.3
            median_x = self.x + dx
            median_y = self.y + dy
        
        # Stage 3: EMA with momentum
        self.vx = self.alpha * (median_x - self.x) + (1 - self.alpha) * self.vx * 0.5
        self.vy = self.alpha * (median_y - self.y) + (1 - self.alpha) * self.vy * 0.5
        
        self.x += self.vx
        self.y += self.vy
        self.lost_frames = 0
        
        return (int(self.x), int(self.y))
    
    def mark_lost(self) -> Optional[Tuple[int, int]]:
        """Called when tracking is lost. Returns last position or None."""
        self.lost_frames += 1
        # Decay velocity when lost
        self.vx *= 0.8
        self.vy *= 0.8
        if self.lost_frames > LOST_TRACKING_FRAMES:
            return None
        if self.x is not None:
            return (int(self.x), int(self.y))
        return None
    
    def reset(self):
        self.x = self.y = None
        self.vx = self.vy = 0.0
        self.lost_frames = 0
        self.history.clear()


@dataclass
class GazeState:
    """Thread-safe container for gaze state shared between threads."""
    point: Optional[Tuple[int, int]] = None
    is_fixation: bool = False
    calib_point: Optional[Tuple[int, int]] = None
    calib_progress: float = 0.0  # Progress as float (point + fraction)
    calibrating: bool = True
    lock: threading.Lock = None
    
    def __post_init__(self):
        self.lock = threading.Lock()
    
    def update_gaze(self, point: Tuple[int, int], fixation: bool):
        with self.lock:
            self.point = point
            self.is_fixation = fixation
            self.calib_point = None
            self.calibrating = False
    
    def update_calibration(self, point: Tuple[int, int], progress: float):
        with self.lock:
            self.calib_point = point
            self.calib_progress = progress
            self.point = None
            self.calibrating = True
    
    def clear_gaze(self):
        with self.lock:
            self.point = None
    
    def get(self) -> dict:
        with self.lock:
            return {
                'point': self.point,
                'is_fixation': self.is_fixation,
                'calib_point': self.calib_point,
                'calib_progress': self.calib_progress,
                'calibrating': self.calibrating
            }


class Heatmap:
    """Gaussian heatmap with decay and peak detection."""
    
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.data = np.zeros((height, width), dtype=np.float32)
        self._kernel = self._make_kernel(HEATMAP_KERNEL_SIZE, HEATMAP_SIGMA)
        self._half = HEATMAP_KERNEL_SIZE // 2
    
    @staticmethod
    def _make_kernel(size: int, sigma: float) -> np.ndarray:
        x = np.arange(size) - size // 2
        k1d = np.exp(-x**2 / (2 * sigma**2))
        k2d = np.outer(k1d, k1d)
        return (k2d / k2d.max()).astype(np.float32)
    
    def add(self, x: int, y: int, weight: float = 1.0):
        x = int(np.clip(x, 0, self.width - 1))
        y = int(np.clip(y, 0, self.height - 1))
        
        x1, x2 = max(0, x - self._half), min(self.width, x + self._half + 1)
        y1, y2 = max(0, y - self._half), min(self.height, y + self._half + 1)
        kx1, kx2 = self._half - (x - x1), self._half + (x2 - x)
        ky1, ky2 = self._half - (y - y1), self._half + (y2 - y)
        
        self.data[y1:y2, x1:x2] += self._kernel[ky1:ky2, kx1:kx2] * weight
    
    def decay(self):
        self.data *= HEATMAP_DECAY
    
    def get_peaks(self, max_count: int = 10) -> List[Tuple[int, int, float]]:
        """Find local maxima using non-maximum suppression. Returns [(x, y_flipped, intensity)]."""
        max_val = self.data.max()
        if max_val < PEAK_THRESHOLD:
            return []
        
        # Downsample for faster processing
        scale = 4
        small = cv2.resize(self.data, (self.width // scale, self.height // scale), 
                          interpolation=cv2.INTER_AREA)
        
        # Find local maxima: pixel is max in its neighborhood
        kernel_size = max(3, PEAK_MIN_DISTANCE // scale)
        if kernel_size % 2 == 0:
            kernel_size += 1
        dilated = cv2.dilate(small, np.ones((kernel_size, kernel_size)))
        local_max = (small == dilated) & (small > PEAK_THRESHOLD)
        
        # Get coordinates of local maxima
        ys, xs = np.where(local_max)
        if len(xs) == 0:
            return []
        
        # Get intensities and sort by intensity (descending)
        intensities = small[ys, xs]
        order = np.argsort(-intensities)
        
        # Non-maximum suppression: keep peaks that are far enough apart
        peaks = []
        min_dist_sq = (PEAK_MIN_DISTANCE // scale) ** 2
        
        for idx in order:
            px, py = int(xs[idx] * scale), int(ys[idx] * scale)
            intensity = float(intensities[idx] / max_val)
            
            # Check distance to existing peaks
            too_close = False
            for ex, ey, _ in peaks:
                if (px - ex) ** 2 + (py - ey) ** 2 < min_dist_sq * scale * scale:
                    too_close = True
                    break
            
            if not too_close:
                peaks.append((px, self.height - py, intensity))
                if len(peaks) >= max_count:
                    break
        
        return peaks
    
    def get_rois(self) -> List[dict]:
        """Get ROIs as dicts with center, radius, weight for mask generation."""
        peaks = self.get_peaks()
        return [{"center": [x, self.height - (self.height - y)], 
                 "radius": int(HEATMAP_SIGMA * 2 * intensity + 50),
                 "weight": intensity} for x, y, intensity in peaks]
    
    def clear(self):
        self.data.fill(0)


class OverlayView(NSView):
    """NSView that renders gaze visualization."""
    
    def initWithFrame_(self, frame):
        self = objc.super(OverlayView, self).initWithFrame_(frame)
        if self is None:
            return None
        self._heat_peaks: List[Tuple[int, int, float]] = []
        self._gaze: Optional[Tuple[int, int]] = None
        self._fixation: bool = False
        self._calib: Optional[Tuple[int, int]] = None
        self._calib_progress: float = 0.0
        self._trail: deque = deque(maxlen=TRAIL_MAX_LENGTH)
        return self
    
    def setData_fixation_calib_peaks_progress_(self, gaze, fixation, calib, peaks, progress):
        """Update view state. Uses ObjC naming to avoid NSView.update conflict."""
        self._heat_peaks = peaks or []
        self._calib = calib
        self._calib_progress = progress
        self._gaze = gaze
        self._fixation = fixation
        if gaze:
            self._trail.append((gaze[0], gaze[1], fixation, time.time()))
    
    def drawRect_(self, rect):
        ctx = NSGraphicsContext.currentContext().CGContext()
        Quartz.CGContextClearRect(ctx, rect)
        
        self._draw_heatmap(ctx)
        
        if self._calib:
            self._draw_calibration(ctx, self._calib)
        else:
            self._draw_trail(ctx)
            if self._gaze:
                self._draw_gaze(ctx, self._gaze, self._fixation)
    
    def _draw_heatmap(self, ctx):
        for x, y, intensity in self._heat_peaks:
            if intensity < 0.1:
                continue
            r, g, b = self._heat_color(intensity)
            
            # Draw 4 concentric circles for smooth falloff
            base_radius = 35 + intensity * 65
            for i in range(4, 0, -1):
                alpha = intensity * (i / 4) * 0.4
                radius = base_radius * (1 + (4 - i) * 0.35)
                Quartz.CGContextSetRGBFillColor(ctx, r, g, b, alpha)
                Quartz.CGContextFillEllipseInRect(
                    ctx, Quartz.CGRectMake(x - radius, y - radius, radius * 2, radius * 2))
    
    @staticmethod
    def _heat_color(t: float) -> Tuple[float, float, float]:
        """Smooth gradient: blue -> cyan -> green -> yellow -> orange -> red."""
        t = max(0.0, min(1.0, t))
        if t < 0.2:  # Blue to Cyan
            p = t / 0.2
            return (0.0, p * 0.8, 1.0)
        elif t < 0.4:  # Cyan to Green
            p = (t - 0.2) / 0.2
            return (0.0, 0.8 + p * 0.2, 1.0 - p)
        elif t < 0.6:  # Green to Yellow
            p = (t - 0.4) / 0.2
            return (p, 1.0, 0.0)
        elif t < 0.8:  # Yellow to Orange
            p = (t - 0.6) / 0.2
            return (1.0, 1.0 - p * 0.35, 0.0)
        else:  # Orange to Red
            p = (t - 0.8) / 0.2
            return (1.0, 0.65 - p * 0.65, 0.0)
    
    def _draw_calibration(self, ctx, point: Tuple[int, int]):
        x, y = point
        frame = self.frame()
        screen_w, screen_h = frame.size.width, frame.size.height
        
        # Overall progress bar at top of screen
        bar_width = 300
        bar_height = 8
        bar_x = (screen_w - bar_width) / 2
        bar_y = screen_h - 60  # Near top in Cocoa coords
        
        overall_progress = self._calib_progress / CALIBRATION_POINTS
        
        # Bar background
        Quartz.CGContextSetRGBFillColor(ctx, 0.2, 0.2, 0.3, 0.8)
        Quartz.CGContextFillRect(ctx, Quartz.CGRectMake(bar_x - 2, bar_y - 2, bar_width + 4, bar_height + 4))
        
        # Bar fill
        Quartz.CGContextSetRGBFillColor(ctx, 0.3, 0.8, 0.5, 0.9)
        Quartz.CGContextFillRect(ctx, Quartz.CGRectMake(bar_x, bar_y, bar_width * overall_progress, bar_height))
        
        # Target circle background
        Quartz.CGContextSetRGBFillColor(ctx, 0.2, 0.1, 0.4, 0.7)
        Quartz.CGContextFillEllipseInRect(ctx, Quartz.CGRectMake(x - 45, y - 45, 90, 90))
        
        # Progress ring for current point
        point_progress = self._calib_progress % 1.0
        if point_progress > 0:
            Quartz.CGContextSetRGBStrokeColor(ctx, 0.3, 1.0, 0.5, 0.9)
            Quartz.CGContextSetLineWidth(ctx, 5)
            start_angle = math.pi / 2
            end_angle = start_angle - point_progress * 2 * math.pi
            Quartz.CGContextAddArc(ctx, x, y, 35, start_angle, end_angle, 1)
            Quartz.CGContextStrokePath(ctx)
        
        # Inner target
        Quartz.CGContextSetRGBFillColor(ctx, 0.5, 0.3, 1.0, 0.9)
        Quartz.CGContextFillEllipseInRect(ctx, Quartz.CGRectMake(x - 20, y - 20, 40, 40))
        
        # Center dot
        Quartz.CGContextSetRGBFillColor(ctx, 1.0, 1.0, 1.0, 1.0)
        Quartz.CGContextFillEllipseInRect(ctx, Quartz.CGRectMake(x - 5, y - 5, 10, 10))
    
    def _draw_trail(self, ctx):
        now = time.time()
        for x, y, fix, ts in self._trail:
            age = now - ts
            if age > TRAIL_MAX_AGE:
                continue
            t = 1 - age / TRAIL_MAX_AGE
            alpha, radius = t * 0.5, t * 12 + 3
            color = (1.0, 0.3, 0.3, alpha) if fix else (0.3, 1.0, 0.3, alpha)
            Quartz.CGContextSetRGBFillColor(ctx, *color)
            Quartz.CGContextFillEllipseInRect(
                ctx, Quartz.CGRectMake(x - radius, y - radius, radius * 2, radius * 2))
    
    def _draw_gaze(self, ctx, point: Tuple[int, int], fixation: bool):
        x, y = point
        if fixation:
            layers = [(30, (1.0, 0.2, 0.2, 0.3)), (18, (1.0, 0.3, 0.3, 0.7)), (6, (1.0, 0.9, 0.9, 1.0))]
        else:
            layers = [(16, (0.2, 1.0, 0.2, 0.6)), (5, (0.8, 1.0, 0.8, 1.0))]
        for radius, (r, g, b, a) in layers:
            Quartz.CGContextSetRGBFillColor(ctx, r, g, b, a)
            Quartz.CGContextFillEllipseInRect(
                ctx, Quartz.CGRectMake(x - radius, y - radius, radius * 2, radius * 2))


class GazeTracker:
    """Eye tracking using EyeGestures in a background thread."""
    
    def __init__(self, camera: int, width: int, height: int, state: GazeState, heatmap: Heatmap):
        self.camera = camera
        self.width = width
        self.height = height
        self.state = state
        self.heatmap = heatmap
        self.running = True
        self._prev_calib = (0, 0)
        self._calib_count = 0
        self._samples_at_point = 0
        self._smoother = GazeSmoother()
        self._gestures: Optional[EyeGestures_v3] = None
        self._recalibrate = False
    
    def start(self):
        threading.Thread(target=self._run, daemon=True).start()
    
    def stop(self):
        self.running = False
    
    def recalibrate(self):
        """Request recalibration - will be picked up by tracking thread."""
        self._recalibrate = True
    
    def _setup_calibration(self, gestures):
        """Setup calibration grid and reset state."""
        xx, yy = np.meshgrid(CALIBRATION_GRID, CALIBRATION_GRID)
        calib_map = np.column_stack([xx.ravel(), yy.ravel()])
        np.random.shuffle(calib_map)
        gestures.uploadCalibrationMap(calib_map, context="overlay")
        gestures.setFixation(0.7)
        
        # Reset library's internal state completely
        gestures.reset(context="overlay")
        
        # Force re-initialization of head tracking on next frame
        gestures.starting_head_position = np.zeros((1, 2))
        gestures.starting_size = np.zeros((1, 2))
        
        # Clear the broken averaging buffer by setting filled_points high
        # This makes library use our smoothing instead
        gestures.filled_points["overlay"] = 20
        gestures.average_points["overlay"] = np.zeros((20, 2))
        
        self._calib_count = 0
        self._samples_at_point = 0
        self._prev_calib = (0, 0)
        self._smoother.reset()
        self.heatmap.clear()
        print(f"Calibration: {CALIBRATION_POINTS} points, {SAMPLES_PER_POINT} samples each. Look at the dots.")
    
    def _run(self):
        global _recalibrate_requested
        
        gestures = EyeGestures_v3()
        self._gestures = gestures
        
        # Patch MediaPipe for better tracking
        try:
            import mediapipe as mp
            gestures.finder.mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
                refine_landmarks=True,
                static_image_mode=False,
                max_num_faces=1,
                min_detection_confidence=MIN_DETECTION_CONFIDENCE,
                min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
            )
        except Exception as e:
            print(f"MediaPipe patch failed: {e}")
        
        cap = VideoCapture(self.camera)
        self._setup_calibration(gestures)
        
        while self.running:
            # Check for recalibration request
            if self._recalibrate or _recalibrate_requested:
                self._recalibrate = False
                _recalibrate_requested = False
                self._setup_calibration(gestures)
            
            ret, frame = cap.read()
            if not ret or frame is None:
                time.sleep(0.01)
                continue
            
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = np.flip(frame, axis=1)
            
            need_calib = self._calib_count < CALIBRATION_POINTS
            event, cevent = gestures.step(frame, need_calib, self.width, self.height, context="overlay")
            
            # Handle lost tracking
            if event is None:
                last_pos = self._smoother.mark_lost()
                if last_pos is None:
                    self.state.clear_gaze()
                time.sleep(0.01)
                continue
            
            if need_calib and cevent:
                px, py = int(cevent.point[0]), int(cevent.point[1])
                
                # Track samples at current calibration point
                if (px, py) != self._prev_calib:
                    # New calibration point
                    if self._samples_at_point >= SAMPLES_PER_POINT:
                        self._calib_count += 1
                    self._samples_at_point = 0
                    self._prev_calib = (px, py)
                else:
                    self._samples_at_point += 1
                
                # Show progress: point number and sample count
                progress = self._calib_count + min(1.0, self._samples_at_point / SAMPLES_PER_POINT)
                self.state.update_calibration((px, self.height - py), progress)
                self._smoother.reset()
            else:
                raw_x, raw_y = event.point[0], event.point[1]
                
                # Our multi-stage smoothing (library's is broken)
                sx, sy = self._smoother.update(raw_x, raw_y)
                sy_cocoa = self.height - sy
                
                self.state.update_gaze((sx, sy_cocoa), event.fixation > 0.5)
                
                # Add to heatmap
                weight = 0.5 if event.fixation > 0.5 else 0.15
                self.heatmap.add(sx, sy, weight)
            
            time.sleep(1 / FPS)


class App(NSObject):
    """Main application delegate."""
    
    def init(self):
        self = objc.super(App, self).init()
        if self is None:
            return None
        
        screen = NSScreen.mainScreen().frame()
        self.width = int(screen.size.width)
        self.height = int(screen.size.height)
        self.frame = screen
        
        self.heatmap = Heatmap(self.width, self.height)
        self.state = GazeState()
        self.tracker: Optional[GazeTracker] = None
        self.view: Optional[OverlayView] = None
        self.show_heatmap = True
        self.camera = 0
        self._event_monitor = None
        
        return self
    
    def applicationDidFinishLaunching_(self, _):
        self._create_window()
        self._setup_key_monitor()
        self.tracker = GazeTracker(self.camera, self.width, self.height, self.state, self.heatmap)
        self.tracker.start()
        
        NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
            1.0 / FPS, self, 'tick:', None, True)
        
        print(f"Gaze overlay running. Press 'q' to recalibrate, Ctrl+C to quit.")
    
    def _setup_key_monitor(self):
        """Setup global key event monitor for 'q' key."""
        from AppKit import NSEvent, NSKeyDownMask
        
        def handler(event):
            global _recalibrate_requested
            chars = event.charactersIgnoringModifiers()
            if chars and chars.lower() == 'q':
                print("\nRecalibration requested...")
                _recalibrate_requested = True
            return event
        
        self._event_monitor = NSEvent.addGlobalMonitorForEventsMatchingMask_handler_(
            NSKeyDownMask, handler
        )
    
    def _create_window(self):
        window = NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
            self.frame, NSBorderlessWindowMask, NSBackingStoreBuffered, False)
        window.setLevel_(Quartz.kCGOverlayWindowLevel)
        window.setOpaque_(False)
        window.setBackgroundColor_(NSColor.clearColor())
        window.setIgnoresMouseEvents_(True)
        window.setCollectionBehavior_(
            Quartz.NSWindowCollectionBehaviorCanJoinAllSpaces |
            Quartz.NSWindowCollectionBehaviorStationary)
        
        self.view = OverlayView.alloc().initWithFrame_(self.frame)
        window.setContentView_(self.view)
        window.makeKeyAndOrderFront_(None)
        self.window = window
    
    def tick_(self, _):
        self.heatmap.decay()
        s = self.state.get()
        peaks = self.heatmap.get_peaks() if self.show_heatmap and not s['calibrating'] else []
        self.view.setData_fixation_calib_peaks_progress_(
            s['point'], s['is_fixation'], s['calib_point'], peaks, s['calib_progress'])
        self.view.setNeedsDisplay_(True)
    
    def cleanup(self):
        """Cleanup event monitor."""
        if self._event_monitor:
            from AppKit import NSEvent
            NSEvent.removeMonitor_(self._event_monitor)
            self._event_monitor = None


def _signal_handler(sig, frame):
    """Handle Ctrl+C gracefully."""
    global _app_delegate
    print("\nShutting down...")
    if _app_delegate:
        if _app_delegate.tracker:
            _app_delegate.tracker.stop()
        _app_delegate.cleanup()
    AppHelper.stopEventLoop()
    sys.exit(0)


def main():
    global _app_delegate
    
    import argparse
    p = argparse.ArgumentParser(description='Transparent gaze overlay')
    p.add_argument('--camera', type=int, default=0)
    p.add_argument('--no-heatmap', action='store_true')
    args = p.parse_args()
    
    # Setup signal handler for Ctrl+C
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)
    
    app = NSApplication.sharedApplication()
    app.setActivationPolicy_(NSApplicationActivationPolicyAccessory)
    
    _app_delegate = App.alloc().init()
    _app_delegate.camera = args.camera
    _app_delegate.show_heatmap = not args.no_heatmap
    app.setDelegate_(_app_delegate)
    
    print("Press Ctrl+C to quit")
    AppHelper.runEventLoop()


if __name__ == '__main__':
    main()
