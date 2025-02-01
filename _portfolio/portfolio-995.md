---
title: "Auto Bang Dream"
excerpt: "YOLO11-based target detection for automatic Music game playing<br/><img src='/images/bang.png'>"
collection: portfolio
---

Project Overview
=====
Auto Bang Dream is an innovative application that automates gameplay in the rhythm game Bang Dream using computer vision and machine learning techniques. The project combines YOLO (You Only Look Once) object detection with precise input simulation to create a responsive and accurate auto-player system.

<video src="{{cxh42.github.io}}/assets/videos/good.mp4" width="640" height="360" controls>
    您的浏览器不支持视频播放。
</video>

<video src="{{cxh42.github.io}}/assets/videos/bad.mp4" width="640" height="360" controls>
    您的浏览器不支持视频播放。
</video>


Technical Implementation
=====
Object Detection Model
-----
Utilized YOLOv11-nano architecture for real-time note detection
Created a custom dataset by capturing and annotating game footage using Roboflow
Trained the model to recognize multiple note types:
* Standard notes (Purple, Blue, Pink, Red)
* Hold notes (Green/Yellow)
* Different lane positions (7 lanes)

Core Features
-----
* Real-time Screen Capture
   * Implements Windows API for efficient screen capture
   * Processes game window specifically using win32gui
   * Maintains high capture framerate for accurate detection
* Note Detection System

   * Detects multiple note types simultaneously
   * Processes notes based on their vertical position relative to judgment lines
   * Implements separate detection logic for standard notes and hold notes
* Input Simulation
   * Uses pydirectinput for reliable game input simulation
   * Implements multi-threaded key press management
   * Handles both short presses and long-hold notes
* Lane Management
   * Divides screen width into 7 distinct lanes
   * Maps each lane to corresponding keyboard inputs
   * Implements intelligent collision detection for overlapping notes

Technical Highlights
-----
* Precision Timing: Implements dual judgment lines for enhanced accuracy. Primary judgment line for standard notes. Secondary line for hold note release timing
* Multi-threading: Utilizes threaded key press handlers for responsive input
* Debug Visualization: Includes real-time visual feedback showing:
   * Note detection boundaries
   * Judgment lines
   * Current key states
   * Detection confidence scores

Architecture
=====
The system follows a modular design with distinct components:
* Screen Capture Module
* YOLO Detection Pipeline
* Note Processing Logic
* Input Simulation System
* Debug Visualization Interface

Performance Optimization
-----
* Minimized capture-to-input latency
* Optimized screen capture frequency
* Implemented efficient memory management
* Added configurable tolerance settings for different note types*

Future Developments
-----
* Support for additional rhythm game titles
* Enhanced accuracy through model fine-tuning
* Advanced pattern recognition for complex note sequences
* Performance optimization for lower-end systems

Source code
------
```python
import cv2
from ultralytics import YOLO
import numpy as np
import win32gui
import win32con
import time
import win32ui
from ctypes import windll, Structure, Union, POINTER, byref, sizeof, c_ulong, c_ushort, c_uint
import ctypes
import logging
import pydirectinput  # Used to simulate in-game key presses
import threading

# Suppress Ultralytics' information output
logging.getLogger("ultralytics").setLevel(logging.WARNING)

# --------------------------
# Define structures required for SendInput (for backup)
# --------------------------
PUL = POINTER(c_ulong)

class KEYBDINPUT(Structure):
    _fields_ = [
        ("wVk", c_ushort),
        ("wScan", c_ushort),
        ("dwFlags", c_ulong),
        ("time", c_ulong),
        ("dwExtraInfo", PUL)
    ]

class _INPUT(Union):
    _fields_ = [("ki", KEYBDINPUT)]
    
class INPUT(Structure):
    _fields_ = [
        ("type", c_ulong),
        ("_input", _INPUT)
    ]

INPUT_KEYBOARD = 1
KEYEVENTF_KEYUP = 0x0002

def send_key_input(hexKeyCode, press=True):
    extra = c_ulong(0)
    flags = 0 if press else (KEYEVENTF_KEYUP | 0x0001)
    ii = _INPUT()
    ii.ki = KEYBDINPUT(wVk=hexKeyCode, wScan=0, dwFlags=flags, time=0, dwExtraInfo=ctypes.pointer(extra))
    x = INPUT(type=INPUT_KEYBOARD, _input=ii)
    windll.user32.SendInput(1, byref(x), sizeof(x))

# --------------------------
# Main program code
# --------------------------
class MusicGameAutoplayer:
    def __init__(self, model_path, window_name):
        # Load YOLO model
        self.model = YOLO(model_path)
        self.window_name = window_name
        self.hwnd = self.find_window()
        # Fixed output size (rescaled game area size)
        self.width = 1280
        self.height = 720
        # Judgment line (head judgment) Y-coordinate (after cropping the top border in the resized image)
        self.judgement_line_y = 590
        # Green note release judgment line, slightly above the judgment line (i.e., smaller value) for early release
        self.green_release_line_y = self.judgement_line_y - 10
        # Tolerance settings
        self.tolerance_head = 15   # Used to detect green head notes
        self.tolerance_tail = 15   # Used to detect green tail notes
        self.tolerance_short = 15  # Used for non-green short press notes

        # Define key mappings for each note type (7 horizontal lanes)
        # Here, we assume PURPLE and GREEN share the same set of keys
        self.key_mappings = {
            'PURPLE': ['A', 'B', 'C', 'D', 'E', 'F', 'G'],
            'GREEN':  ['A', 'B', 'C', 'D', 'E', 'F', 'G'],
            'BLUE':   ['H', 'I', 'J', 'K', 'L', 'M', 'N'],
            'PINK':   ['O', 'P', 'Q', 'R', 'S', 'T', 'U'],
            'RED':    ['O', 'P', 'Q', 'R', 'S', 'T', 'U']
        }
        # Debug image display
        self.debug_frame = None
        # Record active green long-press notes per lane, where the key is the lane number and the value is the currently pressed key
        self.green_active = {}

    def find_window(self):
        hwnd = win32gui.FindWindow(None, self.window_name)
        if not hwnd:
            def enum_windows_callback(hwnd, windows):
                if self.window_name.lower() in win32gui.GetWindowText(hwnd).lower():
                    windows.append(hwnd)
                return True
            windows = []
            win32gui.EnumWindows(enum_windows_callback, windows)
            if windows:
                hwnd = windows[0]
        if not hwnd:
            raise Exception(f"Window not found: {self.window_name}")
        return hwnd

    def capture_screen(self):
        try:
            left, top, right, bottom = win32gui.GetWindowRect(self.hwnd)
            window_width = right - left
            window_height = bottom - top

            hwndDC = win32gui.GetWindowDC(self.hwnd)
            mfcDC = win32ui.CreateDCFromHandle(hwndDC)
            saveDC = mfcDC.CreateCompatibleDC()
            saveBitMap = win32ui.CreateBitmap()
            saveBitMap.CreateCompatibleBitmap(mfcDC, window_width, window_height)
            saveDC.SelectObject(saveBitMap)

            result = windll.user32.PrintWindow(self.hwnd, saveDC.GetSafeHdc(), 0)
            if result != 1:
                print("PrintWindow failed")
            
            bmpstr = saveBitMap.GetBitmapBits(True)
            img = np.frombuffer(bmpstr, dtype='uint8')
            img.shape = (window_height, window_width, 4)

            # Release bitmap resources to prevent GDI object leaks
            win32gui.DeleteObject(saveBitMap.GetHandle())
            saveDC.DeleteDC()
            mfcDC.DeleteDC()
            win32gui.ReleaseDC(self.hwnd, hwndDC)

            frame = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
            # Crop a fixed top border (44 pixels)
            if frame.shape[0] > 44:
                frame = frame[44:, :]
            else:
                print("Warning: Screenshot height is insufficient, unable to crop top border")
            frame = cv2.resize(frame, (self.width, self.height))
            return frame
        except Exception as e:
            print(f"Screenshot failed: {e}")
            return None

    def run(self):
        try:
            self.auto_play()
        except KeyboardInterrupt:
            print("Program stopped")
        except Exception as e:
            print(f"An error occurred: {e}")
        finally:
            cv2.destroyAllWindows()

if __name__ == "__main__":
    # Ensure the program runs with administrator privileges and that the model path and window name are correctly set
    autoplayer = MusicGameAutoplayer(
        model_path='best.pt',
        window_name='BlueStacks App Player'
    )
    autoplayer.run()
```