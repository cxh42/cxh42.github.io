---
title: "Auto Bang"
excerpt: "YOLO11-based target detection for automatic Music game playing<br/><img src='/images/bang.png'>"
collection: portfolio
---

<br/><img src='/images/bang.png'>

Project Overview
=====
Auto Bang Dream is an innovative application that automates gameplay in the rhythm game Bang Dream using computer vision and machine learning techniques. The project combines YOLO (You Only Look Once) object detection with precise input simulation to create a responsive and accurate auto-player system.

AutoBang v2 Dataset————300 high quality manually labeled music game screenshots: https://app.roboflow.com/ds/k4ULdEl9yr?key=muxSteJ6OZ

AutoBang v2 YOLO11 Bang Dream Note Detection
-----
<iframe width="640" height="360" src="https://www.youtube.com/embed/ttC2UBdkAFY?si=OK5pz-ofh4taVCoh" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>
AutoBang v2 used Android Debug Bridge (adb) to simulate human touch behaviors.

AutoBang v1 YOLO11 Bang Dream Note Detection
-----
<iframe width="640" height="360" src="https://www.youtube.com/embed/HAtK6kSt6HY?si=sHOzbEj_HPwYVTDY" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>

AutoBang v1 YOLO11 Bang Dream Auto Player
-----
<iframe width="640" height="360" src="https://www.youtube.com/embed/RF30UTMcYHI?si=vALWwN186AqRtnoM" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>
Due to the complex visuals, extremely strict real-time requirements, and intricate note-triggering logic of the rhythm game BanG Dream!, the current implementation has not demonstrated satisfactory practical usability. As a result, I have decided to put this project on hold for now and revisit it in the future when better solutions become available.

Technical Implementation
=====
Object Detection Model
-----
Utilized YOLOv11-nano architecture for real-time note detection.
Created a custom dataset by capturing and annotating game footage using Roboflow.
Trained the model to recognize multiple note types.

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

Source code of v1
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

Source code of v2
------
```python
import cv2
import torch
from ultralytics import YOLO
import numpy as np
import subprocess
import win32gui
import win32con
import time
import win32ui
from ctypes import windll
import ctypes
import logging
import threading

# 屏蔽 Ultralytics 的信息输出
logging.getLogger("ultralytics").setLevel(logging.WARNING)

class MusicGameAutoplayer:
    def __init__(self, model_path, window_name):
        # 使用 Ultralytics 加载 YOLO 模型
        self.model = YOLO(model_path)
        self.window_name = window_name
        self.hwnd = self.find_window()
        self.width = 1280
        self.height = 720
        self.judgement_line_y = 590  # 判定线的 y 坐标
        self.debug_frame = None

        # 定义判定区域：音符下边缘与判定线距离小于 judge_margin 的视为进入判定区域
        self.judge_margin = 50

        # 假设游戏横向有效区域为 [140, 1140]，分为 7 个轨道，计算各轨道中点作为固定点击点
        self.left_bound = 140
        self.right_bound = 1140
        lane_width = (self.right_bound - self.left_bound) / 7
        self.fixed_lane_points = {i: int(self.left_bound + lane_width * (i + 0.5)) for i in range(7)}
        self.cooldown = 0.2  # 每个轨道点击冷却（秒）
        self.last_click_time = {i: 0 for i in range(7)}

        # 针对 GREEN 及拖尾状态管理
        self.green_active = False
        self.green_x = None
        self.green_last_time = 0  # 最近一次拖动更新时间
        self.green_duration = 1500  # 模拟长按拖动的持续时间（毫秒）

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
            raise Exception(f"未找到窗口: {self.window_name}")
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
                print("PrintWindow 失败")
            
            bmpstr = saveBitMap.GetBitmapBits(True)
            img = np.frombuffer(bmpstr, dtype='uint8')
            img.shape = (window_height, window_width, 4)

            # 释放资源
            win32gui.DeleteObject(saveBitMap.GetHandle())
            saveDC.DeleteDC()
            mfcDC.DeleteDC()
            win32gui.ReleaseDC(self.hwnd, hwndDC)

            frame = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
            frame = cv2.resize(frame, (self.width, self.height))
            return frame
        except Exception as e:
            print(f"截图失败: {e}")
            return None

    def detect_notes(self, frame):
        if frame is None:
            return []
        results = self.model(frame)[0]
        self.debug_frame = frame.copy()
        notes = []
        for det in results.boxes.data:
            x1, y1, x2, y2 = det[:4]
            conf = det[4]
            cls = int(det[5])
            label = self.model.names[cls]
            if label in ['BLUE', 'L', 'R', 'PINK', 'GREEN', 'YELLOW', 'BELT', 'MID']:
                note = {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2, 'conf': conf, 'label': label}
                notes.append(note)
                cv2.rectangle(self.debug_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(self.debug_frame, f"{label} {conf:.2f}", (int(x1), int(y1)-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.line(self.debug_frame, (0, self.judgement_line_y), (self.width, self.judgement_line_y), (0, 0, 255), 3)
        return notes

    def adb_command(self, cmd):
        print(f"执行命令: {cmd}")
        subprocess.Popen(cmd, shell=True)

    def tap(self, x, y):
        cmd = f"adb shell input tap {int(x)} {int(y)}"
        self.adb_command(cmd)

    def swipe(self, start_x, start_y, end_x, end_y, duration):
        cmd = f"adb shell input swipe {int(start_x)} {int(start_y)} {int(end_x)} {int(end_y)} {int(duration)}"
        self.adb_command(cmd)

    def determine_lane(self, note_center_x):
        if note_center_x < self.left_bound or note_center_x > self.right_bound:
            return None
        lane_width = (self.right_bound - self.left_bound) / 7
        lane_index = int((note_center_x - self.left_bound) / lane_width)
        return max(0, min(6, lane_index))

    def auto_play(self):
        print("开始自动演奏...")
        while True:
            try:
                win32gui.SetForegroundWindow(self.hwnd)
                win32gui.ShowWindow(self.hwnd, win32con.SW_RESTORE)
            except Exception as e:
                print(f"设置窗口前台失败: {e}")
            frame = self.capture_screen()
            if frame is None:
                print("获取屏幕失败，重试中...")
                time.sleep(1)
                continue
            notes = self.detect_notes(frame)
            current_time = time.time()

            # 将检测结果按标签分组
            group = {'BLUE': [], 'L': [], 'R': [], 'PINK': [],
                     'GREEN': [], 'YELLOW': [], 'BELT': [], 'MID': []}
            for note in notes:
                if note['label'] in group:
                    group[note['label']].append(note)

            # 判定区域：音符下边缘与判定线距离在 judge_margin 内
            def in_judge_region(n):
                diff = self.judgement_line_y - n['y2']
                return diff >= 0 and diff < self.judge_margin

            # 处理 L 型和 R 型音符（左右滑动）
            l_group = [n for n in group['L'] if in_judge_region(n)]
            if l_group:
                centers = [((n['x1']+n['x2'])/2) for n in l_group]
                self.swipe(max(centers), self.judgement_line_y, min(centers), self.judgement_line_y, duration=150)
            r_group = [n for n in group['R'] if in_judge_region(n)]
            if r_group:
                centers = [((n['x1']+n['x2'])/2) for n in r_group]
                self.swipe(min(centers), self.judgement_line_y, max(centers), self.judgement_line_y, duration=150)

            # 处理 BLUE 类型：普通点击
            for n in group['BLUE']:
                if in_judge_region(n):
                    center_x = (n['x1']+n['x2'])/2
                    lane = self.determine_lane(center_x)
                    if lane is not None and (current_time - self.last_click_time[lane] >= self.cooldown):
                        self.tap(self.fixed_lane_points[lane], self.judgement_line_y)
                        self.last_click_time[lane] = current_time

            # 处理 PINK 类型：立即滑动模拟快速松手
            for n in group['PINK']:
                if in_judge_region(n):
                    center_x = (n['x1']+n['x2'])/2
                    self.swipe(center_x, self.judgement_line_y, center_x + 20, self.judgement_line_y, duration=1)

            # 处理 GREEN 头音符及拖尾逻辑
            drag_group = group['BELT'] + group['MID']
            tail_reached = any(n['y2'] >= self.judgement_line_y for n in drag_group)

            green_candidates = [n for n in group['GREEN'] if in_judge_region(n)]
            if green_candidates:
                candidate = max(green_candidates, key=lambda n: n['y2'])
                target_x = (candidate['x1']+candidate['x2'])/2
                if not self.green_active:
                    # 合并 tap 与 swipe 命令为一个 adb shell 调用，确保连续执行
                    if not tail_reached:
                        cmd = f'adb shell "input tap {target_x} {self.judgement_line_y} && input swipe {target_x} {self.judgement_line_y} {target_x} {self.judgement_line_y} {self.green_duration}"'
                        self.adb_command(cmd)
                        self.green_active = True
                        self.green_x = target_x
                        self.green_last_time = current_time
                    else:
                        cmd = f'adb shell "input tap {target_x} {self.judgement_line_y} && input swipe {target_x} {self.judgement_line_y} {target_x} {self.judgement_line_y} 1"'
                        self.adb_command(cmd)
                        self.green_active = False
                else:
                    if drag_group and not tail_reached:
                        drag_centers = [((n['x1']+n['x2'])/2) for n in drag_group if in_judge_region(n)]
                        if drag_centers:
                            new_target = sum(drag_centers) / len(drag_centers)
                            # 缓慢更新：每次更新采用较小步长（alpha=0.1）
                            alpha = 0.1
                            intermediate_target = self.green_x + alpha * (new_target - self.green_x)
                            if abs(intermediate_target - self.green_x) > 1:
                                self.swipe(self.green_x, self.judgement_line_y, intermediate_target, self.judgement_line_y, duration=500)
                                self.green_x = intermediate_target
                                self.green_last_time = current_time
                    if tail_reached:
                        self.swipe(self.green_x, self.judgement_line_y, self.green_x, self.judgement_line_y, duration=1)
                        self.green_active = False
                    if current_time - self.green_last_time > 0.8:
                        self.green_active = False

            elif group['YELLOW'] and drag_group:
                yellow_candidates = [n for n in group['YELLOW'] if in_judge_region(n)]
                if yellow_candidates:
                    candidate = max(yellow_candidates, key=lambda n: n['y2'])
                    target_x = (candidate['x1']+candidate['x2'])/2
                    if not self.green_active:
                        if not tail_reached:
                            self.tap(target_x, self.judgement_line_y)
                            self.swipe(target_x, self.judgement_line_y, target_x, self.judgement_line_y, duration=self.green_duration)
                            self.green_active = True
                            self.green_x = target_x
                            self.green_last_time = current_time
                        else:
                            self.swipe(target_x, self.judgement_line_y, target_x, self.judgement_line_y, duration=1)
                            self.green_active = False
                    else:
                        if drag_group and not tail_reached:
                            drag_centers = [((n['x1']+n['x2'])/2) for n in drag_group if in_judge_region(n)]
                            if drag_centers:
                                new_target = sum(drag_centers) / len(drag_centers)
                                alpha = 0.1
                                intermediate_target = self.green_x + alpha * (new_target - self.green_x)
                                if abs(intermediate_target - self.green_x) > 1:
                                    self.swipe(self.green_x, self.judgement_line_y, intermediate_target, self.judgement_line_y, duration=500)
                                    self.green_x = intermediate_target
                                    self.green_last_time = current_time
                        if tail_reached:
                            self.swipe(self.green_x, self.judgement_line_y, self.green_x, self.judgement_line_y, duration=1)
                            self.green_active = False
                        if current_time - self.green_last_time > 0.8:
                            self.green_active = False
            else:
                # 没有拖尾的黄色音符按普通点击处理
                for n in group['YELLOW']:
                    if in_judge_region(n) and not drag_group:
                        center_x = (n['x1']+n['x2'])/2
                        lane = self.determine_lane(center_x)
                        if lane is None:
                            continue
                        if current_time - self.last_click_time[lane] >= self.cooldown:
                            self.tap(self.fixed_lane_points[lane], self.judgement_line_y)
                            self.last_click_time[lane] = current_time
                self.green_active = False

            # 显示调试窗口
            if self.debug_frame is not None:
                cv2.imshow('Debug', self.debug_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            time.sleep(0.005)
    
    def run(self):
        try:
            self.auto_play()
        except KeyboardInterrupt:
            print("程序已停止")
        except Exception as e:
            print(f"发生错误: {e}")
        finally:
            cv2.destroyAllWindows()

if __name__ == "__main__":
    # 请确保以管理员权限运行程序，并正确设置模型路径和窗口名称
    autoplayer = MusicGameAutoplayer(
        model_path='best.pt',  # YOLO 模型文件路径
        window_name='BlueStacks App Player'  # 目标窗口名称（例如 BlueStacks 模拟器）
    )
    autoplayer.run()
```