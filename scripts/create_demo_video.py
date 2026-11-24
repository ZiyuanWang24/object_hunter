"""Create a self-contained visual explainer for ObjectHunter.

The animation is intentionally synthetic: it demonstrates the data flow and
uncertainty/lifecycle behavior without claiming to be a camera recording.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import cv2
import numpy as np


W, H, FPS = 1280, 720, 30
BG = (15, 20, 29)
WHITE = (240, 244, 248)
MUTED = (155, 166, 180)
CYAN = (82, 205, 230)
GREEN = (80, 220, 130)
YELLOW = (60, 210, 240)
GRAY = (125, 135, 145)
RED = (80, 100, 235)


def text(img, value, xy, size=0.7, color=WHITE, thickness=1, scale=1.0):
    cv2.putText(img, value, xy, cv2.FONT_HERSHEY_SIMPLEX, size * scale, color, thickness, cv2.LINE_AA)


def panel(img, x1, y1, x2, y2, title, color=(45, 54, 68)):
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
    text(img, title, (x1 + 16, y1 + 30), 0.62, WHITE, 1)


def arrow(img, start, end, color=CYAN, thickness=3):
    cv2.arrowedLine(img, start, end, color, thickness, cv2.LINE_AA, tipLength=0.16)


def draw_camera(img, rect, object_visible=True, phase=0.0, label="RGB camera"):
    x1, y1, x2, y2 = rect
    panel(img, x1, y1, x2, y2, label)
    cv2.rectangle(img, (x1 + 16, y1 + 48), (x2 - 16, y2 - 16), (38, 48, 59), -1)
    for x in range(x1 + 25, x2 - 20, 45):
        cv2.line(img, (x, y2 - 45), (x + 75, y1 + 70), (48, 61, 72), 1)
    cv2.line(img, (x1 + 16, y2 - 75), (x2 - 16, y2 - 75), (55, 68, 78), 1)
    if object_visible:
        cx = int((x1 + x2) / 2 + math.sin(phase) * 80)
        cy = int((y1 + y2) / 2 + math.cos(phase * 0.7) * 25)
        cv2.ellipse(img, (cx, cy), (52, 65), 0, 0, 360, (92, 132, 175), -1)
        cv2.ellipse(img, (cx, cy - 53), (38, 12), 0, 0, 360, (120, 170, 205), -1)
        cv2.ellipse(img, (cx, cy - 54), (27, 7), 0, 0, 360, (28, 40, 54), -1)
        box = (cx - 70, cy - 78, cx + 70, cy + 82)
        cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), GREEN, 2)
        cv2.ellipse(img, (cx, cy + 74), (43, 8), 0, 0, 360, GREEN, 2)
        text(img, "ID 1  cup  0.95", (box[0], box[1] - 10), 0.54, GREEN, 1)
        text(img, "contact point", (cx - 58, cy + 104), 0.42, GREEN, 1)
    else:
        text(img, "object occluded", (x1 + 36, (y1 + y2) // 2), 0.75, YELLOW, 2)


def draw_map(img, rect, phase=0.0, state="OBSERVED", uncertainty=12):
    x1, y1, x2, y2 = rect
    color = GREEN if state == "OBSERVED" else YELLOW if state == "PREDICTED" else GRAY
    panel(img, x1, y1, x2, y2, "Metric tabletop map")
    left, top, right, bottom = x1 + 38, y1 + 62, x2 - 25, y2 - 30
    cv2.rectangle(img, (left, top), (right, bottom), (33, 56, 45), -1)
    for gx in np.linspace(left, right, 7).astype(int):
        cv2.line(img, (gx, top), (gx, bottom), (50, 87, 66), 1)
    for gy in np.linspace(top, bottom, 5).astype(int):
        cv2.line(img, (left, gy), (right, gy), (50, 87, 66), 1)
    text(img, "x (m)", (right - 45, bottom + 22), 0.4, MUTED, 1)
    text(img, "y (m)", (left - 28, top + 5), 0.4, MUTED, 1)
    px = int(left + 80 + (right - left - 160) * (0.5 + 0.35 * math.sin(phase)))
    py = int(top + 70 + (bottom - top - 130) * (0.5 + 0.20 * math.cos(phase * 0.8)))
    for i in range(8):
        hx = int(px - 75 + i * 11)
        hy = int(py + 45 * math.sin(i * 0.7 + phase))
        cv2.circle(img, (hx, hy), 2, (90, 150, 112), -1)
    cv2.ellipse(img, (px, py), (uncertainty * 2, uncertainty), -18, 0, 360, color, 2)
    cv2.circle(img, (px, py), 7, color, -1)
    text(img, f"ID 1  {state.lower()}", (px + 14, py - 13), 0.52, color, 1)
    if state == "PREDICTED":
        arrow(img, (px - 55, py + 30), (px + 55, py - 25), color, 2)
    if state == "DORMANT":
        text(img, "last reliable position", (left + 12, bottom - 10), 0.48, GRAY, 1)


def draw_pipeline(img, t):
    steps = [
        ("YOLO11n-seg", "instances", CYAN),
        ("contact point", "homography", GREEN),
        ("DINOv2", "appearance", (205, 145, 235)),
        ("Hungarian", "association", YELLOW),
        ("Kalman", "uncertainty", (235, 170, 80)),
        ("memory", "persistent ID", GREEN),
    ]
    start_x, y = 70, 255
    gap = 194
    for i, (title, subtitle, color) in enumerate(steps):
        x = start_x + i * gap
        alpha = min(1.0, max(0.0, (t - i * 0.35) / 0.45))
        if alpha <= 0:
            continue
        cv2.rectangle(img, (x, y), (x + 150, y + 105), (35, 45, 57), -1)
        cv2.rectangle(img, (x, y), (x + 150, y + 105), color, 2)
        text(img, title, (x + 12, y + 42), 0.54, color, 1)
        text(img, subtitle, (x + 12, y + 74), 0.48, WHITE, 1)
        if i < len(steps) - 1:
            arrow(img, (x + 154, y + 52), (x + gap - 8, y + 52), MUTED, 2)
    text(img, "Every observation becomes a time-stamped, uncertainty-aware world-model update.", (70, 455), 0.72, WHITE, 1)


def draw_header(img, section, subtitle):
    text(img, "OBJECTHUNTER", (42, 48), 0.85, CYAN, 2)
    text(img, section, (285, 48), 0.62, WHITE, 1)
    text(img, subtitle, (42, 82), 0.48, MUTED, 1)
    cv2.line(img, (42, 100), (1238, 100), (53, 65, 80), 1)


def frame_for(t: float) -> np.ndarray:
    img = np.full((H, W, 3), BG, dtype=np.uint8)
    if t < 7:
        draw_header(img, "MONOCULAR RGB TO SPATIAL MEMORY", "A synthetic walkthrough of the data flow; the live demo uses your webcam.")
        draw_camera(img, (55, 145, 650, 620), True, t * 0.7)
        draw_map(img, (690, 145, 1225, 620), t * 0.7, "OBSERVED", 12)
        text(img, "Detect", (92, 672), 0.68, GREEN, 2)
        arrow(img, (190, 668), (1080, 668), CYAN, 2)
        text(img, "localize", (525, 672), 0.68, CYAN, 2)
        text(img, "track", (855, 672), 0.68, YELLOW, 2)
        text(img, "remember", (1050, 672), 0.68, GREEN, 2)
    elif t < 15:
        local = t - 7
        draw_header(img, "THE PERCEPTION PIPELINE", "Neural perception is isolated from deterministic geometry, filtering, association, and memory.")
        draw_pipeline(img, local)
        text(img, "No depth camera. No LiDAR. A stationary camera and a planar tabletop are the MVP assumptions.", (70, 575), 0.62, YELLOW, 1)
    elif t < 31:
        local = t - 15
        draw_header(img, "ONE TARGET, ONE PERSISTENT ID", "Observed positions are green; short occlusions are predicted in yellow; dormant memory is gray.")
        occluded = 6.0 <= local < 10.0
        recovered = local >= 10.0
        state = "OBSERVED" if not occluded else "PREDICTED"
        if recovered:
            state = "OBSERVED"
        draw_camera(img, (45, 135, 690, 590), not occluded, local * 0.8)
        draw_map(img, (720, 135, 1235, 590), local * 0.8, state, 12 + int(max(0, local - 6) * 8) if occluded else 12)
        if occluded:
            text(img, "OCCLUDED: Kalman prediction continues", (80, 642), 0.70, YELLOW, 2)
            text(img, "uncertainty grows", (930, 642), 0.60, YELLOW, 1)
        elif recovered:
            text(img, "RE-IDENTIFIED: ID 1 preserved", (80, 642), 0.70, GREEN, 2)
            text(img, "gallery + semantic match", (930, 642), 0.60, GREEN, 1)
        else:
            text(img, "OBSERVED: contact point projected to meters", (80, 642), 0.70, GREEN, 2)
            text(img, "95% covariance ellipse", (930, 642), 0.60, CYAN, 1)
    elif t < 39:
        local = t - 31
        draw_header(img, "QUANTIFIED VALIDATION", "Synthetic, deterministic core benchmark; neural-camera metrics are intentionally not fabricated.")
        rows = [
            ("Unit tests", "10 passed", GREEN),
            ("Homography world RMSE", "2.61e-8 m", CYAN),
            ("Kalman warm-up RMSE", "0.00872 m", CYAN),
            ("10-frame occlusion RMSE", "0.00626 m", YELLOW),
            ("Identity consistency", "100% / 60 frames", GREEN),
            ("Re-identification", "1/1 success", GREEN),
        ]
        x1, y1 = 130, 155
        for i, (name, value, color) in enumerate(rows):
            yy = y1 + i * 65
            cv2.rectangle(img, (x1, yy), (x1 + 1020, yy + 48), (31, 40, 51), -1)
            text(img, name, (x1 + 22, yy + 32), 0.62, WHITE, 1)
            text(img, value, (x1 + 700, yy + 32), 0.62, color, 2)
        text(img, "Reproduce with: python scripts/evaluate.py --config configs/default.yaml", (130, 590), 0.56, MUTED, 1)
    else:
        draw_header(img, "RUN THE REAL DEMO", "The project is on GitHub and supports webcam, prerecorded video, CPU fallback, and CUDA.")
        panel(img, 100, 160, 1180, 520, "Launch commands")
        text(img, "python scripts/run_demo.py --camera 0 --device cuda --single-target", (145, 245), 0.68, GREEN, 1)
        text(img, "or use the guided desktop UI:", (145, 315), 0.65, WHITE, 1)
        text(img, "powershell -ExecutionPolicy Bypass -File scripts/run_ui_gpu.ps1", (145, 370), 0.68, CYAN, 1)
        text(img, "Enter an item such as cup, then watch its ID, location, lifecycle,", (145, 445), 0.62, WHITE, 1)
        text(img, "trajectory, velocity, and covariance on the dashboard.", (145, 480), 0.62, WHITE, 1)
        text(img, "github.com/ZiyuanWang24/object_hunter", (145, 575), 0.62, CYAN, 1)
    return img


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="outputs/objecthunter_demo.mp4")
    parser.add_argument("--seconds", type=float, default=46.0)
    args = parser.parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    if not writer.isOpened():
        raise RuntimeError("Could not open MP4 writer; install an OpenCV build with video codecs")
    for index in range(int(args.seconds * FPS)):
        writer.write(frame_for(index / FPS))
    writer.release()
    print(f"Wrote {output} ({args.seconds:.1f}s, {W}x{H}, {FPS} FPS)")


if __name__ == "__main__":
    main()
