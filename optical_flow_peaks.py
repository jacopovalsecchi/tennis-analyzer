#!/usr/bin/env python3
"""Find motion peaks in a sports video with dense Farneback optical flow.

Exploration tool. It does not talk to the Express backend.

Example:
  .venv/bin/python optical_flow_peaks.py backend/uploads/VIDEO.mov
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path, help="Video file to scan")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Result folder. Default: output/optical_flow/<video stem>",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=480,
        help="Width used for the flow. The saved frames stay full size.",
    )
    parser.add_argument(
        "--min-gap",
        type=float,
        default=0.5,
        help="Minimum seconds between two peaks. The stronger one is kept.",
    )
    parser.add_argument(
        "--k",
        type=float,
        default=3.0,
        help="Peak threshold is median + k * MAD of the smoothed series.",
    )
    parser.add_argument(
        "--radius",
        type=int,
        default=3,
        help="Frames saved before and after each peak.",
    )
    return parser.parse_args()


def resize_gray(frame: np.ndarray, width: int) -> np.ndarray:
    height, frame_width = frame.shape[:2]
    if frame_width <= width:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return gray
    scale = width / frame_width
    small = cv2.resize(frame, (width, max(1, int(height * scale))), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)


def flow_magnitude(previous: np.ndarray, current: np.ndarray) -> np.ndarray:
    flow = cv2.calcOpticalFlowFarneback(
        previous,
        current,
        None,
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0,
    )
    magnitude, _angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    return magnitude


def moving_average(values: np.ndarray, radius: int = 2) -> np.ndarray:
    if len(values) == 0:
        return values
    kernel = np.ones(radius * 2 + 1, dtype=np.float64)
    padded = np.pad(values, radius, mode="edge")
    smoothed = np.convolve(padded, kernel, mode="valid") / kernel.sum()
    return smoothed


def find_peaks(times: np.ndarray, values: np.ndarray, min_gap: float, k: float) -> list[int]:
    if len(values) < 3:
        return []
    center = float(np.median(values))
    spread = float(np.median(np.abs(values - center)))
    threshold = center + k * max(spread, 1e-6)
    candidates = []
    for index in range(1, len(values) - 1):
        value = values[index]
        if value < threshold:
            continue
        if value >= values[index - 1] and value > values[index + 1]:
            candidates.append(index)
    if not candidates:
        return []
    kept = [candidates[0]]
    for index in candidates[1:]:
        if times[index] - times[kept[-1]] <= min_gap:
            if values[index] > values[kept[-1]]:
                kept[-1] = index
        else:
            kept.append(index)
    return kept


def scan_video(video_path: Path, width: int) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise SystemExit(f"Cannot open video: {video_path}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    previous = None
    frame_index = 0
    times: list[float] = []
    mean_motion: list[float] = []
    high_motion: list[float] = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        gray = resize_gray(frame, width)
        if previous is not None:
            magnitude = flow_magnitude(previous, gray)
            times.append(frame_index / fps)
            mean_motion.append(float(np.mean(magnitude)))
            high_motion.append(float(np.percentile(magnitude, 90)))
        previous = gray
        frame_index += 1
        if frame_index % 30 == 0:
            print(f"Frame {frame_index}...", flush=True)
    capture.release()
    if not times:
        raise SystemExit("The video has fewer than two frames.")
    return fps, np.asarray(times), np.asarray(mean_motion), np.asarray(high_motion)


def save_series(
    path: Path,
    times: np.ndarray,
    mean_motion: np.ndarray,
    high_motion: np.ndarray,
    smoothed: np.ndarray,
    peak_indexes: set[int],
) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["time_s", "mean_magnitude", "p90_magnitude", "smoothed_p90", "is_peak"])
        for index, time_s in enumerate(times):
            writer.writerow(
                [
                    f"{time_s:.3f}",
                    f"{mean_motion[index]:.4f}",
                    f"{high_motion[index]:.4f}",
                    f"{smoothed[index]:.4f}",
                    "1" if index in peak_indexes else "0",
                ]
            )


def save_plot(path: Path, times: np.ndarray, smoothed: np.ndarray, peaks: list[int], threshold: float) -> None:
    figure, axis = plt.subplots(figsize=(12, 4))
    axis.plot(times, smoothed, color="#1f4b3a", linewidth=1.2, label="smoothed p90 flow")
    axis.axhline(threshold, color="#b45309", linewidth=0.8, linestyle="--", label="threshold")
    if peaks:
        axis.scatter(times[peaks], smoothed[peaks], color="#b91c1c", zorder=3, label="peaks")
    axis.set_xlabel("time (s)")
    axis.set_ylabel("motion")
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(path, dpi=120)
    plt.close(figure)


def save_peak_frames(video_path: Path, output_dir: Path, fps: float, peaks: list[int], radius: int) -> None:
    wanted = {}
    for peak_number, peak in enumerate(peaks, start=1):
        # The series starts at frame 1, because flow needs a previous frame.
        center = peak + 1
        for offset in range(-radius, radius + 1):
            frame_index = center + offset
            if frame_index >= 0:
                wanted.setdefault(frame_index, []).append((peak_number, offset, center / fps))
    if not wanted:
        return
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(video_path))
    frame_index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        for peak_number, offset, time_s in wanted.get(frame_index, []):
            name = f"peak_{peak_number:02d}_t{time_s:07.3f}s_offset{offset:+d}.jpg"
            cv2.imwrite(str(frames_dir / name), frame)
        frame_index += 1
    capture.release()


def main() -> None:
    args = parse_args()
    video_path = args.video.expanduser().resolve()
    if not video_path.is_file():
        raise SystemExit(f"Video not found: {video_path}")
    output_dir = args.output or Path("output") / "optical_flow" / video_path.stem
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    fps, times, mean_motion, high_motion = scan_video(video_path, args.width)
    smoothed = moving_average(high_motion)
    peaks = find_peaks(times, smoothed, args.min_gap, args.k)
    center = float(np.median(smoothed))
    spread = float(np.median(np.abs(smoothed - center)))
    threshold = center + args.k * max(spread, 1e-6)

    save_series(output_dir / "motion.csv", times, mean_motion, high_motion, smoothed, set(peaks))
    save_plot(output_dir / "motion.png", times, smoothed, peaks, threshold)
    save_peak_frames(video_path, output_dir, fps, peaks, args.radius)

    print(f"Video: {video_path.name}")
    print(f"Duration: {times[-1]:.2f}s  fps: {fps:.2f}  samples: {len(times)}")
    print(f"Peaks: {len(peaks)}")
    with (output_dir / "peaks.txt").open("w") as handle:
        for number, index in enumerate(peaks, start=1):
            line = (
                f"{number:02d}  t={times[index]:.3f}s  "
                f"frame={index + 1}  motion={smoothed[index]:.3f}"
            )
            print(f"  {line}")
            handle.write(line + "\n")
    print(f"Series: {output_dir / 'motion.csv'}")
    print(f"Plot: {output_dir / 'motion.png'}")
    print(f"Frames: {output_dir / 'frames'}")


if __name__ == "__main__":
    main()
