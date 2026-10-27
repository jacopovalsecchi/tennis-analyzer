#!/usr/bin/env python3
"""Prova MediaPipe Pose su un video filmato da dietro il giocatore.

Disegna spalle, anche, ginocchia, caviglie, gomiti e polsi. Non giudica il colpo.
"""

from __future__ import annotations

import argparse
import csv
import sys
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "pose_landmarker_full.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_full/float16/latest/pose_landmarker_full.task"
)

# Indici del Pose Landmarker. Sinistra e destra sono quelle del giocatore.
POINTS = {
    11: "l_shoulder",
    12: "r_shoulder",
    13: "l_elbow",
    14: "r_elbow",
    15: "l_wrist",
    16: "r_wrist",
    23: "l_hip",
    24: "r_hip",
    25: "l_knee",
    26: "r_knee",
    27: "l_ankle",
    28: "r_ankle",
}

BONES = (
    (11, 12),
    (23, 24),
    (11, 23),
    (12, 24),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
    (23, 25),
    (25, 27),
    (24, 26),
    (26, 28),
)

VISIBILITY_OK = 0.5


def ensure_model() -> None:
    if MODEL_PATH.exists() and MODEL_PATH.stat().st_size > 0:
        return
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"Scarico il modello in {MODEL_PATH} ...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)


def visibility(landmark) -> float:
    value = getattr(landmark, "visibility", None)
    if value is None:
        value = getattr(landmark, "presence", 0.0)
    return float(value or 0.0)


def pixel(landmark, width: int, height: int) -> tuple[int, int]:
    return int(landmark.x * width), int(landmark.y * height)


def draw_pose(frame, landmarks) -> None:
    height, width = frame.shape[:2]
    found = {index: landmarks[index] for index in POINTS if index < len(landmarks)}

    for start, end in BONES:
        if start not in found or end not in found:
            continue
        first, second = found[start], found[end]
        confident = visibility(first) >= VISIBILITY_OK and visibility(second) >= VISIBILITY_OK
        color = (40, 180, 40) if confident else (40, 40, 220)
        cv2.line(frame, pixel(first, width, height), pixel(second, width, height), color, 2)

    for landmark in found.values():
        color = (40, 180, 40) if visibility(landmark) >= VISIBILITY_OK else (40, 40, 220)
        cv2.circle(frame, pixel(landmark, width, height), 5, color, -1)

    left = found.get(15)
    right = found.get(16)
    left_vis = visibility(left) if left is not None else 0.0
    right_vis = visibility(right) if right is not None else 0.0
    cv2.putText(
        frame,
        f"polso sx {left_vis:.2f}   polso dx {right_vis:.2f}",
        (16, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def empty_row(frame_index: int, time_s: float) -> dict[str, str | int]:
    row: dict[str, str | int] = {
        "frame": frame_index,
        "time_s": f"{time_s:.3f}",
        "pose_found": 0,
    }
    for name in POINTS.values():
        row[f"{name}_x"] = ""
        row[f"{name}_y"] = ""
        row[f"{name}_vis"] = ""
    return row


def fill_row(row: dict[str, str | int], landmarks) -> None:
    row["pose_found"] = 1
    for index, name in POINTS.items():
        if index >= len(landmarks):
            continue
        landmark = landmarks[index]
        row[f"{name}_x"] = f"{landmark.x:.5f}"
        row[f"{name}_y"] = f"{landmark.y:.5f}"
        row[f"{name}_vis"] = f"{visibility(landmark):.3f}"


def open_writer(path: Path, fps: float, width: int, height: int) -> cv2.VideoWriter | None:
    for codec in ("avc1", "mp4v"):
        writer = cv2.VideoWriter(
            str(path), cv2.VideoWriter_fourcc(*codec), fps, (width, height)
        )
        if writer.isOpened():
            return writer
        writer.release()
    return None


def summarize(rows: list[dict[str, str | int]]) -> None:
    total = len(rows)
    found = sum(int(row["pose_found"]) for row in rows)
    print()
    print(f"Frame letti: {total}")
    print(f"Frame con una pose: {found}")
    if total:
        print(f"Copertura: {found / total:.0%}")

    for key, label in (
        ("l_wrist_vis", "Polso sinistro"),
        ("r_wrist_vis", "Polso destro"),
        ("l_hip_vis", "Anca sinistra"),
        ("r_hip_vis", "Anca destra"),
        ("l_knee_vis", "Ginocchio sinistro"),
        ("r_knee_vis", "Ginocchio destro"),
        ("l_ankle_vis", "Caviglia sinistra"),
        ("r_ankle_vis", "Caviglia destra"),
    ):
        values = [float(row[key]) for row in rows if row[key] != ""]
        if not values:
            print(f"{label}: mai visibile")
            continue
        low = sum(value < VISIBILITY_OK for value in values)
        mean = sum(values) / len(values)
        print(f"{label}: visibilità media {mean:.2f}, frame sotto {VISIBILITY_OK:.1f}: {low}")

    print()
    print("Apri il video overlay. Le gambe sono il segmento anca–ginocchio–caviglia.")
    print("Verde: punto affidabile. Rosso: visibilità sotto 0.5.")


def run(video_path: Path, output_dir: Path) -> int:
    ensure_model()
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        print(f"Non riesco ad aprire il video: {video_path}", file=sys.stderr)
        return 1

    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_dir.mkdir(parents=True, exist_ok=True)
    overlay_path = output_dir / f"{video_path.stem}_overlay.mp4"
    csv_path = output_dir / f"{video_path.stem}_keypoints.csv"
    writer = open_writer(overlay_path, fps, width, height)
    if writer is None:
        print("Non riesco a creare il video in output.", file=sys.stderr)
        capture.release()
        return 1

    options = vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    rows: list[dict[str, str | int]] = []
    fieldnames = list(empty_row(0, 0).keys())
    frame_index = 0
    last_timestamp = -1

    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        with csv_path.open("w", newline="") as handle:
            csv_writer = csv.DictWriter(handle, fieldnames=fieldnames)
            csv_writer.writeheader()
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                timestamp = int(frame_index * 1000 / fps)
                if timestamp <= last_timestamp:
                    timestamp = last_timestamp + 1
                last_timestamp = timestamp
                result = landmarker.detect_for_video(mp_image, timestamp)
                row = empty_row(frame_index, frame_index / fps)
                if result.pose_landmarks:
                    landmarks = result.pose_landmarks[0]
                    fill_row(row, landmarks)
                    draw_pose(frame, landmarks)
                else:
                    cv2.putText(
                        frame,
                        "nessuna pose",
                        (16, 32),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (40, 40, 220),
                        2,
                        cv2.LINE_AA,
                    )
                writer.write(frame)
                csv_writer.writerow(row)
                rows.append(row)
                frame_index += 1
                if frame_index % 30 == 0:
                    print(f"Frame {frame_index}...", flush=True)

    capture.release()
    writer.release()
    print(f"Video con scheletro: {overlay_path}")
    print(f"Coordinate: {csv_path}")
    summarize(rows)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prova MediaPipe Pose su un video filmato da dietro."
    )
    parser.add_argument("video", type=Path, help="Percorso del video, per esempio videos/sample.mp4")
    parser.add_argument("--output", type=Path, default=ROOT / "output", help="Cartella dei risultati")
    args = parser.parse_args()
    video = args.video.expanduser().resolve()
    if not video.is_file():
        print(f"Video non trovato: {video}", file=sys.stderr)
        print("Copia il file in videos/ e rilancia il comando.", file=sys.stderr)
        return 1
    return run(video, args.output.resolve())


if __name__ == "__main__":
    raise SystemExit(main())
