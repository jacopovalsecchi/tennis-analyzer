#!/usr/bin/env python3
"""Extract per-stroke features from a MediaPipe pose CSV.

The racket is not a keypoint: the estimated tip extends the forearm
one forearm-length past the wrist. Each stroke is labeled forehand or
backhand from the player's dominant arm.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VISIBILITY_OK = 0.5
PEAK_SPEED = 8.0
VALLEY_SPEED = 2.5
MERGE_SECONDS = 0.35
MIN_FRAMES = 5
RACKET_EXTENSION = 1.0
JUMP_TORSO = 0.35
SEGMENT_PRESENT = 0.25

JOINTS = (
    "l_shoulder",
    "r_shoulder",
    "l_elbow",
    "r_elbow",
    "l_wrist",
    "r_wrist",
    "l_hip",
    "r_hip",
)
ARMS = {
    "left": ("l_shoulder", "l_elbow", "l_wrist"),
    "right": ("r_shoulder", "r_elbow", "r_wrist"),
}

STROKE_FIELDS = [
    "shot",
    "striking_arm",
    "stroke_type",
    "backhand_style",
    "t_start",
    "t_contact",
    "t_end",
    "elbow_angle_deg",
    "racket_torso_angle_deg",
    "wrist_speed",
    "shoulder_elbow_distance",
    "elbow_wrist_distance",
    "wrist_racket_distance",
    "shoulder_racket_distance",
    "swing_direction_deg",
    "shoulder_rotation_deg",
    "hip_rotation_deg",
    "contact_side",
    "crosses_center",
    "left_knee_angle_deg",
    "right_knee_angle_deg",
    "needs_review",
]
FRAME_FIELDS = [
    "shot",
    "frame",
    "time_s",
    "striking_arm",
    "elbow_angle_deg",
    "racket_torso_angle_deg",
    "wrist_speed",
    "wrist_visibility",
    "left_knee_angle_deg",
    "right_knee_angle_deg",
    "jump",
]


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def video_geometry(path: Path) -> tuple[float, float, float]:
    import cv2

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise SystemExit(f"Cannot open the video: {path}")
    width = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    capture.release()
    if width <= 0 or height <= 0:
        raise SystemExit(f"Invalid video size: {path}")
    return width, height, fps


def number(row: dict[str, str], key: str) -> float | None:
    raw = row.get(key, "")
    if raw == "":
        return None
    return float(raw)


def point(row: dict[str, str], name: str, scale_y: float) -> tuple[float, float, float] | None:
    x = number(row, f"{name}_x")
    y = number(row, f"{name}_y")
    visibility = number(row, f"{name}_vis")
    if x is None or y is None or visibility is None:
        return None
    return x, y * scale_y, visibility


def length(vector: tuple[float, float]) -> float:
    return math.hypot(vector[0], vector[1])


def subtract(a: tuple[float, float], b: tuple[float, float]) -> tuple[float, float]:
    return a[0] - b[0], a[1] - b[1]


def distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return length(subtract(a, b))


def angle_between(first: tuple[float, float], second: tuple[float, float]) -> float | None:
    first_length = length(first)
    second_length = length(second)
    if first_length < 1e-6 or second_length < 1e-6:
        return None
    cosine = (first[0] * second[0] + first[1] * second[1]) / (first_length * second_length)
    cosine = max(-1.0, min(1.0, cosine))
    return math.degrees(math.acos(cosine))


def wrap_degrees(value: float) -> float:
    while value > 180:
        value -= 360
    while value <= -180:
        value += 360
    return value


def line_angle(start: tuple[float, float], end: tuple[float, float]) -> float | None:
    vector = subtract(end, start)
    if length(vector) < 1e-6:
        return None
    return math.degrees(math.atan2(vector[1], vector[0]))


def torso(row: dict[str, str], scale_y: float) -> tuple[tuple[float, float], tuple[float, float], float] | None:
    left_shoulder = point(row, "l_shoulder", scale_y)
    right_shoulder = point(row, "r_shoulder", scale_y)
    left_hip = point(row, "l_hip", scale_y)
    right_hip = point(row, "r_hip", scale_y)
    if not all((left_shoulder, right_shoulder, left_hip, right_hip)):
        return None
    mid_shoulder = (
        (left_shoulder[0] + right_shoulder[0]) / 2,
        (left_shoulder[1] + right_shoulder[1]) / 2,
    )
    mid_hip = ((left_hip[0] + right_hip[0]) / 2, (left_hip[1] + right_hip[1]) / 2)
    torso_length = distance(mid_shoulder, mid_hip)
    if torso_length < 1e-4:
        return None
    return mid_shoulder, mid_hip, torso_length


def smooth(values: list[float], radius: int = 2) -> list[float]:
    smoothed: list[float] = []
    for index in range(len(values)):
        window = values[max(0, index - radius) : index + radius + 1]
        smoothed.append(sum(window) / len(window))
    return smoothed


def wrist_speeds(
    rows: list[dict[str, str]], scale_y: float, fps: float
) -> list[dict[str, float]]:
    previous: dict[str, tuple[float, float] | None] = {"left": None, "right": None}
    series: list[dict[str, float]] = []
    for row in rows:
        body = torso(row, scale_y)
        current: dict[str, float] = {}
        for arm, (_, _, wrist_name) in ARMS.items():
            wrist = point(row, wrist_name, scale_y)
            if body is None or wrist is None or wrist[2] < VISIBILITY_OK or row["pose_found"] != "1":
                current[arm] = 0.0
                previous[arm] = None
                continue
            prior = previous[arm]
            step = 0.0 if prior is None else distance(wrist[:2], prior) / body[2]
            current[arm] = step * fps
            current[f"{arm}_step"] = step
            previous[arm] = wrist[:2]
        current["max"] = max(current["left"], current["right"])
        series.append(current)
    maxima = smooth([item["max"] for item in series])
    for item, value in zip(series, maxima):
        item["smooth"] = value
    return series


def find_peaks(series: list[dict[str, float]], times: list[float]) -> list[int]:
    candidates: list[int] = []
    for index in range(1, len(series) - 1):
        speed = series[index]["smooth"]
        if speed < PEAK_SPEED:
            continue
        if speed >= series[index - 1]["smooth"] and speed > series[index + 1]["smooth"]:
            candidates.append(index)
    if not candidates:
        return []
    merged = [candidates[0]]
    for index in candidates[1:]:
        if times[index] - times[merged[-1]] <= MERGE_SECONDS:
            if series[index]["smooth"] > series[merged[-1]]["smooth"]:
                merged[-1] = index
        else:
            merged.append(index)
    return merged


def window_bounds(series: list[dict[str, float]], peak: int) -> tuple[int, int]:
    start = peak
    while start > 0 and series[start - 1]["smooth"] > VALLEY_SPEED:
        start -= 1
    end = peak
    while end < len(series) - 1 and series[end + 1]["smooth"] > VALLEY_SPEED:
        end += 1
    return start, end


def hitting_arm(series: list[dict[str, float]], start: int, end: int, peak: int) -> str:
    totals = {"left": 0.0, "right": 0.0}
    for index in range(start, end + 1):
        totals["left"] += series[index]["left"]
        totals["right"] += series[index]["right"]
    if abs(totals["left"] - totals["right"]) < 1e-6:
        return "right" if series[peak]["right"] >= series[peak]["left"] else "left"
    return "right" if totals["right"] > totals["left"] else "left"


def segment_visible(row: dict[str, str], name: str) -> bool:
    raw = row.get(f"{name}_vis", "")
    if raw == "":
        return False
    return float(raw) >= SEGMENT_PRESENT


def arm_segment_visible(row: dict[str, str], arm: str) -> bool:
    """Shoulder–elbow and elbow–wrist are both present for this arm."""
    shoulder, elbow, wrist = ARMS[arm]
    return segment_visible(row, shoulder) and segment_visible(row, elbow) and segment_visible(row, wrist)


def both_arms_visible(rows: list[dict[str, str]], indexes: list[int]) -> bool:
    """True when both arm segments are visible on most frames of the shot."""
    if not indexes:
        return False
    present = sum(
        1
        for index in indexes
        if arm_segment_visible(rows[index], "left") and arm_segment_visible(rows[index], "right")
    )
    return present / len(indexes) >= 0.5


def stroke_type_for(striking_arm: str, dominant_arm: str) -> str:
    """Dominant arm on the ball is a forehand. The other arm is a backhand."""
    if striking_arm == dominant_arm:
        return "forehand"
    return "backhand"


def backhand_style_for(both_arms: bool, preference: str | None) -> str:
    """Two-handed when both arm segments are visible. Otherwise use the profile, or one-handed."""
    if both_arms:
        return "two-handed"
    if preference in ("one-handed", "two-handed"):
        return preference
    return "one-handed"


def format_knee(rows: list[dict[str, str]], indexes: list[int], scale_y: float, side: str) -> str:
    values = [value for index in indexes if (value := knee_angle(rows[index], scale_y, side)) is not None]
    if not values:
        return ""
    values.sort()
    middle = len(values) // 2
    if len(values) % 2:
        chosen = values[middle]
    else:
        chosen = (values[middle - 1] + values[middle]) / 2
    return f"{chosen:.0f}"


def knee_angle(row: dict[str, str], scale_y: float, side: str) -> float | None:
    hip = point(row, f"{side}_hip", scale_y)
    knee = point(row, f"{side}_knee", scale_y)
    ankle = point(row, f"{side}_ankle", scale_y)
    if hip is None or knee is None or ankle is None:
        return None
    if min(hip[2], knee[2], ankle[2]) < VISIBILITY_OK:
        return None
    return angle_between(subtract(hip[:2], knee[:2]), subtract(ankle[:2], knee[:2]))


def arm_geometry(
    row: dict[str, str], scale_y: float, arm: str
) -> dict[str, float] | None:
    body = torso(row, scale_y)
    shoulder_name, elbow_name, wrist_name = ARMS[arm]
    shoulder = point(row, shoulder_name, scale_y)
    elbow = point(row, elbow_name, scale_y)
    wrist = point(row, wrist_name, scale_y)
    if body is None or not all((shoulder, elbow, wrist)):
        return None
    if min(shoulder[2], elbow[2], wrist[2]) < VISIBILITY_OK:
        return None
    forearm = subtract(wrist[:2], elbow[:2])
    forearm_length = length(forearm)
    if forearm_length < 1e-6:
        return None
    tip = (
        wrist[0] + RACKET_EXTENSION * forearm[0],
        wrist[1] + RACKET_EXTENSION * forearm[1],
    )
    torso_vector = subtract(body[1], body[0])
    elbow_angle = angle_between(subtract(shoulder[:2], elbow[:2]), forearm)
    racket_angle = angle_between(forearm, torso_vector)
    if elbow_angle is None or racket_angle is None:
        return None
    scale = body[2]
    return {
        "elbow_angle_deg": elbow_angle,
        "racket_torso_angle_deg": racket_angle,
        "shoulder_elbow_distance": distance(shoulder[:2], elbow[:2]) / scale,
        "elbow_wrist_distance": forearm_length / scale,
        "wrist_racket_distance": RACKET_EXTENSION * forearm_length / scale,
        "shoulder_racket_distance": distance(shoulder[:2], tip) / scale,
        "contact_side": (wrist[0] - body[0][0]) / scale,
        "wrist_visibility": wrist[2],
        "tip_x": tip[0],
        "tip_y": tip[1],
        "wrist_x": wrist[0],
        "wrist_y": wrist[1],
        "elbow_x": elbow[0],
        "elbow_y": elbow[1],
    }


def rotation(rows: list[dict[str, str]], indexes: list[int], scale_y: float, left: str, right: str) -> float | None:
    angles: list[float] = []
    for index in indexes:
        start = point(rows[index], left, scale_y)
        end = point(rows[index], right, scale_y)
        if start is None or end is None:
            continue
        if start[2] < VISIBILITY_OK or end[2] < VISIBILITY_OK:
            continue
        value = line_angle(start[:2], end[:2])
        if value is not None:
            angles.append(value)
    if len(angles) < 2:
        return None
    return wrap_degrees(angles[-1] - angles[0])


def median_geometry(items: list[dict[str, float] | None]) -> dict[str, float] | None:
    valid = [item for item in items if item is not None]
    if not valid:
        return None
    keys = (
        "elbow_angle_deg",
        "racket_torso_angle_deg",
        "shoulder_elbow_distance",
        "elbow_wrist_distance",
        "wrist_racket_distance",
        "shoulder_racket_distance",
        "contact_side",
        "wrist_visibility",
    )
    merged: dict[str, float] = {}
    for key in keys:
        values = sorted(item[key] for item in valid)
        middle = len(values) // 2
        if len(values) % 2:
            merged[key] = values[middle]
        else:
            merged[key] = (values[middle - 1] + values[middle]) / 2
    return merged


def crosses_center(rows: list[dict[str, str]], indexes: list[int], scale_y: float, arm: str) -> str:
    wrist_name = ARMS[arm][2]
    signs: list[float] = []
    for index in indexes:
        body = torso(rows[index], scale_y)
        wrist = point(rows[index], wrist_name, scale_y)
        if body is None or wrist is None or wrist[2] < VISIBILITY_OK:
            continue
        signs.append(wrist[0] - body[0][0])
    if len(signs) < 2:
        return "no"
    if min(signs) < 0 < max(signs):
        return "yes"
    return "no"


def direction_degrees(
    rows: list[dict[str, str]], indexes: list[int], contact: int, scale_y: float, arm: str
) -> float | None:
    wrist_name = ARMS[arm][2]
    start = None
    for index in indexes:
        if index > contact:
            break
        wrist = point(rows[index], wrist_name, scale_y)
        if wrist is not None and wrist[2] >= VISIBILITY_OK:
            start = wrist
            break
    end = point(rows[contact], wrist_name, scale_y)
    if start is None or end is None:
        return None
    return line_angle(start[:2], end[:2])


def build_strokes(
    rows: list[dict[str, str]],
    scale_y: float,
    fps: float,
    dominant_arm: str,
    backhand_preference: str | None,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    times = [float(row["time_s"]) for row in rows]
    series = wrist_speeds(rows, scale_y, fps)
    peaks = find_peaks(series, times)
    strokes: list[dict[str, str]] = []
    frames: list[dict[str, str]] = []
    for stroke_id, peak in enumerate(peaks, start=1):
        start, end = window_bounds(series, peak)
        indexes = list(range(start, end + 1))
        arm = hitting_arm(series, start, end, peak)
        usable = [
            index
            for index in indexes
            if arm_geometry(rows[index], scale_y, arm) is not None
        ]
        if len(usable) < MIN_FRAMES:
            continue
        stable = [
            index
            for index in usable
            if series[index].get(f"{arm}_step", 0.0) <= JUMP_TORSO
        ]
        pool = stable or usable
        contact = max(pool, key=lambda index: (series[index]["smooth"], -abs(index - peak)))
        nearby = [index for index in pool if abs(index - contact) <= 2]
        geometry = median_geometry([arm_geometry(rows[index], scale_y, arm) for index in nearby])
        if geometry is None:
            continue
        swing = direction_degrees(rows, indexes, contact, scale_y, arm)
        shoulders = rotation(rows, indexes, scale_y, "l_shoulder", "r_shoulder")
        hips = rotation(rows, indexes, scale_y, "l_hip", "r_hip")
        jump = any(
            series[index].get(f"{arm}_step", 0.0) > JUMP_TORSO for index in indexes
        ) or not stable
        both_arms = both_arms_visible(rows, indexes)
        kind = stroke_type_for(arm, dominant_arm)
        style = ""
        if kind == "backhand":
            style = backhand_style_for(both_arms, backhand_preference)
        stroke = {
            "shot": str(stroke_id),
            "striking_arm": arm,
            "stroke_type": kind,
            "backhand_style": style,
            "t_start": f"{times[start]:.3f}",
            "t_contact": f"{times[contact]:.3f}",
            "t_end": f"{times[end]:.3f}",
            "elbow_angle_deg": f"{geometry['elbow_angle_deg']:.0f}",
            "racket_torso_angle_deg": f"{geometry['racket_torso_angle_deg']:.0f}",
            "wrist_speed": f"{series[contact]['smooth']:.1f}",
            "shoulder_elbow_distance": f"{geometry['shoulder_elbow_distance']:.2f}",
            "elbow_wrist_distance": f"{geometry['elbow_wrist_distance']:.2f}",
            "wrist_racket_distance": f"{geometry['wrist_racket_distance']:.2f}",
            "shoulder_racket_distance": f"{geometry['shoulder_racket_distance']:.2f}",
            "swing_direction_deg": "" if swing is None else f"{swing:.0f}",
            "shoulder_rotation_deg": "" if shoulders is None else f"{shoulders:.0f}",
            "hip_rotation_deg": "" if hips is None else f"{hips:.0f}",
            "contact_side": f"{geometry['contact_side']:.2f}",
            "crosses_center": crosses_center(rows, indexes, scale_y, arm),
            "left_knee_angle_deg": format_knee(rows, nearby, scale_y, "l"),
            "right_knee_angle_deg": format_knee(rows, nearby, scale_y, "r"),
            "needs_review": "yes" if jump else "no",
        }
        strokes.append(stroke)
        for index in indexes:
            frame_geometry = arm_geometry(rows[index], scale_y, arm)
            frames.append(
                {
                    "shot": str(stroke_id),
                    "frame": rows[index]["frame"],
                    "time_s": f"{times[index]:.3f}",
                    "striking_arm": arm,
                    "elbow_angle_deg": ""
                    if frame_geometry is None
                    else f"{frame_geometry['elbow_angle_deg']:.0f}",
                    "racket_torso_angle_deg": ""
                    if frame_geometry is None
                    else f"{frame_geometry['racket_torso_angle_deg']:.0f}",
                    "wrist_speed": f"{series[index]['smooth']:.1f}",
                    "wrist_visibility": ""
                    if frame_geometry is None
                    else f"{frame_geometry['wrist_visibility']:.2f}",
                    "left_knee_angle_deg": format_knee(rows, [index], scale_y, "l"),
                    "right_knee_angle_deg": format_knee(rows, [index], scale_y, "r"),
                    "jump": "yes" if series[index].get(f"{arm}_step", 0.0) > JUMP_TORSO else "no",
                }
            )
    return strokes, frames


def write_csv(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def direction_phrase(raw: str) -> str:
    if raw == "":
        return "direction not measured"
    degrees = float(raw)
    if abs(degrees) > 135:
        horizontal = "to the left"
    elif abs(degrees) <= 45:
        horizontal = "to the right"
    else:
        horizontal = ""
    if degrees < -25:
        vertical = "and upward"
    elif degrees > 25:
        vertical = "and downward"
    else:
        vertical = ""
    return " ".join(part for part in (horizontal, vertical) if part) or "nearly along the axis"


def print_summary(strokes: list[dict[str, str]]) -> None:
    print()
    if not strokes:
        print("No stroke above the wrist-speed threshold.")
        return
    forehands = sum(1 for stroke in strokes if stroke["stroke_type"] == "forehand")
    backhands = len(strokes) - forehands
    one_handed = sum(1 for stroke in strokes if stroke["backhand_style"] == "one-handed")
    two_handed = sum(1 for stroke in strokes if stroke["backhand_style"] == "two-handed")
    print(f"Strokes found: {len(strokes)}")
    print(f"Forehands: {forehands}. Backhands: {backhands} (one-handed {one_handed}, two-handed {two_handed}).")
    print("Distances are in torso lengths. Speed is in torso lengths per second.")
    print("0° of swing points to the right of the image. Negative is upward.")
    print("The racket tip is the forearm extended past the wrist, not a seen point.")
    print()
    for stroke in strokes:
        check = " Needs review: a wrist frame jumped." if stroke["needs_review"] == "yes" else ""
        style = f", {stroke['backhand_style']} backhand" if stroke["backhand_style"] else ""
        print(
            f"Shot {stroke['shot']}, {stroke['stroke_type']}{style}, "
            f"{stroke['striking_arm']} arm, "
            f"contact at {stroke['t_contact']}s "
            f"({stroke['t_start']}–{stroke['t_end']})."
        )
        print(
            f"  Elbow {stroke['elbow_angle_deg']}°. "
            f"Racket-torso {stroke['racket_torso_angle_deg']}°. "
            f"Speed {stroke['wrist_speed']}."
        )
        print(
            f"  Swing {direction_phrase(stroke['swing_direction_deg'])}. "
            f"Shoulders {stroke['shoulder_rotation_deg'] or 'n/a'}°, "
            f"hips {stroke['hip_rotation_deg'] or 'n/a'}°.{check}"
        )
        print(
            "  Knee, 180° is a straight leg: "
            f"left {stroke['left_knee_angle_deg'] or 'n/a'} degrees, "
            f"right {stroke['right_knee_angle_deg'] or 'n/a'} degrees."
        )


def draw_overlay(
    video_path: Path,
    rows: list[dict[str, str]],
    strokes: list[dict[str, str]],
    frame_rows: list[dict[str, str]],
    destination: Path,
    scale_y: float,
) -> None:
    import cv2

    windows: list[tuple[float, float, dict[str, str]]] = [
        (float(stroke["t_start"]), float(stroke["t_end"]), stroke) for stroke in strokes
    ]
    capture = cv2.VideoCapture(str(video_path))
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    destination.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    for codec in ("avc1", "mp4v"):
        candidate = cv2.VideoWriter(
            str(destination), cv2.VideoWriter_fourcc(*codec), fps, (width, height)
        )
        if candidate.isOpened():
            writer = candidate
            break
        candidate.release()
    if writer is None:
        capture.release()
        print("Could not write the features video.", file=sys.stderr)
        return

    jumped = {
        int(row["frame"]): row["jump"] == "yes" for row in frame_rows if row["frame"] != ""
    }

    def to_pixel(x: float, y_scaled: float) -> tuple[int, int]:
        return int(x * width), int(y_scaled / scale_y * height)

    frame_index = 0
    while True:
        ok, frame = capture.read()
        if not ok or frame_index >= len(rows):
            break
        row = rows[frame_index]
        time_s = float(row["time_s"])
        active = next((stroke for start, end, stroke in windows if start <= time_s <= end), None)
        if active is not None and jumped.get(frame_index, False):
            cv2.putText(
                frame,
                f"shot {active['shot']}  uncertain wrist",
                (16, 36),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (80, 80, 255),
                2,
                cv2.LINE_AA,
            )
        elif active is not None:
            geometry = arm_geometry(row, scale_y, active["striking_arm"])
            if geometry is not None:
                elbow = to_pixel(geometry["elbow_x"], geometry["elbow_y"])
                wrist = to_pixel(geometry["wrist_x"], geometry["wrist_y"])
                tip = to_pixel(geometry["tip_x"], geometry["tip_y"])
                cv2.line(frame, elbow, wrist, (0, 220, 255), 3)
                cv2.line(frame, wrist, tip, (0, 220, 255), 2)
                cv2.circle(frame, tip, 6, (0, 220, 255), -1)
                label = active["stroke_type"]
                if active["backhand_style"]:
                    label = f"{active['backhand_style']} {label}"
                cv2.putText(
                    frame,
                    (
                        f"shot {active['shot']}  {label}"
                        f"  elbow {geometry['elbow_angle_deg']:.0f}"
                        f"  racket {geometry['racket_torso_angle_deg']:.0f}"
                    ),
                    (16, 36),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
        writer.write(frame)
        frame_index += 1
    capture.release()
    writer.release()
    print(f"Estimated racket video: {destination}")


def default_video(csv_path: Path) -> Path | None:
    stem = csv_path.name.replace("_keypoints.csv", "")
    for folder in (ROOT / "videos", ROOT / "output"):
        for path in sorted(folder.glob(f"{stem}*")):
            if path.suffix.lower() in {".mov", ".mp4", ".m4v"} and "overlay" not in path.name and "features" not in path.name:
                return path
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract stroke features from a pose CSV.")
    parser.add_argument("csv", type=Path, help="CSV written by try_mediapipe.py")
    parser.add_argument("--video", type=Path, default=None, help="Original video, for scale and overlay")
    parser.add_argument("--output", type=Path, default=ROOT / "output", help="Folder for the results")
    parser.add_argument(
        "--dominant-arm",
        choices=("left", "right"),
        default="right",
        help="Registered dominant arm. A stroke with that arm is a forehand.",
    )
    parser.add_argument(
        "--backhand",
        choices=("one-handed", "two-handed"),
        default=None,
        help="Profile backhand. Used when both arms do not swing together.",
    )
    args = parser.parse_args()
    csv_path = args.csv.expanduser().resolve()
    if not csv_path.is_file():
        print(f"CSV not found: {csv_path}", file=sys.stderr)
        return 1
    video_path = args.video.expanduser().resolve() if args.video else default_video(csv_path)
    if video_path is None or not video_path.is_file():
        print("The original video is required for width and height.", file=sys.stderr)
        print(
            "Example: python extract_features.py output/sample_keypoints.csv --video videos/sample.mov",
            file=sys.stderr,
        )
        return 1

    width, height, fps = video_geometry(video_path)
    scale_y = height / width
    rows = load_rows(csv_path)
    strokes, frames = build_strokes(rows, scale_y, fps, args.dominant_arm, args.backhand)
    stem = csv_path.name.replace("_keypoints.csv", "")
    output_dir = args.output.resolve()
    strokes_path = output_dir / f"{stem}_strokes.csv"
    frames_path = output_dir / f"{stem}_stroke_frames.csv"
    write_csv(strokes_path, STROKE_FIELDS, strokes)
    write_csv(frames_path, FRAME_FIELDS, frames)
    print(f"Stroke table: {strokes_path}")
    print(f"Stroke frames: {frames_path}")
    draw_overlay(
        video_path,
        rows,
        strokes,
        frames,
        output_dir / f"{stem}_features.mp4",
        scale_y,
    )
    print_summary(strokes)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
