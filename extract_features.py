#!/usr/bin/env python3
"""Estrae le feature di ogni colpo dal CSV di MediaPipe.

La racchetta non è un keypoint: la punta stimata prolunga l'avambraccio
di una lunghezza oltre il polso. Non giudica se il colpo è corretto.
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
    "sinistro": ("l_shoulder", "l_elbow", "l_wrist"),
    "destro": ("r_shoulder", "r_elbow", "r_wrist"),
}

STROKE_FIELDS = [
    "colpo",
    "braccio",
    "t_inizio",
    "t_contatto",
    "t_fine",
    "angolo_gomito_deg",
    "angolo_racchetta_busto_deg",
    "velocita_polso",
    "distanza_spalla_gomito",
    "distanza_gomito_polso",
    "distanza_polso_racchetta",
    "distanza_spalla_racchetta",
    "direzione_oscillazione_deg",
    "rotazione_spalle_deg",
    "rotazione_anche_deg",
    "lato_contatto",
    "attraversa_centro",
    "angolo_ginocchio_sx_deg",
    "angolo_ginocchio_dx_deg",
    "da_controllare",
]
FRAME_FIELDS = [
    "colpo",
    "frame",
    "time_s",
    "braccio",
    "angolo_gomito_deg",
    "angolo_racchetta_busto_deg",
    "velocita_polso",
    "visibilita_polso",
    "angolo_ginocchio_sx_deg",
    "angolo_ginocchio_dx_deg",
    "salto",
]


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def video_geometry(path: Path) -> tuple[float, float, float]:
    import cv2

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise SystemExit(f"Non riesco ad aprire il video: {path}")
    width = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    capture.release()
    if width <= 0 or height <= 0:
        raise SystemExit(f"Dimensioni video non valide: {path}")
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
    previous: dict[str, tuple[float, float] | None] = {"sinistro": None, "destro": None}
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
        current["massima"] = max(current["sinistro"], current["destro"])
        series.append(current)
    maxima = smooth([item["massima"] for item in series])
    for item, value in zip(series, maxima):
        item["liscia"] = value
    return series


def find_peaks(series: list[dict[str, float]], times: list[float]) -> list[int]:
    candidates: list[int] = []
    for index in range(1, len(series) - 1):
        speed = series[index]["liscia"]
        if speed < PEAK_SPEED:
            continue
        if speed >= series[index - 1]["liscia"] and speed > series[index + 1]["liscia"]:
            candidates.append(index)
    if not candidates:
        return []
    merged = [candidates[0]]
    for index in candidates[1:]:
        if times[index] - times[merged[-1]] <= MERGE_SECONDS:
            if series[index]["liscia"] > series[merged[-1]]["liscia"]:
                merged[-1] = index
        else:
            merged.append(index)
    return merged


def window_bounds(series: list[dict[str, float]], peak: int) -> tuple[int, int]:
    start = peak
    while start > 0 and series[start - 1]["liscia"] > VALLEY_SPEED:
        start -= 1
    end = peak
    while end < len(series) - 1 and series[end + 1]["liscia"] > VALLEY_SPEED:
        end += 1
    return start, end


def hitting_arm(series: list[dict[str, float]], start: int, end: int, peak: int) -> str:
    totals = {"sinistro": 0.0, "destro": 0.0}
    for index in range(start, end + 1):
        totals["sinistro"] += series[index]["sinistro"]
        totals["destro"] += series[index]["destro"]
    if abs(totals["sinistro"] - totals["destro"]) < 1e-6:
        return "destro" if series[peak]["destro"] >= series[peak]["sinistro"] else "sinistro"
    return "destro" if totals["destro"] > totals["sinistro"] else "sinistro"


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
        "angolo_gomito_deg": elbow_angle,
        "angolo_racchetta_busto_deg": racket_angle,
        "distanza_spalla_gomito": distance(shoulder[:2], elbow[:2]) / scale,
        "distanza_gomito_polso": forearm_length / scale,
        "distanza_polso_racchetta": RACKET_EXTENSION * forearm_length / scale,
        "distanza_spalla_racchetta": distance(shoulder[:2], tip) / scale,
        "lato_contatto": (wrist[0] - body[0][0]) / scale,
        "visibilita_polso": wrist[2],
        "punta_x": tip[0],
        "punta_y": tip[1],
        "polso_x": wrist[0],
        "polso_y": wrist[1],
        "gomito_x": elbow[0],
        "gomito_y": elbow[1],
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
        "angolo_gomito_deg",
        "angolo_racchetta_busto_deg",
        "distanza_spalla_gomito",
        "distanza_gomito_polso",
        "distanza_polso_racchetta",
        "distanza_spalla_racchetta",
        "lato_contatto",
        "visibilita_polso",
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
        return "sì"
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
    rows: list[dict[str, str]], scale_y: float, fps: float
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
        contact = max(pool, key=lambda index: (series[index]["liscia"], -abs(index - peak)))
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
        stroke = {
            "colpo": str(stroke_id),
            "braccio": arm,
            "t_inizio": f"{times[start]:.3f}",
            "t_contatto": f"{times[contact]:.3f}",
            "t_fine": f"{times[end]:.3f}",
            "angolo_gomito_deg": f"{geometry['angolo_gomito_deg']:.0f}",
            "angolo_racchetta_busto_deg": f"{geometry['angolo_racchetta_busto_deg']:.0f}",
            "velocita_polso": f"{series[contact]['liscia']:.1f}",
            "distanza_spalla_gomito": f"{geometry['distanza_spalla_gomito']:.2f}",
            "distanza_gomito_polso": f"{geometry['distanza_gomito_polso']:.2f}",
            "distanza_polso_racchetta": f"{geometry['distanza_polso_racchetta']:.2f}",
            "distanza_spalla_racchetta": f"{geometry['distanza_spalla_racchetta']:.2f}",
            "direzione_oscillazione_deg": "" if swing is None else f"{swing:.0f}",
            "rotazione_spalle_deg": "" if shoulders is None else f"{shoulders:.0f}",
            "rotazione_anche_deg": "" if hips is None else f"{hips:.0f}",
            "lato_contatto": f"{geometry['lato_contatto']:.2f}",
            "attraversa_centro": crosses_center(rows, indexes, scale_y, arm),
            "angolo_ginocchio_sx_deg": format_knee(rows, nearby, scale_y, "l"),
            "angolo_ginocchio_dx_deg": format_knee(rows, nearby, scale_y, "r"),
            "da_controllare": "sì" if jump else "no",
        }
        strokes.append(stroke)
        for index in indexes:
            frame_geometry = arm_geometry(rows[index], scale_y, arm)
            frames.append(
                {
                    "colpo": str(stroke_id),
                    "frame": rows[index]["frame"],
                    "time_s": f"{times[index]:.3f}",
                    "braccio": arm,
                    "angolo_gomito_deg": ""
                    if frame_geometry is None
                    else f"{frame_geometry['angolo_gomito_deg']:.0f}",
                    "angolo_racchetta_busto_deg": ""
                    if frame_geometry is None
                    else f"{frame_geometry['angolo_racchetta_busto_deg']:.0f}",
                    "velocita_polso": f"{series[index]['liscia']:.1f}",
                    "visibilita_polso": ""
                    if frame_geometry is None
                    else f"{frame_geometry['visibilita_polso']:.2f}",
                    "angolo_ginocchio_sx_deg": format_knee(rows, [index], scale_y, "l"),
                    "angolo_ginocchio_dx_deg": format_knee(rows, [index], scale_y, "r"),
                    "salto": "sì" if series[index].get(f"{arm}_step", 0.0) > JUMP_TORSO else "no",
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
        return "direzione non misurata"
    degrees = float(raw)
    if abs(degrees) > 135:
        horizontal = "verso sinistra"
    elif abs(degrees) <= 45:
        horizontal = "verso destra"
    else:
        horizontal = ""
    if degrees < -25:
        vertical = "e verso l'alto"
    elif degrees > 25:
        vertical = "e verso il basso"
    else:
        vertical = ""
    return " ".join(part for part in (horizontal, vertical) if part) or "quasi in asse"


def print_summary(strokes: list[dict[str, str]]) -> None:
    print()
    if not strokes:
        print("Nessun colpo sopra la soglia di velocità del polso.")
        return
    print(f"Colpi trovati: {len(strokes)}")
    print("Distanze in lunghezze del busto. Velocità in busti al secondo.")
    print("0° di oscillazione è verso destra dell'immagine, negativo è verso l'alto.")
    print("La punta della racchetta è il prolungamento dell'avambraccio, non un punto visto.")
    print()
    for stroke in strokes:
        check = " Da controllare: un frame del polso salta." if stroke["da_controllare"] == "sì" else ""
        print(
            f"Colpo {stroke['colpo']}, braccio {stroke['braccio']}, "
            f"contatto a {stroke['t_contatto']}s "
            f"({stroke['t_inizio']}–{stroke['t_fine']})."
        )
        print(
            f"  Gomito {stroke['angolo_gomito_deg']}°. "
            f"Racchetta-busto {stroke['angolo_racchetta_busto_deg']}°. "
            f"Velocità {stroke['velocita_polso']}."
        )
        print(
            f"  Oscillazione {direction_phrase(stroke['direzione_oscillazione_deg'])}. "
            f"Spalle {stroke['rotazione_spalle_deg'] or 'n/d'}°, "
            f"anche {stroke['rotazione_anche_deg'] or 'n/d'}°.{check}"
        )
        print(
            "  Ginocchio, 180° è la gamba distesa: "
            f"sinistro {stroke['angolo_ginocchio_sx_deg'] or 'n/d'} gradi, "
            f"destro {stroke['angolo_ginocchio_dx_deg'] or 'n/d'} gradi."
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
        (float(stroke["t_inizio"]), float(stroke["t_fine"]), stroke) for stroke in strokes
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
        print("Non riesco a scrivere il video delle feature.", file=sys.stderr)
        return

    jumped = {
        int(row["frame"]): row["salto"] == "sì" for row in frame_rows if row["frame"] != ""
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
                f"colpo {active['colpo']}  polso incerto",
                (16, 36),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (80, 80, 255),
                2,
                cv2.LINE_AA,
            )
        elif active is not None:
            geometry = arm_geometry(row, scale_y, active["braccio"])
            if geometry is not None:
                elbow = to_pixel(geometry["gomito_x"], geometry["gomito_y"])
                wrist = to_pixel(geometry["polso_x"], geometry["polso_y"])
                tip = to_pixel(geometry["punta_x"], geometry["punta_y"])
                cv2.line(frame, elbow, wrist, (0, 220, 255), 3)
                cv2.line(frame, wrist, tip, (0, 220, 255), 2)
                cv2.circle(frame, tip, 6, (0, 220, 255), -1)
                cv2.putText(
                    frame,
                    (
                        f"colpo {active['colpo']}  gomito {geometry['angolo_gomito_deg']:.0f}"
                        f"  racchetta {geometry['angolo_racchetta_busto_deg']:.0f}"
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
    print(f"Video con racchetta stimata: {destination}")


def default_video(csv_path: Path) -> Path | None:
    stem = csv_path.name.replace("_keypoints.csv", "")
    for folder in (ROOT / "videos", ROOT / "output"):
        for path in sorted(folder.glob(f"{stem}*")):
            if path.suffix.lower() in {".mov", ".mp4", ".m4v"} and "overlay" not in path.name and "features" not in path.name:
                return path
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Estrae le feature dei colpi da un CSV di pose.")
    parser.add_argument("csv", type=Path, help="CSV scritto da try_mediapipe.py")
    parser.add_argument("--video", type=Path, default=None, help="Video originale, per scala e overlay")
    parser.add_argument("--output", type=Path, default=ROOT / "output", help="Cartella dei risultati")
    args = parser.parse_args()
    csv_path = args.csv.expanduser().resolve()
    if not csv_path.is_file():
        print(f"CSV non trovato: {csv_path}", file=sys.stderr)
        return 1
    video_path = args.video.expanduser().resolve() if args.video else default_video(csv_path)
    if video_path is None or not video_path.is_file():
        print("Serve il video originale per conoscere larghezza e altezza.", file=sys.stderr)
        print("Esempio: python extract_features.py output/sample_keypoints.csv --video videos/sample.mov", file=sys.stderr)
        return 1

    width, height, fps = video_geometry(video_path)
    scale_y = height / width
    rows = load_rows(csv_path)
    strokes, frames = build_strokes(rows, scale_y, fps)
    stem = csv_path.name.replace("_keypoints.csv", "")
    output_dir = args.output.resolve()
    strokes_path = output_dir / f"{stem}_strokes.csv"
    frames_path = output_dir / f"{stem}_stroke_frames.csv"
    write_csv(strokes_path, STROKE_FIELDS, strokes)
    write_csv(frames_path, FRAME_FIELDS, frames)
    print(f"Tabella colpi: {strokes_path}")
    print(f"Frame dei colpi: {frames_path}")
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
