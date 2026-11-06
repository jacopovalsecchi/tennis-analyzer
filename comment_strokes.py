#!/usr/bin/env python3
"""Comment each detected stroke against the right-handed forehand checklist.

The contact time comes from the wrist-speed peak in the strokes CSV.
Body position is read on that frame. Whether the right arm was already
open is read on the frames just before it, where the ball bounce sits.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import extract_features as features

# Lengths are fractions of torso length. The camera is behind the player:
# smaller y is toward the net, smaller x is the player's right.
LEFT_LEG_FORWARD = 0.10
ARM_BEHIND_X = 0.08
ARM_BEHIND_Y = 0.20
ARM_EXTENDED = 150.0
ARM_OPEN = 110.0
LEFT_ARM_REACH = 0.35
LEFT_ARM_LEVEL = 0.40
CHEST_WIDTH = 0.25
CHEST_FLAT_DEG = 30.0
CHEST_TURN = 0.08
PREP_LEAD = 0.40
PREP_LAG = 0.08


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def nearest(rows: list[dict[str, str]], time_s: float) -> dict[str, str] | None:
    if not rows:
        return None
    return min(rows, key=lambda row: abs(float(row["time_s"]) - time_s))


def torso_length(row: dict[str, str], scale_y: float) -> float | None:
    body = features.torso(row, scale_y)
    if body is None:
        return None
    return body[2]


def shoulder_line(row: dict[str, str], scale_y: float) -> tuple[float, float, float] | None:
    left = features.point(row, "l_shoulder", scale_y)
    right = features.point(row, "r_shoulder", scale_y)
    length = torso_length(row, scale_y)
    if left is None or right is None or length is None:
        return None
    if left[2] < features.VISIBILITY_OK or right[2] < features.VISIBILITY_OK:
        return None
    angle = features.line_angle(left[:2], right[:2])
    if angle is None:
        return None
    flat = min(abs(angle), 180 - abs(angle))
    width = abs(left[0] - right[0]) / length
    turn = (right[1] - left[1]) / length
    return width, flat, turn


def right_arm(row: dict[str, str], scale_y: float) -> tuple[str, str]:
    """Extended, and open behind the body. Each is yes, no, or unknown."""
    geometry = features.arm_geometry(row, scale_y, "right")
    body = features.torso(row, scale_y)
    wrist = features.point(row, "r_wrist", scale_y)
    shoulder = features.point(row, "r_shoulder", scale_y)
    if geometry is None or body is None or wrist is None or shoulder is None:
        return "unknown", "unknown"
    extended = "yes" if geometry["elbow_angle_deg"] >= ARM_EXTENDED else "no"
    length = body[2]
    behind_x = (body[0][0] - wrist[0]) / length
    behind_y = (wrist[1] - shoulder[1]) / length
    behind = "yes" if behind_x >= ARM_BEHIND_X and behind_y >= ARM_BEHIND_Y else "no"
    return extended, behind


def left_leg_forward(row: dict[str, str], scale_y: float) -> str:
    left = features.point(row, "l_ankle", scale_y)
    right = features.point(row, "r_ankle", scale_y)
    length = torso_length(row, scale_y)
    if left is None or right is None or length is None:
        return "unknown"
    if left[2] < features.VISIBILITY_OK or right[2] < features.VISIBILITY_OK:
        return "unknown"
    gap = (right[1] - left[1]) / length
    return "yes" if gap >= LEFT_LEG_FORWARD else "no"


def left_arm_horizontal(row: dict[str, str], scale_y: float) -> str:
    geometry = features.arm_geometry(row, scale_y, "left")
    wrist = features.point(row, "l_wrist", scale_y)
    shoulder = features.point(row, "l_shoulder", scale_y)
    length = torso_length(row, scale_y)
    if geometry is None or wrist is None or shoulder is None or length is None:
        return "unknown"
    reach = (wrist[0] - shoulder[0]) / length
    level = abs(wrist[1] - shoulder[1]) / length
    open_enough = geometry["elbow_angle_deg"] >= ARM_OPEN
    if reach >= LEFT_ARM_REACH and level <= LEFT_ARM_LEVEL and open_enough:
        return "yes"
    return "no"


def chest(row: dict[str, str], scale_y: float) -> tuple[str, str]:
    """Parallel to the baseline, and turned toward the player's right."""
    line = shoulder_line(row, scale_y)
    if line is None:
        return "unknown", "unknown"
    width, flat, turn = line
    parallel = "yes" if width >= CHEST_WIDTH and flat <= CHEST_FLAT_DEG else "no"
    turned = "yes" if parallel == "yes" and turn >= CHEST_TURN else "no"
    if parallel == "no":
        turned = "no"
    return parallel, turned


def arm_open_before(rows: list[dict[str, str]], scale_y: float, contact: float) -> str:
    window = [
        row
        for row in rows
        if contact - PREP_LEAD <= float(row["time_s"]) <= contact - PREP_LAG
    ]
    if not window:
        return "unknown"
    seen = [right_arm(row, scale_y)[1] for row in window]
    if "yes" in seen:
        return "yes"
    if all(value == "unknown" for value in seen):
        return "unknown"
    return "no"


GOOD = {
    "left_leg": "gamba sinistra avanti",
    "extended": "braccio destro disteso",
    "behind": "braccio destro aperto dietro il corpo",
    "left_arm": "braccio sinistro aperto in orizzontale",
    "parallel": "petto quasi parallelo alla linea di fondo",
    "turned": "corpo girato verso destra",
    "early": "braccio destro già aperto prima dell'impatto",
}
BAD = {
    "left_leg": "la gamba sinistra non è avanti",
    "extended": "il braccio destro non è disteso",
    "behind": "il braccio destro non è aperto dietro il corpo",
    "left_arm": "il braccio sinistro non è aperto in orizzontale",
    "parallel": "il petto non è quasi parallelo alla linea di fondo",
    "turned": "il corpo non è girato verso destra",
    "early": "il braccio destro non era già aperto prima dell'impatto",
}
MISSING = {
    "left_leg": "la gamba sinistra",
    "extended": "se il braccio destro è disteso",
    "behind": "se il braccio destro è dietro il corpo",
    "left_arm": "il braccio sinistro",
    "parallel": "il petto",
    "turned": "la rotazione del corpo",
    "early": "se il braccio era aperto prima dell'impatto",
}


def join_phrases(parts: list[str]) -> str:
    if not parts:
        return ""
    if len(parts) == 1:
        return parts[0]
    return ", ".join(parts[:-1]) + " e " + parts[-1]


def forehand_comment(row: dict[str, str], scale_y: float, rows: list[dict[str, str]], contact: float) -> str:
    extended, behind = right_arm(row, scale_y)
    parallel, turned = chest(row, scale_y)
    checks = {
        "left_leg": left_leg_forward(row, scale_y),
        "extended": extended,
        "behind": behind,
        "left_arm": left_arm_horizontal(row, scale_y),
        "parallel": parallel,
        "turned": turned,
        "early": arm_open_before(rows, scale_y, contact),
    }
    good = [GOOD[name] for name, value in checks.items() if value == "yes"]
    bad = [BAD[name] for name, value in checks.items() if value == "no"]
    missing = [MISSING[name] for name, value in checks.items() if value == "unknown"]
    sentences = [f"Impatto a {contact:.2f} s."]
    if good:
        sentences.append("Bene: " + join_phrases(good) + ".")
    if bad:
        sentences.append("No: " + join_phrases(bad) + ".")
    if missing:
        sentences.append("Non si vede: " + join_phrases(missing) + ".")
    if not good and not bad and not missing:
        sentences.append("La pose all'impatto non basta per giudicare il dritto.")
    return " ".join(sentences)


def other_comment(stroke: dict[str, str], contact: float) -> str:
    arm = "destro" if stroke.get("striking_arm") == "right" else "sinistro"
    kind = "rovescio" if stroke.get("stroke_type") == "backhand" else "colpo"
    return (
        f"Impatto a {contact:.2f} s, {kind} di braccio {arm}. "
        "I criteri valgono per il dritto di un giocatore destro."
    )


def unique_strokes(strokes: list[dict[str, str]]) -> list[dict[str, str]]:
    seen: set[str] = set()
    kept = []
    for stroke in strokes:
        contact = stroke.get("t_contact", "")
        if contact in seen:
            continue
        seen.add(contact)
        kept.append(stroke)
    return kept


def build_comments(
    rows: list[dict[str, str]],
    strokes: list[dict[str, str]],
    scale_y: float,
) -> list[dict[str, str | float]]:
    comments = []
    for stroke in unique_strokes(strokes):
        contact = float(stroke["t_contact"])
        row = nearest(rows, contact)
        forehand = stroke.get("stroke_type") == "forehand" and stroke.get("striking_arm") == "right"
        if row is None:
            text = f"Impatto a {contact:.2f} s. Frame di contatto non trovato."
            label = "Colpo"
        elif forehand:
            text = forehand_comment(row, scale_y, rows, contact)
            label = "Dritto di destro"
        else:
            text = other_comment(stroke, contact)
            label = "Rovescio" if stroke.get("stroke_type") == "backhand" else "Colpo"
        comments.append(
            {
                "time": round(contact, 3),
                "frame": None if row is None else int(row["frame"]),
                "label": label,
                "arm": stroke.get("striking_arm", ""),
                "comment": text,
                "image": None,
            }
        )
    return comments


def save_contact_frames(video: Path, frames_dir: Path, comments: list[dict[str, str | float | None]]) -> None:
    import cv2

    names: dict[int, list[str]] = {}
    for shot_index, comment in enumerate(comments, start=1):
        frame = comment.get("frame")
        if frame is None:
            continue
        name = f"shot_{shot_index:02d}_t{float(comment['time']):07.3f}s.jpg"
        names.setdefault(int(frame), []).append(name)
        comment["image"] = name
    if not names:
        return
    frames_dir.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(video))
    index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        for name in names.get(index, []):
            cv2.imwrite(str(frames_dir / name), frame)
        index += 1
    capture.release()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("keypoints", type=Path, help="Keypoints CSV from try_mediapipe.py")
    parser.add_argument("strokes", type=Path, help="Strokes CSV from extract_features.py")
    parser.add_argument("--video", type=Path, required=True, help="Original video, for scale")
    parser.add_argument("--output", type=Path, required=True, help="JSON file to write")
    args = parser.parse_args()
    keypoints = args.keypoints.expanduser().resolve()
    strokes_path = args.strokes.expanduser().resolve()
    video = args.video.expanduser().resolve()
    if not keypoints.is_file() or not strokes_path.is_file() or not video.is_file():
        print("Keypoints, strokes, and video are required.", flush=True)
        return 1
    width, height, _fps = features.video_geometry(video)
    comments = build_comments(load_rows(keypoints), load_rows(strokes_path), height / width)
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    save_contact_frames(video, output.parent / "comment_frames", comments)
    output.write_text(json.dumps({"comments": comments}, ensure_ascii=False, indent=2))
    print(f"Comments: {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
