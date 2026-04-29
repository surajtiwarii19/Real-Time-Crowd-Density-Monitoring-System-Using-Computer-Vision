from pathlib import Path
import time
from collections import deque

try:
    import cv2
except ImportError as exc:
    raise SystemExit(
        "OpenCV is not installed for this Python. Run with: ./venv/bin/python new_crowd.py"
    ) from exc

import numpy as np
import yaml
from ultralytics import YOLO


CONFIG_PATH = Path(__file__).resolve().with_name("config.yaml")


def load_config():
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def parse_camera_source(source):
    if isinstance(source, str):
        s = source.strip()
        if s.isdigit():
            return int(s)
    return source


def open_video_capture(source):
    cap = cv2.VideoCapture(source, cv2.CAP_V4L2)
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(source)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    return cap


def get_status(count, green_max, yellow_max):
    if count <= green_max:
        return "GREEN", (0, 255, 0)
    if count <= yellow_max:
        return "YELLOW", (0, 255, 255)
    return "RED", (0, 0, 255)


def detect_people(model, frame, confidence, person_class=0):
    results = model(frame, conf=confidence, verbose=False)
    people = []
    for result in results:
        if result.boxes is None:
            continue
        for box in result.boxes:
            if int(box.cls[0]) == person_class:
                people.append(box)
    return people


def clip_bbox(bbox, shape):
    h, w = shape[:2]
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(0, min(x2, w))
    y2 = max(0, min(y2, h))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def bbox_iou(bbox_a, bbox_b):
    ax1, ay1, ax2, ay2 = bbox_a
    bx1, by1, bx2, by2 = bbox_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area == 0:
        return 0.0

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union_area = area_a + area_b - inter_area
    if union_area <= 0:
        return 0.0

    return float(inter_area) / float(union_area)


def bbox_center(bbox):
    x1, y1, x2, y2 = bbox
    return (x1 + x2) * 0.5, (y1 + y2) * 0.5


def bbox_center_distance(bbox_a, bbox_b):
    ax, ay = bbox_center(bbox_a)
    bx, by = bbox_center(bbox_b)
    return float(np.hypot(ax - bx, ay - by))


def shift_bbox(bbox, dx, dy):
    x1, y1, x2, y2 = bbox
    return (
        int(round(x1 + dx)),
        int(round(y1 + dy)),
        int(round(x2 + dx)),
        int(round(y2 + dy)),
    )


def predict_track_bbox(track):
    bbox = track["bbox"]
    vx, vy = track.get("vel", (0.0, 0.0))
    missed = int(track.get("missed", 0))
    lead = min(4, missed + 1)
    return shift_bbox(bbox, vx * lead, vy * lead)


def update_track_velocity(track, new_bbox, momentum=0.7):
    old_bbox = track.get("bbox", new_bbox)
    old_cx, old_cy = bbox_center(old_bbox)
    new_cx, new_cy = bbox_center(new_bbox)
    inst_vx = new_cx - old_cx
    inst_vy = new_cy - old_cy

    prev_vx, prev_vy = track.get("vel", (0.0, 0.0))
    smoothed_vx = momentum * prev_vx + (1.0 - momentum) * inst_vx
    smoothed_vy = momentum * prev_vy + (1.0 - momentum) * inst_vy
    track["vel"] = (smoothed_vx, smoothed_vy)


def match_detections_to_tracks(
    detections,
    tracks,
    iou_threshold=0.15,
    max_center_distance=80.0,
    max_missed_for_match=3,
):
    if not detections:
        return {}, set(), set(tracks.keys())

    if not tracks:
        return {}, set(range(len(detections))), set()

    active_tracks = {
        track_id: track
        for track_id, track in tracks.items()
        if int(track.get("missed", 0)) <= max_missed_for_match
    }
    if not active_tracks:
        return {}, set(range(len(detections))), set(tracks.keys())

    unmatched_detections = set(range(len(detections)))
    unmatched_tracks = set(tracks.keys())
    matches = {}

    candidates = []
    for det_idx, det_bbox in enumerate(detections):
        w = max(1, det_bbox[2] - det_bbox[0])
        h = max(1, det_bbox[3] - det_bbox[1])
        det_diag = float(np.hypot(w, h))
        adaptive_dist = max(22.0, min(max_center_distance, 0.8 * det_diag))

        for track_id, track in active_tracks.items():
            predicted_bbox = predict_track_bbox(track)
            iou = bbox_iou(det_bbox, predicted_bbox)
            center_dist = bbox_center_distance(det_bbox, predicted_bbox)

            if iou < iou_threshold and center_dist > adaptive_dist:
                continue

            missed_penalty = 0.04 * float(track.get("missed", 0))
            score = 1.4 * iou - 0.32 * (center_dist / max(1.0, adaptive_dist)) - missed_penalty
            candidates.append((score, det_idx, track_id))

    for score, det_idx, track_id in sorted(candidates, reverse=True):
        if det_idx in unmatched_detections and track_id in unmatched_tracks:
            matches[det_idx] = track_id
            unmatched_detections.remove(det_idx)
            unmatched_tracks.remove(track_id)

    return matches, unmatched_detections, unmatched_tracks


def motion_ratio(curr_frame, prev_frame, bbox):
    if prev_frame is None:
        return 0.0

    x1, y1, x2, y2 = bbox
    curr_roi = curr_frame[y1:y2, x1:x2]
    prev_roi = prev_frame[y1:y2, x1:x2]
    if curr_roi.size == 0 or prev_roi.size == 0:
        return 0.0

    curr_gray = cv2.cvtColor(curr_roi, cv2.COLOR_BGR2GRAY)
    prev_gray = cv2.cvtColor(prev_roi, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.GaussianBlur(curr_gray, (5, 5), 0)
    prev_gray = cv2.GaussianBlur(prev_gray, (5, 5), 0)

    diff = cv2.absdiff(curr_gray, prev_gray)
    _, diff = cv2.threshold(diff, 22, 255, cv2.THRESH_BINARY)
    return float(np.count_nonzero(diff)) / float(diff.size)


def frame_motion_ratio(curr_frame, prev_frame):
    if prev_frame is None:
        return 0.0

    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    curr_small = cv2.resize(curr_gray, (160, 120), interpolation=cv2.INTER_AREA)
    prev_small = cv2.resize(prev_gray, (160, 120), interpolation=cv2.INTER_AREA)
    curr_small = cv2.GaussianBlur(curr_small, (5, 5), 0)
    prev_small = cv2.GaussianBlur(prev_small, (5, 5), 0)

    diff = cv2.absdiff(curr_small, prev_small)
    _, diff = cv2.threshold(diff, 20, 255, cv2.THRESH_BINARY)
    return float(np.count_nonzero(diff)) / float(diff.size)


def laplacian_variance(frame, bbox):
    x1, y1, x2, y2 = bbox
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return 0.0
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def get_anti_spoof_config(config):
    anti = config.get("anti_spoof", {})
    return {
        "history_size": int(anti.get("history_size", 20)),
        "warmup_frames": int(anti.get("warmup_frames", 18)),
        "motion_threshold": float(anti.get("motion_threshold", 0.014)),
        "low_motion_threshold": float(anti.get("low_motion_threshold", 0.006)),
        "sharp_blur_threshold": float(anti.get("sharp_blur_threshold", 230.0)),
        "static_center_jitter": float(anti.get("static_center_jitter", 3.5)),
        "static_area_jitter": float(anti.get("static_area_jitter", 0.03)),
        "global_motion_pause": float(anti.get("global_motion_pause", 0.05)),
        "fake_min_streak": int(anti.get("fake_min_streak", 12)),
        "real_recover_streak": int(anti.get("real_recover_streak", 3)),
        "match_iou_threshold": float(anti.get("match_iou_threshold", 0.15)),
        "match_max_center_distance": float(anti.get("match_max_center_distance", 80.0)),
        "max_missed_for_match": int(anti.get("max_missed_for_match", 3)),
        "max_missed_frames": int(anti.get("max_missed_frames", 8)),
    }


def build_track_state(history_size, bbox):
    return {
        "motion": deque(maxlen=history_size),
        "blur": deque(maxlen=history_size),
        "center": deque(maxlen=history_size),
        "area": deque(maxlen=history_size),
        "bbox": bbox,
        "vel": (0.0, 0.0),
        "seen_frames": 0,
        "missed": 0,
        "is_real": True,
        "fake_streak": 0,
        "real_streak": 0,
    }


def classify_real_or_fake(frame, prev_frame, bbox, track, global_motion, anti_cfg):
    motion_hist = track["motion"]
    blur_hist = track["blur"]
    center_hist = track["center"]
    area_hist = track["area"]

    m = motion_ratio(frame, prev_frame, bbox)
    b = laplacian_variance(frame, bbox)
    cx, cy = bbox_center(bbox)
    area = float(max(1, (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])))

    motion_hist.append(m)
    blur_hist.append(b)
    center_hist.append((cx, cy))
    area_hist.append(area)

    if track["seen_frames"] < anti_cfg["warmup_frames"]:
        track["is_real"] = True
        track["fake_streak"] = 0
        track["real_streak"] = 0
        return True

    recent_motion = list(motion_hist)[-15:]
    recent_blur = list(blur_hist)[-15:]
    recent_centers = list(center_hist)[-15:]
    recent_areas = list(area_hist)[-15:]

    motion_hits = sum(1 for v in recent_motion if v >= anti_cfg["motion_threshold"])
    low_motion_hits = sum(1 for v in recent_motion if v <= anti_cfg["low_motion_threshold"])
    avg_motion = float(sum(recent_motion)) / float(len(recent_motion)) if recent_motion else 0.0
    avg_blur = float(sum(recent_blur)) / float(len(recent_blur)) if recent_blur else 0.0

    if len(recent_centers) >= 6:
        center_array = np.array(recent_centers, dtype=np.float32)
        center_mean = np.mean(center_array, axis=0)
        center_offsets = np.linalg.norm(center_array - center_mean, axis=1)
        center_jitter = float(np.median(center_offsets))
    else:
        center_jitter = 999.0

    if len(recent_areas) >= 6:
        area_mean = float(np.mean(recent_areas))
        area_std = float(np.std(recent_areas))
        area_jitter = area_std / max(1.0, area_mean)
    else:
        area_jitter = 1.0

    global_stable = global_motion <= anti_cfg["global_motion_pause"]
    near_static = (
        center_jitter < anti_cfg["static_center_jitter"]
        and area_jitter < anti_cfg["static_area_jitter"]
    )
    low_liveliness = (
        low_motion_hits >= max(10, len(recent_motion) - 2)
        and avg_motion < (anti_cfg["low_motion_threshold"] + 0.0015)
    )
    sharp_subject = avg_blur > anti_cfg["sharp_blur_threshold"]
    active = (
        motion_hits >= 2
        or avg_motion >= anti_cfg["motion_threshold"]
        or center_jitter > (anti_cfg["static_center_jitter"] + 1.8)
        or area_jitter > max(0.05, anti_cfg["static_area_jitter"] * 2.2)
    )

    suspicious = global_stable and near_static and low_liveliness and sharp_subject

    if suspicious:
        track["fake_streak"] += 1
        track["real_streak"] = max(0, track["real_streak"] - 1)
    else:
        track["fake_streak"] = max(0, track["fake_streak"] - 1)
        if active:
            track["real_streak"] += 1
        else:
            track["real_streak"] = max(0, track["real_streak"] - 1)

    mature = track["seen_frames"] >= (anti_cfg["warmup_frames"] + 10)
    is_real = bool(track.get("is_real", True))

    if is_real:
        if mature and track["fake_streak"] >= anti_cfg["fake_min_streak"]:
            is_real = False
    else:
        if track["real_streak"] >= anti_cfg["real_recover_streak"]:
            is_real = True
            track["fake_streak"] = 0

    track["is_real"] = is_real
    return is_real


def draw_info(
    frame,
    real_count,
    fake_count,
    status,
    color,
    green_max,
    yellow_max,
    alert_msg,
    alert_enabled,
    show_boxes,
    real_boxes,
    fake_boxes,
    font_scale,
    box_thickness,
    show_fps,
    fps,
):
    panel_font = max(0.4, float(font_scale))
    thickness = max(1, int(box_thickness))

    cv2.rectangle(frame, (10, 10), (430, 190), (0, 0, 0), -1)
    cv2.rectangle(frame, (10, 10), (430, 190), color, thickness)

    cv2.putText(
        frame,
        f"Real Count: {real_count}",
        (20, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        panel_font,
        (255, 255, 255),
        thickness,
    )
    cv2.putText(
        frame,
        f"Ignored/Fake: {fake_count}",
        (20, 76),
        cv2.FONT_HERSHEY_SIMPLEX,
        panel_font,
        (255, 80, 180),
        thickness,
    )
    cv2.putText(
        frame,
        f"Status: {status}",
        (20, 108),
        cv2.FONT_HERSHEY_SIMPLEX,
        panel_font,
        color,
        thickness,
    )
    cv2.putText(
        frame,
        f"Green: 0-{green_max}",
        (20, 142),
        cv2.FONT_HERSHEY_SIMPLEX,
        panel_font * 0.8,
        (0, 255, 0),
        1,
    )
    cv2.putText(
        frame,
        f"Yellow: {green_max + 1}-{yellow_max}",
        (190, 142),
        cv2.FONT_HERSHEY_SIMPLEX,
        panel_font * 0.8,
        (0, 255, 255),
        1,
    )

    if show_fps:
        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (20, 172),
            cv2.FONT_HERSHEY_SIMPLEX,
            panel_font * 0.8,
            (220, 220, 220),
            1,
        )

    if alert_enabled and status == "RED":
        h = frame.shape[0]
        cv2.rectangle(frame, (10, h - 60), (frame.shape[1] - 10, h - 10), (0, 0, 255), -1)
        cv2.putText(
            frame,
            alert_msg,
            (20, h - 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            panel_font * 0.85,
            (255, 255, 255),
            thickness,
        )

    if show_boxes:
        for box in real_boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), thickness)
            cv2.putText(
                frame,
                "REAL",
                (x1, max(14, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                panel_font * 0.7,
                (0, 255, 0),
                thickness,
            )

        for x1, y1, x2, y2 in fake_boxes:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), thickness)
            cv2.putText(
                frame,
                "FAKE",
                (x1, max(14, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                panel_font * 0.7,
                (255, 0, 255),
                thickness,
            )

    return frame


def main():
    config = load_config()
    model_cfg = config.get("model", {})
    density_cfg = config.get("density", {})
    alerts_cfg = config.get("alerts", {})
    video_cfg = config.get("video", {})
    display_cfg = config.get("display", {})
    anti_cfg = get_anti_spoof_config(config)

    model_name = model_cfg.get("name", "yolo12n.pt")
    confidence = float(model_cfg.get("confidence", 0.5))
    person_class = int(model_cfg.get("person_class", 0))

    green_max = int(density_cfg.get("green_max", 1))
    yellow_max = int(density_cfg.get("yellow_max", 2))

    alert_enabled = bool(alerts_cfg.get("enabled", True))
    alert_msg = str(alerts_cfg.get("message", "ALERT: Area overcrowded!"))

    source = parse_camera_source(video_cfg.get("source", 0))
    show_fps = bool(video_cfg.get("show_fps", True))
    show_boxes = bool(video_cfg.get("show_boxes", True))

    window_name = str(display_cfg.get("window_name", "Crowd Monitoring System"))
    font_scale = float(display_cfg.get("font_scale", 0.7))
    box_thickness = int(display_cfg.get("box_thickness", 2))

    model = YOLO(model_name)
    cap = open_video_capture(source)

    if not cap.isOpened():
        print("Error: Cannot open camera")
        return

    print("Starting crowd monitor with liveliness filter...")
    print("Press 'q' to quit")

    prev_frame = None
    tracks = {}
    next_track_id = 0

    prev_t = time.perf_counter()
    while True:
        now = time.perf_counter()
        dt = now - prev_t
        prev_t = now
        fps = (1.0 / dt) if dt > 0 else 0.0

        ret, frame = cap.read()
        if not ret:
            continue

        raw_frame = frame.copy()

        people = detect_people(model, frame, confidence, person_class)
        global_motion = frame_motion_ratio(frame, prev_frame)

        detections = []
        for person in people:
            raw = tuple(map(int, person.xyxy[0]))
            bbox = clip_bbox(raw, frame.shape)
            if bbox is not None:
                detections.append((person, bbox))

        matches, unmatched_det_idxs, unmatched_track_ids = match_detections_to_tracks(
            [bbox for _, bbox in detections],
            tracks,
            iou_threshold=anti_cfg["match_iou_threshold"],
            max_center_distance=anti_cfg["match_max_center_distance"],
            max_missed_for_match=anti_cfg["max_missed_for_match"],
        )

        for track_id in unmatched_track_ids:
            track = tracks[track_id]
            track["missed"] += 1
            vx, vy = track.get("vel", (0.0, 0.0))
            track["vel"] = (0.85 * vx, 0.85 * vy)

        stale_track_ids = [
            track_id
            for track_id, track in tracks.items()
            if int(track.get("missed", 0)) > anti_cfg["max_missed_frames"]
        ]
        for track_id in stale_track_ids:
            del tracks[track_id]

        detection_to_track = dict(matches)
        for det_idx in sorted(unmatched_det_idxs):
            _, bbox = detections[det_idx]
            detection_to_track[det_idx] = next_track_id
            tracks[next_track_id] = build_track_state(anti_cfg["history_size"], bbox)
            next_track_id += 1

        real_count = 0
        fake_count = 0
        real_boxes = []
        fake_boxes = []

        for det_idx, (person, bbox) in enumerate(detections):
            track_id = detection_to_track.get(det_idx)
            if track_id is None or track_id not in tracks:
                track_id = next_track_id
                tracks[next_track_id] = build_track_state(anti_cfg["history_size"], bbox)
                next_track_id += 1

            track = tracks[track_id]
            update_track_velocity(track, bbox)
            track["bbox"] = bbox
            track["missed"] = 0
            track["seen_frames"] += 1

            is_real = classify_real_or_fake(
                frame,
                prev_frame,
                bbox,
                track,
                global_motion,
                anti_cfg,
            )

            if is_real:
                real_count += 1
                real_boxes.append(person)
            else:
                fake_count += 1
                fake_boxes.append(bbox)

        status, color = get_status(real_count, green_max, yellow_max)
        frame = draw_info(
            frame,
            real_count,
            fake_count,
            status,
            color,
            green_max,
            yellow_max,
            alert_msg,
            alert_enabled,
            show_boxes,
            real_boxes,
            fake_boxes,
            font_scale,
            box_thickness,
            show_fps,
            fps,
        )

        cv2.imshow(window_name, frame)
        prev_frame = raw_frame

        if cv2.waitKey(10) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
