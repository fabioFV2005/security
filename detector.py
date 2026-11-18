import time
from collections import Counter, deque
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from notifier import AlertManager, TelegramNotifier
from severity import LEVELS, Person, SeverityEngine

FACE_COLORS = {
    "face_visible": (0, 200, 0),
    "face_mask": (0, 200, 255),
    "face_covered": (0, 0, 255),
    "unknown": (180, 180, 180),
}

LEVEL_COLORS = {
    0: (120, 120, 120),
    1: (0, 200, 0),
    2: (0, 200, 255),
    3: (0, 120, 255),
    4: (0, 0, 255),
}


class Track:
    def __init__(self, window, now):
        self.first_seen = now
        self.last_seen = now
        self.frames = 0
        self.faces = deque(maxlen=window)

    def face(self, min_votes):
        votes = Counter(f for f in self.faces if f)
        if votes["face_covered"] >= min_votes:
            return "face_covered"
        if votes["face_mask"] >= min_votes:
            return "face_mask"
        if votes["face_visible"]:
            return "face_visible"
        return "unknown"


class SecuritySystem:
    def __init__(self, cfg):
        self.cfg = cfg
        dev = cfg.get("device", "auto")
        self.device = (0 if torch.cuda.is_available() else "cpu") if dev == "auto" else dev
        print(f"Device: {'cuda' if self.device == 0 else self.device}")

        self.person_model = YOLO(cfg.get("person_model", "yolo26n.pt"))
        face_path = cfg.get("face_model")
        self.face_model = None
        if face_path and Path(face_path).exists():
            self.face_model = YOLO(face_path)
            print(f"Face model loaded: {face_path} {self.face_model.names}")
        else:
            print("Face model not found, running in person-only mode. Run: python train.py")

        t = cfg.get("tracking") or {}
        self.window = t.get("smooth_window", 8)
        self.min_votes = t.get("min_votes", 3)
        self.min_track_frames = t.get("min_track_frames", 5)
        self.forget_seconds = t.get("forget_seconds", 5)
        self.tracks = {}

        self.zone = np.array(cfg.get("zone") or [], dtype=np.float32)
        self.severity = SeverityEngine(cfg.get("severity") or {}, zone_enabled=len(self.zone) >= 3)
        self.notifier = TelegramNotifier(cfg.get("telegram") or {})
        if not self.notifier.enabled:
            print("Telegram disabled: set token and chat_id in config.yaml")
        self.alerts = AlertManager(cfg.get("alerts") or {}, self.notifier, cfg.get("save_dir", "alerts"))

    def in_zone(self, box, shape):
        if len(self.zone) < 3:
            return True
        h, w = shape[:2]
        poly = (self.zone * [w, h]).astype(np.int32)
        foot = (float((box[0] + box[2]) / 2), float(box[3]))
        return cv2.pointPolygonTest(poly, foot, False) >= 0

    def head_crop(self, frame, box):
        x1, y1, x2, y2 = box
        bw, bh = x2 - x1, y2 - y1
        pad = int(bw * 0.1)
        cx1, cx2 = max(0, x1 - pad), min(frame.shape[1], x2 + pad)
        cy1, cy2 = max(0, y1 - pad), min(frame.shape[0], y1 + int(bh * 0.45))
        if cx2 - cx1 < 16 or cy2 - cy1 < 16:
            return None
        return frame[cy1:cy2, cx1:cx2]

    def classify_faces(self, frame, boxes):
        if self.face_model is None or not boxes:
            return [(None, 0.0)] * len(boxes)
        crops, idx = [], []
        for i, b in enumerate(boxes):
            c = self.head_crop(frame, b)
            if c is not None:
                crops.append(c)
                idx.append(i)
        out = [(None, 0.0)] * len(boxes)
        if not crops:
            return out
        results = self.face_model.predict(
            crops,
            conf=self.cfg.get("face_conf", 0.4),
            imgsz=self.cfg.get("face_imgsz", 320),
            device=self.device,
            verbose=False,
        )
        for i, r in zip(idx, results):
            if r.boxes is None or len(r.boxes) == 0:
                continue
            j = int(r.boxes.conf.argmax())
            out[i] = (r.names[int(r.boxes.cls[j])], float(r.boxes.conf[j]))
        return out

    def process(self, frame):
        now = time.time()
        res = self.person_model.track(
            frame,
            persist=True,
            classes=[0],
            conf=self.cfg.get("person_conf", 0.5),
            imgsz=self.cfg.get("imgsz", 640),
            device=self.device,
            tracker="bytetrack.yaml",
            verbose=False,
        )[0]

        boxes, ids, confs = [], [], []
        if res.boxes is not None and len(res.boxes):
            boxes = res.boxes.xyxy.cpu().numpy().astype(int).tolist()
            confs = res.boxes.conf.cpu().numpy().tolist()
            ids = res.boxes.id.cpu().numpy().astype(int).tolist() if res.boxes.id is not None else [-1 - i for i in range(len(boxes))]

        faces = self.classify_faces(frame, boxes)
        people = []
        for box, tid, conf, (face, fconf) in zip(boxes, ids, confs, faces):
            tr = self.tracks.get(tid)
            if tr is None:
                tr = self.tracks[tid] = Track(self.window, now)
            tr.last_seen = now
            tr.frames += 1
            tr.faces.append(face)
            if tr.frames < self.min_track_frames:
                continue
            people.append(Person(
                track_id=tid,
                box=tuple(box),
                conf=conf,
                face=tr.face(self.min_votes),
                face_conf=fconf,
                in_zone=self.in_zone(box, frame.shape),
                dwell=now - tr.first_seen,
            ))

        for tid in [k for k, v in self.tracks.items() if now - v.last_seen > self.forget_seconds]:
            del self.tracks[tid]

        assessment = self.severity.assess(people)
        annotated = self.draw(frame, people, assessment)
        if assessment.level > 0:
            self.alerts.handle(assessment, annotated, len(people))
        return annotated, assessment

    def draw(self, frame, people, assessment):
        img = frame.copy()
        if len(self.zone) >= 3:
            h, w = img.shape[:2]
            cv2.polylines(img, [(self.zone * [w, h]).astype(np.int32)], True, (255, 0, 255), 2)
        for p in people:
            x1, y1, x2, y2 = p.box
            color = FACE_COLORS.get(p.face, FACE_COLORS["unknown"])
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            label = f"#{p.track_id} {p.face} {int(p.dwell)}s"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(img, (x1, max(0, y1 - th - 8)), (x1 + tw + 6, y1), color, -1)
            cv2.putText(img, label, (x1 + 3, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)
        banner = f"{LEVELS[assessment.level]}  {assessment.event}  people={len(people)}"
        cv2.rectangle(img, (0, 0), (img.shape[1], 32), LEVEL_COLORS[assessment.level], -1)
        cv2.putText(img, banner, (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(img, time.strftime("%Y-%m-%d %H:%M:%S"), (img.shape[1] - 230, 23),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        return img

    def open_source(self):
        src = self.cfg.get("source", 0)
        if isinstance(src, str) and src.isdigit():
            src = int(src)
        cap = cv2.VideoCapture(src, cv2.CAP_DSHOW) if isinstance(src, int) else cv2.VideoCapture(src)
        if not cap.isOpened():
            cap = cv2.VideoCapture(src)
        return src, cap

    def run(self):
        src, cap = self.open_source()
        if not cap.isOpened():
            raise SystemExit(f"Cannot open source: {src}")
        is_file = isinstance(src, str) and Path(src).exists()
        show = self.cfg.get("show_window", True)
        every = max(1, int(self.cfg.get("process_every", 1)))
        n, fps_t, fps = 0, time.time(), 0.0
        print("Running. Press q to quit.")
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    if is_file:
                        break
                    print("Stream lost, reconnecting...")
                    cap.release()
                    time.sleep(2)
                    src, cap = self.open_source()
                    continue
                n += 1
                if n % every:
                    continue
                annotated, _ = self.process(frame)
                if n % 30 == 0:
                    fps = 30 / every / (time.time() - fps_t)
                    fps_t = time.time()
                if show:
                    cv2.putText(annotated, f"{fps:.1f} fps", (10, annotated.shape[0] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    cv2.imshow("Security", annotated)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
        finally:
            cap.release()
            cv2.destroyAllWindows()
