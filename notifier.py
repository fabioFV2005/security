import os
import queue
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import requests

from severity import level_index


class TelegramNotifier:
    def __init__(self, cfg):
        self.token = os.getenv("TELEGRAM_BOT_TOKEN") or cfg.get("token") or ""
        self.chat_id = os.getenv("TELEGRAM_CHAT_ID") or str(cfg.get("chat_id") or "")
        self.enabled = bool(cfg.get("enabled", True) and self.token and self.chat_id)
        self.queue = queue.Queue(maxsize=20)
        if self.enabled:
            threading.Thread(target=self._worker, daemon=True).start()

    def send(self, image, caption):
        if not self.enabled:
            return
        try:
            self.queue.put_nowait((image.copy(), caption))
        except queue.Full:
            print("[telegram] queue full, alert dropped")

    def send_sync(self, image, caption):
        ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ok:
            return False
        url = f"https://api.telegram.org/bot{self.token}/sendPhoto"
        for attempt in range(3):
            try:
                r = requests.post(
                    url,
                    data={"chat_id": self.chat_id, "caption": caption[:1024]},
                    files={"photo": ("alert.jpg", buf.tobytes(), "image/jpeg")},
                    timeout=15,
                )
                if r.ok:
                    return True
                print(f"[telegram] error {r.status_code}: {r.text[:200]}")
                if r.status_code < 500 and r.status_code != 429:
                    return False
            except requests.RequestException as e:
                print(f"[telegram] request failed: {e}")
            time.sleep(2 * (attempt + 1))
        return False

    def _worker(self):
        while True:
            image, caption = self.queue.get()
            self.send_sync(image, caption)


class AlertManager:
    def __init__(self, cfg, notifier, save_dir):
        self.min_level = level_index(cfg.get("min_level", "LOW"))
        cooldowns = cfg.get("cooldown_seconds") or {}
        self.cooldown = {level_index(k): float(v) for k, v in cooldowns.items()}
        self.notifier = notifier
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.last_time = 0.0
        self.last_level = 0

    def should_alert(self, level, now):
        if level < self.min_level:
            return False
        if now - self.last_time > max(self.cooldown.values(), default=60):
            self.last_level = 0
        if level > self.last_level:
            return True
        return now - self.last_time >= self.cooldown.get(level, 60)

    def handle(self, assessment, image, people_count):
        now = time.time()
        if not self.should_alert(assessment.level, now):
            return False
        self.last_time = now
        self.last_level = assessment.level

        stamp = datetime.now()
        path = self.save_dir / f"{stamp:%Y%m%d_%H%M%S}_{assessment.name}_{assessment.event}.jpg"
        cv2.imwrite(str(path), image)

        title = "PERSON WITH COVERED FACE" if assessment.event == "person_face_covered" else "PERSON DETECTED"
        caption = (
            f"[{assessment.name}] {title}\n"
            f"Time: {stamp:%Y-%m-%d %H:%M:%S}\n"
            f"People in view: {people_count}\n"
            f"Score: {assessment.score}\n"
            f"Reasons: {', '.join(assessment.reasons)}"
        )
        print(caption.replace("\n", " | "))
        self.notifier.send(image, caption)
        return True
