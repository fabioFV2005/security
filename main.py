import argparse

import numpy as np
import yaml


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--source", help="Camera index, video file or rtsp:// url")
    ap.add_argument("--no-show", action="store_true")
    ap.add_argument("--test-telegram", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.source is not None:
        cfg["source"] = args.source
    if args.no_show:
        cfg["show_window"] = False

    if args.test_telegram:
        from notifier import TelegramNotifier

        n = TelegramNotifier(cfg.get("telegram") or {})
        if not n.enabled:
            raise SystemExit("Telegram is not configured: set token and chat_id in config.yaml")
        img = np.full((240, 420, 3), 40, dtype=np.uint8)
        print("Sent" if n.send_sync(img, "Security system: Telegram test OK") else "Failed")
        return

    from detector import SecuritySystem

    SecuritySystem(cfg).run()


if __name__ == "__main__":
    main()
