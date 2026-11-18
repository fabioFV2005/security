# YOLO Security System

Detects people with YOLO, checks whether each face is visible, masked or covered (balaclava, ski mask, scarf), scores the severity and sends Telegram alerts with the image.

## 1. Install

```
.venv\Scripts\pip install -r requirements.txt
.venv\Scripts\pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126 --force-reinstall
```

The second line enables your NVIDIA GPU. Without it everything runs on CPU.

## 2. Telegram

1. Talk to `@BotFather` in Telegram, send `/newbot`, copy the token.
2. Send any message to your new bot, then open `https://api.telegram.org/bot<TOKEN>/getUpdates` and copy `chat.id`.
3. Put both in `config.yaml` under `telegram`.
4. Test: `python main.py --test-telegram`

## 3. Train the covered-face model

Get a free API key at https://app.roboflow.com/settings/api, then:

```
set ROBOFLOW_API_KEY=your_key
python train.py
```

This downloads three public Roboflow datasets, merges them into 3 classes (`face_visible`, `face_mask`, `face_covered`), trains and saves `models/face_cover.pt`.

Options: `--epochs 150`, `--model yolo26n.pt` (faster) or `yolo26m.pt` (more accurate), `--local path\to\dataset` to add your own labeled images.

## 4. Run

```
python main.py
python main.py --source video.mp4
python main.py --source rtsp://user:pass@192.168.1.10/stream
```

Press `q` to quit. Alert images are also saved in `alerts/`.

Without a trained model it still works in person-only mode.

## Severity

| Factor | Points |
|---|---|
| Person | 1 |
| Face mask | +1 |
| Face covered | +3 |
| Night (22:00-06:00) | +1 |
| Inside restricted zone | +1 |
| Loitering over 20 s | +1 |
| 2+ people with covered faces | +1 |

1 = LOW, 2 = MEDIUM, 3-4 = HIGH, 5+ = CRITICAL. Weights, hours and cooldowns are in `config.yaml`.

Restricted zone: polygon of normalized points, e.g. `zone: [[0.1,0.5],[0.9,0.5],[0.9,1.0],[0.1,1.0]]`.
