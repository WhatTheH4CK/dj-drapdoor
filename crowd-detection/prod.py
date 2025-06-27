import argparse
import threading
import time
import json

import cv2
import torch
import numpy as np
from collections import deque

from fastapi import FastAPI, Response, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from models import build_model
latest_dims = {"width": 0, "height": 0}

# --- Config & globals ---
WEIGHT_PATH        = "pretrained_model/best_mae.pth"
THRESHOLD          = 0.3
MAX_DIM            = 512
FPS                = 10
ROLLING_AVG_FRAMES = 10

latest_count  = 0
latest_frame  = None
latest_points = []
_frame_lock   = threading.Lock()
recent_counts = deque(maxlen=ROLLING_AVG_FRAMES)

# --- Args ---
parser = argparse.ArgumentParser(description="People-counting server")
parser.add_argument("--cam",      type=int,   default=0,            help="OpenCV camera index (0,1,2…)")
parser.add_argument("--backbone", type=str,   default="vgg16_bn",  help="model backbone")
parser.add_argument("--row",      type=int,   default=2,            help="grid rows")
parser.add_argument("--line",     type=int,   default=2,            help="grid lines")
args = parser.parse_args()

# --- Model load ---
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    torch.cuda.set_device(device)

model = build_model(args)
ckpt  = torch.load(WEIGHT_PATH, map_location=device, weights_only=True)
model.load_state_dict(ckpt["model"])
model.to(device).eval()

# --- Camera thread ---
def camera_loop():
    global latest_count, latest_frame, latest_points, latest_dims
    cap = cv2.VideoCapture(args.cam)
    while True:
        ret, img_bgr = cap.read()
        if not ret:
            time.sleep(0.1)
            continue

        H, W, _ = img_bgr.shape
        scale    = min(MAX_DIM/W, MAX_DIM/H)
        Wn, Hn   = (int(W*scale)//128*128, int(H*scale)//128*128)

        img_rgb  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_res  = cv2.resize(img_rgb, (Wn, Hn))
        tensor   = (
            torch.from_numpy(img_res.astype(np.float32)/255.0)
                 .permute(2,0,1).unsqueeze(0).to(device)
        )

        with torch.no_grad():
            out    = model(tensor)
            scores = torch.nn.functional.softmax(out["pred_logits"], -1)[0,:,1]
            pts    = out["pred_points"][0][scores > THRESHOLD].cpu().numpy()

        count = len(pts)
        recent_counts.append(count)

        _, jpg = cv2.imencode('.jpg', img_bgr)
        
        with _frame_lock:
            latest_frame  = jpg.tobytes()
            latest_count  = count
            latest_points = pts.tolist()
            latest_dims   = {"width": Wn, "height": Hn}


        print(f" Frame updated count={count} w={Wn} h={Hn}", flush=True)
        time.sleep(max(1/FPS, 0.01))

    cap.release()

threading.Thread(target=camera_loop, daemon=True).start()

# --- FastAPI app ---
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/snapshot")
def snapshot():
    with _frame_lock:
        f = latest_frame
    if not f:
        return Response(status_code=503)
    return Response(f, media_type="image/jpeg")

@app.get("/video")
def video_feed():
    def gen():
        while True:
            with _frame_lock:
                f = latest_frame
            if f:
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + f + b"\r\n"
            time.sleep(1/FPS)
    return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/count")
def get_count():
    with _frame_lock:
        return {"count": latest_count, "points": latest_points, **latest_dims}

# --- SSE endpoint ---
def sse_generator():
    while True:
        with _frame_lock:
            data = json.dumps({"count": latest_count, "points": latest_points, **latest_dims})
        yield f"data: {data}\n\n"
        time.sleep(1/FPS)

@app.get("/sse/count")
def sse_count():
    return StreamingResponse(
        sse_generator(),
        media_type="text/event-stream"
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
