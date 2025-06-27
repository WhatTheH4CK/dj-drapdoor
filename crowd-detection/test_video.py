import torch
import torchvision.transforms as transforms
import numpy as np
import cv2
from models import build_model
import argparse
import os
from collections import deque
import time

# --- Config ---
INPUT_VIDEO_PATH = "input5.mp4"
OUTPUT_VIDEO_PATH = "output_with_dots5.mp4"
WEIGHT_PATH = "pretrained_model/best_mae.pth"
MAX_DIM = 512
THRESHOLD = 0.3
ROLLING_AVG_FRAMES = 10
SHOW_IMAGE = True

# --- Setup ---
torch.backends.cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
args = argparse.Namespace(backbone="vgg16_bn", row=2, line=2)

print("🔧 Loading model...")
start_load = time.time()
model = build_model(args)
model.to(device)
checkpoint = torch.load(WEIGHT_PATH, map_location=device)
model.load_state_dict(checkpoint["model"])
model.eval()
print(f"✅ Model loaded in {time.time() - start_load:.2f}s")

# --- Video Setup ---
cap = cv2.VideoCapture(INPUT_VIDEO_PATH)
if not cap.isOpened():
    raise RuntimeError(f"❌ Cannot open video: {INPUT_VIDEO_PATH}")

fps = cap.get(cv2.CAP_PROP_FPS)
W_orig = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H_orig = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (W_orig, H_orig))

recent_counts = deque(maxlen=ROLLING_AVG_FRAMES)

print("🎞️  Processing video...")
frame_idx = 0
while True:
    ret, img_bgr = cap.read()
    if not ret:
        break

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Resize to fit model
    H, W, _ = img_rgb.shape
    scaling = min(MAX_DIM / W, MAX_DIM / H)
    W_new = int(W * scaling) // 128 * 128
    H_new = int(H * scaling) // 128 * 128
    img_resized = cv2.resize(img_rgb, (W_new, H_new), interpolation=cv2.INTER_LINEAR)

    # Preprocess
    img_normalized = img_resized.astype(np.float32) / 255.0
    img_normalized = (img_normalized - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    img_transposed = np.transpose(img_normalized, (2, 0, 1))
    img_tensor = torch.from_numpy(img_transposed).unsqueeze(0).float().to(device)

    # Inference
    with torch.no_grad():
        outputs = model(img_tensor)
        scores = torch.nn.functional.softmax(outputs["pred_logits"], -1)[0, :, 1]
        points = outputs["pred_points"][0][scores > THRESHOLD].cpu().numpy()

    # Draw on orifginal-sized frame
    count = len(points)
    recent_counts.append(count)
    avg_count = int(round(sum(recent_counts) / len(recent_counts)))
    image_bgr = cv2.resize(img_rgb, (W_orig, H_orig))
    image_bgr = cv2.cvtColor(image_bgr, cv2.COLOR_RGB2BGR)

    # Scale points from resized to original dimensions
    scale_x = W_orig / W_new
    scale_y = H_orig / H_new
    for p in points:
        x = int(p[0] * scale_x)
        y = int(p[1] * scale_y)
        cv2.circle(image_bgr, (x, y), 2, (255, 0, 0), -1)

    cv2.putText(image_bgr,
                f"Current: {count}  Avg({ROLLING_AVG_FRAMES}): {avg_count}",
                (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2)

    out.write(image_bgr)

    if SHOW_IMAGE:
        cv2.imshow("Processed Video", image_bgr)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    frame_idx += 1

# Cleanup
cap.release()
out.release()
cv2.destroyAllWindows()
print(f"🎉 Done. {frame_idx} frames written to {OUTPUT_VIDEO_PATH}")
