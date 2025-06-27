import torch
import torchvision.transforms as transforms
import numpy as np
import cv2
from PIL import Image
from models import build_model
import argparse
import os
import time
from collections import deque

# --- Config ---
WEIGHT_PATH = "pretrained_model/best_mae.pth"
THRESHOLD = 0.3
MAX_DIM = 512
FPS = 10
OUTPUT_IMAGE = "prediction_output.jpg"
SHOW_IMAGE = True  # Set to False to disable live display
ROLLING_AVG_FRAMES = 10  # number of recent frames to average

# --- Setup ---
torch.backends.cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
args = argparse.Namespace(backbone="vgg16_bn", row=2, line=2)

recent_counts = deque(maxlen=ROLLING_AVG_FRAMES)

print("🔧 Loading model...")
start_load = time.time()
model = build_model(args)
model.to(device)

checkpoint = torch.load(WEIGHT_PATH, map_location=device)
model.load_state_dict(checkpoint["model"])
model.eval()
end_load = time.time()
print(f"✅ Model loaded in {end_load - start_load:.2f}s")

# --- Transforms ---
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

def find_highest_working_camera(max_index=10):
    best_index = None
    for i in range(max_index):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)  # CAP_DSHOW = DirectShow (Windows)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None and frame.size > 0:
                best_index = i  # update to latest working index
            cap.release()
    return best_index

# --- Camera Init ---
camera_index = find_highest_working_camera()
if camera_index is None:
    raise RuntimeError("❌ No working camera found.")
camera = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
print(f"📷 Using camera at index {camera_index}")

print("🎥 Starting camera loop at ~1 FPS...")
try:
    while True:
        loop_start = time.time()

        # --- Capture image ---
        ret, img_bgr = camera.read()
        if not ret:
            print("⚠️ No image captured")
            continue

        # --- Convert to PIL ---
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        # img_pil = Image.fromarray(img_rgb)

        # --- Resize ---
        resize_start = time.time()
        H, W, _ = img_rgb.shape
        scaling = min(MAX_DIM / W, MAX_DIM / H)
        W_new = int(W * scaling) // 128 * 128
        H_new = int(H * scaling) // 128 * 128
        # img_resized = img_pil.resize((W_new, H_new), Image.LANCZOS)
        img_resized = cv2.resize(img_rgb, (W_new, H_new), interpolation=cv2.INTER_LINEAR)
        resize_end = time.time()

        # --- Preprocess ---
        print("Preprocessing image")
        prep_start = time.time()
        img_normalized = img_resized.astype(np.float32) / 255.0
        img_normalized = (img_normalized - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        img_transposed = np.transpose(img_normalized, (2, 0, 1))
        img_tensor = (
            torch.from_numpy(img_transposed)
                .unsqueeze(0)
                .float()                 # stays on CPU
                .to(device)              # now copy to GPU
        )

        print("img_tensor dtype", img_tensor.dtype)
        # img_tensor = transform(img_resized).unsqueeze(0).to(device, non_blocking=True)
        # img_tensor = jetson_utils.cudaFromNumpy(img_np).Clone()
        prep_end = time.time()

        # --- Inference ---
        inf_start = time.time()
        with torch.no_grad():
            outputs = model(img_tensor)
            scores = torch.nn.functional.softmax(outputs["pred_logits"], -1)[0, :, 1]
            points = outputs["pred_points"][0][scores > THRESHOLD].cpu().numpy()
        inf_end = time.time()

        # --- Draw ---
        draw_start = time.time()
        image_np = np.array(img_resized)
        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        for p in points:
            cv2.circle(image_bgr, (int(p[0]), int(p[1])), 2, (255, 0, 0), -1)

        count = len(points)
        recent_counts.append(count)
        avg_count = int(round(sum(recent_counts) / len(recent_counts)))

        cv2.putText(image_bgr,
                    f"Current: {count}  Avg: {avg_count}",
                    (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                    (0, 255, 255), 2)
        cv2.imwrite(OUTPUT_IMAGE, image_bgr)
        draw_end = time.time()

        if SHOW_IMAGE:
            cv2.imshow("Prediction", image_bgr)
            cv2.waitKey(1)  # Just lets OpenCV update the window

        # --- Logs ---
        print(f"🧠 Inference: {inf_end - inf_start:.2f}s | "
              f"Resize: {resize_end - resize_start:.2f}s | "
              f"Preprocess: {prep_end - prep_start:.2f}s | "
              f"Draw+Save: {draw_end - draw_start:.2f}s | "
              f"Total: {time.time() - loop_start:.2f}s | "
              f"Count: {count}")

        # --- FPS Control ---
        time.sleep(max(1.0 / FPS - (time.time() - loop_start), 0))
except KeyboardInterrupt:
    print("🛑 Interrupted. Releasing camera...")
finally:
    camera.release()
    cv2.destroyAllWindows()
    print("✅ Camera released.")