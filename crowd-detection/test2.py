import torch
import torchvision.transforms as transforms
import numpy as np
import cv2
from PIL import Image
from models import build_model
import argparse
import os

# --- Configuration ---
IMAGE_PATH = "test_images/test2.jpg"
WEIGHT_PATH = "pretrained_model/best_mae.pth"
OUTPUT_IMAGE = "prediction_output.jpg"
THRESHOLD = 0.3

# --- Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
args = argparse.Namespace(backbone="vgg16_bn", row=2, line=2)

# --- Load model ---
model = build_model(args)
model.to(device)
checkpoint = torch.load(WEIGHT_PATH, map_location=device)
model.load_state_dict(checkpoint["model"])
model.eval()

# --- Load and preprocess image ---
img_raw = Image.open(IMAGE_PATH).convert("RGB")
MAX_DIM = 800

W, H = img_raw.size
scaling = min(MAX_DIM / W, MAX_DIM / H)
W_new = int(W * scaling) // 128 * 128
H_new = int(H * scaling) // 128 * 128
img_resized = img_raw.resize((W_new, H_new), Image.Resampling.LANCZOS)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])
img_tensor = transform(img_resized).unsqueeze(0).to(device)

# --- Inference ---
with torch.no_grad():
    outputs = model(img_tensor)
    scores = torch.nn.functional.softmax(outputs["pred_logits"], -1)[0, :, 1]
    points = outputs["pred_points"][0][scores > THRESHOLD].cpu().numpy()

# --- Draw output ---
image_np = np.array(img_resized)
image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
for p in points:
    cv2.circle(image_bgr, (int(p[0]), int(p[1])), 2, (255, 0, 0), -1)

count = len(points)
cv2.putText(image_bgr, f"Count: {count}", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2)
cv2.imwrite(OUTPUT_IMAGE, image_bgr)

print(f"Estimated count: {count}")
print(f"Output saved to {OUTPUT_IMAGE}")
