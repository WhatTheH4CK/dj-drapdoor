import torch
import torchvision.transforms as standard_transforms
import numpy as np
from PIL import Image, ImageOps  # add at the top if not already
import cv2
from models import build_model
import os
import warnings

warnings.filterwarnings('ignore')

# --- Hardcoded parameters ---
params = {
    "backbone": "vgg16_bn",
    "threshold": 0.8,
    "row": 2,
    "line": 2,
    "gpu_id": 0,
    "images_path": "./test_images",
    "weight_path": "./pretrained_model/best_mae.pth",
    "output_dir": "./prediction/images/",
    "predicts_txt_dir": "./prediction/predict_txt.txt",
    "predicts_point_dir": "./prediction/new_thr=0.8"
}


# --- Main function ---
def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = str(params["gpu_id"])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")

    # Load model
    import argparse
    args = argparse.Namespace(**params)
    model = build_model(args)
    model.to(device)
    checkpoint = torch.load(params["weight_path"], map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    model.eval()

    transform = standard_transforms.Compose([
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    all_point = []
    c = 1

    image_files = sorted([f for f in os.listdir(params["images_path"]) if f.lower().endswith('.jpg')])
    for i, filename in enumerate(image_files, start=1):
        print(f'Processing image {filename}')
        img_raw = Image.open(os.path.join(params["images_path"], filename)).convert("RGB")

        # Resize
        width, height = img_raw.size
        if width > 2000 or height > 2000:
            r = width / height
            width = 2000
            height = int(2000 / r)
        new_width = width // 128 * 128
        new_height = height // 128 * 128

        img_raw = img_raw.resize((new_width, new_height), Image.Resampling.LANCZOS)

        # Transform
        img = transform(img_raw)
        samples = img.unsqueeze(0).to(device)

        # Inference
        outputs = model(samples)
        outputs_scores = torch.nn.functional.softmax(outputs['pred_logits'], -1)[:, :, 1][0]
        outputs_points = outputs['pred_points'][0]
        points = outputs_points[outputs_scores > params["threshold"]].detach().cpu().numpy().tolist()
        predict_cnt = len(points)

        # Draw points
        img_to_draw = cv2.cvtColor(np.array(img_raw), cv2.COLOR_RGB2BGR)
        all_point.append(points)
        dot_size = 1 if new_width < 1000 else 2 if new_width < 1500 else 3
        for p in points:
            cv2.circle(img_to_draw, (int(p[0]), int(p[1])), dot_size, (0, 0, 255), -1)

        # Annotate
        font = cv2.FONT_HERSHEY_SIMPLEX
        color = (0, 250, 255)
        cv2.putText(img_to_draw, f'predict={predict_cnt}', (50, 50), font, 0.5, color, 2)

        # Save results
        os.makedirs(params["output_dir"], exist_ok=True)
        cv2.imwrite(os.path.join(params["output_dir"], f'pred{i:03}.jpg'), img_to_draw)

        if c == 1:
            file = open(params["predicts_txt_dir"], "w")
        c += 1
        file.write(f"img{i}  pre={predict_cnt}  gr=?\n")

        if c == 201:
            all_point = np.array(all_point)
            os.makedirs(params["predicts_point_dir"], exist_ok=True)
            np.save(os.path.join(params["predicts_point_dir"], 'points.npy'), all_point)

    file.close()


if __name__ == "__main__":
    main()
