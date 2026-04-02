import os
import cv2
import torch
import torchvision.transforms as transforms
import numpy as np
import argparse
import time
from PIL import Image
from ultralytics import YOLO

import config
from model import get_model

# load_checkpoint copied from utils.py due to NumPy compatibility issues with matplotlib,
# which is imported in utils.py.
def load_checkpoint(model, model_type='efficientnet'):
    prefix_map = {'simple': 'baseline', 'resnet50': 'resnet50', 'efficientnet': 'efficientnet'}
    prefix = prefix_map.get(model_type, model_type)
    checkpoint_path = os.path.join(config.CHECKPOINT_DIR, f'{prefix}_best_model.pth')
    checkpoint = torch.load(checkpoint_path, map_location=config.DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Checkpoint loaded from {checkpoint_path}")
    return checkpoint['epoch']


CLASS_COLOURS = {
    "military truck":      (0,   60, 220),
    "military tank":       (0,   30, 160),
    "military aircraft":   (20, 130, 255),
    "military helicopter": (0,  200, 255),
    "civilian car":        (60, 200,  60),
    "civilian aircraft":   (220, 170,   0),
}

# Transform for model input
INFER_TRANSFORM = transforms.Compose([
    transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def load_classifier(model_type):
    print(f"Loading {model_type} classifier")
    model = get_model(model_type=model_type)
    load_checkpoint(model, model_type=model_type)
    model.eval()
    print(f"{model_type} classifier loaded.\n")
    return model


def load_yolo():
    print("Loading YOLOv8n detector")
    yolo = YOLO('yolov8n.pt')
    print("YOLOv8n loaded.\n")
    return yolo



# Built-in class IDs for vehicles in YOLOv8
# 2=car, 3=motorcycle, 4=airplane, 5=bus, 7=truck, 8=boat
YOLO_VEHICLE_CLASSES = {2, 3, 4, 5, 7, 8}


def get_yolo_boxes(yolo, frame, yolo_conf_threshold=0.35):
    results = yolo(frame, verbose=False, conf=yolo_conf_threshold, device='mps')
    boxes = []
    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            if cls_id in YOLO_VEHICLE_CLASSES:
                x1, y1, x2, y2 = box.xyxy[0].int().tolist()
                boxes.append((x1, y1, x2, y2))
    return boxes


def classify_crop(classifier, frame_bgr, x1, y1, x2, y2, device):
    if (x2 - x1) < 20 or (y2 - y1) < 20:
        return None, 0.0

    h, w = frame_bgr.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    crop_bgr = frame_bgr[y1:y2, x1:x2]
    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    pil_img  = Image.fromarray(crop_rgb)

    tensor = INFER_TRANSFORM(pil_img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = classifier(tensor)
        probs  = torch.softmax(logits, dim=1)[0]
        conf, idx = torch.max(probs, dim=0)

    return config.CLASSES[idx.item()], conf.item()


# Drawing functions

def draw_detection(frame, class_name, confidence, x1, y1, x2, y2):
    colour = CLASS_COLOURS.get(class_name, (180, 180, 180))
    cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)

    label = f"{class_name}  {confidence*100:.1f}%"
    font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1
    (tw, th), baseline = cv2.getTextSize(label, font, scale, thick)

    bg_y1 = max(y1 - th - baseline - 6, 0)
    bg_y2 = bg_y1 + th + baseline + 4
    cv2.rectangle(frame, (x1, bg_y1), (x1 + tw + 6, bg_y2), colour, cv2.FILLED)
    cv2.putText(frame, label, (x1 + 3, bg_y2 - baseline - 1),
                font, scale, (255, 255, 255), thick, cv2.LINE_AA)


def draw_hud(frame, fps, frame_idx, total_frames, n_detections):
    h, w = frame.shape[:2]
    lines = [f"FPS: {fps:.1f}", f"Frame: {frame_idx}/{total_frames}", f"Vehicles: {n_detections}"]
    font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1
    pad    = 8
    line_h = cv2.getTextSize("A", font, scale, thick)[0][1] + pad
    panel_w = max(cv2.getTextSize(l, font, scale, thick)[0][0] for l in lines) + pad * 2
    panel_h = line_h * len(lines) + pad

    cv2.rectangle(frame, (w - panel_w - 4, 4), (w - 4, 4 + panel_h), (20, 20, 20), cv2.FILLED)
    for i, line in enumerate(lines):
        cv2.putText(frame, line, (w - panel_w, 4 + pad + (i + 1) * line_h - 4),
                    font, scale, (210, 210, 210), thick, cv2.LINE_AA)
        

# Detection pipeline
def run_detection(video_path, model_type='efficientnet', yolo_conf=0.35, cls_conf=0.40):
    yolo       = load_yolo()
    classifier = load_classifier(model_type)
    device     = config.DEVICE
    print(f"Running on: {device}\n")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps    = cap.get(cv2.CAP_PROP_FPS) or 25
    width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Video  : {video_path}")
    print(f"Size   : {width}x{height}  |  FPS: {video_fps:.1f}  |  Frames: {total_frames}")
    print(f"Model  : {model_type}  |  YOLO conf: {yolo_conf}  |  Classifier conf: {cls_conf}\n")
    print("Controls:  Q = quit   SPACE = pause/resume\n")

    window = f"Convoy Detection — YOLO + {model_type}"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, min(width, 1280), min(height, 720))

    frame_idx = 0
    fps       = 0.0
    prev_time = time.time()
    frame     = np.zeros((height, width, 3), dtype=np.uint8)

    while cap.isOpened():

        ret, frame = cap.read()
        if not ret:
            print("End of video.")
            break
        frame_idx += 1

        # Stage 1: YOLO — find all vehicles in the frame
        boxes = get_yolo_boxes(yolo, frame, yolo_conf_threshold=yolo_conf)

        # Stage 2: classifier — identify each crop
        n_detections = 0
        for (x1, y1, x2, y2) in boxes:
            class_name, confidence = classify_crop(classifier, frame, x1, y1, x2, y2, device)
            if class_name is None:
                continue
            n_detections += 1
            if confidence >= cls_conf:
                draw_detection(frame, class_name, confidence, x1, y1, x2, y2)
            else:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (120, 120, 120), 1)
                cv2.putText(frame, f"uncertain {confidence*100:.0f}%",
                            (x1 + 2, max(y1 - 4, 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                            (160, 160, 160), 1, cv2.LINE_AA)

        # HUD
        now       = time.time()
        fps       = 0.9 * fps + 0.1 / max(now - prev_time, 1e-6)
        prev_time = now
        draw_hud(frame, fps, frame_idx, total_frames, n_detections)

        cv2.imshow(window, frame)

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nDone. Processed {frame_idx} frames.")


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Real-time vehicle detection using YOLO + classifier')
    parser.add_argument('--video',     type=str,   required=True,
                        help='Path to input MP4 video')
    parser.add_argument('--model',     type=str,   default='efficientnet',
                        choices=['simple', 'resnet50', 'efficientnet'],
                        help='Classifier model to use (default: efficientnet)')
    parser.add_argument('--yolo_conf', type=float, default=0.35,
                        help='YOLO detection confidence threshold (default: 0.35)')
    parser.add_argument('--cls_conf',  type=float, default=0.40,
                        help='Classifier confidence threshold (default: 0.40)')
    args = parser.parse_args()

    # Run detection with the specified model
    run_detection(
        video_path=args.video,
        model_type=args.model,
        yolo_conf=args.yolo_conf,
        cls_conf=args.cls_conf,
    )