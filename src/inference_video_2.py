import torch
import time
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from huggingface_hub import hf_hub_download
from ultralytics import YOLO

_old_load = torch.load
def unsafe_load(*args, **kwargs):
    kwargs["weights_only"] = False
    return _old_load(*args, **kwargs)
torch.load = unsafe_load

from src.models.components.simple_resnet import FaceLandmarkModel
from src.models.wflw_module import WFLWLitModule

def live_cam_inference(checkpoint_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Loading Landmark model on {device}...")
    my_net = FaceLandmarkModel()
    landmark_model = WFLWLitModule.load_from_checkpoint(
        checkpoint_path,
        net=my_net,
    )
    landmark_model.to(device)
    landmark_model.eval()

    print("Loading YOLOv8 Face Detection model...")
    yolo_path = hf_hub_download(repo_id="arnabdhar/YOLOv8-Face-Detection", filename="model.pt")
    yolo_model = YOLO(yolo_path)
    yolo_model.to(device) 

    transform = A.Compose([
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam.")

    print("Press 'q' to exit the camera.")

    frame_count = 0
    fps_display = 0
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        frame_count += 1
        if frame_count == 10:
            end_time = time.time()
            fps_display = 10 / (end_time - start_time)
            start_time = time.time()
            frame_count = 0
            
        cv2.putText(frame, f"FPS: {int(fps_display)}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        results = yolo_model(frame, verbose=False) 

        for box in results[0].boxes.xyxy:
            x1, y1, x2, y2 = map(int, box[:4])

            orig_w = x2 - x1
            orig_h = y2 - y1

            x1 -= int(0.2 * orig_w)
            x2 += int(0.2 * orig_w)
            y1 -= int(0.2 * orig_h)
            y2 += int(0.2 * orig_h)

            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            box_w = x2 - x1
            box_h = y2 - y1

            if box_w <= 20 or box_h <= 20:
                continue

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            crop_bgr = frame[y1:y2, x1:x2]
            
            model_image = cv2.resize(crop_bgr, (256, 256))

            input_tensor = transform(image=model_image)["image"].unsqueeze(0).to(device)

            with torch.no_grad():
                outputs = landmark_model(input_tensor)

            landmarks = outputs.cpu().numpy().squeeze()
            landmarks = landmarks.reshape(-1, 2)

            for (lx, ly) in landmarks:
                real_x = int(lx * box_w) + x1
                real_y = int(ly * box_h) + y1
                
                cv2.circle(frame, (real_x, real_y), 2, (180, 105, 255), -1)

        cv2.imshow("Kaggle Face Landmarks", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    CKPT_PATH = "logs/train/wflw_v0/model.ckpt"
    live_cam_inference(CKPT_PATH)