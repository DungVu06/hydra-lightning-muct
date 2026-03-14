import torch

_old_load = torch.load
def unsafe_load(*args, **kwargs):
    kwargs["weights_only"] = False
    return _old_load(*args, **kwargs)
torch.load = unsafe_load

import numpy as np
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

from src.models.components.simple_resnet import FaceLandmarkModel
from src.models.muct_module import MUCTLitModule

def live_cam_inference(checkpoint_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model on {device}...")

    my_net = FaceLandmarkModel()
    model = MUCTLitModule.load_from_checkpoint(checkpoint_path, net=my_net)
    model.to(device)
    model.eval()

    transform = A.Compose([
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ToTensorV2()
    ])

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Không thể mở webcam.")

    box_size = 300

    print("Nhấn phím 'q' để thoát camera.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)

        h, w = frame.shape[:2]

        x1 = int(w / 2 - box_size / 2)
        y1 = int(h / 2 - box_size / 2)
        x2 = x1 + box_size
        y2 = y1 + box_size

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, "Giu khuon mat trong khung", (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        crop_bgr = frame[y1:y2, x1:x2]
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        
        model_image = cv2.resize(crop_rgb, (224, 224))
        input_tensor = transform(image=model_image)["image"].unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(input_tensor)

        landmarks = outputs.cpu().numpy().squeeze().reshape(-1, 2)

        landmarks_real = np.zeros_like(landmarks)
        landmarks_real[:, 0] = landmarks[:, 0] * (box_size / 224.0)
        landmarks_real[:, 1] = landmarks[:, 1] * (box_size / 224.0)

        for (lx, ly) in landmarks_real:
            real_x = int(lx) + x1
            real_y = int(ly) + y1
            
            cv2.circle(frame, (real_x, real_y), 2, (180, 105, 255), -1)

        cv2.imshow("Face Landmarks Live Cam", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    CKPT_PATH = "logs/train/model.ckpt"
    live_cam_inference(CKPT_PATH)
