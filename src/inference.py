import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import functools
import torch.optim
import torch.optim.lr_scheduler
import wandb

from src.models.components.simple_resnet import FaceLandmarkModel
from src.models.wflw_module import WFLWLitModule

api = wandb.Api()

artifact = api.artifact("vuhoangdung3103_yorha/face-landmark-muct/model-8v7nsfcz:v0")

save_dir = "logs/train/wflw_v0"
artifact_dir = artifact.download(root=save_dir)

# torch.serialization.add_safe_globals([
#     functools.partial, 
#     torch.optim.Adam,
#     torch.optim.lr_scheduler.ReduceLROnPlateau 
# ])

# def predict_and_draw(image_path, checkpoint_path):
#     image = np.array(Image.open(image_path).convert("RGB"))
    
#     transform = A.Compose([
#         A.Resize(256, 256),
#         A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
#         ToTensorV2()
#     ])
    
#     input_tensor = transform(image=image)["image"].unsqueeze(0)
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
#     my_net = FaceLandmarkModel()
    
#     model = WFLWLitModule.load_from_checkpoint(checkpoint_path, net=my_net)
#     model.eval()
#     model.to(device)
#     input_tensor = input_tensor.to(device)

#     with torch.no_grad():
#         outputs = model(input_tensor)
        
#     landmarks = outputs.cpu().numpy().squeeze()
#     landmarks = landmarks.reshape(-1, 2)
#     landmarks = landmarks * 256.0
#     img_resized = Image.fromarray(image).resize((256, 256))
    
#     plt.figure(figsize=(8, 8))
#     plt.imshow(img_resized)
    
#     plt.scatter(landmarks[:, 0], landmarks[:, 1], s=15, c='hotpink', marker='o')
    
#     plt.axis('off')
#     plt.show()

# TEST_IMAGE_PATH = "data/inference_data/test_img_7.png" 
# CKPT_PATH = "logs/train/wflw_v0/model.ckpt"

# predict_and_draw(TEST_IMAGE_PATH, CKPT_PATH)